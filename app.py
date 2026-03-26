import sys
import os
import tempfile
import shutil
import logging
import subprocess
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QGroupBox, QPushButton, QProgressBar,
                               QHBoxLayout, QFormLayout, QLineEdit, QComboBox,
                               QCheckBox, QSpinBox, QFileDialog, QMessageBox, QLabel,
                               QTabWidget)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import QThread, QThreadPool, QTimer, Qt, Signal
from core.settings import load_settings, save_settings, Settings
from engine.ffmpeg_helper import probe_media, extract_audio, split_audio
from core.jobs import Job, Task, JobStatus, TaskStatus
from core.worker import TranscriptionWorker
from engine.gpu_detector import detect_all_gpus
from ui.live_panel import LiveTranscriptionPanel
from ui.settings_panel import SettingsPanel


def apply_theme(theme: str):
    """Apply the specified theme to the application."""
    app = QApplication.instance()
    if app is None:
        return

    hints = app.styleHints()
    if theme == "dark":
        hints.setColorScheme(Qt.ColorScheme.Dark)
    elif theme == "light":
        hints.setColorScheme(Qt.ColorScheme.Light)
    else:  # system
        hints.setColorScheme(Qt.ColorScheme.Unknown)


def format_duration(seconds):
    """Format duration in seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


class FileLoadWorker(QThread):
    """Worker thread for loading and processing media files without blocking the UI."""
    status_updated = Signal(str)
    file_info_ready = Signal(float, bool)  # duration, is_video
    finished = Signal(object)  # Job object
    error = Signal(str)

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self._temp_dir = None

    def run(self):
        try:
            self.status_updated.emit("Probing file...")
            duration, media_info = probe_media(self.file_path)
            if duration is None:
                self.error.emit(f"Could not probe file: {media_info}")
                return

            is_video = bool(media_info)
            self.file_info_ready.emit(duration, is_video)

            self._temp_dir = tempfile.mkdtemp(prefix="ivrit_transcriber_job_")

            audio_to_split_path = self.file_path

            if is_video:
                self.status_updated.emit("Extracting audio from video...")
                extracted_audio_path = os.path.join(
                    self._temp_dir, os.path.basename(self.file_path) + ".wav"
                )
                err = extract_audio(self.file_path, extracted_audio_path)
                if err:
                    raise RuntimeError(f"Audio extraction failed: {err}")
                audio_to_split_path = extracted_audio_path

            self.status_updated.emit("Splitting audio into chunks...")
            chunk_paths, err = split_audio(audio_to_split_path, 1, self._temp_dir)
            if err:
                raise RuntimeError(f"Audio splitting failed: {err}")

            self.status_updated.emit("Preparing chunks...")
            tasks = []
            for i, chunk_path in enumerate(chunk_paths):
                chunk_duration, _ = probe_media(chunk_path)
                if chunk_duration is None:
                    chunk_duration = 60
                task = Task(self.file_path, chunk_path, i)
                task.duration = chunk_duration
                tasks.append(task)

            job = Job(self.file_path, tasks)
            job.temp_dir = self._temp_dir
            self._temp_dir = None  # Job now owns the temp dir

            self.finished.emit(job)

        except Exception as e:
            self.error.emit(str(e))
            if self._temp_dir and os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ivrit Transcriber")
        self.resize(800, 600)

        # Set window icon
        icon_path = os.path.join(
            getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))),
            'ICON.png'
        )
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.settings = load_settings()
        self.current_job = None  # Single Job object or None
        self.active_worker = None  # Single worker or None
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)  # Process jobs sequentially

        # Detect GPU availability
        self.gpu_info = detect_all_gpus()

        # Apply saved theme
        apply_theme(self.settings.theme)

        # Menu bar
        self._create_menu_bar()

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Tab 1: File Transcription ---
        file_tab = QWidget()
        file_tab_layout = QVBoxLayout()
        file_tab.setLayout(file_tab_layout)

        self._create_input_pane(file_tab_layout)
        self._create_output_pane(file_tab_layout)
        self._create_status_pane(file_tab_layout)
        self._create_run_pane(file_tab_layout)

        self.tab_widget.addTab(file_tab, "File Transcription")

        # --- Tab 2: Live Transcription ---
        self.live_panel = LiveTranscriptionPanel(self.settings)
        self.tab_widget.addTab(self.live_panel, "Live Transcription")

        # --- Tab 3: Settings ---
        self.settings_panel = SettingsPanel(self.settings, self.gpu_info)
        self.settings_panel.theme_changed.connect(self._on_theme_changed)
        self.tab_widget.addTab(self.settings_panel, "Settings")

        self.setCentralWidget(main_widget)

        self._load_settings_to_ui()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        help_menu = menu_bar.addMenu("&Help")

        select_file_action = QAction("Select File...", self)
        select_file_action.triggered.connect(self._select_file)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(select_file_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        about_action = QAction("About", self)
        help_menu.addAction(about_action)

    def _create_input_pane(self, layout):
        input_group_box = QGroupBox("Input File")
        input_layout = QFormLayout()

        # File path display with select button
        file_row_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.select_file_button = QPushButton("Select File...")
        self.select_file_button.clicked.connect(self._select_file)
        file_row_layout.addWidget(self.file_path_edit)
        file_row_layout.addWidget(self.select_file_button)
        input_layout.addRow("File:", file_row_layout)

        # File info
        self.file_type_label = QLabel("-")
        self.file_duration_label = QLabel("-")
        input_layout.addRow("Type:", self.file_type_label)
        input_layout.addRow("Duration:", self.file_duration_label)

        input_group_box.setLayout(input_layout)
        layout.addWidget(input_group_box)

    def _create_output_pane(self, layout):
        output_group_box = QGroupBox("Output")
        output_layout = QFormLayout()

        # Output folder
        output_folder_layout = QHBoxLayout()
        self.output_folder_edit = QLineEdit()
        self.output_folder_button = QPushButton("Browse...")
        self.output_folder_button.clicked.connect(self._browse_output_folder)
        output_folder_layout.addWidget(self.output_folder_edit)
        output_folder_layout.addWidget(self.output_folder_button)
        output_layout.addRow("Output Folder:", output_folder_layout)

        # Output filename (custom)
        self.output_filename_edit = QLineEdit()
        self.output_filename_edit.setPlaceholderText("Leave empty to use input filename")
        output_layout.addRow("Output Filename:", self.output_filename_edit)

        # Help text
        help_label = QLabel("Filename without extension (e.g., 'my_transcription')")
        help_label.setStyleSheet("color: gray; font-size: 10pt;")
        output_layout.addRow("", help_label)

        output_group_box.setLayout(output_layout)
        layout.addWidget(output_group_box)

    def _create_status_pane(self, layout):
        status_group_box = QGroupBox("Transcription Status")
        status_layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        # Status text
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)

        # ETA
        self.eta_label = QLabel("")
        status_layout.addWidget(self.eta_label)

        status_group_box.setLayout(status_layout)
        layout.addWidget(status_group_box)

    def _create_run_pane(self, layout):
        run_group_box = QGroupBox("Run")
        run_layout = QHBoxLayout()
        run_group_box.setLayout(run_layout)

        self.start_button = QPushButton("Start Transcription")
        self.start_button.clicked.connect(self._start_transcription)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._pause_transcription)
        self.pause_button.setEnabled(False)

        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self._resume_transcription)
        self.resume_button.setEnabled(False)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_transcription)
        self.cancel_button.setEnabled(False)

        self.open_folder_button = QPushButton("Open Output Folder")
        self.open_folder_button.clicked.connect(self._open_output_folder)

        run_layout.addWidget(self.start_button)
        run_layout.addWidget(self.pause_button)
        run_layout.addWidget(self.resume_button)
        run_layout.addWidget(self.cancel_button)
        run_layout.addWidget(self.open_folder_button)

        layout.addWidget(run_group_box)

    def _load_settings_to_ui(self):
        self.output_folder_edit.setText(self.settings.output_folder or "")
        # Do NOT load default_output_filename - always start empty per user preference
        self.output_filename_edit.setText("")

    def _save_settings_from_ui(self):
        self.settings.output_folder = self.output_folder_edit.text()
        self.settings_panel.save_settings()
        self.live_panel.save_settings()
        save_settings(self.settings)

    def _on_theme_changed(self, theme):
        apply_theme(theme)
        self.settings.theme = theme

    def _select_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select File", "",
            "Media Files (*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a *.mp4 *.mkv *.avi *.mov *.webm *.wmv *.flv *.ts);;Audio Files (*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a);;Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.wmv *.flv *.ts);;All Files (*)"
        )
        if not file:
            return

        # Clear any existing job
        if self.current_job:
            if hasattr(self.current_job, 'temp_dir') and os.path.exists(self.current_job.temp_dir):
                shutil.rmtree(self.current_job.temp_dir)
            self.current_job = None

        # Clear custom filename when selecting new file
        self.output_filename_edit.setText("")

        # Update UI to show loading state
        self.file_path_edit.setText(file)
        self.file_type_label.setText("Loading...")
        self.file_duration_label.setText("Loading...")
        self.status_label.setText("Processing file...")
        self.progress_bar.setValue(0)
        self.eta_label.setText("")
        self.select_file_button.setEnabled(False)
        self.start_button.setEnabled(False)

        # Start background worker to process the file
        self._file_load_worker = FileLoadWorker(file, self)
        self._file_load_worker.status_updated.connect(self._on_file_load_status)
        self._file_load_worker.file_info_ready.connect(self._on_file_info_ready)
        self._file_load_worker.finished.connect(self._on_file_loaded)
        self._file_load_worker.error.connect(self._on_file_load_error)
        self._file_load_worker.start()

    def _on_file_load_status(self, message):
        self.status_label.setText(message)

    def _on_file_info_ready(self, duration, is_video):
        self.file_type_label.setText("Video" if is_video else "Audio")
        self.file_duration_label.setText(format_duration(duration))

    def _on_file_loaded(self, job):
        self.current_job = job
        self.status_label.setText("Ready to transcribe")
        self.select_file_button.setEnabled(True)
        self.start_button.setEnabled(True)

    def _on_file_load_error(self, message):
        QMessageBox.warning(self, "Error", f"Failed to process file: {message}")
        self.file_path_edit.setText("")
        self.file_type_label.setText("-")
        self.file_duration_label.setText("-")
        self.status_label.setText("Error processing file")
        self.select_file_button.setEnabled(True)
        self.current_job = None

    def _browse_output_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_folder_edit.setText(dir_path)

    def _open_output_folder(self):
        """Open the output folder in the system file explorer"""
        output_dir = self.output_folder_edit.text()
        if not output_dir:
            QMessageBox.information(self, "Open Output Folder", "Please select an output folder first.")
            return

        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "Open Output Folder", f"Output folder does not exist:\n{output_dir}")
            return

        try:
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':
                subprocess.run(['open', output_dir], check=True)
            else:  # Linux and other Unix-like systems
                subprocess.run(['xdg-open', output_dir], check=True)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open output folder:\n{e}")

    def _start_transcription(self):
        if self.current_job is None:
            QMessageBox.warning(self, "Start Transcription", "Please select a file first.")
            return

        if self.current_job.status == JobStatus.RUNNING:
            QMessageBox.information(self, "Already Running", "Transcription is already in progress.")
            return

        # Verify chunk files still exist (they may be gone if a previous run failed and cleaned up)
        for task in self.current_job.tasks:
            if not os.path.exists(task.chunk_path):
                QMessageBox.warning(
                    self, "Start Transcription",
                    "Temporary audio files no longer exist.\nPlease re-select the input file."
                )
                self.current_job = None
                self.status_label.setText("Ready")
                self.progress_bar.setValue(0)
                return

        output_dir = self.output_folder_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "Start Transcription", "Please select an output folder.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Validate output directory is writable
        test_file = os.path.join(output_dir, '.ivrit_write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (IOError, OSError) as e:
            QMessageBox.warning(
                self, "Permission Error",
                f"Cannot write to output folder:\n{output_dir}\n\nError: {e}"
            )
            return

        # Handle custom filename (if provided)
        custom_filename = self.output_filename_edit.text().strip()
        if custom_filename:
            # Sanitize: keep alphanumeric, spaces, hyphens, underscores
            custom_filename = "".join(c for c in custom_filename if c.isalnum() or c in (' ', '-', '_'))
            # Remove multiple consecutive spaces
            custom_filename = " ".join(custom_filename.split())
            # Limit length
            custom_filename = custom_filename[:200]

            if not custom_filename:
                QMessageBox.warning(self, "Invalid Filename", "Please enter a valid filename.")
                return

            self.current_job.custom_output_filename = custom_filename
        else:
            self.current_job.custom_output_filename = None

        # Validate sufficient disk space
        total_duration = sum(task.duration for task in self.current_job.tasks)
        # Rough estimate: 0.5 MB per second of audio for intermediate files
        estimated_size_mb = total_duration * 0.5

        import shutil as shutil_disk
        try:
            stat = shutil_disk.disk_usage(output_dir)
            available_mb = stat.free / (1024 * 1024)

            # Require 1.5x the estimated size for safety margin
            required_mb = estimated_size_mb * 1.5

            if available_mb < required_mb:
                reply = QMessageBox.question(
                    self, "Low Disk Space",
                    f"Warning: Low disk space detected.\n\n"
                    f"Estimated space needed: {estimated_size_mb:.1f} MB\n"
                    f"Available space: {available_mb:.1f} MB\n"
                    f"Recommended: {required_mb:.1f} MB\n\n"
                    f"Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
        except Exception as e:
            # If we can't check disk space, just log and continue
            logging.warning(f"Could not check disk space: {e}")

        # Configure logging - save in app's logs folder
        if getattr(sys, 'frozen', False):
            app_dir = os.path.dirname(sys.executable)
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(app_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        input_name = os.path.splitext(os.path.basename(self.current_job.original_file_path))[0]
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = os.path.join(log_dir, f'{input_name}_{timestamp}.log')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

        self.current_job.output_dir = output_dir

        # Sync UI settings to self.settings before passing to worker
        self._save_settings_from_ui()

        # Create worker (no job_index parameter)
        self.active_worker = TranscriptionWorker(self.current_job, self.settings)
        self.active_worker.signals.job_status_updated.connect(self._update_job_status)
        self.active_worker.signals.task_status_updated.connect(self._update_task_status)
        self.active_worker.signals.progress_updated.connect(self._update_progress)
        self.active_worker.signals.eta_updated.connect(self._update_eta)
        self.active_worker.signals.finished.connect(self._worker_finished)

        self.thread_pool.start(self.active_worker)

        # Update button states
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.select_file_button.setEnabled(False)

    def _pause_transcription(self):
        if self.active_worker:
            self.active_worker.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)

    def _resume_transcription(self):
        if self.active_worker:
            self.active_worker.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)

    def _cancel_transcription(self):
        if self.active_worker:
            self.active_worker.cancel()
            self.cancel_button.setEnabled(False)

    def _update_job_status(self, status, message):
        if self.current_job:
            self.current_job.status = status
            self.current_job.error_message = message if status == JobStatus.ERROR else None
            self.status_label.setText(f"{status.value}: {message}")

            if status in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELED]:
                self.start_button.setEnabled(True)
                self.pause_button.setEnabled(False)
                self.resume_button.setEnabled(False)
                self.cancel_button.setEnabled(False)
                self.select_file_button.setEnabled(True)

                if status == JobStatus.DONE:
                    QMessageBox.information(self, "Success", "Transcription completed!")
                elif status == JobStatus.ERROR:
                    QMessageBox.warning(self, "Error", f"Transcription failed:\n{message}")

    def _update_task_status(self, task_index, status, message):
        if self.current_job and task_index < len(self.current_job.tasks):
            self.current_job.tasks[task_index].status = status
            self.current_job.tasks[task_index].error_message = message if status == TaskStatus.ERROR else None
            # Update status label with current task info
            self.status_label.setText(message)

    def _update_progress(self, task_index, progress):
        if self.current_job and task_index < len(self.current_job.tasks):
            self.current_job.tasks[task_index].progress = progress
            self.current_job.update_progress()
            # Update progress bar with overall job progress
            overall_progress_percent = int(self.current_job.progress * 100)
            self.progress_bar.setValue(overall_progress_percent)

    def _update_eta(self, eta_string):
        self.eta_label.setText(eta_string)

    def _worker_finished(self):
        # Worker has finished, clear active worker
        self.active_worker = None
        # Note: Temp directory cleanup is now handled exclusively in worker's finally block

    def closeEvent(self, event):
        self._save_settings_from_ui()
        # Stop live transcription session if active
        self.live_panel.stop_session()
        # Wait for file load worker if running
        if hasattr(self, '_file_load_worker') and self._file_load_worker is not None:
            self._file_load_worker.wait(5000)
        # Cancel active worker before waiting for the thread pool to finish
        if self.active_worker:
            self.active_worker.cancel()
        self.thread_pool.waitForDone()  # Wait for all threads to finish before exiting
        logging.shutdown()
        QApplication.instance().quit()
        event.accept()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler()],
    )
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
