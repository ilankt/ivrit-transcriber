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
from PySide6.QtGui import QAction
from PySide6.QtCore import QThreadPool, QTimer
from core.settings import load_settings, save_settings, Settings
from engine.ffmpeg_helper import probe_media, extract_audio, split_audio
from core.jobs import Job, Task, JobStatus, TaskStatus
from core.worker import TranscriptionWorker
from engine.gpu_detector import detect_all_gpus
from ui.live_panel import LiveTranscriptionPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ivrit Transcriber")
        self.resize(800, 600)

        self.settings = load_settings()
        self.current_job = None  # Single Job object or None
        self.active_worker = None  # Single worker or None
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)  # Process jobs sequentially

        # Detect GPU availability
        self.gpu_info = detect_all_gpus()
        self.nvidia_available = self.gpu_info["nvidia_cuda"]["available"]
        self.amd_available = self.gpu_info["amd_vulkan"]["available"]

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

        # Input Pane
        self._create_input_pane(file_tab_layout)

        # Status Pane
        self._create_status_pane(file_tab_layout)

        # Run Pane
        self._create_run_pane(file_tab_layout)

        self.tab_widget.addTab(file_tab, "File Transcription")

        # --- Tab 2: Live Transcription ---
        self.live_panel = LiveTranscriptionPanel(self.settings)
        self.tab_widget.addTab(self.live_panel, "Live Transcription")

        # Options Pane (shared, below tabs)
        self._create_options_pane(main_layout)

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

    def _create_options_pane(self, layout):
        options_group_box = QGroupBox("Options")
        options_layout = QFormLayout()
        options_group_box.setLayout(options_layout)

        # Model Type
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Fast", "Accurate"])
        options_layout.addRow("Model:", self.model_type_combo)

        # VAD
        self.vad_checkbox = QCheckBox("Enable VAD (Voice Activity Detection)")
        self.vad_checkbox.setChecked(True)
        options_layout.addRow(self.vad_checkbox)

        # Device selection (CPU/GPU)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (Try GPU, fallback to CPU)", "auto")
        self.device_combo.addItem("CPU Only", "cpu")

        if self.nvidia_available:
            self.device_combo.addItem(
                f"NVIDIA GPU ({self.gpu_info['nvidia_cuda']['info']})", "nvidia"
            )
        if self.amd_available:
            self.device_combo.addItem(
                f"AMD GPU ({self.gpu_info['amd_vulkan']['info']})", "amd"
            )

        # If GPU was previously selected but is no longer available, reset to auto
        if self.settings.device == "nvidia" and not self.nvidia_available:
            self.settings.device = "auto"
        if self.settings.device == "amd" and not self.amd_available:
            self.settings.device = "auto"

        options_layout.addRow("Device:", self.device_combo)

        # Output format
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("SRT (Subtitles)", "srt")
        self.output_format_combo.addItem("TXT (Plain Text)", "txt")
        self.output_format_combo.addItem("Both (SRT + TXT)", "both")
        options_layout.addRow("Output Format:", self.output_format_combo)

        # Output folder
        output_folder_layout = QHBoxLayout()
        self.output_folder_edit = QLineEdit()
        self.output_folder_button = QPushButton("Browse...")
        self.output_folder_button.clicked.connect(self._browse_output_folder)
        output_folder_layout.addWidget(self.output_folder_edit)
        output_folder_layout.addWidget(self.output_folder_button)
        options_layout.addRow("Output Folder:", output_folder_layout)

        # Output filename (custom)
        self.output_filename_edit = QLineEdit()
        self.output_filename_edit.setPlaceholderText("Leave empty to use input filename")
        options_layout.addRow("Output Filename:", self.output_filename_edit)

        # Help text
        help_label = QLabel("Filename without extension (e.g., 'my_transcription')")
        help_label.setStyleSheet("color: gray; font-size: 10pt;")
        options_layout.addRow("", help_label)

        layout.addWidget(options_group_box)

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
        self.model_type_combo.setCurrentText(self.settings.model_type)
        self.vad_checkbox.setChecked(self.settings.vad_enabled)
        self.output_folder_edit.setText(self.settings.output_folder or "")

        # Do NOT load default_output_filename - always start empty per user preference
        self.output_filename_edit.setText("")

        # Load device setting
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == self.settings.device:
                self.device_combo.setCurrentIndex(i)
                break

        # Load output format setting
        for i in range(self.output_format_combo.count()):
            if self.output_format_combo.itemData(i) == self.settings.output_format:
                self.output_format_combo.setCurrentIndex(i)
                break

    def _save_settings_from_ui(self):
        self.settings.model_type = self.model_type_combo.currentText()
        self.settings.vad_enabled = self.vad_checkbox.isChecked()
        self.settings.output_folder = self.output_folder_edit.text()
        self.settings.device = self.device_combo.currentData()
        self.settings.output_format = self.output_format_combo.currentData()
        self.live_panel.save_settings()
        save_settings(self.settings)

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

        # Reset UI
        self.status_label.setText("Processing file...")
        self.progress_bar.setValue(0)
        self.eta_label.setText("")

        temp_job_dir = None

        try:
            duration, media_info = probe_media(file)
            if duration is None:
                QMessageBox.warning(self, "Error", f"Could not probe file {os.path.basename(file)}: {media_info}")
                self.file_path_edit.setText("")
                self.file_type_label.setText("-")
                self.file_duration_label.setText("-")
                self.status_label.setText("Error: Could not probe file")
                return

            temp_job_dir = tempfile.mkdtemp(prefix="ivrit_transcriber_job_")

            # Update UI with file info
            self.file_path_edit.setText(file)
            self.file_type_label.setText("Video" if media_info else "Audio")
            self.file_duration_label.setText(f"{duration:.2f}s")

            audio_to_split_path = file

            if media_info:  # It's a video, extract audio
                self.status_label.setText("Extracting audio...")
                extracted_audio_path = os.path.join(temp_job_dir, os.path.basename(file) + ".wav")
                error = extract_audio(file, extracted_audio_path)
                if error:
                    raise RuntimeError(f"Audio extraction failed: {error}")
                audio_to_split_path = extracted_audio_path

            # Split audio into chunks
            self.status_label.setText("Splitting audio...")
            chunk_paths, error = split_audio(audio_to_split_path, 1, temp_job_dir)  # Hardcoded 1 minute
            if error:
                raise RuntimeError(f"Audio splitting failed: {error}")

            tasks = []
            for i, chunk_path in enumerate(chunk_paths):
                chunk_duration, _ = probe_media(chunk_path)
                if chunk_duration is None:
                    chunk_duration = 60  # Default to 1 minute as a fallback
                task = Task(file, chunk_path, i)
                task.duration = chunk_duration
                tasks.append(task)

            # Create job
            job = Job(file, tasks)
            job.temp_dir = temp_job_dir
            self.current_job = job

            self.status_label.setText("Ready to transcribe")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to process {os.path.basename(file)}: {str(e)}")
            # Clean up temp directory if it was created
            if temp_job_dir and os.path.exists(temp_job_dir):
                shutil.rmtree(temp_job_dir)
            self.file_path_edit.setText("")
            self.file_type_label.setText("-")
            self.file_duration_label.setText("-")
            self.status_label.setText("Error processing file")
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
        # Cancel active worker before waiting for the thread pool to finish
        if self.active_worker:
            self.active_worker.cancel()
        self.thread_pool.waitForDone()  # Wait for all threads to finish before exiting
        logging.shutdown()
        QApplication.instance().quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
