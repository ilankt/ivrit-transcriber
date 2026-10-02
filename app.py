import sys
import os
import tempfile
import shutil
import logging
import subprocess
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QGroupBox, QPushButton, QProgressBar,
                               QHBoxLayout, QFormLayout, QLineEdit,
                               QFileDialog, QMessageBox, QLabel,
                               QTabWidget, QScrollArea)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import QThread, QThreadPool, QTimer, Qt, Signal
from core.settings import get_settings_path, load_settings, save_settings
from engine.ffmpeg_helper import probe_media, extract_audio, split_audio
from core.jobs import Job, Task, JobStatus
from core.job_store import JobStore, save_job, source_signature
from core.filenames import sanitize_output_stem
from core.runtime import determine_engine, get_base_path
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


def _get_icon_path() -> str:
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'ICON.png')


class StartupWorker(QThread):
    """Detect GPUs without blocking the main window."""
    gpu_info_ready = Signal(object)  # dict from detect_all_gpus()

    def run(self):
        from engine.gpu_detector import detect_all_gpus
        gpu_info = detect_all_gpus()
        self.gpu_info_ready.emit(gpu_info)


def _detecting_gpu_info() -> dict:
    return {
        "detecting": True,
        "nvidia_cuda": {"available": False, "info": "Detecting"},
        "amd_vulkan": {"available": False, "info": "Detecting"},
        "apple_metal": {"available": False, "info": "Detecting"},
    }


class FileLoadWorker(QThread):
    """Worker thread for loading and processing media files without blocking the UI."""
    status_updated = Signal(str)
    file_info_ready = Signal(float, bool)  # duration, is_video
    loaded = Signal(object)  # Job object; finished remains QThread's lifecycle signal
    error = Signal(str)

    def __init__(self, file_path, parent=None, job=None):
        super().__init__(parent)
        self.file_path = file_path
        self._temp_dir = None
        self.job = job

    def run(self):
        try:
            if self.job and source_signature(self.file_path) != self.job.source_signature:
                raise ValueError("The source file changed. Add it as a new job to avoid mixing transcripts.")
            self.status_updated.emit("Probing file...")
            duration, media_info = probe_media(self.file_path)
            if duration is None:
                self.error.emit(f"Could not probe file: {media_info}")
                return

            is_video = bool(media_info)
            self.file_info_ready.emit(duration, is_video)

            if self.job:
                self._temp_dir = os.path.join(os.path.dirname(self.job.record_path), "audio")
                os.makedirs(self._temp_dir, exist_ok=True)
            else:
                self._temp_dir = tempfile.mkdtemp(prefix="ivrit_transcriber_job_")
            required_bytes = duration * 16000 * 2 * 2 + 64 * 1024 * 1024
            if shutil.disk_usage(self._temp_dir).free < required_bytes:
                raise OSError(f"Preparing this recording needs approximately {required_bytes / 1024 ** 2:.0f} MB "
                              f"of free cache space in {self._temp_dir}.")

            # Compressed audio containers (including audio-only MP4/M4A) need
            # decoding too: copying AAC/MP3 packets into WAV is not PCM audio.
            self.status_updated.emit("Converting audio to 16 kHz mono WAV...")
            audio_to_split_path = os.path.join(self._temp_dir, "audio.wav")
            err = extract_audio(self.file_path, audio_to_split_path)
            if err:
                raise RuntimeError(f"Audio extraction failed: {err}")

            self.status_updated.emit("Splitting audio into chunks...")
            chunk_paths, err = split_audio(audio_to_split_path, 1, self._temp_dir)
            if err:
                raise RuntimeError(f"Audio splitting failed: {err}")
            if not chunk_paths:
                raise RuntimeError("No audio chunks were produced from this file.")

            self.status_updated.emit("Preparing chunks...")
            tasks = []
            for i, chunk_path in enumerate(chunk_paths):
                chunk_duration, _ = probe_media(chunk_path)
                if chunk_duration is None:
                    chunk_duration = 60
                task = Task(chunk_path, i)
                task.duration = chunk_duration
                tasks.append(task)

            if self.job and self.job.tasks:
                if len(tasks) != len(self.job.tasks) or any(
                        abs(old.duration - new.duration) > 0.1 for old, new in zip(self.job.tasks, tasks)):
                    raise ValueError("Prepared audio no longer matches this job. Add the file as a new job.")
                for old, new in zip(self.job.tasks, tasks):
                    new.status, new.progress = old.status, old.progress
                    new.text, new.srt_segments = old.text, old.srt_segments
            job = self.job or Job(self.file_path, tasks)
            job.tasks = tasks
            job.temp_dir = self._temp_dir
            save_job(job)
            os.remove(audio_to_split_path)
            self._temp_dir = None  # Job now owns the temp dir

            self.loaded.emit(job)

        except Exception as e:
            if self.job:
                self.job.status, self.job.error_message = JobStatus.ERROR, str(e)
                try:
                    save_job(self.job)
                except OSError:
                    logging.exception("Could not save preparation failure")
            self.error.emit(str(e))
            if self._temp_dir and os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir, ignore_errors=True)


class ModelDownloadWorker(QThread):
    """Downloads a model from HuggingFace in a background thread."""
    progress_updated = Signal(int)   # 0-100
    result = Signal(bool, str)     # success, message

    def __init__(self, download_info: dict, model_path: str, parent=None):
        super().__init__(parent)
        self.download_info = download_info
        self.model_path = model_path
        self._canceled = False

    def cancel(self):
        self._canceled = True

    def run(self):
        try:
            from engine.model_downloader import download_ct2_model, download_ggml_file

            def progress_cb(pct):
                self.progress_updated.emit(pct)

            def cancel_check():
                return self._canceled

            if self.download_info["type"] == "ct2":
                download_ct2_model(
                    self.download_info["repo_id"],
                    self.model_path,
                    progress_cb,
                    cancel_check,
                )
            else:
                dest_dir = os.path.dirname(self.model_path)
                download_ggml_file(
                    self.download_info["repo_id"],
                    self.download_info["filename"],
                    dest_dir,
                    progress_cb,
                    cancel_check,
                    os.path.basename(self.model_path),
                )
            if self._canceled:
                raise InterruptedError("Download canceled")
            self.result.emit(True, "")
        except InterruptedError:
            self.result.emit(False, "canceled")
        except Exception as e:
            # Keep existing files and Hub's partial downloads for a later retry.
            self.result.emit(False, str(e))


class MainWindow(QMainWindow):
    def __init__(self, gpu_info: dict):
        super().__init__()
        self.setWindowTitle("Ivrit Transcriber")
        self.resize(800, 600)

        icon_path = _get_icon_path()
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.settings = load_settings()
        self.gpu_info = gpu_info
        self.startup_worker = None
        self.current_job = None
        self.active_worker = None
        self._file_load_worker = None
        self._download_worker = None
        self._closing = False
        self.job_store = JobStore()
        self._close_timer = QTimer(self)
        self._close_timer.setInterval(100)
        self._close_timer.timeout.connect(self.close)
        self.live_panel = None  # built lazily on first tab access
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)

        apply_theme(self.settings.theme)

        self._create_menu_bar()

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 0: File Transcription
        file_tab = QWidget()
        file_tab_layout = QVBoxLayout()
        file_tab.setLayout(file_tab_layout)

        self._create_input_pane(file_tab_layout)
        self._create_output_pane(file_tab_layout)
        self._create_status_pane(file_tab_layout)
        self._create_run_pane(file_tab_layout)

        self.tab_widget.addTab(file_tab, "File Transcription")

        # Tab 1: Live Transcription — placeholder until first click
        self._live_placeholder = QWidget()
        self.tab_widget.addTab(self._live_placeholder, "Live Transcription")
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Tab 2: Settings
        self.settings_panel = SettingsPanel(self.settings, self.gpu_info)
        self.settings_panel.theme_changed.connect(self._on_theme_changed)
        self.settings_panel.download_requested.connect(self._on_download_requested)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(self.settings_panel)
        self.tab_widget.addTab(settings_scroll, "Settings")

        from ui.job_library import JobLibrary
        self.job_library = JobLibrary(self.job_store, self.settings, FileLoadWorker,
                                      self._library_can_start, self.settings_panel.save_settings, self)
        self.tab_widget.addTab(self.job_library, "Jobs && Review")

        self.setCentralWidget(main_widget)

        self._load_settings_to_ui()

    def update_gpu_info(self, gpu_info: dict):
        """Apply background GPU detection results after the window is already visible."""
        self.gpu_info = gpu_info
        self.settings_panel.update_gpu_info(gpu_info)

    def _library_can_start(self):
        return not (self._closing or self.active_worker or self._file_load_worker or self._download_worker
                    or (self.live_panel is not None and self.live_panel.is_busy))

    def _on_tab_changed(self, index: int):
        """Lazily build the Live Transcription panel on first visit."""
        if index == 1 and self.live_panel is None:
            from ui.live_panel import LiveTranscriptionPanel
            self.live_panel = LiveTranscriptionPanel(self.settings)
            self.live_panel.can_start = lambda: self._library_can_start() and not self.job_library.is_busy
            self.live_panel.session_starting.connect(self.settings_panel.save_settings)
            self.live_panel.model_download_needed.connect(self._on_live_model_download_needed)
            layout = QVBoxLayout(self._live_placeholder)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.live_panel)
        elif index == 3:
            self.job_library.refresh()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        help_menu = menu_bar.addMenu("&Help")

        self.select_file_action = QAction("Select File...", self)
        self.select_file_action.triggered.connect(self._select_file)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(self.select_file_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        about_action = QAction("About", self)
        about_action.triggered.connect(lambda: QMessageBox.about(
            self, "About Ivrit Transcriber",
            "Ivrit Transcriber\nLocal Hebrew and English file and live transcription.\n"
            "Powered by Faster-Whisper and whisper.cpp."))
        help_menu.addAction(about_action)

    def _create_input_pane(self, layout):
        input_group_box = QGroupBox("Input File")
        input_layout = QFormLayout()

        file_row_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.select_file_button = QPushButton("Select File...")
        self.select_file_button.clicked.connect(self._select_file)
        file_row_layout.addWidget(self.file_path_edit)
        file_row_layout.addWidget(self.select_file_button)
        input_layout.addRow("File:", file_row_layout)

        self.file_type_label = QLabel("-")
        self.file_duration_label = QLabel("-")
        input_layout.addRow("Type:", self.file_type_label)
        input_layout.addRow("Duration:", self.file_duration_label)

        input_group_box.setLayout(input_layout)
        layout.addWidget(input_group_box)

    def _create_output_pane(self, layout):
        output_group_box = QGroupBox("Output")
        output_layout = QFormLayout()

        output_folder_layout = QHBoxLayout()
        self.output_folder_edit = QLineEdit()
        self.output_folder_button = QPushButton("Browse...")
        self.output_folder_button.clicked.connect(self._browse_output_folder)
        output_folder_layout.addWidget(self.output_folder_edit)
        output_folder_layout.addWidget(self.output_folder_button)
        output_layout.addRow("Output Folder:", output_folder_layout)

        self.output_filename_edit = QLineEdit()
        self.output_filename_edit.setPlaceholderText("Leave empty to use input filename")
        output_layout.addRow("Output Filename:", self.output_filename_edit)

        help_label = QLabel("Filename without extension (e.g., 'my_transcription')")
        help_label.setStyleSheet("color: gray; font-size: 10pt;")
        output_layout.addRow("", help_label)

        output_group_box.setLayout(output_layout)
        layout.addWidget(output_group_box)

    def _create_status_pane(self, layout):
        status_group_box = QGroupBox("Transcription Status")
        status_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)

        self.eta_label = QLabel("")
        self.eta_label.setWordWrap(True)
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
        self.output_filename_edit.setText("")

    def _save_settings_from_ui(self):
        self.settings.output_folder = self.output_folder_edit.text()
        self.settings_panel.save_settings()
        if self.live_panel is not None:
            self.live_panel.save_settings()
        try:
            save_settings(self.settings)
        except OSError:
            logging.exception("Could not save application settings")
            self.status_label.setText("Could not save preferences; settings remain active for this session.")

    def _on_theme_changed(self, theme):
        apply_theme(theme)
        self.settings.theme = theme

    def _select_file(self):
        if self._closing or self.active_worker is not None or self._file_load_worker is not None or self.job_library.is_busy:
            return
        file, _ = QFileDialog.getOpenFileName(
            self, "Select File", "",
            "Media Files (*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a *.mp4 *.mkv *.avi *.mov *.webm *.wmv *.flv *.ts);;Audio Files (*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a);;Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.wmv *.flv *.ts);;All Files (*)"
        )
        if not file:
            return

        if self.current_job:
            if not self.current_job.record_path and self.current_job.temp_dir and os.path.exists(self.current_job.temp_dir):
                shutil.rmtree(self.current_job.temp_dir, ignore_errors=True)
            self.current_job = None

        self.output_filename_edit.setText("")

        self.file_path_edit.setText(file)
        self.file_type_label.setText("Loading...")
        self.file_duration_label.setText("Loading...")
        self.status_label.setText("Processing file...")
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setValue(0)
        self.eta_label.setText("")
        self.select_file_button.setEnabled(False)
        self.select_file_action.setEnabled(False)
        self.start_button.setEnabled(False)

        try:
            job = self.job_store.create(file)
        except OSError as error:
            self._on_file_load_error(str(error))
            self.select_file_button.setEnabled(True)
            self.select_file_action.setEnabled(True)
            return
        self._file_load_worker = FileLoadWorker(file, self, job=job)
        self._file_load_worker.status_updated.connect(self._on_file_load_status)
        self._file_load_worker.file_info_ready.connect(self._on_file_info_ready)
        self._file_load_worker.loaded.connect(self._on_file_loaded)
        self._file_load_worker.finished.connect(self._file_load_finished)
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
        self.start_button.setText("Start Transcription")
        self.job_library.refresh()

    def _file_load_finished(self):
        self._file_load_worker.deleteLater()
        self._file_load_worker = None
        self.select_file_button.setEnabled(not self._closing)
        self.select_file_action.setEnabled(not self._closing)
        self.start_button.setEnabled(self.current_job is not None and not self._closing)

    def _on_file_load_error(self, message):
        if not self._closing:
            QMessageBox.warning(self, "Error", f"Failed to process file: {message}")
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
            else:
                subprocess.run(['xdg-open', output_dir], check=True)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open output folder:\n{e}")

    def _ensure_model_available(self, settings=None) -> bool:
        """Check if the selected model is present; offer to download English models if not.

        Returns True if the model is ready to use, False if the user should abort.
        """
        from engine.model_loader import (
            get_download_required_gb,
            get_model_download_info,
            is_model_available,
            resolve_model_path,
        )

        settings = settings or self.settings
        engine = determine_engine(settings.device)
        base_path = get_base_path()
        models_dir = settings.models_folder or None
        model_path = resolve_model_path(settings.language, engine, base_path, models_dir)

        if is_model_available(model_path, engine):
            return True

        download_info = get_model_download_info(settings.language, engine)
        if not download_info:
            QMessageBox.warning(
                self, "Model Missing",
                f"Required model not found:\n{model_path}\n\n"
                "This is a bundled model. Please reinstall the application."
            )
            return False

        # Check disk space before downloading (~3 GB for English CT2, ~1.5 GB for GGML)
        required_gb = get_download_required_gb(download_info)
        space_check_dir = (models_dir if models_dir and os.path.exists(models_dir) else base_path)
        try:
            stat = shutil.disk_usage(space_check_dir)
            if stat.free < required_gb * 1024 ** 3:
                reply = QMessageBox.question(
                    self, "Low Disk Space",
                    f"Downloading this model requires ~{required_gb:.0f} GB of free disk space.\n"
                    f"Available: {stat.free / 1024 ** 3:.1f} GB\n\nContinue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return False
        except Exception:
            pass

        return self._run_model_download(download_info, model_path)

    def _run_model_download(self, download_info: dict, model_path: str) -> bool:
        """Show a progress dialog and download the model. Returns True on success."""
        from engine.model_loader import get_download_size_label
        from ui.model_download import ModelDownloadDialog

        if self._download_worker is not None or self._closing:
            return False

        repo_id = download_info["repo_id"]
        size_note = get_download_size_label(download_info)

        self._download_worker = ModelDownloadWorker(download_info, model_path, parent=self)
        progress_dialog = ModelDownloadDialog(
            self._download_worker, f"Downloading / verifying model ({size_note})…\n{repo_id}", self)
        self._download_worker.start()
        progress_dialog.exec()
        self._download_worker.wait()
        self._download_worker.deleteLater()
        self._download_worker = None
        progress_dialog.deleteLater()

        if not progress_dialog.success:
            if progress_dialog.message and "canceled" not in progress_dialog.message.lower() and not self._closing:
                QMessageBox.warning(
                    self, "Download Failed",
                    f"Failed to download model:\n{progress_dialog.message}"
                )
            return False

        return not self._closing

    def _on_live_model_download_needed(self):
        """Download the missing live-transcription model then retry starting the session."""
        self._save_settings_from_ui()

        from engine.model_loader import resolve_model_path, get_model_download_info

        models_dir = self.settings.models_folder or None
        model_path = resolve_model_path(
            self.settings.language, "faster-whisper", get_base_path(), models_dir
        )
        download_info = get_model_download_info(self.settings.language, "faster-whisper")
        if download_info and self._run_model_download(download_info, model_path):
            self.settings_panel._refresh_model_status()
            self.live_panel._start_session()

    def _on_download_requested(self):
        """Handle the Download button in the Settings panel."""
        if not self._library_can_start() or self.job_library.is_busy:
            QMessageBox.information(self, "Model In Use", "Finish active jobs or live capture before downloading or repairing models.")
            return
        self._save_settings_from_ui()

        from engine.model_loader import resolve_model_path, get_model_download_info

        device = self.settings.device
        engine = determine_engine(device, self.gpu_info)
        base_path = get_base_path()
        models_dir = self.settings.models_folder or None
        model_path = resolve_model_path(self.settings.language, engine, base_path, models_dir)
        download_info = get_model_download_info(self.settings.language, engine)

        if not download_info:
            QMessageBox.information(
                self, "No Download Available",
                "This model is bundled with the application and cannot be downloaded separately."
            )
            return

        self._run_model_download(download_info, model_path)
        self.settings_panel._refresh_model_status()

    def _start_transcription(self):
        if self._closing or self.active_worker is not None or self._file_load_worker is not None or self.job_library.is_busy:
            return
        if self.live_panel is not None and self.live_panel.is_busy:
            QMessageBox.information(self, "Live Session Active", "Finish live capture before starting file transcription.")
            return
        if self.current_job is None:
            QMessageBox.warning(self, "Start Transcription", "Please select a file first.")
            return

        if self.current_job.status == JobStatus.RUNNING:
            QMessageBox.information(self, "Already Running", "Transcription is already in progress.")
            return

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

        try:
            os.makedirs(output_dir, exist_ok=True)
            with tempfile.TemporaryFile(dir=output_dir) as f:
                f.write(b'test')
        except (IOError, OSError) as e:
            QMessageBox.warning(
                self, "Permission Error",
                f"Cannot write to output folder:\n{output_dir}\n\nError: {e}"
            )
            return

        custom_filename = self.output_filename_edit.text().strip()
        if custom_filename:
            custom_filename = sanitize_output_stem(custom_filename)

            if not custom_filename:
                QMessageBox.warning(self, "Invalid Filename", "Please enter a valid filename.")
                return

            self.current_job.custom_output_filename = custom_filename
        else:
            if self.current_job.settings_snapshot is None:
                existing = [job for job in self.job_store.load() if job.record_path != self.current_job.record_path]
                self.current_job.custom_output_filename = self.job_store.unique_stem(
                    self.current_job.original_file_path, output_dir, existing)

        # Sync settings before model check (language setting must be current)
        self._save_settings_from_ui()
        from core.settings import Settings
        job_settings = (Settings.model_validate(self.current_job.settings_snapshot)
                        if self.current_job.settings_snapshot else self.settings.model_copy(deep=True))

        # Ensure the selected model is available (download if needed)
        if job_settings.diarization_enabled:
            from engine.diarization import dependency_error, model_available, model_path
            error = dependency_error()
            if not error and not model_available(model_path(get_base_path(), job_settings.models_folder)):
                error = "Download the speaker model using Settings > Set Up Speakers."
            if error:
                QMessageBox.warning(self, "Speaker Setup Required", error)
                return
        if not self._ensure_model_available(job_settings):
            return

        total_duration = sum(task.duration for task in self.current_job.tasks)
        estimated_size_mb = 16 + total_duration * 0.001

        try:
            stat = shutil.disk_usage(output_dir)
            available_mb = stat.free / (1024 * 1024)
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
            logging.warning(f"Could not check disk space: {e}")

        if getattr(sys, 'frozen', False):
            app_dir = os.path.dirname(get_settings_path())
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(app_dir, 'logs')
        input_name = os.path.splitext(os.path.basename(self.current_job.original_file_path))[0]
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = os.path.join(log_dir, f'{sanitize_output_stem(input_name, 100)}_{timestamp}.log')
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            logging.basicConfig(handlers=[file_handler], level=logging.INFO,
                                format='%(asctime)s - %(message)s', force=True)
        except OSError:
            logging.exception("Could not create a log file; continuing transcription")

        self.current_job.output_dir = output_dir
        if self.current_job.settings_snapshot is None:
            self.current_job.settings_snapshot = self.settings.model_dump()
        from core.settings import Settings
        job_settings = Settings.model_validate(self.current_job.settings_snapshot)
        try:
            save_job(self.current_job)
        except OSError as error:
            QMessageBox.warning(self, "Cannot Save Job", str(error))
            return

        from core.worker import TranscriptionWorker
        self.active_worker = TranscriptionWorker(self.current_job, job_settings)
        self.active_worker.signals.job_status_updated.connect(self._update_job_status)
        self.active_worker.signals.task_status_updated.connect(self._update_task_status)
        self.active_worker.signals.progress_updated.connect(self._update_progress)
        self.active_worker.signals.stage_progress_updated.connect(self._update_stage_progress)
        self.active_worker.signals.finished.connect(self._worker_finished)

        self.current_job.status = JobStatus.RUNNING
        self.current_job.progress = 0.0
        self.thread_pool.start(self.active_worker)

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.select_file_button.setEnabled(False)
        self.select_file_action.setEnabled(False)

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
            self.status_label.setText("Canceling at the next inference boundary...")

    def _update_job_status(self, status, message):
        if self.current_job:
            self.status_label.setText(f"{status.value}: {message}")

            if status in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELED]:
                self.eta_label.setText("")
                self.pause_button.setEnabled(False)
                self.resume_button.setEnabled(False)
                self.cancel_button.setEnabled(False)
                if self._closing:
                    return
                if status == JobStatus.DONE:
                    QMessageBox.information(self, "Success", "Transcription completed!")
                elif status == JobStatus.ERROR:
                    QMessageBox.warning(self, "Error", f"Transcription failed:\n{message}")

    def _update_task_status(self, task_index, status, message):
        if self.current_job and task_index < len(self.current_job.tasks):
            self.status_label.setText(message)

    def _update_progress(self, task_index, progress):
        if self.current_job and task_index < len(self.current_job.tasks):
            self.current_job.update_progress()

    def _update_stage_progress(self, percent, description):
        self.progress_bar.setFormat("%p% of current step")
        self.progress_bar.setValue(percent)
        self.eta_label.setText(description)

    def _worker_finished(self):
        self.active_worker = None
        done = self.current_job is not None and self.current_job.status == JobStatus.DONE
        self.start_button.setEnabled(not self._closing and not done)
        self.start_button.setText("Retry Unfinished Chunks" if not done else "Completed — see Jobs && Review")
        self.select_file_button.setEnabled(not self._closing)
        self.select_file_action.setEnabled(not self._closing)
        self.job_library.refresh()

    def closeEvent(self, event):
        if not self._closing:
            if self.live_panel is not None and self.live_panel.has_unsaved_session:
                reply = QMessageBox.question(
                    self, "Unsaved Session", "The live transcript could not be saved. Close and discard it?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return
                self.live_panel.discard_unsaved_session()
            self._closing = True
            self._save_settings_from_ui()
            self.centralWidget().setEnabled(False)
            self.select_file_action.setEnabled(False)
            if self.live_panel is not None:
                self.live_panel.stop_session()
            if self.active_worker:
                self.active_worker.cancel()
            self.job_library.stop()
            if self._download_worker:
                self._download_worker.cancel()
        # Keep the event loop alive to receive completion and save live output.
        pending = (
            self.active_worker is not None or self.thread_pool.activeThreadCount() > 0
            or self._file_load_worker is not None or self._download_worker is not None
            or (self.startup_worker is not None and self.startup_worker.isRunning())
            or (self.live_panel is not None and self.live_panel.is_busy)
            or self.job_library.is_busy
        )
        if pending:
            self.status_label.setText("Finishing background work before closing...")
            self._close_timer.start()
            event.ignore()
            return
        self._close_timer.stop()
        if self.live_panel is not None and self.live_panel.has_unsaved_session:
            self._closing = False
            self.centralWidget().setEnabled(True)
            self.select_file_action.setEnabled(True)
            self.select_file_button.setEnabled(True)
            self.start_button.setEnabled(self.current_job is not None)
            self.status_label.setText("Save the live session before closing, or close again to discard it.")
            event.ignore()
            return
        if self.current_job and self.current_job.temp_dir and not self.current_job.record_path:
            shutil.rmtree(self.current_job.temp_dir, ignore_errors=True)
        logging.shutdown()
        event.accept()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--smoke-test-report')
    startup_args, _ = parser.parse_known_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler()],
    )
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow(_detecting_gpu_info())
    if startup_args.smoke_test_report:
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()

    if startup_args.smoke_test_report:
        from core.smoke_test import schedule_smoke_test
        schedule_smoke_test(app, window, FileLoadWorker, startup_args.smoke_test_report)
    else:
        startup_worker = StartupWorker(window)
        window.startup_worker = startup_worker
        startup_worker.gpu_info_ready.connect(window.update_gpu_info)
        startup_worker.finished.connect(lambda: setattr(window, "startup_worker", None))
        startup_worker.start()

    sys.exit(app.exec())
