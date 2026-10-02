"""Persistent job queue and entry point for transcript review."""
from pathlib import Path

from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QTableWidget, QTableWidgetItem, QAbstractItemView,
                               QHeaderView, QFileDialog, QLabel, QMessageBox)

from core.job_store import save_job, release_audio
from core.jobs import JobStatus, TaskStatus
from core.settings import Settings
from core.worker import TranscriptionWorker


class JobLibrary(QWidget):
    def __init__(self, store, settings, loader_type, can_start, sync_settings, parent=None):
        super().__init__(parent)
        self.store, self.settings = store, settings
        self.loader_type, self.can_start, self.sync_settings = loader_type, can_start, sync_settings
        self.jobs, self.queue = [], []
        self.worker = self.loader = self.active_job = None
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.stopping = False
        layout = QVBoxLayout(self)
        help_text = QLabel("Resume interrupted jobs or retry failed chunks. Completed chunks are kept. "
                           "Files use the settings selected when added. Select a completed or partial job to review its transcript.")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Chunks saved", "Language", "Output"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        self.add_button = QPushButton("Add Files…")
        self.run_button = QPushButton("Run Queue")
        self.resume_button = QPushButton("Resume / Retry Selected")
        self.stop_button = QPushButton("Stop Queue")
        self.stop_button.setEnabled(False)
        for button in (self.add_button, self.run_button, self.resume_button, self.stop_button):
            row.addWidget(button)
        layout.addLayout(row)
        actions = QHBoxLayout()
        self.review_button = QPushButton("Review Transcript…")
        self.cache_button = QPushButton("Release Selected Audio Cache")
        refresh = QPushButton("Refresh")
        for button in (self.review_button, self.cache_button, refresh):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.add_button.clicked.connect(self.add_files)
        self.run_button.clicked.connect(lambda: self.start_jobs([job for job in self.jobs if job.status == JobStatus.QUEUED]))
        self.resume_button.clicked.connect(lambda: self.start_jobs(self.selected_jobs()))
        self.stop_button.clicked.connect(self.stop)
        self.review_button.clicked.connect(self.review)
        self.cache_button.clicked.connect(self.release_selected_audio)
        refresh.clicked.connect(self.refresh)
        self.table.itemDoubleClicked.connect(self.review)
        self.refresh()

    @property
    def is_busy(self):
        return self.worker is not None or self.loader is not None or bool(self.queue) or self.pool.activeThreadCount() > 0

    def selected_jobs(self):
        return [self.jobs[index.row()] for index in self.table.selectionModel().selectedRows()]

    def refresh(self):
        if self.is_busy:
            return
        selected = {job.record_path for job in self.selected_jobs()}
        self.jobs = self.store.load()
        self._render(selected)
        if self.store.errors:
            self.status.setText(f"{len(self.store.errors)} unreadable job record(s); original records were retained.")

    def _render(self, selected=()):
        self.table.setRowCount(len(self.jobs))
        for row, job in enumerate(self.jobs):
            completed = sum(task.status == TaskStatus.DONE for task in job.tasks)
            values = [Path(job.original_file_path).name, job.status.value,
                      f"{completed}/{len(job.tasks)}", (job.settings_snapshot or {}).get("language", "—"),
                      str(Path(job.output_dir) / (job.custom_output_filename or Path(job.original_file_path).stem))
                      if job.output_dir else "Choose on resume"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(job.error_message or job.original_file_path)
                self.table.setItem(row, col, item)
            if job.record_path in selected:
                self.table.selectRow(row)

    def add_files(self):
        if self.is_busy:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Add Media Files", "",
            "Media (*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a *.mp4 *.mkv *.avi *.mov *.webm *.wmv *.flv *.ts);;All Files (*)")
        if not files:
            return
        output = QFileDialog.getExistingDirectory(self, "Batch Output Folder", self.settings.output_folder or "")
        if not output:
            return
        self.sync_settings()
        self.refresh()
        for source in files:
            try:
                stem = self.store.unique_stem(source, output, self.jobs)
                job = self.store.create(source, self.settings, output, output_stem=stem)
                self.jobs.append(job)
            except OSError as error:
                QMessageBox.warning(self, "Could Not Add File", f"{source}\n{error}")
        self._render()

    def start_jobs(self, jobs):
        if self.is_busy:
            return
        if not self.can_start():
            self.status.setText("Finish the current file, live session, or model download before running the queue.")
            return
        self.sync_settings()
        candidates = [job for job in jobs if job.status != JobStatus.DONE]
        if not candidates:
            self.status.setText("Select unfinished jobs, or add files to the queue.")
            return
        output = None
        if any(not job.output_dir for job in candidates):
            output = QFileDialog.getExistingDirectory(self, "Output Folder", self.settings.output_folder or "")
            if not output:
                return
        try:
            for job in candidates:
                if not job.output_dir:
                    job.output_dir = output
                    job.custom_output_filename = self.store.unique_stem(job.original_file_path, output,
                                                                        [j for j in self.jobs if j is not job])
                if job.settings_snapshot is None:
                    job.settings_snapshot = self.settings.model_dump()
                Settings.model_validate(job.settings_snapshot)
                save_job(job)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could Not Start Queue", str(error))
            return
        self.queue = list(candidates)
        self.stopping = False
        self._set_running(True)
        self._next()

    def _set_running(self, running):
        for button in (self.add_button, self.run_button, self.resume_button, self.review_button, self.cache_button):
            button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _next(self):
        if self.stopping or not self.queue:
            self.queue.clear()
            self.active_job = None
            self._set_running(False)
            self.status.setText("Queue stopped. Unfinished jobs can be resumed." if self.stopping else
                                "Queue finished. Check each file's status for errors.")
            self._render()
            return
        self.active_job = self.queue.pop(0)
        job = self.active_job
        # Completed chunks remain available in the record even after cache cleanup.
        missing_audio = not job.tasks or any(not Path(task.chunk_path).is_file() for task in job.tasks
                                             if task.status != TaskStatus.DONE)
        if missing_audio:
            self.loader = self.loader_type(job.original_file_path, self, job=job)
            self.loader.status_updated.connect(self.status.setText)
            self.loader.error.connect(self._load_error)
            self.loader.loaded.connect(self._loaded)
            self.loader.finished.connect(self._load_finished)
            self._prepared = False
            self.loader.start()
        else:
            self._transcribe()

    def _loaded(self, job):
        self._prepared = True

    def _load_error(self, message):
        self.status.setText(message)
        self.active_job.status, self.active_job.error_message = JobStatus.ERROR, message

    def _load_finished(self):
        self.loader.deleteLater()
        self.loader = None
        if self._prepared and not self.stopping:
            self._transcribe()
        else:
            self._render()
            QTimer.singleShot(0, self._next)

    def _transcribe(self):
        self.worker = TranscriptionWorker(self.active_job, Settings.model_validate(self.active_job.settings_snapshot))
        self.worker.signals.job_status_updated.connect(self._status)
        self.worker.signals.stage_progress_updated.connect(
            lambda percent, message: self.status.setText(f"{Path(self.active_job.original_file_path).name}: {message}"))
        self.worker.signals.finished.connect(self._finished)
        self.pool.start(self.worker)

    def _status(self, status, message):
        self.status.setText(f"{Path(self.active_job.original_file_path).name}: {message}")
        self._render()

    def _finished(self):
        self.worker = None
        self._render()
        QTimer.singleShot(0, self._next)

    def stop(self):
        self.stopping = True
        self.queue.clear()
        if self.worker:
            self.worker.cancel()
        self.status.setText("Stopping after current inference or file preparation finishes…")

    def review(self, *_):
        if self.is_busy or not self.can_start():
            return
        jobs = self.selected_jobs()
        if len(jobs) != 1:
            self.status.setText("Select one job to review.")
            return
        job = jobs[0]
        if not any(task.status == TaskStatus.DONE for task in job.tasks):
            self.status.setText("This job has no completed transcript chunks yet.")
            return
        from ui.transcript_editor import TranscriptEditor
        dialog = TranscriptEditor(job, self)
        dialog.exec()
        dialog.deleteLater()

    def release_selected_audio(self):
        if self.is_busy or not self.can_start():
            return
        try:
            for job in self.selected_jobs():
                release_audio(job)
            self.status.setText("Audio cache released. Resuming requires the unchanged original media file.")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could Not Release Cache", str(error))
