"""
Live Transcription Panel UI.

Provides the UI for real-time audio capture and transcription from
system audio (WASAPI loopback), typically used for Zoom meetings.
"""
import os
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QPushButton, QProgressBar, QLabel, QTextEdit,
    QLineEdit, QFileDialog, QMessageBox,
)
from PySide6.QtCore import QTimer, Qt

from engine.audio_capture import list_loopback_devices
from core.live_worker import LiveTranscriptionWorker, save_live_session


class LiveTranscriptionPanel(QWidget):
    """Panel for live audio transcription from system audio loopback."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._worker: LiveTranscriptionWorker | None = None
        self._session_timer = QTimer(self)
        self._session_timer.setInterval(1000)
        self._session_timer.timeout.connect(self._update_elapsed_time)
        self._session_start_time = None
        self._devices: list[dict] = []

        self._build_ui()
        self._refresh_devices()
        self._load_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Audio Source ---
        source_group = QGroupBox("Audio Source")
        source_layout = QFormLayout()

        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        self.refresh_button = QPushButton("Refresh Devices")
        self.refresh_button.clicked.connect(self._refresh_devices)
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_button)
        source_layout.addRow("Device:", device_row)

        # VU meter
        self.vu_meter = QProgressBar()
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setValue(0)
        self.vu_meter.setTextVisible(False)
        self.vu_meter.setFixedHeight(16)
        source_layout.addRow("Level:", self.vu_meter)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # --- Session Controls ---
        session_group = QGroupBox("Session")
        session_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Session")
        self.start_button.clicked.connect(self._start_session)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_session)
        self.stop_button.setEnabled(False)

        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setStyleSheet("font-size: 14pt; font-weight: bold;")

        self.status_label = QLabel("Ready")

        session_layout.addWidget(self.start_button)
        session_layout.addWidget(self.stop_button)
        session_layout.addWidget(self.elapsed_label)
        session_layout.addStretch()
        session_layout.addWidget(self.status_label)

        session_group.setLayout(session_layout)
        layout.addWidget(session_group)

        # --- Live Transcript ---
        transcript_group = QGroupBox("Live Transcript")
        transcript_layout = QVBoxLayout()

        self.transcript_edit = QTextEdit()
        self.transcript_edit.setReadOnly(True)
        self.transcript_edit.setMinimumHeight(200)
        # Right-to-left for Hebrew
        self.transcript_edit.setStyleSheet("QTextEdit { font-size: 12pt; }")
        self.transcript_edit.setLayoutDirection(Qt.RightToLeft)
        transcript_layout.addWidget(self.transcript_edit)

        clear_row = QHBoxLayout()
        clear_row.addStretch()
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.transcript_edit.clear)
        clear_row.addWidget(self.clear_button)
        transcript_layout.addLayout(clear_row)

        transcript_group.setLayout(transcript_layout)
        layout.addWidget(transcript_group, 1)  # stretch factor for transcript area

        # --- Output ---
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()

        folder_row = QHBoxLayout()
        self.output_folder_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_output_folder)
        folder_row.addWidget(self.output_folder_edit, 1)
        folder_row.addWidget(self.browse_button)
        output_layout.addRow("Folder:", folder_row)

        self.session_name_edit = QLineEdit()
        self.session_name_edit.setPlaceholderText("Auto-generated if empty")
        output_layout.addRow("Name:", self.session_name_edit)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

    def _refresh_devices(self):
        """Refresh the list of available loopback audio devices."""
        self.device_combo.clear()
        self._devices = list_loopback_devices()

        if not self._devices:
            self.device_combo.addItem("No loopback devices found", None)
            self.start_button.setEnabled(False)
        else:
            for dev in self._devices:
                label = f"{dev['name']} ({dev['sample_rate']}Hz, {dev['channels']}ch)"
                self.device_combo.addItem(label, dev['index'])
            self.start_button.setEnabled(True)

            # Restore previously selected device
            if self.settings.live_audio_device:
                for i, dev in enumerate(self._devices):
                    if dev['name'] == self.settings.live_audio_device:
                        self.device_combo.setCurrentIndex(i)
                        break

    def _load_settings(self):
        """Load live-mode settings to UI."""
        if self.settings.live_output_folder:
            self.output_folder_edit.setText(self.settings.live_output_folder)

    def save_settings(self):
        """Save live-mode settings from UI."""
        idx = self.device_combo.currentIndex()
        if idx >= 0 and idx < len(self._devices):
            self.settings.live_audio_device = self._devices[idx]['name']
        folder = self.output_folder_edit.text()
        if folder:
            self.settings.live_output_folder = folder

    def _browse_output_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_folder_edit.setText(dir_path)

    def _start_session(self):
        """Validate inputs and start live transcription."""
        idx = self.device_combo.currentIndex()
        device_data = self.device_combo.currentData()
        if device_data is None:
            QMessageBox.warning(self, "No Device", "Please select an audio device first.")
            return

        output_folder = self.output_folder_edit.text()
        if not output_folder:
            QMessageBox.warning(self, "No Output Folder", "Please select an output folder.")
            return

        os.makedirs(output_folder, exist_ok=True)

        dev = self._devices[idx]

        self._worker = LiveTranscriptionWorker(
            device_index=dev['index'],
            device_sample_rate=dev['sample_rate'],
            device_channels=dev['channels'],
            settings=self.settings,
        )
        self._worker.segment_ready.connect(self._on_segment_ready)
        self._worker.audio_level.connect(self._on_audio_level)
        self._worker.status_updated.connect(self._on_status)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.session_finished.connect(self._on_session_finished)

        self._worker.start()

        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.device_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.output_folder_edit.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.session_name_edit.setEnabled(False)

        self._session_start_time = datetime.now()
        self._session_timer.start()

    def stop_session(self):
        """Stop the current live transcription session."""
        if self._worker is None:
            return

        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping...")
        self._worker.stop()

    def _on_segment_ready(self, start: float, end: float, text: str):
        """Append a transcribed segment to the transcript view."""
        h = int(start // 3600)
        m = int((start % 3600) // 60)
        s = int(start % 60)
        timestamp = f"[{h:02d}:{m:02d}:{s:02d}]"
        self.transcript_edit.append(f"{timestamp} {text}")
        # Auto-scroll to bottom
        scrollbar = self.transcript_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_audio_level(self, level: float):
        """Update VU meter."""
        self.vu_meter.setValue(int(level * 100))

    def _on_status(self, message: str):
        """Update status label."""
        self.status_label.setText(message)

    def _on_error(self, message: str):
        """Show error and reset UI."""
        QMessageBox.warning(self, "Live Transcription Error", message)

    def _on_session_finished(self):
        """Save output and reset UI after session ends."""
        self._session_timer.stop()

        # Save output files
        if self._worker and self._worker.session_segments:
            output_folder = self.output_folder_edit.text()
            session_name = self.session_name_edit.text().strip()
            if not session_name:
                session_name = f"meeting-{self._session_start_time.strftime('%Y-%m-%d-%H%M%S')}"
            # Sanitize filename
            session_name = "".join(
                c for c in session_name if c.isalnum() or c in (' ', '-', '_')
            )

            output_format = self.settings.output_format
            save_live_session(
                self._worker.session_segments,
                output_folder, session_name, output_format
            )
            self.status_label.setText(f"Saved: {session_name}")
        else:
            self.status_label.setText("Session ended (no segments)")

        self._worker = None

        # Reset UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.device_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.output_folder_edit.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.session_name_edit.setEnabled(True)
        self.vu_meter.setValue(0)

    def _update_elapsed_time(self):
        """Update the elapsed time label every second."""
        if self._session_start_time is None:
            return
        elapsed = (datetime.now() - self._session_start_time).total_seconds()
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        self.elapsed_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
