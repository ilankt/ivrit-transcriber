"""One-time setup UI for optional speaker labels."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QProgressBar
from core.speaker_setup_worker import SpeakerSetupWorker
from engine.diarization import MODEL_URL, dependency_error, model_available


class SpeakerSetupDialog(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.worker = None
        self.setWindowTitle("Set Up Speakers")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        help_text = QLabel(
            'Label recorded audio as Speaker 1, Speaker 2, etc. Processing stays on this computer.<br><br>'
            f'Download the <a href="{MODEL_URL}">ivrit.ai speaker model</a> once, '
            'then use it offline.<br>No Hugging Face account or token is required for these public model files.'
        )
        help_text.setWordWrap(True)
        help_text.setOpenExternalLinks(True)
        layout.addWidget(help_text)
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("Optional Hugging Face token (normally leave blank; not saved)")
        layout.addWidget(self.token_edit)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.download_button = QPushButton("Download Speaker Model")
        self.download_button.clicked.connect(self._download)
        layout.addWidget(self.download_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        layout.addWidget(self.close_button)
        error = dependency_error()
        if error:
            self.status.setText(error)
            self.download_button.setEnabled(False)
        elif model_available(path):
            self.status.setText("Speaker model is ready. Downloads and repairs use the selected Models folder.")

    def _download(self):
        self.worker = SpeakerSetupWorker(self.path, self.token_edit.text().strip(), self)
        self.token_edit.clear()
        self.token_edit.setEnabled(False)
        self.download_button.setEnabled(False)
        self.close_button.setText("Cancel Download")
        self.status.setText("Downloading model files. Progress updates after each file finishes.")
        self.progress.setValue(0)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.result.connect(lambda success, message: self.status.setText(message))
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _finished(self):
        self.token_edit.setEnabled(True)
        self.download_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.close_button.setText("Close")

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.status.setText("Canceling after the current file finishes...")
            self.close_button.setEnabled(False)
            return
        super().reject()
