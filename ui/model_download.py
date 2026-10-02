"""Keep a model download alive until its worker has really stopped."""
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton


class ModelDownloadDialog(QDialog):
    def __init__(self, worker, label, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.success = False
        self.message = ""
        self.setWindowTitle("Downloading Model")
        layout = QVBoxLayout(self)
        self.label = QLabel(label)
        layout.addWidget(self.label)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        worker.progress_updated.connect(self.progress.setValue)
        worker.result.connect(self._result)
        worker.finished.connect(self.accept)

    def _result(self, success, message):
        self.success, self.message = success, message

    def reject(self):
        if self.worker.isRunning():
            self.worker.cancel()
            self.label.setText("Canceling after the current file finishes...")
            self.cancel_button.setEnabled(False)
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.reject()
            event.ignore()
        else:
            event.accept()
