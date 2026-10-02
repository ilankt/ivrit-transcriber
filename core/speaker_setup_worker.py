"""Background speaker-model download; credentials live only in memory."""
from PySide6.QtCore import QThread, Signal


class SpeakerSetupWorker(QThread):
    progress = Signal(int)
    result = Signal(bool, str)

    def __init__(self, path, token, parent=None):
        super().__init__(parent)
        self.path = path
        self.token = token

    def run(self):
        from engine.diarization import download_model
        try:
            download_model(self.path, self.token, self.progress.emit, self.isInterruptionRequested)
            self.result.emit(True, "Speaker model downloaded. You can now enable Detect speakers.")
        except InterruptedError:
            self.result.emit(False, "Download canceled. You can resume it later.")
        except Exception:
            # Hub exceptions may contain HTTP details; do not display/log tokens.
            self.result.emit(False,
                "Download failed. Check your connection and available disk space, then retry. "
                "The public ivrit.ai speaker model normally needs no access token.")
        finally:
            self.token = None
