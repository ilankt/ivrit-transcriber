from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                               QTableWidgetItem, QHeaderView, QPushButton, QLabel,
                               QSlider, QComboBox, QFileDialog, QMessageBox, QInputDialog)

from core.filenames import sanitize_output_stem
from core.transcript import transcript_rows, save_review, export_review


class TranscriptEditor(QDialog):
    def __init__(self, job, parent=None):
        super().__init__(parent)
        self.job = job
        self.setWindowTitle(f"Review — {Path(job.original_file_path).name}")
        self.resize(1000, 660)
        self.dirty = False
        layout = QVBoxLayout(self)
        self.status = QLabel("Select a row to seek. Edit text, speaker, or timestamps in seconds. "
                             "Corrections are saved separately from the original transcript.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Start (s)", "End (s)", "Speaker", "Text"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.8)
        self.player.errorOccurred.connect(lambda *_: self.status.setText(self.player.errorString()))
        self.play_button = QPushButton("Play / Pause")
        self.play_button.clicked.connect(self._play)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.setRange(0, 0)
        self.position.sliderMoved.connect(self.player.setPosition)
        self.player.durationChanged.connect(lambda value: self.position.setMaximum(value))
        self.player.positionChanged.connect(self._position_changed)
        speed = QComboBox()
        speed.addItems(["0.75×", "1×", "1.25×", "1.5×", "2×"])
        speed.setCurrentIndex(1)
        speed.currentIndexChanged.connect(lambda i: self.player.setPlaybackRate([.75, 1, 1.25, 1.5, 2][i]))
        playback = QHBoxLayout()
        for widget in (self.play_button, self.position, speed):
            playback.addWidget(widget)
        layout.addLayout(playback)
        buttons = QHBoxLayout()
        rename = QPushButton("Rename Speaker…")
        save = QPushButton("Save Corrections")
        export = QPushButton("Export Corrected…")
        close = QPushButton("Close")
        for button in (rename, save, export, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        rename.clicked.connect(self._rename_speaker)
        save.clicked.connect(self._save)
        export.clicked.connect(self._export)
        close.clicked.connect(self.reject)
        try:
            rows = transcript_rows(job)
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows = []
            self.status.setText(f"Could not read transcript: {error}")
            save.setEnabled(False)
            export.setEnabled(False)
        self.row_ids = [row["id"] for row in rows]
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for col, value in enumerate((f"{row['start']:.3f}", f"{row['end']:.3f}", row["speaker"], row["text"])):
                self.table.setItem(index, col, QTableWidgetItem(value))
        self.table.itemChanged.connect(lambda *_: setattr(self, "dirty", True))
        self.table.cellClicked.connect(self._seek_row)
        if Path(job.original_file_path).is_file():
            self.player.setSource(QUrl.fromLocalFile(job.original_file_path))
        else:
            self.play_button.setEnabled(False)
            self.status.setText("Original media is missing. Text editing and export are still available.")

    def _rows(self):
        return [dict(id=self.row_ids[i], start=float(self.table.item(i, 0).text()),
                     end=float(self.table.item(i, 1).text()), speaker=self.table.item(i, 2).text().strip(),
                     text=self.table.item(i, 3).text().strip()) for i in range(self.table.rowCount())]

    def _seek_row(self, row, column):
        try:
            self.player.setPosition(round(float(self.table.item(row, 0).text()) * 1000))
        except (ValueError, OverflowError):
            pass

    def _position_changed(self, value):
        if not self.position.isSliderDown():
            self.position.setValue(value)

    def _play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _rename_speaker(self):
        speakers = sorted({self.table.item(i, 2).text() for i in range(self.table.rowCount())} - {""})
        if not speakers:
            self.status.setText("Enter a speaker in the Speaker column first.")
            return
        old, ok = QInputDialog.getItem(self, "Rename Speaker", "Speaker:", speakers, editable=False)
        if not ok:
            return
        new, ok = QInputDialog.getText(self, "Rename Speaker", "New name:", text=old)
        if ok and new.strip():
            for i in range(self.table.rowCount()):
                if self.table.item(i, 2).text() == old:
                    self.table.item(i, 2).setText(new.strip())

    def _save(self):
        try:
            save_review(self.job, self._rows())
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could Not Save Corrections", str(error))
            return False
        self.dirty = False
        self.status.setText("Corrections saved.")
        return True

    def _export(self):
        if not self._save():
            return
        stem = self.job.custom_output_filename or Path(self.job.original_file_path).stem
        path, _ = QFileDialog.getSaveFileName(self, "Export Corrected Transcript",
                                            str(Path(self.job.output_dir or ".") / f"{stem}_edited.srt"),
                                            "SubRip subtitles (*.srt);;Text (*.txt)")
        if not path:
            return
        path = Path(path)
        try:
            export_review(self._rows(), str(path.parent), sanitize_output_stem(path.stem),
                          "txt" if path.suffix.lower() == ".txt" else "srt")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Export Failed", str(error))
            return
        self.status.setText(f"Exported {path}")

    def reject(self):
        if self.dirty:
            choice = QMessageBox.question(self, "Unsaved Corrections", "Save your corrections before closing?",
                                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                          QMessageBox.Save)
            if choice == QMessageBox.Cancel or (choice == QMessageBox.Save and not self._save()):
                return
        self.player.stop()
        super().reject()
