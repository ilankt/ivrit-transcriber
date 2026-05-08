"""
Settings Panel UI.

Provides shared application settings: theme, language, VAD, device, output format,
model status with manual download, and custom models folder.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QCheckBox, QLabel, QPushButton, QLineEdit,
    QHBoxLayout, QFileDialog,
)
from PySide6.QtCore import Signal


class SettingsPanel(QWidget):
    """Panel for application-wide settings."""

    theme_changed = Signal(str)
    download_requested = Signal()  # user clicked "Download" button

    def __init__(self, settings, gpu_info, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.gpu_info = gpu_info
        self._build_ui()
        self._load_settings()
        self._refresh_model_status()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Appearance ---
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("System", "system")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        appearance_layout.addRow("Theme:", self.theme_combo)

        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)

        # --- Transcription ---
        transcription_group = QGroupBox("Transcription")
        transcription_layout = QFormLayout()

        self.language_combo = QComboBox()
        self.language_combo.addItem("Hebrew", "he")
        self.language_combo.addItem("English", "en")
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        transcription_layout.addRow("Language:", self.language_combo)

        self.english_info_label = QLabel(
            "English uses standard OpenAI Whisper large-v3.\n"
            "Model will be downloaded on first use (~3 GB)."
        )
        self.english_info_label.setWordWrap(True)
        self.english_info_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.english_info_label.setVisible(False)
        transcription_layout.addRow("", self.english_info_label)

        self.vad_checkbox = QCheckBox("Enable VAD (Voice Activity Detection)")
        self.vad_checkbox.setChecked(True)
        transcription_layout.addRow(self.vad_checkbox)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (Try GPU, fallback to CPU)", "auto")
        self.device_combo.addItem("CPU Only", "cpu")

        nvidia_available = self.gpu_info["nvidia_cuda"]["available"]
        amd_available = self.gpu_info["amd_vulkan"]["available"]

        if nvidia_available:
            self.device_combo.addItem(
                f"NVIDIA GPU ({self.gpu_info['nvidia_cuda']['info']})", "nvidia"
            )
        if amd_available:
            self.device_combo.addItem(
                f"AMD GPU ({self.gpu_info['amd_vulkan']['info']})", "amd"
            )

        if self.settings.device == "nvidia" and not nvidia_available:
            self.settings.device = "auto"
        if self.settings.device == "amd" and not amd_available:
            self.settings.device = "auto"

        self.device_combo.currentIndexChanged.connect(self._refresh_model_status)
        transcription_layout.addRow("Device:", self.device_combo)

        # Model status row: status label + Download button + Refresh button
        model_status_widget = QWidget()
        model_status_row = QHBoxLayout(model_status_widget)
        model_status_row.setContentsMargins(0, 0, 0, 0)

        self.model_status_label = QLabel("Checking…")
        self.download_button = QPushButton("Download")
        self.download_button.setVisible(False)
        self.download_button.clicked.connect(self._on_download_clicked)
        self.refresh_status_button = QPushButton("↻")
        self.refresh_status_button.setFixedWidth(32)
        self.refresh_status_button.setToolTip("Refresh model status")
        self.refresh_status_button.clicked.connect(self._refresh_model_status)

        model_status_row.addWidget(self.model_status_label, 1)
        model_status_row.addWidget(self.download_button)
        model_status_row.addWidget(self.refresh_status_button)
        transcription_layout.addRow("Model:", model_status_widget)

        transcription_group.setLayout(transcription_layout)
        layout.addWidget(transcription_group)

        # --- Output ---
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("SRT (Subtitles)", "srt")
        self.output_format_combo.addItem("TXT (Plain Text)", "txt")
        self.output_format_combo.addItem("Both (SRT + TXT)", "both")
        output_layout.addRow("Output Format:", self.output_format_combo)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # --- Models ---
        models_group = QGroupBox("Models")
        models_layout = QFormLayout()

        folder_row = QHBoxLayout()
        self.models_folder_edit = QLineEdit()
        self.models_folder_edit.setPlaceholderText("Default (app directory/Models)")
        self.models_folder_edit.textChanged.connect(self._refresh_model_status)
        models_folder_browse = QPushButton("Browse…")
        models_folder_browse.clicked.connect(self._browse_models_folder)
        folder_row.addWidget(self.models_folder_edit, 1)
        folder_row.addWidget(models_folder_browse)
        models_layout.addRow("Models Folder:", folder_row)

        models_group.setLayout(models_layout)
        layout.addWidget(models_group)

        layout.addStretch()

    def _load_settings(self):
        """Load settings values into UI controls."""
        for i in range(self.theme_combo.count()):
            if self.theme_combo.itemData(i) == self.settings.theme:
                self.theme_combo.setCurrentIndex(i)
                break

        for i in range(self.language_combo.count()):
            if self.language_combo.itemData(i) == self.settings.language:
                self.language_combo.setCurrentIndex(i)
                break
        self.english_info_label.setVisible(self.settings.language == "en")

        self.vad_checkbox.setChecked(self.settings.vad_enabled)

        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == self.settings.device:
                self.device_combo.setCurrentIndex(i)
                break

        for i in range(self.output_format_combo.count()):
            if self.output_format_combo.itemData(i) == self.settings.output_format:
                self.output_format_combo.setCurrentIndex(i)
                break

        if self.settings.models_folder:
            self.models_folder_edit.setText(self.settings.models_folder)

    def save_settings(self):
        """Write current UI values back to the settings object."""
        self.settings.theme = self.theme_combo.currentData()
        self.settings.language = self.language_combo.currentData()
        self.settings.vad_enabled = self.vad_checkbox.isChecked()
        self.settings.device = self.device_combo.currentData()
        self.settings.output_format = self.output_format_combo.currentData()
        folder = self.models_folder_edit.text().strip()
        self.settings.models_folder = folder if folder else None

    def _refresh_model_status(self):
        """Check whether the currently selected model is present and update the status row."""
        from engine.model_loader import resolve_model_path, get_model_download_info, validate_model_path
        from engine.whisper_cpp_runner import validate_ggml_model
        from core.worker import get_base_path

        language = self.language_combo.currentData() or "he"
        device = self.device_combo.currentData() or "auto"
        # Avoid slow GPU detection: derive engine from saved device value directly.
        engine = "whisper-cpp" if device == "amd" else "faster-whisper"

        models_dir = self.models_folder_edit.text().strip() or None
        model_path = resolve_model_path(language, engine, get_base_path(), models_dir)

        model_ready = (
            validate_ggml_model(model_path) if engine == "whisper-cpp"
            else validate_model_path(model_path)
        )

        download_info = get_model_download_info(language, engine)

        if model_ready:
            self.model_status_label.setText("Ready ✓")
            self.model_status_label.setStyleSheet("color: green;")
            self.download_button.setVisible(False)
        elif download_info:
            size = "~3 GB" if download_info["type"] == "ct2" else "~1.5 GB"
            lang_label = "English" if language == "en" else "Hebrew"
            self.model_status_label.setText(f"Not downloaded ({size})")
            self.model_status_label.setStyleSheet("color: orange;")
            self.download_button.setText(f"Download {lang_label} Model")
            self.download_button.setVisible(True)
        else:
            self.model_status_label.setText("Not found (bundled model missing)")
            self.model_status_label.setStyleSheet("color: red;")
            self.download_button.setVisible(False)

    def _on_download_clicked(self):
        self.download_requested.emit()

    def _browse_models_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Models Folder")
        if dir_path:
            self.models_folder_edit.setText(dir_path)

    def _on_theme_changed(self):
        theme = self.theme_combo.currentData()
        if theme:
            self.theme_changed.emit(theme)

    def _on_language_changed(self):
        lang = self.language_combo.currentData()
        self.english_info_label.setVisible(lang == "en")
        self._refresh_model_status()
