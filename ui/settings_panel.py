"""
Settings Panel UI.

Provides shared application settings: theme, model, VAD, device, output format.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QCheckBox,
)
from PySide6.QtCore import Signal


class SettingsPanel(QWidget):
    """Panel for application-wide settings."""

    theme_changed = Signal(str)

    def __init__(self, settings, gpu_info, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.gpu_info = gpu_info
        self._build_ui()
        self._load_settings()

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

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Fast", "Accurate"])
        transcription_layout.addRow("Model:", self.model_type_combo)

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

        # Reset to auto if previously selected GPU is no longer available
        if self.settings.device == "nvidia" and not nvidia_available:
            self.settings.device = "auto"
        if self.settings.device == "amd" and not amd_available:
            self.settings.device = "auto"

        transcription_layout.addRow("Device:", self.device_combo)

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

        layout.addStretch()

    def _load_settings(self):
        """Load settings values into UI controls."""
        # Theme
        for i in range(self.theme_combo.count()):
            if self.theme_combo.itemData(i) == self.settings.theme:
                self.theme_combo.setCurrentIndex(i)
                break

        # Model
        self.model_type_combo.setCurrentText(self.settings.model_type)

        # VAD
        self.vad_checkbox.setChecked(self.settings.vad_enabled)

        # Device
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == self.settings.device:
                self.device_combo.setCurrentIndex(i)
                break

        # Output format
        for i in range(self.output_format_combo.count()):
            if self.output_format_combo.itemData(i) == self.settings.output_format:
                self.output_format_combo.setCurrentIndex(i)
                break

    def save_settings(self):
        """Write current UI values back to the settings object."""
        self.settings.theme = self.theme_combo.currentData()
        self.settings.model_type = self.model_type_combo.currentText()
        self.settings.vad_enabled = self.vad_checkbox.isChecked()
        self.settings.device = self.device_combo.currentData()
        self.settings.output_format = self.output_format_combo.currentData()

    def _on_theme_changed(self):
        theme = self.theme_combo.currentData()
        if theme:
            self.theme_changed.emit(theme)
