import json
import logging
import os
from typing import Literal
from pydantic import BaseModel, Field
from core.storage import atomic_text_writer

class Settings(BaseModel):
    language: Literal["he", "en"] = "he"
    vad_enabled: bool = True
    diarization_enabled: bool = False
    diarization_speakers: int = Field(default=0, ge=0, le=50)
    threads: int = Field(default=0, ge=0)
    compute_type: str = "auto"
    output_folder: str | None = None
    models_folder: str | None = None  # Custom models directory; None = default (app dir/Models)
    device: Literal["auto", "cpu", "nvidia", "amd", "metal"] = "auto"
    output_format: Literal["srt", "txt", "both"] = "srt"
    theme: Literal["system", "light", "dark"] = "system"
    live_audio_device: str | None = None
    live_output_folder: str | None = None

def get_settings_path() -> str:
    if os.name == 'nt':
        appdata = os.getenv('APPDATA') or os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming')
        return os.path.join(appdata, 'IvritTranscriber', 'settings.json')
    else:
        return os.path.join(os.path.expanduser('~'), '.ivrit_transcriber', 'settings.json')

def load_settings() -> Settings:
    path = get_settings_path()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get('device') == 'gpu':
            data['device'] = 'nvidia'
        return Settings.model_validate(data)
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError, TypeError):
        logging.warning("Could not read valid settings; using defaults", exc_info=True)
        return Settings()

def save_settings(settings: Settings):
    path = get_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with atomic_text_writer(path) as f:
        json.dump(settings.model_dump(), f, indent=4)
