import json
import os
from pydantic import BaseModel
from typing import Optional

class Settings(BaseModel):
    model_type: str = "Fast"
    vad_enabled: bool = True
    threads: int = 0
    compute_type: str = "auto"
    output_folder: Optional[str] = None
    device: str = "auto"  # "auto", "cpu", or "gpu" (NVIDIA CUDA only)
    output_format: str = "srt"  # "srt", "txt", or "both"
    default_output_filename: Optional[str] = None  # Not persisted per user preference

def get_settings_path() -> str:
    if os.name == 'nt':
        return os.path.join(os.getenv('APPDATA'), 'IvritTranscriber', 'settings.json')
    else:
        return os.path.join(os.path.expanduser('~'), '.ivrit_transcriber', 'settings.json')

def load_settings() -> Settings:
    path = get_settings_path()
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                return Settings.parse_obj(json.load(f))
            except (json.JSONDecodeError, TypeError):
                return Settings()
    return Settings()

def save_settings(settings: Settings):
    path = get_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(settings.dict(), f, indent=4)
