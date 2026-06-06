import json
import os
from pydantic import BaseModel

class Settings(BaseModel):
    language: str = "he"  # "he" (Hebrew) or "en" (English)
    vad_enabled: bool = True
    threads: int = 0
    compute_type: str = "auto"
    output_folder: str | None = None
    models_folder: str | None = None  # Custom models directory; None = default (app dir/Models)
    device: str = "auto"  # "auto", "cpu", "nvidia", or "amd"
    output_format: str = "srt"  # "srt", "txt", or "both"
    theme: str = "system"  # "system", "light", or "dark"
    live_audio_device: str | None = None
    live_output_folder: str | None = None

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
                data = json.load(f)
                # Migrate old "gpu" device value to "nvidia"
                if data.get('device') == 'gpu':
                    data['device'] = 'nvidia'
                return Settings.parse_obj(data)
            except (json.JSONDecodeError, TypeError):
                return Settings()
    return Settings()

def save_settings(settings: Settings):
    path = get_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(settings.dict(), f, indent=4)
