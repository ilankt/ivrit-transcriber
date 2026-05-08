import os
from faster_whisper import WhisperModel

_MODEL_REGISTRY = {
    ("he", "faster-whisper"): {
        "path": ("Models", "ivrit-large-v3-ct2"),
        "repo_id": None,
    },
    ("he", "whisper-cpp"): {
        "path": ("Models", "ggml-ivrit-large-v3.bin"),
        "repo_id": None,
    },
    ("en", "faster-whisper"): {
        "path": ("Models", "en-large-v3-ct2"),
        "repo_id": "Systran/faster-whisper-large-v3",
        "type": "ct2",
    },
    ("en", "whisper-cpp"): {
        "path": ("Models", "ggml-large-v3.bin"),
        "repo_id": "ggerganov/whisper.cpp",
        "filename": "ggml-large-v3.bin",
        "type": "ggml",
    },
}


def resolve_model_path(language: str, engine: str, base_path: str) -> str:
    """Return the absolute local path for the given language/engine combination."""
    key = (language, engine)
    entry = _MODEL_REGISTRY.get(key) or _MODEL_REGISTRY[("he", "faster-whisper")]
    return os.path.join(base_path, *entry["path"])


def get_model_download_info(language: str, engine: str) -> dict | None:
    """Return download metadata dict, or None if the model is bundled."""
    key = (language, engine)
    entry = _MODEL_REGISTRY.get(key)
    if entry and entry.get("repo_id"):
        return entry
    return None


def validate_model_path(path: str) -> bool:
    required_files = ["model.bin", "tokenizer.json", "vocabulary.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            return False
    return True


def load_whisper_model(model_path: str, device: str, compute_type: str, threads: int):
    actual_device = device.lower()

    if actual_device == "auto":
        try:
            model = WhisperModel(model_path, device="cuda", compute_type=compute_type, cpu_threads=threads)
            return model, None
        except Exception as e_cuda:
            try:
                model = WhisperModel(model_path, device="cpu", compute_type=compute_type, cpu_threads=threads)
                return model, None
            except Exception as e_cpu:
                return None, f"Failed to load model on both CUDA and CPU: CUDA error: {e_cuda}, CPU error: {e_cpu}"
    elif actual_device in ("gpu", "nvidia"):
        try:
            model = WhisperModel(model_path, device="cuda", compute_type=compute_type, cpu_threads=threads)
            return model, None
        except Exception as e:
            return None, f"Failed to load model on GPU (cuda): {e}"
    else:
        try:
            model = WhisperModel(model_path, device=actual_device, compute_type=compute_type, cpu_threads=threads)
            return model, None
        except Exception as e:
            return None, f"Failed to load model on {actual_device}: {e}"
