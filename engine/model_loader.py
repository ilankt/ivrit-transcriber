import os
from faster_whisper import WhisperModel

_CT2_REQUIRED_FILES = ("model.bin", "tokenizer.json", "vocabulary.json")

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
        "size_label": "~3 GB",
        "required_gb": 3.5,
    },
    ("en", "whisper-cpp"): {
        "path": ("Models", "ggml-large-v3.bin"),
        "repo_id": "ggerganov/whisper.cpp",
        "filename": "ggml-large-v3.bin",
        "type": "ggml",
        "size_label": "~1.5 GB",
        "required_gb": 1.8,
    },
}


def resolve_model_path(
    language: str,
    engine: str,
    base_path: str,
    models_dir: str | None = None,
) -> str:
    """Return the absolute local path for the given language/engine combination.

    If *models_dir* is provided it is used as the models directory directly.
    Otherwise the default ``<base_path>/Models`` tree is used.
    """
    key = (language, engine)
    entry = _MODEL_REGISTRY.get(key) or _MODEL_REGISTRY[("he", "faster-whisper")]
    # entry["path"] is e.g. ("Models", "ivrit-large-v3-ct2").
    # When a custom models_dir is supplied, skip the "Models" prefix.
    if models_dir:
        return os.path.join(models_dir, *entry["path"][1:])
    return os.path.join(base_path, *entry["path"])


def get_all_known_model_names() -> set[str]:
    """Return the set of model directory/file names that are registered for any language/engine."""
    return {entry["path"][-1] for entry in _MODEL_REGISTRY.values()}


def get_model_download_info(language: str, engine: str) -> dict | None:
    """Return download metadata dict, or None if the model is bundled."""
    key = (language, engine)
    entry = _MODEL_REGISTRY.get(key)
    if entry and entry.get("repo_id"):
        return entry
    return None


def get_download_size_label(download_info: dict) -> str:
    return download_info.get("size_label", "unknown size")


def get_download_required_gb(download_info: dict) -> float:
    return float(download_info.get("required_gb", 0.0))


def is_model_available(model_path: str, engine: str) -> bool:
    if engine == "whisper-cpp":
        from engine.whisper_cpp_runner import validate_ggml_model

        return validate_ggml_model(model_path)
    return validate_model_path(model_path)


def validate_model_path(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, file)) for file in _CT2_REQUIRED_FILES)


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
