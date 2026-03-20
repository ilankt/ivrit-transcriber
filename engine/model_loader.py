import os
from faster_whisper import WhisperModel

def validate_model_path(path: str) -> bool:
    # Check for the presence of key files that indicate a valid CTranslate2 model directory
    # This is a basic check and might need to be more robust depending on model specifics
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
    else:  # "cpu" or any other explicit device
        try:
            model = WhisperModel(model_path, device=actual_device, compute_type=compute_type, cpu_threads=threads)
            return model, None
        except Exception as e:
            return None, f"Failed to load model on {actual_device}: {e}"
