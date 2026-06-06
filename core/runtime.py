"""Runtime helpers shared by UI and worker modules."""
import os
import sys


def get_base_path() -> str:
    """Return the app root for source runs and the executable directory for builds."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def determine_engine(device: str) -> str:
    """Return the transcription engine for a saved device setting."""
    if device == "amd":
        return "whisper-cpp"
    if device == "auto":
        from engine.gpu_detector import detect_cuda_gpu, detect_vulkan_gpu

        cuda_ok, _ = detect_cuda_gpu()
        if cuda_ok:
            return "faster-whisper"
        amd_ok, _ = detect_vulkan_gpu()
        if amd_ok:
            return "whisper-cpp"
    return "faster-whisper"
