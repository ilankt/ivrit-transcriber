"""Runtime helpers shared by UI and worker modules."""
import os
import sys


def get_base_path() -> str:
    """Return the app root for source runs and the bundled resource root for builds."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def determine_engine(device: str, gpu_info: dict | None = None) -> str:
    """Return the transcription engine for a saved device setting."""
    if device in ("amd", "metal"):
        return "whisper-cpp"
    if device == "auto":
        if gpu_info is None:
            from engine.gpu_detector import detect_all_gpus
            gpu_info = detect_all_gpus()
        if gpu_info.get("apple_metal", {}).get("available"):
            return "whisper-cpp"
        if gpu_info.get("nvidia_cuda", {}).get("available"):
            return "faster-whisper"
        if gpu_info.get("amd_vulkan", {}).get("available"):
            return "whisper-cpp"
    return "faster-whisper"
