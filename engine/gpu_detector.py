"""
GPU Detection Utility

This module detects NVIDIA CUDA-compatible GPUs for hardware acceleration.
Note: AMD GPUs are NOT supported by faster-whisper/CTranslate2.
"""

def detect_cuda_gpu() -> tuple[bool, str]:
    """
    Detect if a NVIDIA CUDA-compatible GPU is available.

    Returns:
        tuple[bool, str]: (is_available, info_message)
        - is_available: True if NVIDIA CUDA GPU is detected and functional
        - info_message: Human-readable message about GPU status

    Note:
        This function checks for NVIDIA CUDA support only.
        AMD GPUs are not supported by faster-whisper/CTranslate2.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "No NVIDIA CUDA GPU detected"

        # Get GPU information
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return False, "CUDA is available but no GPU devices found"

        # Get GPU name
        gpu_name = torch.cuda.get_device_name(0)

        # Check if it's an NVIDIA GPU (torch.cuda only supports NVIDIA)
        return True, f"NVIDIA GPU detected: {gpu_name}"

    except ImportError:
        # PyTorch not installed
        return False, "PyTorch not installed (required for GPU detection)"
    except Exception as e:
        # Any other error
        return False, f"GPU detection error: {str(e)}"


def get_gpu_info() -> dict:
    """
    Get detailed GPU information.

    Returns:
        dict: GPU information including:
            - available: bool
            - count: int
            - devices: list of device names
            - cuda_version: str
    """
    info = {
        "available": False,
        "count": 0,
        "devices": [],
        "cuda_version": None
    }

    try:
        import torch

        if torch.cuda.is_available():
            info["available"] = True
            info["count"] = torch.cuda.device_count()
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(info["count"])]
            info["cuda_version"] = torch.version.cuda

    except ImportError:
        pass
    except Exception:
        pass

    return info
