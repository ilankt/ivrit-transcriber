"""
GPU Detection Utility

Detects NVIDIA CUDA, AMD Vulkan, and Apple Silicon Metal hardware.
"""
import subprocess
import sys
import platform

_POPEN_EXTRA_KWARGS = {}
if sys.platform == 'win32':
    _POPEN_EXTRA_KWARGS['creationflags'] = subprocess.CREATE_NO_WINDOW


def detect_cuda_gpu() -> tuple[bool, str]:
    """
    Detect if a NVIDIA CUDA-compatible GPU is available.

    Returns:
        tuple[bool, str]: (is_available, info_message)
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() == 0:
            return False, "No NVIDIA CUDA GPU detected"
        # CTranslate2 is the transcription runtime; optional CPU-only Torch
        # must not hide a GPU that Faster-Whisper can use.
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, timeout=3, check=False, **_POPEN_EXTRA_KWARGS)
            names = result.stdout.decode('utf-8', errors='replace').strip().splitlines()
            if result.returncode == 0 and names:
                return True, names[0]
        except (OSError, subprocess.TimeoutExpired):
            pass
        return True, "NVIDIA CUDA"

    except ImportError:
        return False, "CTranslate2 is unavailable"
    except Exception as e:
        return False, f"GPU detection error: {str(e)}"


def detect_vulkan_gpu() -> tuple[bool, str]:
    """
    Detect if a Vulkan-capable AMD GPU is available.

    Uses vulkaninfo if available, falls back to WMI on Windows.

    Returns:
        tuple[bool, str]: (is_available, gpu_name)
    """
    # Try vulkaninfo first
    try:
        result = subprocess.run(
            ['vulkaninfo', '--summary'],
            capture_output=True, timeout=10, check=False,
            **_POPEN_EXTRA_KWARGS
        )
        if result.returncode == 0:
            text = result.stdout.decode('utf-8', errors='replace')
            # Look for AMD device in vulkaninfo output
            for line in text.splitlines():
                if 'deviceName' in line:
                    name = line.split('=')[-1].strip()
                    if any(kw in name.upper() for kw in ('AMD', 'RADEON')):
                        return True, name
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Fallback: WMI on Windows
    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                capture_output=True, timeout=10, check=False,
                **_POPEN_EXTRA_KWARGS
            )
            if result.returncode == 0:
                text = result.stdout.decode('utf-8', errors='replace')
                for line in text.splitlines():
                    line = line.strip()
                    if any(kw in line.upper() for kw in ('AMD', 'RADEON')):
                        return True, line
        except (OSError, subprocess.TimeoutExpired):
            pass

    return False, "No AMD Vulkan GPU detected"


def detect_metal_gpu() -> tuple[bool, str]:
    """Detect Apple Silicon, including an x86 Python running under Rosetta."""
    if sys.platform != 'darwin':
        return False, "Apple Metal requires macOS on Apple Silicon"
    if platform.machine().lower() in ('arm64', 'aarch64'):
        return True, "Apple Silicon (Metal)"
    try:
        result = subprocess.run(
            ['/usr/sbin/sysctl', '-n', 'hw.optional.arm64'],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == '1':
            return True, "Apple Silicon (Metal; Python running under Rosetta)"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return False, "No Apple Silicon GPU detected"


def detect_all_gpus() -> dict:
    """
    Detect all available GPUs.

    Returns:
        dict with keys:
            nvidia_cuda: {"available": bool, "info": str}
            amd_vulkan: {"available": bool, "info": str}
            apple_metal: {"available": bool, "info": str}
    """
    metal_available, metal_info = detect_metal_gpu()
    if sys.platform == 'darwin':
        cuda_available, cuda_info = False, "CUDA is unavailable on macOS"
        amd_available, amd_info = False, "Vulkan backend is not used on macOS"
    else:
        cuda_available, cuda_info = detect_cuda_gpu()
        amd_available, amd_info = detect_vulkan_gpu()
    return {
        "nvidia_cuda": {"available": cuda_available, "info": cuda_info},
        "amd_vulkan": {"available": amd_available, "info": amd_info},
        "apple_metal": {"available": metal_available, "info": metal_info},
    }
