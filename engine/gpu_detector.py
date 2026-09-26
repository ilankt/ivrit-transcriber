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
        import torch

        if not torch.cuda.is_available():
            return False, "No NVIDIA CUDA GPU detected"

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return False, "CUDA is available but no GPU devices found"

        gpu_name = torch.cuda.get_device_name(0)
        return True, gpu_name

    except ImportError:
        return False, "PyTorch not installed (required for GPU detection)"
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
        p = subprocess.Popen(
            ['vulkaninfo', '--summary'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            **_POPEN_EXTRA_KWARGS
        )
        out, _ = p.communicate(timeout=10)
        if p.returncode == 0:
            text = out.decode('utf-8', errors='replace')
            # Look for AMD device in vulkaninfo output
            for line in text.splitlines():
                if 'deviceName' in line:
                    name = line.split('=')[-1].strip()
                    if any(kw in name.upper() for kw in ('AMD', 'RADEON')):
                        return True, name
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Fallback: WMI on Windows
    if sys.platform == 'win32':
        try:
            p = subprocess.Popen(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                **_POPEN_EXTRA_KWARGS
            )
            out, _ = p.communicate(timeout=10)
            if p.returncode == 0:
                text = out.decode('utf-8', errors='replace')
                for line in text.splitlines():
                    line = line.strip()
                    if any(kw in line.upper() for kw in ('AMD', 'RADEON')):
                        return True, line
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
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
