"""Install the host's optional speaker accelerator into the current environment.

Run after requirements.txt and requirements-speakers.txt. --dry-run prints the
commands without installing anything. No drivers or system settings are changed.
"""
import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def has_nvidia():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        result = subprocess.run([executable, "-L"], capture_output=True, text=True, timeout=10, **options)
        return result.returncode == 0 and "GPU " in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def installation_commands(backend, cuda_version):
    pip = [sys.executable, "-m", "pip", "install"]
    if backend == "cuda":
        # An explicit local version replaces CPU-only torch even at the same 2.7.1 version.
        return [pip + ["--upgrade", f"torch==2.7.1+{cuda_version}",
                       f"torchaudio==2.7.1+{cuda_version}", "--index-url",
                       f"https://download.pytorch.org/whl/{cuda_version}"]]
    if backend == "directml":
        return [pip + ["-r", str(ROOT / "requirements-speakers-amd.txt")],
                pip + ["--force-reinstall", "--no-deps", "onnxruntime-directml==1.24.4"]]
    # Native macOS PyTorch wheels already contain MPS; CPU needs no accelerator.
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "metal", "directml"], default="auto")
    parser.add_argument("--cuda-version", choices=["cu118", "cu126", "cu128"], default="cu128")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backend = args.backend
    if backend == "auto":
        if sys.platform == "darwin":
            backend = "metal" if platform.machine().lower() == "arm64" else "cpu"
        elif has_nvidia():
            backend = "cuda"
        else:
            backend = "directml" if sys.platform == "win32" else "cpu"
    if backend == "directml" and sys.platform != "win32":
        parser.error("DirectML requires Windows")
    if backend == "cuda" and sys.platform == "darwin":
        parser.error("CUDA is not supported on macOS")
    if backend == "metal" and (sys.platform != "darwin" or platform.machine().lower() != "arm64"):
        parser.error("Metal speaker acceleration requires native ARM64 Python on Apple Silicon")
    print(f"Speaker acceleration setup: {backend}", flush=True)
    for command in installation_commands(backend, args.cuda_version):
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, check=True, cwd=ROOT)
    if backend == "metal":
        print("Metal is included in the native PyTorch installation; no DirectML/CUDA packages needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
