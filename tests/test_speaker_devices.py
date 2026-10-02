from types import SimpleNamespace

import pytest

from engine.speaker_devices import select_speaker_backend
from scripts.setup_speaker_acceleration import installation_commands


@pytest.mark.parametrize("platform,requested,cuda,mps,expected", [
    ("win32", "cpu", True, False, "cpu"),
    ("win32", "auto", True, False, "cuda"),
    ("win32", "nvidia", True, False, "cuda"),
    ("win32", "nvidia", False, False, "cpu"),
    ("win32", "amd", True, False, "directml"),
    ("win32", "auto", False, False, "directml"),
    ("linux", "auto", True, False, "cuda"),
    ("linux", "nvidia", True, False, "cuda"),
    ("linux", "auto", False, False, "cpu"),
    ("darwin", "auto", False, True, "mps"),
    ("darwin", "metal", False, True, "mps"),
    ("darwin", "cpu", False, True, "cpu"),
    ("darwin", "auto", False, False, "cpu"),
    ("darwin", "metal", False, False, "cpu"),
    ("darwin", "nvidia", True, True, "cpu"),
    ("win32", "metal", False, True, "cpu"),
])
def test_device_policy_respects_explicit_selection(platform, requested, cuda, mps, expected):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))
    assert select_speaker_backend(requested, torch, platform) == expected


def test_cpu_selection_does_not_probe_optional_gpu_apis():
    assert select_speaker_backend("cpu", object(), "darwin") == "cpu"


def test_metal_install_does_not_pull_windows_or_cuda_packages():
    assert installation_commands("metal", "cu128") == []
    assert installation_commands("cpu", "cu128") == []


@pytest.mark.parametrize("version", ["cu118", "cu126", "cu128"])
def test_nvidia_install_replaces_cpu_torch_with_matching_cuda_audio(version):
    commands = installation_commands("cuda", version)
    assert len(commands) == 1
    assert f"torch==2.7.1+{version}" in commands[0]
    assert f"torchaudio==2.7.1+{version}" in commands[0]
    assert commands[0][-1] == f"https://download.pytorch.org/whl/{version}"
    assert "directml" not in " ".join(commands[0])
