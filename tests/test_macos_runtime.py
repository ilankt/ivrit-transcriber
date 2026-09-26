"""Platform-independent regressions for Mac backend routing and subprocess setup."""
import io
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from app import ModelDownloadWorker
from core import runtime
from core.settings import Settings
from engine import ffmpeg_helper, gpu_detector, model_loader, whisper_cpp_runner
from ui.settings_panel import SettingsPanel


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def metal_info():
    return {"apple_metal": {"available": True, "info": "Apple Silicon"}}


@pytest.mark.parametrize("platform_name,architecture,arm64,expected", [
    ("darwin", "arm64", None, True),
    ("darwin", "x86_64", "1", True),
    ("darwin", "x86_64", "0", False),
    ("win32", "ARM64", None, False),
    ("linux", "aarch64", None, False),
])
def test_detect_apple_silicon(monkeypatch, platform_name, architecture, arm64, expected):
    monkeypatch.setattr(gpu_detector, "sys", SimpleNamespace(platform=platform_name))
    monkeypatch.setattr(gpu_detector.platform, "machine", lambda: architecture)
    probe = Mock(return_value=SimpleNamespace(returncode=0, stdout=arm64))
    monkeypatch.setattr(gpu_detector.subprocess, "run", probe)
    assert gpu_detector.detect_metal_gpu()[0] is expected
    if arm64 is None:
        probe.assert_not_called()


def test_macos_skips_cuda_and_vulkan_probes(monkeypatch):
    monkeypatch.setattr(gpu_detector, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(gpu_detector, "detect_metal_gpu", lambda: (True, "Apple Silicon"))
    cuda, vulkan = Mock(), Mock()
    monkeypatch.setattr(gpu_detector, "detect_cuda_gpu", cuda)
    monkeypatch.setattr(gpu_detector, "detect_vulkan_gpu", vulkan)
    assert gpu_detector.detect_all_gpus()["apple_metal"]["available"]
    cuda.assert_not_called()
    vulkan.assert_not_called()


@pytest.mark.parametrize("device,info,engine", [
    ("metal", {}, "whisper-cpp"),
    ("auto", metal_info(), "whisper-cpp"),
    ("cpu", metal_info(), "faster-whisper"),
    ("auto", {}, "faster-whisper"),
    ("amd", {}, "whisper-cpp"),
    ("nvidia", {}, "faster-whisper"),
    ("auto", {"amd_vulkan": {"available": True}}, "whisper-cpp"),
    ("auto", {"nvidia_cuda": {"available": True}}, "faster-whisper"),
])
def test_engine_selection(device, info, engine):
    assert runtime.determine_engine(device, info) == engine


def test_worker_auto_engine_uses_detected_metal(monkeypatch):
    monkeypatch.setattr(gpu_detector, "detect_all_gpus", metal_info)
    assert runtime.determine_engine("auto") == "whisper-cpp"


def test_settings_auto_uses_ggml_download_and_cpu_uses_ct2(qt_app, tmp_path):
    settings = Settings(language="en", models_folder=str(tmp_path))
    panel = SettingsPanel(settings, metal_info())
    assert panel.device_combo.findData("metal") >= 0
    assert "Apple GPU" in panel.device_combo.currentText()
    assert panel.download_button.isHidden() is False
    (tmp_path / "ggml-large-v3.bin").write_bytes(b"\0" * (11 * 1024 * 1024))
    panel._refresh_model_status()
    assert panel.model_status_label.text().startswith("Ready")
    panel.device_combo.setCurrentIndex(panel.device_combo.findData("cpu"))
    assert panel.model_status_label.text().startswith("Not downloaded")
    panel.deleteLater()


def test_saved_metal_selection_survives_startup_detection(qt_app, tmp_path):
    settings = Settings(device="metal", models_folder=str(tmp_path))
    panel = SettingsPanel(settings, {"detecting": True})
    assert panel.device_combo.currentData() == "metal"
    panel.update_gpu_info(metal_info())
    assert panel.device_combo.currentData() == "metal"
    panel.save_settings()
    assert settings.device == "metal"
    panel.deleteLater()


def test_homebrew_binary_found_without_shell_path(monkeypatch):
    monkeypatch.setattr(whisper_cpp_runner, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(whisper_cpp_runner.shutil, "which", lambda name: None)
    monkeypatch.setattr(whisper_cpp_runner.os.path, "isfile", lambda path: path == "/opt/homebrew/bin/whisper-cli")
    monkeypatch.setattr(whisper_cpp_runner.os, "access", lambda path, mode: True)
    assert whisper_cpp_runner.get_whispercpp_binary_path("/app") == "/opt/homebrew/bin/whisper-cli"


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
def test_homebrew_media_tools_found_without_shell_path(monkeypatch, name):
    monkeypatch.setattr(ffmpeg_helper, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(ffmpeg_helper.shutil, "which", lambda name: None)
    expected = os.path.join("/opt/homebrew/bin", name)
    monkeypatch.setattr(ffmpeg_helper.os.path, "isfile", lambda path: path == expected)
    monkeypatch.setattr(ffmpeg_helper.os, "access", lambda path, mode: True)
    assert ffmpeg_helper._find_executable(name) == expected


def test_nonzero_whisper_help_is_rejected(monkeypatch):
    monkeypatch.setattr(whisper_cpp_runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    assert not whisper_cpp_runner.validate_whispercpp_binary("broken-whisper-cli")


def test_hebrew_ggml_download_is_renamed_to_registered_path(monkeypatch, tmp_path, qt_app):
    from engine import model_downloader
    info = model_loader.get_model_download_info("he", "whisper-cpp")
    target = model_loader.resolve_model_path("he", "whisper-cpp", str(tmp_path), str(tmp_path))
    def download(repo_id, filename, directory, *args):
        assert repo_id == "ivrit-ai/whisper-large-v3-ggml"
        (Path(directory) / filename).write_bytes(b"test model")
    monkeypatch.setattr(model_downloader, "download_ggml_file", download)
    worker = ModelDownloadWorker(info, target)
    results = []
    worker.finished.connect(lambda success, message: results.append((success, message)))
    worker.run()
    assert results == [(True, "")]
    assert Path(target).read_bytes() == b"test model"
    assert not (tmp_path / "ggml-model.bin").exists()


def test_ct2_auto_on_mac_never_attempts_cuda(monkeypatch):
    monkeypatch.setattr(model_loader, "sys", SimpleNamespace(platform="darwin"))
    model = Mock()
    constructor = Mock(return_value=model)
    monkeypatch.setattr(model_loader, "WhisperModel", constructor)
    assert model_loader.load_whisper_model("model", "auto", "auto", 4) == (model, None)
    constructor.assert_called_once_with("model", device="cpu", compute_type="auto", cpu_threads=4)


@pytest.mark.parametrize("gpu_available", [True, False])
def test_metal_runner_keeps_gpu_enabled_and_reports_cpu_fallback(monkeypatch, gpu_available):
    calls = []
    process = SimpleNamespace(returncode=0, wait=Mock(), terminate=Mock())
    process.stderr = io.BytesIO(
        b"whisper_backend_init_gpu: using Metal backend\n" if gpu_available
        else b"whisper_backend_init_gpu: no GPU found\n"
    )
    def popen(args, **kwargs):
        calls.append(args)
        assert kwargs["stdout"] == whisper_cpp_runner.subprocess.DEVNULL
        output = args[args.index("--output-file") + 1]
        Path(output + ".txt").write_text("Test transcript", encoding="utf-8")
        Path(output + ".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nTest transcript\n", encoding="utf-8")
        return process
    monkeypatch.setattr(whisper_cpp_runner.subprocess, "Popen", popen)
    if gpu_available:
        text, segments = whisper_cpp_runner.transcribe_chunk_whispercpp(
            "input.wav", "model.bin", "whisper-cli", threads=6, require_metal=True,
        )
        assert text == "Test transcript"
        assert len(segments) == 1
    else:
        with pytest.raises(RuntimeError, match="Metal GPU initialization failed"):
            whisper_cpp_runner.transcribe_chunk_whispercpp(
                "input.wav", "model.bin", "whisper-cli", threads=6, require_metal=True,
            )
        process.terminate.assert_called_once()
    assert "--no-gpu" not in calls[0]
    assert calls[0][calls[0].index("--threads") + 1] == "6"


def test_live_worker_maps_metal_to_supported_cpu_device(monkeypatch, qt_app):
    from core import live_worker
    worker = live_worker.LiveTranscriptionWorker(0, 16000, 1, Settings(device="metal"))
    monkeypatch.setattr(worker, "_start_capture", lambda: (object(), lambda: None))
    monkeypatch.setattr(live_worker, "validate_model_path", lambda path: True)
    load = Mock(return_value=(object(), None))
    monkeypatch.setattr(live_worker, "load_whisper_model", load)
    monkeypatch.setattr(worker, "_capture_loop", lambda *args: None)
    worker.run()
    assert load.call_args.args[1] == "cpu"


def test_file_worker_auto_metal_uses_ggml_and_gpu(monkeypatch, qt_app, tmp_path):
    from core import worker as worker_module
    from core.jobs import Job, Task, JobStatus
    monkeypatch.setattr(gpu_detector, "detect_all_gpus", metal_info)
    monkeypatch.setattr(worker_module, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(worker_module, "get_whispercpp_binary_path", lambda base: "/opt/homebrew/bin/whisper-cli")
    monkeypatch.setattr(worker_module, "validate_whispercpp_binary", lambda path: True)
    monkeypatch.setattr(worker_module, "validate_ggml_model", lambda path: True)
    transcribe = Mock(return_value=("Test transcript", []))
    monkeypatch.setattr(worker_module, "transcribe_chunk_whispercpp", transcribe)
    ct2_load = Mock(side_effect=AssertionError("Metal must not load CTranslate2"))
    monkeypatch.setattr(worker_module, "load_whisper_model", ct2_load)
    settings = Settings(device="auto", threads=6, models_folder=str(tmp_path))
    job = Job("input.m4a", [Task("chunk.wav", 0, duration=1)], output_dir=str(tmp_path / "output"))
    worker = worker_module.TranscriptionWorker(job, settings)
    statuses = []
    worker.signals.job_status_updated.connect(lambda status, message: statuses.append(status))
    worker.run()
    assert statuses[-1] == JobStatus.DONE
    assert transcribe.call_args.kwargs["model_path"].endswith("ggml-ivrit-large-v3.bin")
    assert transcribe.call_args.kwargs["use_gpu"] is True
    assert transcribe.call_args.kwargs["require_metal"] is True
    assert transcribe.call_args.kwargs["threads"] == 6
    ct2_load.assert_not_called()


def test_main_window_constructs_with_metal_settings(monkeypatch, qt_app, tmp_path):
    import app
    monkeypatch.setattr(app, "load_settings", lambda: Settings(device="metal", models_folder=str(tmp_path)))
    monkeypatch.setattr(app, "save_settings", lambda settings: None)
    window = app.MainWindow(metal_info())
    window.show()
    qt_app.processEvents()
    assert window.isVisible()
    assert window.settings_panel.device_combo.currentData() == "metal"
    window.close()
