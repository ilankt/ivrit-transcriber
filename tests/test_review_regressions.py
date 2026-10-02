"""Regression coverage for persistence, cancellation, and Qt worker ownership."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QThread, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

import app
from core import settings, storage, worker as workers
from core.filenames import sanitize_output_stem
from core.jobs import Job, JobStatus, Task, TaskStatus
from core.live_worker import LiveTranscriptionWorker, save_live_session
from engine import checkpoint, ffmpeg_helper, model_downloader, transcriber, whisper_cpp_runner
from engine.audio_capture import AudioBuffer
from ui.model_download import ModelDownloadDialog


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def wait_until(predicate, timeout=3000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


@pytest.mark.parametrize("value", [[], None, {"threads": -1}, {"threads": "invalid"},
                                        {"language": "xx"}, {"output_format": "pdf"}])
def test_invalid_settings_recover_to_defaults(monkeypatch, tmp_path, value):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(settings, "get_settings_path", lambda: str(path))
    assert settings.load_settings() == settings.Settings()


def test_settings_migrate_and_round_trip_unicode(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"device": "gpu"}', encoding="utf-8")
    monkeypatch.setattr(settings, "get_settings_path", lambda: str(path))
    configured = settings.load_settings()
    assert configured.device == "nvidia"
    configured.output_folder = "D:/\u05d0\u05d1"
    settings.save_settings(configured)
    assert settings.load_settings() == configured


def test_failed_atomic_replace_preserves_original(monkeypatch, tmp_path):
    path = tmp_path / "transcript.txt"
    path.write_text("previous result", encoding="utf-8")
    monkeypatch.setattr(storage.os, "replace", Mock(side_effect=PermissionError("locked")))
    with pytest.raises(PermissionError):
        with storage.atomic_text_writer(path) as stream:
            stream.write("new result")
    assert path.read_text() == "previous result"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("seconds,expected", [(59.9996, "00:01:00,000"),
                                              (3599.9996, "01:00:00,000"),
                                              (86400.125, "24:00:00,125")])
def test_srt_rounding_carries(seconds, expected):
    assert checkpoint._format_srt_time(seconds) == expected


@pytest.mark.parametrize("stem", ["CON", "nul", "LPT1", "COM9", "COM\u00b9"])
def test_windows_reserved_output_stems(stem):
    assert sanitize_output_stem(stem).startswith("_")


def test_audio_chunks_sort_numerically(monkeypatch, tmp_path):
    for number in (99, 100, 999, 1000, 1001):
        (tmp_path / f"basename__part-{number:03d}.wav").touch()
    monkeypatch.setattr(ffmpeg_helper, "_run_ffmpeg", lambda *a: None)
    paths, error = ffmpeg_helper.split_audio("audio.wav", 1, str(tmp_path))
    assert error is None
    assert [int(Path(path).stem.rsplit("-", 1)[1]) for path in paths] == [99, 100, 999, 1000, 1001]


def configured_worker(monkeypatch, tmp_path, tasks=1):
    monkeypatch.setattr(workers, "validate_model_path", lambda path: True)
    monkeypatch.setattr(workers, "load_whisper_model", lambda *a: (object(), None))
    audio = tmp_path / "audio"
    audio.mkdir()
    job = Job("sample.wav", [Task(str(audio / f"{i}.wav"), i, duration=1) for i in range(tasks)],
              output_dir=str(tmp_path), temp_dir=str(audio))
    worker = workers.TranscriptionWorker(job, settings.Settings(device="cpu", output_format="txt"))
    statuses, finished = [], []
    worker.signals.job_status_updated.connect(lambda status, text: statuses.append((status, text)),
                                              Qt.ConnectionType.DirectConnection)
    worker.signals.finished.connect(lambda: finished.append(True), Qt.ConnectionType.DirectConnection)
    return worker, statuses, finished


def test_new_run_never_merges_old_checkpoints(monkeypatch, tmp_path, qt_app):
    checkpoint.save_chunk_checkpoint(str(tmp_path), "sample", 5, "OLD", [], 1)
    worker, statuses, finished = configured_worker(monkeypatch, tmp_path)
    monkeypatch.setattr(workers, "transcribe_chunk", lambda *a, **kw: ("NEW", []))
    worker.run()
    assert (tmp_path / "sample.txt").read_text() == "NEW"
    assert checkpoint.load_all_checkpoints(str(tmp_path), "sample")[0]["text"] == "OLD"
    assert not checkpoint.load_all_checkpoints(str(tmp_path), worker._checkpoint_name)
    assert statuses[-1][0] == JobStatus.DONE
    assert finished == [True]


@pytest.mark.parametrize("successful_chunks", [0, 1])
def test_failed_chunks_report_error_and_keep_audio(monkeypatch, tmp_path, qt_app, successful_chunks):
    worker, statuses, finished = configured_worker(monkeypatch, tmp_path, tasks=2)
    def transcribe(path, *a, **kw):
        if successful_chunks and path.endswith("0.wav"):
            return "partial", []
        raise RuntimeError("inference failed")
    monkeypatch.setattr(workers, "transcribe_chunk", transcribe)
    worker.run()
    assert statuses[-1][0] == JobStatus.ERROR
    assert ("Partial results saved" if successful_chunks else "No results") in statuses[-1][1]
    assert worker.job.tasks[-1].status == TaskStatus.ERROR
    assert Path(worker.job.temp_dir).is_dir()
    assert finished == [True]


def test_cleanup_failure_still_emits_finished(monkeypatch, tmp_path, qt_app):
    worker, statuses, finished = configured_worker(monkeypatch, tmp_path)
    monkeypatch.setattr(workers, "transcribe_chunk", lambda *a, **kw: ("result", []))
    monkeypatch.setattr(workers.shutil, "rmtree", Mock(side_effect=PermissionError("locked")))
    worker.run()
    assert finished == [True]
    assert statuses[-1][0] == JobStatus.DONE


def test_cancellation_keeps_inference_owned_until_it_returns(monkeypatch, tmp_path, qt_app):
    worker, statuses, finished = configured_worker(monkeypatch, tmp_path)
    done = threading.Event()
    worker.signals.finished.connect(done.set, Qt.ConnectionType.DirectConnection)
    entered, release = threading.Event(), threading.Event()
    def transcribe(*a, **kw):
        entered.set()
        assert release.wait(3)
        raise InterruptedError("canceled")
    monkeypatch.setattr(workers, "transcribe_chunk", transcribe)
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        assert entered.wait(2)
        worker.cancel()
        assert not done.wait(.4)
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert statuses[-1][0] == JobStatus.CANCELED
    assert finished == [True]


def test_cancel_does_not_return_partial_chunk_as_success():
    cancel = threading.Event()
    segment = SimpleNamespace(text="part", start=0, end=1)
    def segments():
        yield segment
        cancel.set()
        yield segment
    model = SimpleNamespace(transcribe=lambda *a, **kw: (segments(), SimpleNamespace(duration=2)))
    with pytest.raises(InterruptedError):
        transcriber.transcribe_chunk("audio", model, "en", 1, False, cancel_event=cancel)


def test_live_stop_flushes_short_final_buffer_and_uses_snapshot(tmp_path, qt_app):
    configured = settings.Settings(language="en", output_format="both")
    worker = LiveTranscriptionWorker(0, 16000, 1, configured)
    configured.language = "he"
    assert worker.settings.language == "en"
    worker._session_start_time = datetime(2026, 10, 2, 23, 59, 59)
    audio = AudioBuffer(16000, 1)
    audio.write(np.full(8000, .1, dtype=np.float32))
    segment = SimpleNamespace(start=0, end=.5, text="last words")
    model = SimpleNamespace(transcribe=lambda *a, **kw: (iter([segment]), None))
    worker.stop()
    cleanup = Mock()
    worker._capture_loop(audio, model, 1, cleanup)
    cleanup.assert_called_once()
    assert worker.session_segments == [(0, .5, "last words")]
    save_live_session(worker.session_segments, str(tmp_path), "live")
    assert "00:00:00,000 --> 00:00:00,500" in (tmp_path / "live.srt").read_text()


def test_failed_model_download_preserves_existing_data(monkeypatch, tmp_path, qt_app):
    (tmp_path / "model.bin").write_bytes(b"existing model")
    monkeypatch.setattr(model_downloader, "download_ct2_model", Mock(side_effect=OSError("offline")))
    worker = app.ModelDownloadWorker({"type": "ct2", "repo_id": "test/model"}, str(tmp_path))
    results = []
    worker.result.connect(lambda *result: results.append(result))
    worker.run()
    assert (tmp_path / "model.bin").read_bytes() == b"existing model"
    assert results == [(False, "offline")]


def test_download_cancel_waits_for_native_thread_completion(monkeypatch, tmp_path, qt_app):
    entered, release = threading.Event(), threading.Event()
    def download(*args):
        entered.set()
        assert release.wait(3)
    monkeypatch.setattr(model_downloader, "download_ct2_model", download)
    worker = app.ModelDownloadWorker({"type": "ct2", "repo_id": "test/model"}, str(tmp_path))
    dialog = ModelDownloadDialog(worker, "Downloading")
    dialog.show()
    worker.start()
    try:
        assert entered.wait(2)
        dialog.reject()
        assert dialog.isVisible()
        assert worker.isRunning()
    finally:
        release.set()
        worker.wait(3000)
    wait_until(lambda: not dialog.isVisible())
    assert not dialog.success
    assert dialog.message == "canceled"


def test_canceling_final_download_file_is_not_success(monkeypatch, tmp_path):
    canceled = [False]
    def download(*args):
        canceled[0] = True
    monkeypatch.setattr(model_downloader, "_hf_download", download)
    monkeypatch.setattr(model_downloader, "_manifest", lambda *a: dict(repo_id="test/model", revision="abc",
                        files=[dict(name="model.bin", size=10, algorithm="sha256", digest="0" * 64)]))
    with pytest.raises(InterruptedError):
        model_downloader.download_ggml_file("test/model", "model.bin", str(tmp_path),
                                             cancel_check=lambda: canceled[0])


def test_quiet_whisper_process_can_be_canceled(monkeypatch, tmp_path):
    # A real child writes no stderr while blocked, reproducing readline's old hang.
    script = tmp_path / "quiet_child.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    real_popen = subprocess.Popen
    processes = []
    def launch(*args, **kwargs):
        process = real_popen([sys.executable, str(script)], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(whisper_cpp_runner.subprocess, "Popen", launch)
    cancel = threading.Event()
    timer = threading.Timer(.2, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(InterruptedError):
            whisper_cpp_runner.transcribe_chunk_whispercpp("audio", "model", "cli", cancel_event=cancel)
    finally:
        timer.join()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert time.monotonic() - started < 5
    assert processes[0].poll() is not None


def make_window(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "load_settings", lambda: settings.Settings(device="cpu", models_folder=str(tmp_path)))
    monkeypatch.setattr(app, "save_settings", lambda *a: None)
    window = app.MainWindow({})
    window.show()
    return window


def test_file_menu_cannot_replace_active_job(monkeypatch, tmp_path, qt_app):
    window = make_window(monkeypatch, tmp_path)
    dialog = Mock(side_effect=AssertionError("must not open"))
    monkeypatch.setattr(app.QFileDialog, "getOpenFileName", dialog)
    window.active_worker = object()
    window._select_file()
    dialog.assert_not_called()
    window.active_worker = None
    window.close()


def test_close_keeps_window_alive_until_thread_finishes(monkeypatch, tmp_path, qt_app):
    release = threading.Event()
    class SlowStartup(QThread):
        def run(self):
            release.wait(3)
    window = make_window(monkeypatch, tmp_path)
    worker = SlowStartup(window)
    window.startup_worker = worker
    worker.start()
    try:
        window.close()
        assert window.isVisible()
        assert worker.isRunning()
    finally:
        release.set()
        worker.wait(3000)
    wait_until(lambda: not window.isVisible())


def test_live_save_failure_preserves_transcript_and_recovers(monkeypatch, tmp_path, qt_app):
    from ui import live_panel
    monkeypatch.setattr(live_panel, "list_loopback_devices", lambda: [])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: None)
    panel = live_panel.LiveTranscriptionPanel(settings.Settings(language="en"))
    panel._pending_save = ([(0, .5, "preserved")], "meeting", "both")
    save = live_panel.save_live_session
    monkeypatch.setattr(live_panel, "save_live_session", Mock(side_effect=PermissionError("locked")))
    assert not panel._save_pending(str(tmp_path))
    assert panel.has_unsaved_session
    assert panel.save_button.isEnabled()
    monkeypatch.setattr(live_panel, "save_live_session", save)
    assert panel._save_pending(str(tmp_path))
    assert not panel.has_unsaved_session
    assert (tmp_path / "meeting.txt").read_text().strip() == "preserved"


def test_cuda_detection_does_not_require_torch(monkeypatch):
    from engine import gpu_detector
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
    monkeypatch.setattr(gpu_detector.subprocess, "run", Mock(side_effect=FileNotFoundError))
    assert gpu_detector.detect_cuda_gpu() == (True, "NVIDIA CUDA")


def test_macos_loopback_inputs_are_listed(monkeypatch):
    from engine import audio_capture
    monkeypatch.setattr(audio_capture, "sys", SimpleNamespace(platform="darwin"))
    sd = SimpleNamespace(
        query_hostapis=lambda: [{"name": "Core Audio"}],
        query_devices=lambda: [{"name": "BlackHole 2ch", "hostapi": 0, "max_input_channels": 2,
                                "default_samplerate": 48000}])
    assert audio_capture._list_devices_sounddevice(sd)[0]["name"] == "BlackHole 2ch"


def test_ct2_model_requires_nonempty_files(tmp_path):
    from engine.model_loader import validate_model_path
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.json"):
        (tmp_path / name).touch()
    assert not validate_model_path(str(tmp_path))
    for path in tmp_path.iterdir():
        path.write_bytes(b"fixture")
    assert validate_model_path(str(tmp_path))


def test_export_failure_keeps_recoverable_checkpoints(monkeypatch, tmp_path, qt_app):
    worker, statuses, finished = configured_worker(monkeypatch, tmp_path)
    monkeypatch.setattr(workers, "transcribe_chunk", lambda *a, **kw: ("recover me", []))
    monkeypatch.setattr(workers, "merge_checkpoints_to_files", Mock(side_effect=PermissionError("locked output")))
    worker.run()
    assert statuses[-1][0] == JobStatus.ERROR
    assert checkpoint.load_all_checkpoints(str(tmp_path), worker._checkpoint_name)[0]["text"] == "recover me"
    assert finished == [True]


def test_live_cleanup_failure_still_finishes_thread(monkeypatch, qt_app):
    from core import live_worker
    worker = LiveTranscriptionWorker(0, 16000, 1, settings.Settings(device="cpu"))
    monkeypatch.setattr(worker, "_start_capture", lambda: (object(), Mock(side_effect=OSError("device removed"))))
    monkeypatch.setattr(live_worker, "validate_model_path", lambda *a: True)
    monkeypatch.setattr(live_worker, "load_whisper_model", lambda *a: (object(), None))
    monkeypatch.setattr(worker, "_capture_loop", lambda *a: None)
    completed = []
    worker.finished.connect(lambda: completed.append(True))
    worker.start()
    wait_until(lambda: bool(completed))
    assert worker.wait(1000)


def test_output_folder_error_is_handled_without_starting_worker(monkeypatch, tmp_path, qt_app):
    window = make_window(monkeypatch, tmp_path)
    chunk = tmp_path / "audio.wav"
    chunk.touch()
    window.current_job = Job("input.wav", [Task(str(chunk), 0, duration=1)])
    # An existing file is not a writable output directory.
    window.output_folder_edit.setText(str(chunk))
    warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    window._start_transcription()
    warning.assert_called_once()
    assert window.active_worker is None
    window.close()


def test_speaker_count_accepts_the_full_ui_range():
    assert settings.Settings(diarization_speakers=50).diarization_speakers == 50
