"""Recovery and user-workflow tests with small synthetic audio and fake ASR."""
import hashlib
import json
import os
import time
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from app import FileLoadWorker
from core import worker as workers
from core.job_store import JobStore, save_job, release_audio
from core.jobs import JobStatus, Task, TaskStatus
from core.settings import Settings
from core.transcript import transcript_rows, save_review, export_review
from core.live_worker import LiveTranscriptionWorker
from engine.audio_capture import AudioBuffer
from engine.live_timestamps import LiveTimeline, subtitle_cues
from engine.mapped_audio import mapped_audio
from engine import model_downloader as downloads
from engine.model_loader import get_model_download_info, validate_model_path


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def write_wav(path, samples):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def prepared_job(tmp_path, count=2):
    source = tmp_path / "source.wav"
    write_wav(source, np.zeros(16000))
    store = JobStore(tmp_path / "history")
    settings = Settings(language="en", device="cpu", output_format="both")
    job = store.create(source, settings, str(tmp_path))
    Path(job.temp_dir).mkdir()
    for index in range(count):
        path = Path(job.temp_dir) / f"{index}.wav"
        write_wav(path, np.zeros(16000))
        job.tasks.append(Task(str(path), index, duration=1))
    save_job(job)
    return store, job, settings


def fake_asr(monkeypatch, callback):
    monkeypatch.setattr(workers, "validate_model_path", lambda path: True)
    monkeypatch.setattr(workers, "load_whisper_model", lambda *a: (object(), None))
    monkeypatch.setattr(workers, "transcribe_chunk", callback)


def asr_result(text):
    return text, [json.dumps(dict(start=.1, end=.8, text=text))]


def test_restart_retries_only_failed_chunks_and_keeps_timeline(monkeypatch, tmp_path, qt_app):
    store, job, settings = prepared_job(tmp_path)
    def first(path, *args, **kwargs):
        if path.endswith("1.wav"):
            raise RuntimeError("temporary inference failure")
        return asr_result("first")
    fake_asr(monkeypatch, first)
    workers.TranscriptionWorker(job, settings).run()
    reloaded = store.load()[0]
    assert reloaded.status == JobStatus.ERROR
    assert [task.status for task in reloaded.tasks] == [TaskStatus.DONE, TaskStatus.ERROR]
    callback = Mock(return_value=asr_result("second"))
    fake_asr(monkeypatch, callback)
    resumed = workers.TranscriptionWorker(reloaded, settings)
    resumed.run()
    assert callback.call_count == 1
    assert callback.call_args.args[0].endswith("1.wav")
    assert resumed.total_audio_duration == 1
    assert resumed.processed_audio_duration == 1
    assert "00:00:01,100 --> 00:00:01,800" in (tmp_path / "source.srt").read_text()
    assert (tmp_path / "source.txt").read_text() == "first\nsecond"
    assert store.load()[0].status == JobStatus.DONE
    assert not Path(job.temp_dir).exists()


def test_running_job_recovers_as_interrupted_and_freezes_settings(tmp_path, qt_app):
    store, job, settings = prepared_job(tmp_path)
    job.status, job.tasks[0].status = JobStatus.RUNNING, TaskStatus.RUNNING
    save_job(job)
    loaded = store.load()[0]
    assert loaded.status == JobStatus.CANCELED
    assert loaded.tasks[0].status == TaskStatus.PENDING
    worker = workers.TranscriptionWorker(loaded, Settings(language="he", device="amd"))
    assert worker.settings.language == "en"
    assert worker.settings.device == "cpu"


def test_export_retry_needs_no_model_or_audio(monkeypatch, tmp_path, qt_app):
    store, job, settings = prepared_job(tmp_path, 1)
    job.tasks[0].status = TaskStatus.DONE
    job.tasks[0].text, job.tasks[0].srt_segments = asr_result("recovered")
    release_audio(job)
    forbidden = Mock(side_effect=AssertionError("ASR must not run"))
    monkeypatch.setattr(workers, "load_whisper_model", forbidden)
    workers.TranscriptionWorker(store.load()[0], settings).run()
    assert (tmp_path / "source.txt").read_text() == "recovered"
    forbidden.assert_not_called()


def test_corrupt_record_is_retained_and_other_jobs_load(tmp_path):
    store, job, _ = prepared_job(tmp_path)
    bad = store.root / "bad" / "job.json"
    bad.parent.mkdir()
    bad.write_text("{", encoding="utf-8")
    assert len(store.load()) == 1
    assert store.errors == [str(bad)]
    assert bad.exists()


def test_cache_cleanup_rejects_unowned_directory(tmp_path):
    _, job, _ = prepared_job(tmp_path)
    job.temp_dir = str(tmp_path)
    with pytest.raises(ValueError):
        release_audio(job)
    assert (tmp_path / "source.wav").exists()


def test_batch_reserves_output_names(tmp_path):
    store, job, _ = prepared_job(tmp_path)
    job.custom_output_filename = "source"
    (tmp_path / "source - 2.srt").touch()
    assert store.unique_stem(job.original_file_path, str(tmp_path), [job]) == "source - 3"


def test_changed_source_is_rejected_before_preparation(tmp_path, qt_app):
    _, job, _ = prepared_job(tmp_path)
    Path(job.original_file_path).write_bytes(b"changed")
    errors = []
    worker = FileLoadWorker(job.original_file_path, job=job)
    worker.error.connect(errors.append)
    worker.run()
    assert "changed" in errors[0]
    assert job.status == JobStatus.ERROR


def test_preparation_is_durable_and_releases_duplicate_wav(tmp_path, qt_app):
    source = tmp_path / "input.wav"
    write_wav(source, np.ones(8000) * 100)
    store = JobStore(tmp_path / "jobs")
    job = store.create(source)
    worker = FileLoadWorker(str(source), job=job)
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert not errors
    loaded = store.load()[0]
    assert loaded.tasks[0].duration == pytest.approx(.5)
    assert Path(loaded.tasks[0].chunk_path).exists()
    assert not (Path(loaded.temp_dir) / "audio.wav").exists()


def test_corrections_survive_new_completed_chunks_and_keep_originals(tmp_path):
    _, job, _ = prepared_job(tmp_path)
    job.tasks[0].status = TaskStatus.DONE
    job.tasks[0].text, job.tasks[0].srt_segments = asr_result("original")
    rows = transcript_rows(job)
    rows[0].update(text="corrected", speaker="Alex")
    save_review(job, rows)
    job.tasks[1].status = TaskStatus.DONE
    job.tasks[1].text, job.tasks[1].srt_segments = asr_result("next")
    rows = transcript_rows(job)
    assert [row["text"] for row in rows] == ["corrected", "next"]
    assert job.tasks[0].text == "original"
    export_review(rows, str(tmp_path), "edited")
    assert "Alex: corrected" in (tmp_path / "edited.txt").read_text()
    assert "00:00:01,100" in (tmp_path / "edited.srt").read_text()


@pytest.mark.parametrize("start,end", [(float("nan"), 2), (1, float("inf")), (-1, 1), (2, 1)])
def test_invalid_review_timing_does_not_overwrite_saved_edits(tmp_path, start, end):
    _, job, _ = prepared_job(tmp_path)
    rows = [dict(id="0:0", start=0, end=1, text="safe", speaker="")]
    save_review(job, rows)
    rows[0].update(start=start, end=end)
    with pytest.raises(ValueError):
        save_review(job, rows)
    assert json.loads(Path(job.record_path).with_name("review.json").read_text())["segments"][0]["text"] == "safe"


def test_audio_overflow_is_bounded_and_drains_accepted_audio_in_order():
    buffer = AudioBuffer(10, 1, max_seconds=2)
    assert buffer.write(np.arange(12, dtype=np.float32))
    assert not buffer.write(np.arange(12, 25, dtype=np.float32))
    assert buffer.overflowed
    assert buffer.duration_seconds == 2
    first = buffer.read_and_clear(max_seconds=.5)
    rest = buffer.read_and_clear()
    np.testing.assert_array_equal(np.concatenate([first, rest]), np.arange(20))
    assert not buffer.write(np.ones(1))
    assert buffer.read_and_clear() is None


def test_live_overlap_removes_repeat_but_preserves_new_repeated_word():
    timeline = LiveTimeline()
    assert timeline.append([(1.5, 2, "yes")], 0, 0, 2) == [(1.5, 2, "yes")]
    result = timeline.append([(.1, .6, "Yes,"), (.6, .9, "yes"), (.9, 1.2, "again")], 1.5, 2, 3)
    assert [word[2] for word in result] == ["yes", "again"]
    assert result[0][0] == pytest.approx(2.1)


def test_subtitles_preserve_silence_and_limit_length():
    cues = subtitle_cues([(0, .4, "first"), (.5, .9, "second"), (4, 4.4, "third")])
    assert cues == [(0, .9, "first second"), (4, 4.4, "third")]


def test_live_stop_drains_backlog_in_bounded_batches(qt_app):
    worker = LiveTranscriptionWorker(0, 16000, 1, Settings(language="en"))
    worker._session_start_time = datetime.now()
    audio = AudioBuffer(16000, 1)
    audio.write(np.full(25 * 16000, .1, dtype=np.float32))
    worker.stop()
    sizes = []
    def transcribe(samples, *args):
        sizes.append(len(samples) / 16000)
        return [(1, 2, "words")]
    worker._transcribe_buffer = transcribe
    worker._capture_loop(audio, object(), 1)
    assert sizes == [10, 10.5, 5.5]
    assert len(worker.session_segments) == 3
    assert worker.session_segments[1][:2] == (10.5, 11.5)


def test_disk_backed_audio_preserves_samples_and_cleans_up(tmp_path):
    paths = [tmp_path / "one.wav", tmp_path / "two.wav"]
    write_wav(paths[0], [-32768, 0, 32767])
    write_wav(paths[1], [10, -10])
    with mapped_audio(paths) as audio:
        assert isinstance(audio, np.memmap)
        cache = Path(audio.filename)
        np.testing.assert_allclose(audio, np.array([-32768, 0, 32767, 10, -10]) / 32768)
    assert not cache.exists()


def manifest_for(data):
    return dict(repo_id="fixture/model", revision="fixed-revision", files=[
        dict(name=name, size=len(content), algorithm="sha256", digest=hashlib.sha256(content).hexdigest())
        for name, content in data.items()])


def test_verified_download_repairs_only_corrupt_files(monkeypatch, tmp_path):
    data = {name: (name + " data").encode() for name in downloads.CT2_FILES}
    monkeypatch.setattr(downloads, "_manifest", lambda *a: manifest_for(data))
    fetched = []
    def download(repo, name, directory, revision, force):
        assert revision == "fixed-revision"
        fetched.append(name)
        (Path(directory) / name).write_bytes(data[name])
    monkeypatch.setattr(downloads, "_hf_download", download)
    downloads.download_ct2_model("fixture/model", str(tmp_path))
    assert validate_model_path(str(tmp_path))
    assert len(fetched) == 4
    fetched.clear()
    original = data["model.bin"]
    (tmp_path / "model.bin").write_bytes(b"x" * len(original))
    downloads.download_ct2_model("fixture/model", str(tmp_path))
    assert fetched == ["model.bin"]
    assert (tmp_path / "model.bin").read_bytes() == original
    assert not (tmp_path / ".ivrit-incomplete").exists()


def test_bad_download_never_replaces_existing_model(monkeypatch, tmp_path):
    target = tmp_path / "local.bin"
    target.write_bytes(b"previous usable model")
    monkeypatch.setattr(downloads, "_manifest", lambda *a: manifest_for({"remote.bin": b"correct"}))
    monkeypatch.setattr(downloads, "_hf_download", lambda repo, name, directory, *a:
                        (Path(directory) / name).write_bytes(b"corrupt"))
    with pytest.raises(ValueError, match="Checksum"):
        downloads.download_ggml_file("fixture/model", "remote.bin", str(tmp_path), target_name="local.bin")
    assert target.read_bytes() == b"previous usable model"


@pytest.mark.parametrize("language,engine", [("he", "faster-whisper"), ("en", "faster-whisper"),
                                             ("he", "whisper-cpp"), ("en", "whisper-cpp")])
def test_every_supported_model_has_download_metadata(language, engine):
    assert get_model_download_info(language, engine)["repo_id"]


def test_queue_continues_after_failure_and_skips_completed(monkeypatch, tmp_path, qt_app):
    from ui.job_library import JobLibrary
    store, first, settings = prepared_job(tmp_path, 1)
    second = store.create(first.original_file_path, settings, str(tmp_path))
    second.custom_output_filename = "second"
    Path(second.temp_dir).mkdir()
    second.tasks = [Task(first.tasks[0].chunk_path, 0, duration=1)]
    save_job(second)
    def transcribe(*args, **kwargs):
        if panel.active_job.record_path == first.record_path:
            raise RuntimeError("test failure")
        return asr_result("second completed")
    fake_asr(monkeypatch, transcribe)
    panel = JobLibrary(store, settings, FileLoadWorker, lambda: True, lambda: None)
    panel.start_jobs([first, second])
    for _ in range(500):
        QTest.qWait(10)
        if not panel.is_busy:
            break
    assert not panel.is_busy
    assert first.status == JobStatus.ERROR
    assert second.status == JobStatus.DONE
    assert (tmp_path / "second.txt").read_text() == "second completed"
    assert Path(first.tasks[0].chunk_path).exists()


def test_editor_close_saves_corrections_without_mutating_original(monkeypatch, tmp_path, qt_app):
    from PySide6.QtWidgets import QMessageBox
    from ui.transcript_editor import TranscriptEditor
    _, job, _ = prepared_job(tmp_path, 1)
    job.tasks[0].status = TaskStatus.DONE
    job.tasks[0].text, job.tasks[0].srt_segments = asr_result("original")
    dialog = TranscriptEditor(job)
    dialog.table.item(0, 3).setText("edited in the UI")
    assert dialog.dirty
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Save)
    dialog.reject()
    assert transcript_rows(job)[0]["text"] == "edited in the UI"
    assert job.tasks[0].text == "original"
    assert not dialog.dirty
    dialog.deleteLater()


def test_single_file_ui_creates_resumable_job_without_overwriting_output(monkeypatch, tmp_path, qt_app):
    import app
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    source = tmp_path / "single.wav"
    write_wav(source, np.ones(8000) * 100)
    (tmp_path / "single.txt").write_text("previous transcript")
    store = JobStore(tmp_path / "history")
    monkeypatch.setattr(app, "JobStore", lambda: store)
    monkeypatch.setattr(app, "load_settings", lambda: Settings(language="en", device="cpu", output_format="both",
                                                              output_folder=str(tmp_path), models_folder=str(tmp_path)))
    monkeypatch.setattr(app, "save_settings", lambda settings: None)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    monkeypatch.setattr(app.MainWindow, "_ensure_model_available", lambda *args: True)
    fake_asr(monkeypatch, lambda *a, **kw: asr_result("new transcript"))
    window = app.MainWindow({})
    try:
        window._select_file()
        for _ in range(1000):
            QTest.qWait(10)
            time.sleep(.001)
            if window._file_load_worker is None:
                break
        assert window._file_load_worker is None, window.status_label.text()
        assert window.current_job is not None
        window._start_transcription()
        for _ in range(1000):
            QTest.qWait(10)
            time.sleep(.001)
            if window.active_worker is None:
                break
        assert window.active_worker is None
        assert window.current_job.status == JobStatus.DONE
        assert (tmp_path / "single.txt").read_text() == "previous transcript"
        assert (tmp_path / "single - 2.txt").read_text() == "new transcript"
        assert store.load()[0].status == JobStatus.DONE
    finally:
        window.close()
