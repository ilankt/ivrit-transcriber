"""Speaker labeling, export timelines, optional setup, and worker integration."""
import json
import os
from pathlib import Path
import threading
import sys
import wave
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from core.jobs import Job, JobStatus, Task
from core.settings import Settings
from engine import diarization
from engine.checkpoint import merge_checkpoints_to_files, save_chunk_checkpoint


def segment(start, end, text, **extra):
    return json.dumps(dict(start=start, end=end, text=text, **extra))


def test_splits_at_word_speaker_change_without_losing_text():
    turns = [(0, 2, "Speaker 1"), (2, 5, "Speaker 2")]
    source = segment(0, 5, " Hello there. Good morning.", words=[
        {"start": 0, "end": 1, "word": " Hello"},
        {"start": 1, "end": 2, "word": " there."},
        {"start": 2, "end": 3, "word": " Good"},
        {"start": 3, "end": 5, "word": " morning."},
    ])
    text, encoded = diarization.label_segments([source], turns)
    labeled = list(map(json.loads, encoded))
    assert text == "Speaker 1: Hello there.\nSpeaker 2: Good morning."
    assert [(s["start"], s["end"]) for s in labeled] == [(0, 2), (2, 5)]
    assert "".join(s["text"] for s in labeled) == json.loads(source)["text"]


def test_same_speaker_is_preserved_across_chunks():
    turns = [(0, 10, "Speaker 1"), (20, 30, "Speaker 2"), (60, 65, "Speaker 1")]
    text, _ = diarization.label_segments([segment(0, 5, "Back again")], turns, offset=60)
    assert text == "Speaker 1: Back again"


def test_segment_only_backend_uses_total_overlap():
    turns = [(0, 2, "Speaker 1"), (2, 5, "Speaker 2"), (5, 7, "Speaker 1")]
    text, _ = diarization.label_segments([segment(0, 7, "Complete sentence")], turns)
    assert text == "Speaker 1: Complete sentence"


def test_missing_word_text_falls_back_without_dropping_words():
    source = segment(0, 3, "Keep every word", words=[{"start": 0, "end": 1, "word": "Keep"}])
    text, _ = diarization.label_segments([source], [(0, 3, "Speaker 1")])
    assert text == "Speaker 1: Keep every word"


def test_unmatched_speech_is_explicitly_unknown():
    text, _ = diarization.label_segments([segment(10, 11, "Unmatched")], [(0, 2, "Speaker 1")])
    assert text == "Speaker unknown: Unmatched"


def test_export_preserves_silence_empty_chunks_and_speaker_labels(tmp_path):
    checkpoints = [
        {"text": "Speaker 1: First", "duration": 60,
         "srt_segments": [segment(0, 5, "First", speaker="Speaker 1")]},
        {"text": "", "duration": 60, "srt_segments": []},
        {"text": "Speaker 2: Later", "duration": 10,
         "srt_segments": [segment(1, 3, "Later", speaker="Speaker 2")]},
    ]
    txt, srt = merge_checkpoints_to_files(str(tmp_path), "result", checkpoints, "both")
    assert "00:02:01,000 --> 00:02:03,000" in Path(srt).read_text()
    assert "Speaker 2: Later" in Path(txt).read_text()
    assert "Speaker 2: Later" in Path(srt).read_text()


def test_failed_chunk_does_not_shift_next_subtitle(tmp_path):
    save_chunk_checkpoint(str(tmp_path), "result", 0, "first", [segment(0, 4, "first")], 60, 0)
    save_chunk_checkpoint(str(tmp_path), "result", 2, "third", [segment(2, 5, "third")], 10, 120)
    _, srt = merge_checkpoints_to_files(str(tmp_path), "result")
    assert "00:02:02,000 --> 00:02:05,000" in Path(srt).read_text()
    assert "Speaker" not in Path(srt).read_text()


def test_model_requires_complete_manifest(tmp_path):
    (tmp_path / "config.yaml").write_text("config")
    assert not diarization.model_available(tmp_path)
    marker = tmp_path / ".ivrit-ready.json"
    marker.write_text(json.dumps({"files": {"config.yaml": 6, "weights.bin": 3}}))
    assert not diarization.model_available(tmp_path)
    (tmp_path / "weights.bin").write_bytes(b"abc")
    assert diarization.model_available(tmp_path)
    (tmp_path / "weights.bin").write_bytes(b"ab")
    assert not diarization.model_available(tmp_path)


def test_canceled_download_is_not_marked_ready(tmp_path, monkeypatch):
    import huggingface_hub
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha="fixed-revision")
    api.list_repo_files.return_value = ["config.yaml", "weights.bin"]
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: api)
    canceled = [False]

    def download(repo, name, **kwargs):
        assert kwargs["revision"] == "fixed-revision"
        path = tmp_path / name
        path.write_text("asset")
        canceled[0] = True
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(InterruptedError):
        diarization.download_model(str(tmp_path), cancel_check=lambda: canceled[0])
    assert not diarization.model_available(tmp_path)


def test_cancel_before_inference_does_not_load_optional_libraries():
    event = threading.Event()
    event.set()
    with pytest.raises(InterruptedError):
        diarization.diarize_chunks([], "missing", cancel_event=event)


@pytest.mark.parametrize("device,platform,acceleration,expected_device", [
    ("cpu", "win32", "ready", "CPU"),
    ("amd", "win32", "ready", "GPU (DirectML)"),
    ("auto", "win32", "ready", "GPU (DirectML)"),
    ("amd", "win32", "setup_error", "CPU; GPU acceleration unavailable"),
    ("amd", "win32", "runtime_error", "CPU; GPU acceleration unavailable"),
    ("metal", "darwin", "ready", "GPU (Metal)"),
    ("auto", "darwin", "ready", "GPU (Metal)"),
    ("metal", "darwin", "setup_error", "CPU; GPU acceleration unavailable"),
    ("metal", "darwin", "runtime_error", "CPU; GPU acceleration unavailable"),
])
def test_pipeline_receives_entire_recording_and_normalizes_speakers(
        tmp_path, monkeypatch, device, platform, acceleration, expected_device):
    import numpy as np
    chunks = []
    for index in range(2):
        path = tmp_path / f"chunk{index}.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(np.full(16000, index * 1000, dtype="<i2").tobytes())
        chunks.append(str(path))
    annotation = Mock()
    annotation.itertracks.return_value = [
        (SimpleNamespace(start=0.1, end=0.8), "track", "cluster_z"),
        (SimpleNamespace(start=1.0, end=1.5), "track", "cluster_a"),
        (SimpleNamespace(start=1.5, end=2.0), "track", "cluster_z"),
    ]
    calls = []

    def apply(file, hook, num_speakers):
        calls.append(file)
        assert file["waveform"].shape == (1, 32000)
        assert file["sample_rate"] == 16000 and num_speakers == 2
        assert file["waveform"][0, 20000] == pytest.approx(1000 / 32768)
        hook("segmentation", None, total=2, completed=2)
        hook("embeddings", None, total=2, completed=2)
        return annotation

    pipeline = Mock(side_effect=apply)
    loader = Mock(return_value=pipeline)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        from_numpy=lambda array: SimpleNamespace(unsqueeze=lambda dim: array[np.newaxis]),
        inference_mode=nullcontext,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    accelerator = Mock(return_value=SimpleNamespace(accelerated=acceleration == "ready"))
    if acceleration == "setup_error":
        accelerator.side_effect = RuntimeError("GPU unavailable")
    monkeypatch.setitem(sys.modules, "engine.speaker_directml", SimpleNamespace(enable_directml=accelerator))
    monkeypatch.setitem(sys.modules, "engine.speaker_metal", SimpleNamespace(enable_metal=accelerator))
    speech_accelerator = Mock(return_value=SimpleNamespace(accelerated=acceleration == "ready"))
    monkeypatch.setitem(sys.modules, "engine.speaker_segmentation", SimpleNamespace(enable_segmentation_gpu=speech_accelerator))
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(diarization, "_load_pipeline", loader)
    monkeypatch.setattr(diarization, "dependency_error", lambda: None)
    monkeypatch.setattr(diarization, "model_available", lambda path: True)
    statuses = []
    progress = []
    turns = diarization.diarize_chunks(chunks, "local-model", device=device, num_speakers=2,
                                      status_callback=statuses.append, progress_callback=lambda *args: progress.append(args))
    assert len(calls) == 1
    assert pipeline.clustering.constrained_assignment is True
    loader.assert_called_once_with("local-model")
    assert turns == [(0.1, 0.8, "Speaker 1"), (1.0, 1.5, "Speaker 2"), (1.5, 2.0, "Speaker 1")]
    speech_device = expected_device.replace("GPU (", "GPU + CPU (")
    assert statuses[-2] == f"Detecting speakers: finding speech — {speech_device}"
    assert statuses[-1] == f"Detecting speakers: comparing voices — {expected_device}"
    assert progress[0] == (100, "Step 1/3 — 100% of step — Step complete")
    assert progress[1] == (100, "Step 2/3 — 100% of step — Step complete")
    assert accelerator.call_count == (0 if device == "cpu" else 1)


@pytest.mark.parametrize("failure", [None, "setup", "inference", "cancel"])
@pytest.mark.parametrize("num_speakers", [0, 2])
def test_cuda_inference_and_recovery_do_not_swallow_cancellation(tmp_path, monkeypatch, failure, num_speakers):
    import numpy as np
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00" * 32000)
    result = Mock()
    result.itertracks.return_value = [(SimpleNamespace(start=0.0, end=1.0), "track", "a")]

    def infer(file, hook, **kwargs):
        assert kwargs == ({"num_speakers": 2} if num_speakers else {})
        if num_speakers:
            assert gpu.clustering.constrained_assignment is True
        hook("embeddings", None, total=1, completed=1)
        if failure == "inference":
            raise RuntimeError("CUDA out of memory")
        if failure == "cancel":
            raise InterruptedError("Canceled by user")
        return result

    gpu = Mock(side_effect=infer)
    if failure == "setup":
        gpu.to.side_effect = RuntimeError("CUDA initialization failed")

    def cpu_infer(file, hook, **kwargs):
        assert kwargs == ({"num_speakers": 2} if num_speakers else {})
        if num_speakers:
            assert cpu.clustering.constrained_assignment is True
        hook("embeddings", None, total=1, completed=1)
        return result

    cpu = Mock(side_effect=cpu_infer)
    loader = Mock(side_effect=[gpu, cpu])
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=Mock()),
                            device=lambda name: name, inference_mode=nullcontext,
                            from_numpy=lambda array: SimpleNamespace(unsqueeze=lambda dim: array[np.newaxis]))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(diarization, "_load_pipeline", loader)
    monkeypatch.setattr(diarization, "dependency_error", lambda: None)
    monkeypatch.setattr(diarization, "model_available", lambda path: True)
    statuses = []
    if failure == "cancel":
        with pytest.raises(InterruptedError):
            diarization.diarize_chunks([str(audio_path)], "model", device="nvidia", num_speakers=num_speakers, status_callback=statuses.append)
        assert loader.call_count == 1
        cpu.assert_not_called()
    else:
        turns = diarization.diarize_chunks([str(audio_path)], "model", device="nvidia", num_speakers=num_speakers, status_callback=statuses.append)
        assert turns == [(0.0, 1.0, "Speaker 1")]
        if failure:
            assert loader.call_count == 2 and cpu.call_count == 1
            assert "CPU; GPU acceleration unavailable" in statuses[-1]
        else:
            assert loader.call_count == 1
            assert "GPU (CUDA)" in statuses[-1]
    gpu.to.assert_called_once_with("cuda")
    torch.cuda.empty_cache.assert_called_once()


def test_known_speakers_cannot_collapse_distinct_local_tracks():
    import numpy as np
    from pyannote.audio.pipelines.clustering import AgglomerativeClustering
    clustering = AgglomerativeClustering()
    pipeline = SimpleNamespace(clustering=clustering)
    # Two clear voice prototypes, then two less clear but locally distinct tracks.
    embeddings = np.array([[[1., 0.], [0., 1.]], [[.8, .6], [.9, .435]]])
    indices = (np.array([0, 0]), np.array([0, 1]), np.array([0, 1]))
    old, _, _ = clustering.assign_embeddings(embeddings, *indices, constrained=False)
    assert old[1].tolist() == [0, 0]
    diarization._configure_speaker_assignment(pipeline, 2)
    corrected, _, _ = clustering.assign_embeddings(
        embeddings, *indices, constrained=clustering.constrained_assignment)
    assert corrected.tolist() == [[0, 1], [1, 0]]


@pytest.mark.parametrize("count", [0, 1, 3, 4])
def test_other_speaker_counts_keep_original_assignment(count):
    clustering = SimpleNamespace(constrained_assignment=False)
    diarization._configure_speaker_assignment(SimpleNamespace(clustering=clustering), count)
    assert clustering.constrained_assignment is False


def test_zero_length_word_uses_speaker_at_timestamp():
    source = segment(0, 2, " Hi!", words=[
        {"start": 0, "end": 1, "word": " Hi"},
        {"start": 1, "end": 1, "word": "!"},
    ])
    text, _ = diarization.label_segments([source], [(0, 2, "Speaker 1")])
    assert text == "Speaker 1: Hi!"


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_speaker_settings_persist_without_credentials(qt_app, tmp_path):
    from ui.settings_panel import SettingsPanel
    settings = Settings(models_folder=str(tmp_path))
    panel = SettingsPanel(settings, {})
    assert not panel.diarization_checkbox.isChecked()
    panel.diarization_checkbox.setChecked(True)
    panel.speaker_count.setValue(2)
    panel.save_settings()
    restored = Settings.model_validate_json(settings.model_dump_json())
    assert restored.diarization_enabled and restored.diarization_speakers == 2
    assert "token" not in restored.model_dump()
    panel.deleteLater()


def test_speaker_model_is_not_offered_for_cleanup():
    from engine.model_loader import get_all_known_model_names
    assert diarization.MODEL_FOLDER in get_all_known_model_names()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("backend", ["faster-whisper", "whisper-cpp"])
def test_worker_labels_both_backends_only_when_enabled(qt_app, monkeypatch, tmp_path, enabled, backend):
    from core import worker as module
    settings = Settings(device="cpu", diarization_enabled=enabled, output_format="both")
    tasks = [Task("chunk0.wav", 0, duration=60), Task("chunk1.wav", 1, duration=10)]
    job = Job("source.wav", tasks, output_dir=str(tmp_path))
    monkeypatch.setattr(module, "determine_engine", lambda *a: backend)
    monkeypatch.setattr(module, "validate_model_path", lambda *a: True)
    monkeypatch.setattr(module, "validate_ggml_model", lambda *a: True)
    monkeypatch.setattr(module, "resolve_whispercpp_binary", lambda *a: ("cli", None))
    monkeypatch.setattr(module, "load_whisper_model", lambda *a: (object(), None))
    diarize = Mock(return_value=[(0, 3, "Speaker 1"), (60, 63, "Speaker 2")])
    monkeypatch.setattr(diarization, "diarize_chunks", diarize)
    transcribe = Mock(return_value=("Hello", [segment(0, 3, "Hello")]))
    monkeypatch.setattr(module, "transcribe_chunk", transcribe)
    monkeypatch.setattr(module, "transcribe_chunk_whispercpp", transcribe)
    worker = module.TranscriptionWorker(job, settings)
    statuses = []
    worker.signals.job_status_updated.connect(lambda status, _: statuses.append(status), Qt.DirectConnection)
    worker.run()
    assert statuses[-1] == JobStatus.DONE
    assert all(call.kwargs.get("word_timestamps") is enabled for call in transcribe.call_args_list)
    output = (tmp_path / "source.srt").read_text()
    assert "00:01:00,000 --> 00:01:03,000" in output
    if enabled:
        diarize.assert_called_once()
        assert "Speaker 1: Hello" in output and "Speaker 2: Hello" in output
        assert "Speaker 2: Hello" in (tmp_path / "source.txt").read_text()
    else:
        diarize.assert_not_called()
        assert "Speaker" not in output


def test_worker_reports_diarization_cancellation(qt_app, monkeypatch, tmp_path):
    from core import worker as module
    monkeypatch.setattr(module, "determine_engine", lambda *a: "faster-whisper")
    monkeypatch.setattr(diarization, "diarize_chunks", Mock(side_effect=InterruptedError))
    job = Job("source.wav", [Task("chunk.wav", 0, duration=5)], output_dir=str(tmp_path))
    worker = module.TranscriptionWorker(job, Settings(diarization_enabled=True))
    statuses = []
    worker.signals.job_status_updated.connect(lambda status, _: statuses.append(status), Qt.DirectConnection)
    worker.run()
    assert statuses[-1] == JobStatus.CANCELED
    assert not (tmp_path / "source.srt").exists()
