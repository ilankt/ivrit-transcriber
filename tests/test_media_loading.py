"""Exercise real container decoding and the file-loading workflow (requires FFmpeg)."""

import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app import FileLoadWorker
from engine.ffmpeg_helper import extract_audio, probe_media, _select_audio_stream


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and FFprobe must be on PATH",
)


def encode(*args):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True, capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


def load_file(path):
    jobs, errors, info = [], [], []
    worker = FileLoadWorker(str(path))
    worker.loaded.connect(jobs.append, Qt.ConnectionType.DirectConnection)
    worker.error.connect(errors.append, Qt.ConnectionType.DirectConnection)
    worker.file_info_ready.connect(
        lambda duration, video: info.append((duration, video)),
        Qt.ConnectionType.DirectConnection,
    )
    worker.run()
    return jobs, errors, info


def read_pcm(path):
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
        assert wav.getsampwidth() == 2
        assert wav.getcomptype() == "NONE"
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def test_whisper_decoder_accepts_prepared_audio(tmp_path):
    """Exercise the actual ASR decoder API, not just external FFmpeg decoding."""
    from faster_whisper.audio import decode_audio
    source = tmp_path / "prepared.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(np.full(16000, 1000, dtype="<i2").tobytes())
    decoded = decode_audio(str(source))
    assert decoded.shape == (16000,)
    assert decoded[0] == pytest.approx(1000 / 32768)


@pytest.mark.parametrize("extension,video,codec", [
    ("m4a", False, "aac"),
    ("mp4", False, "aac"),
    ("mp4", True, "aac"),
    ("mp3", False, "libmp3lame"),
    ("flac", False, "flac"),
    ("wav", False, "pcm_s24le"),
])
def test_file_loader_decodes_compressed_and_pcm_inputs(tmp_path, qt_app, extension, video, codec):
    source = tmp_path / f"sample with spaces.{extension}"
    args = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1.2"]
    if video:
        args += ["-f", "lavfi", "-i", "color=size=32x32:rate=5:duration=1.2", "-c:v", "mpeg4"]
    encode(*args, "-ac", "2", "-c:a", codec, source)
    jobs, errors, info = load_file(source)
    assert not errors
    assert len(jobs) == 1
    job = jobs[0]
    try:
        assert info[0][1] is video
        assert len(job.tasks) == 1
        samples = read_pcm(job.tasks[0].chunk_path)
        assert len(samples) / 16000 == pytest.approx(1.2, abs=0.1)
        assert np.max(np.abs(samples.astype(float))) > 1000
    finally:
        shutil.rmtree(job.temp_dir)


def test_chunks_preserve_all_samples_across_minute_boundary(tmp_path, qt_app):
    source = tmp_path / "long.m4a"
    encode("-f", "lavfi", "-i", "sine=sample_rate=44100:duration=61.2", "-c:a", "aac", source)
    jobs, errors, _ = load_file(source)
    assert not errors
    job = jobs[0]
    try:
        assert len(job.tasks) == 2
        reference = tmp_path / "reference.wav"
        assert extract_audio(str(source), str(reference)) is None
        original = read_pcm(reference)
        joined = np.concatenate([read_pcm(task.chunk_path) for task in job.tasks])
        np.testing.assert_array_equal(joined, original)
        assert sum(task.duration for task in job.tasks) == pytest.approx(len(original) / 16000, abs=0.001)
    finally:
        shutil.rmtree(job.temp_dir)


def test_extract_uses_default_audio_track(tmp_path):
    source, output = tmp_path / "multiple.mp4", tmp_path / "selected.wav"
    encode(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
        "-map", "0:a", "-map", "1:a", "-c:a", "aac",
        "-disposition:a:0", "0", "-disposition:a:1", "default", source,
    )
    assert extract_audio(str(source), str(output)) is None
    samples = read_pcm(output).astype(float)
    spectrum = np.abs(np.fft.rfft(samples))
    frequency = np.fft.rfftfreq(len(samples), 1 / 16000)[np.argmax(spectrum)]
    assert frequency == pytest.approx(880, abs=3)


def test_audio_selection_without_default_uses_first_track():
    streams = [
        {"index": 0, "codec_type": "video"},
        {"index": 1, "codec_type": "audio", "channels": 1},
        {"index": 2, "codec_type": "audio", "channels": 6},
    ]
    assert _select_audio_stream({"streams": streams})["index"] == 1


def test_m4a_cover_art_is_not_video(tmp_path, qt_app):
    cover, source = tmp_path / "cover.jpg", tmp_path / "cover.m4a"
    encode("-f", "lavfi", "-i", "color=size=32x32", "-frames:v", "1", cover)
    encode(
        "-f", "lavfi", "-i", "sine=duration=1", "-i", cover,
        "-map", "0:a", "-map", "1:v", "-c:a", "aac", "-c:v", "copy",
        "-disposition:v", "attached_pic", source,
    )
    jobs, errors, info = load_file(source)
    assert not errors
    try:
        assert info[0][1] is False
        read_pcm(jobs[0].tasks[0].chunk_path)
    finally:
        shutil.rmtree(jobs[0].temp_dir)


def test_video_without_audio_reports_clear_error(tmp_path, qt_app):
    source = tmp_path / "silent.mp4"
    encode("-f", "lavfi", "-i", "color=size=32x32:duration=1", "-c:v", "mpeg4", source)
    jobs, errors, _ = load_file(source)
    assert not jobs
    assert len(errors) == 1
    assert "does not contain an audio track" in errors[0]


def test_corrupt_media_reports_probe_error(tmp_path):
    source = tmp_path / "broken.m4a"
    source.write_bytes(b"not a media file")
    duration, error = probe_media(str(source))
    assert duration is None
    assert error
