"""Opt-in hardware checks using a generated tone or an explicitly supplied fixture.

Examples:
  python scripts/hardware_smoke.py --loopback --report logs/loopback.json
  python scripts/hardware_smoke.py --media build/speaker-sample/conversation.wav --device amd

Loopback plays a quiet two-second tone and never saves captured audio.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_loopback():
    import numpy as np
    import pyaudiowpatch as pa
    from core.live_worker import LiveTranscriptionWorker
    from core.settings import Settings
    output = None
    cleanup = None
    with pa.PyAudio() as audio:
        device = audio.get_default_wasapi_device(d_out=True)
        loopback = audio.get_default_wasapi_loopback()
        rate = int(loopback["defaultSampleRate"])
        channels = loopback["maxInputChannels"]
        worker = LiveTranscriptionWorker(loopback["index"], rate, channels, Settings(), backend="pyaudiowpatch")
        try:
            buffer, cleanup = worker._start_capture()
            output = audio.open(format=pa.paFloat32, channels=channels, rate=rate, output=True,
                                output_device_index=device["index"], frames_per_buffer=1024)
            tone = .025 * np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate)
            samples = np.repeat(tone.astype(np.float32)[:, None], channels, axis=1)
            output.write(samples.tobytes())
            time.sleep(.2)
            worker.stop()
            cleanup()
            cleanup = None
            captured = buffer.read_and_clear()
            if captured is None or len(captured) < rate:
                raise RuntimeError("Loopback captured less than one second")
            mono = captured.mean(axis=1) if captured.ndim > 1 else captured
            spectrum = np.abs(np.fft.rfft(mono))
            frequencies = np.fft.rfftfreq(len(mono), 1 / rate)
            tone_energy = float(spectrum[(frequencies > 435) & (frequencies < 445)].max())
            if tone_energy < 5:
                raise RuntimeError("The generated loopback tone was not detected")
            return dict(passed=True, seconds=len(captured) / rate, device=loopback["name"],
                        tone_hz=440, overflow=buffer.overflowed, capture_error=worker._capture_error)
        finally:
            if cleanup:
                cleanup()
            if output:
                output.stop_stream()
                output.close()


def check_file(media, device, directory):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from app import FileLoadWorker
    from core.job_store import JobStore
    from core.settings import Settings
    from core.worker import TranscriptionWorker
    from core.jobs import JobStatus
    qt_app = QApplication.instance() or QApplication([])
    settings = Settings(language="en", device=device, output_format="both", threads=8)
    store = JobStore(directory / "jobs")
    job = store.create(str(media.resolve()), settings, str(directory))
    job.custom_output_filename = "hardware-fixture"
    errors = []
    loader = FileLoadWorker(job.original_file_path, job=job)
    loader.error.connect(errors.append)
    loader.run()
    if errors:
        raise RuntimeError(errors[0])
    worker = TranscriptionWorker(job, settings)
    worker.run()
    if job.status != JobStatus.DONE or not any(task.text.strip() for task in job.tasks):
        raise RuntimeError(job.error_message or "No transcript was produced")
    return dict(passed=True, device=device, chunks=len(job.tasks),
                text_characters=sum(len(task.text) for task in job.tasks),
                recovered_status=store.load()[0].status.value)


def check_live_fixture(media, directory):
    from faster_whisper.audio import decode_audio
    from core.live_worker import LiveTranscriptionWorker, save_live_session
    from core.settings import Settings
    from engine.audio_capture import AudioBuffer
    from engine.model_loader import load_whisper_model, resolve_model_path
    samples = decode_audio(str(media), sampling_rate=16000)
    if len(samples) > 120 * 16000:
        raise ValueError("Use a synthetic fixture shorter than two minutes")
    model, error = load_whisper_model(resolve_model_path("en", "faster-whisper", str(ROOT)), "cpu", "auto", 8)
    if error:
        raise RuntimeError(error)
    worker = LiveTranscriptionWorker(0, 16000, 1, Settings(language="en", device="cpu"))
    worker._session_start_time = datetime.now()
    buffer = AudioBuffer(16000, 1)
    buffer.write(samples)
    worker.stop()
    worker._capture_loop(buffer, model, 3)
    segments = worker.session_segments
    if not segments or any(start < 0 or end <= start or end > len(samples) / 16000 + .01
                           for start, end, text in segments):
        raise RuntimeError("Live fixture produced no valid timed transcript")
    save_live_session(segments, str(directory), "live-hardware-fixture")
    return dict(passed=True, cues=len(segments), text_characters=sum(len(text) for _, _, text in segments),
                duration=len(samples) / 16000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--media", type=Path)
    parser.add_argument("--live-fixture", action="store_true", help="Feed --media through the live ASR/timing pipeline without recording")
    parser.add_argument("--device", choices=["cpu", "amd", "nvidia", "metal"], default="cpu")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/hardware-smoke.json")
    args = parser.parse_args()
    if not args.loopback and args.media is None:
        parser.error("Choose --loopback or --media with a non-sensitive speech fixture")
    if args.live_fixture and args.media is None:
        parser.error("--live-fixture requires --media")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    try:
        if args.loopback:
            result["loopback"] = check_loopback()
        if args.media:
            if args.live_fixture:
                result["live_fixture"] = check_live_fixture(args.media, args.report.parent)
            else:
                result["file"] = check_file(args.media, args.device, args.report.parent)
        result["passed"] = True
    except Exception as error:
        result.update(passed=False, error=str(error))
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
