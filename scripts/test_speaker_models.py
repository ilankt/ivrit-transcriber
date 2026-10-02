"""Opt-in inference check with synthetic voices; no user audio or tokens in reports.

Generate samples with make_speaker_sample.ps1 first. Run in a separate virtual
environment for each backend; this is not a pytest unit test or an accuracy
benchmark for Hebrew conversations.
"""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def prepare_sample(directory):
    parts = json.loads((directory / "parts.json").read_text(encoding="utf-8-sig"))
    frames = []
    reference = []
    position = 0
    for part in parts:
        with wave.open(str(directory / part["file"]), "rb") as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
            data = audio.readframes(audio.getnframes())
        duration = len(data) / 32000
        reference.append({"start": position, "end": position + duration,
                          "speaker": part["speaker"], "text": part["text"]})
        frames.extend([data, b"\x00" * 32000])
        position += duration + 1
    path = directory / "conversation.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"".join(frames))
    return path, reference


def evaluate(reference, detected):
    assignments = []
    for item in reference:
        scores = defaultdict(float)
        for start, end, speaker in detected:
            scores[speaker] += max(0, min(item["end"], end) - max(item["start"], start))
        positive = {speaker: score for speaker, score in scores.items() if score > 0}
        assignments.append(max(positive, key=positive.get) if positive else None)
    # The fixture alternates A/B/A/B. Cluster names themselves are arbitrary.
    passed = (len(assignments) == 4 and all(value is not None for value in assignments)
              and assignments[0] == assignments[2] and assignments[1] == assignments[3]
              and assignments[0] != assignments[1])
    return {"passed": passed, "turn_assignments": assignments,
            "detected_speakers": len({speaker for _, _, speaker in detected})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["app", "community", "ivrit"], required=True)
    parser.add_argument("--sample-dir", type=Path, default=ROOT / "build/speaker-sample")
    parser.add_argument("--speakers", type=int, default=0, help="0 = automatic")
    parser.add_argument("--device", choices=["cpu", "auto", "nvidia", "amd", "metal"], default="cpu",
                        help="Device for the integrated app backend")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Fail if app inference does not use a GPU or falls back to CPU")
    parser.add_argument("--ivrit-numpy-workaround", action="store_true",
                        help="Test-only fix for ivrit 0.2.6's missing module-level NumPy import")
    args = parser.parse_args()
    if args.backend != "app" and (args.device != "cpu" or args.require_gpu):
        parser.error("Device selection and --require-gpu are supported by --backend app")
    report = {"backend": args.backend, "requested_speakers": args.speakers,
              "ivrit_numpy_workaround": args.ivrit_numpy_workaround,
              "fixture": "synthetic English A/B/A/B; no overlapping voices", "versions": {}}
    for package in ("ivrit", "pyannote.audio", "torch", "torchaudio"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    started = time.monotonic()
    try:
        audio_path, reference = prepare_sample(args.sample_dir)
        if args.backend == "app":
            from engine.diarization import diarize_chunks, download_model, model_available, model_path
            path = model_path(str(ROOT))
            if not model_available(path):
                download_model(path)
            report["requested_device"] = args.device
            report["statuses"] = []
            detected = diarize_chunks([str(audio_path)], path, device=args.device,
                                     num_speakers=args.speakers, status_callback=report["statuses"].append)
            report["gpu_used"] = any("GPU (" in status for status in report["statuses"])
            report["gpu_fallback"] = any("unavailable" in status or "retrying" in status for status in report["statuses"])
            if args.require_gpu and (not report["gpu_used"] or report["gpu_fallback"]):
                raise RuntimeError("Requested GPU acceleration did not complete successfully")
        elif args.backend == "community":
            import numpy as np
            import torch
            from huggingface_hub import snapshot_download
            from pyannote.audio import Pipeline
            path = snapshot_download("pyannote/speaker-diarization-community-1")
            pipeline = Pipeline.from_pretrained(path)
            with wave.open(str(audio_path), "rb") as audio:
                waveform = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
            kwargs = {"num_speakers": args.speakers} if args.speakers else {}
            result = pipeline({"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": 16000}, **kwargs)
            detected = [(turn.start, turn.end, speaker) for turn, _, speaker
                        in result.exclusive_speaker_diarization.itertracks(yield_label=True)]
        else:
            # The same narrow allowlist used by ivrit.ai's official RunPod image.
            import torch
            from pyannote.audio.core.task import Problem, Resolution, Specifications
            torch.serialization.add_safe_globals([Problem, Resolution, Specifications, torch.torch_version.TorchVersion])
            import numpy as np
            if args.ivrit_numpy_workaround:
                import ivrit.diarization as ivrit_diarization
                ivrit_diarization.np = np
            from ivrit.diarization import diarize
            from ivrit.types import Segment
            with wave.open(str(audio_path), "rb") as audio:
                waveform = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
            segments = [Segment(start=item["start"], end=item["end"], text=item["text"], words=[])
                        for item in reference]
            kwargs = {"num_speakers": args.speakers} if args.speakers else {}
            result = diarize(waveform, segments, engine="pyannote", device="cpu", **kwargs)
            detected = [(seg.start, seg.end, seg.speakers[0]) for seg in result if seg.speakers and seg.speakers[0] is not None]
        report.update(evaluate(reference, detected))
        report["turns"] = detected
    except Exception as error:
        # Record only exception class: Hub error strings can contain HTTP details.
        report.update(passed=False, error_type=type(error).__name__)
        if isinstance(error, ModuleNotFoundError):
            report["missing_module"] = error.name
        if isinstance(error, NameError):
            report["missing_name"] = error.name
        import traceback
        report["error_location"] = [f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                                    for frame in traceback.extract_tb(error.__traceback__)]
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    suffix = "-numpy-workaround" if args.ivrit_numpy_workaround else ""
    if args.device != "cpu":
        suffix += f"-{args.device}"
    destination = args.sample_dir / f"{args.backend}-{args.speakers}{suffix}-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
