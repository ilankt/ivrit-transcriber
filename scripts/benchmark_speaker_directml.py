"""Compare CPU and DirectML speaker inference using synthetic local audio.

Requires the normal speaker dependencies plus requirements-speakers-amd.txt.
Run with --repeat 4 for a longer recording; --threads 8 sets both paths to the
same CPU thread count. Defaults to the app's normal PyTorch thread count.
No audio or model weights leave the machine.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

import numpy as np
import onnxruntime as ort
import torch

from engine.diarization import _load_pipeline, model_path
from engine.speaker_directml import enable_directml
from scripts.test_speaker_models import evaluate, prepare_sample


def timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.threads:
        torch.set_num_threads(args.threads)
    report = {"providers": ort.get_available_providers(), "torch_threads": torch.get_num_threads()}
    assert "DmlExecutionProvider" in report["providers"], report
    pipeline, report["model_load_seconds"] = timed(lambda: _load_pipeline(model_path(str(ROOT))))
    model = pipeline._embedding.model_
    _ = pipeline._embedding.min_num_samples
    path, reference = prepare_sample(ROOT / "build/speaker-sample")
    with wave.open(str(path)) as audio:
        data = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    sample_duration = len(data) / 16000
    waveform = torch.from_numpy(np.tile(data, args.repeat)).reshape(1, -1)
    report["audio_seconds"] = waveform.shape[-1] / 16000
    cache = ROOT / "build/speaker-directml"
    cache.mkdir(exist_ok=True)

    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def run_pipeline(label):
            phase_start = [time.perf_counter(), None]
            phases = {}

            def hook(step, artifact, file=None, total=None, completed=None):
                if step != phase_start[1]:
                    now = time.perf_counter()
                    if phase_start[1]:
                        phases[phase_start[1]] = phases.get(phase_start[1], 0) + now - phase_start[0]
                    phase_start[:] = [now, step]
                    print(label, step, flush=True)

            result, seconds = timed(lambda: pipeline({"waveform": waveform, "sample_rate": 16000}, hook=hook))
            turns = [(t.start, t.end, speaker) for t, _, speaker in result.itertracks(yield_label=True)]
            evaluations = []
            for repeat in range(args.repeat):
                shifted = [{**item, "start": item["start"] + repeat * sample_duration,
                            "end": item["end"] + repeat * sample_duration} for item in reference]
                evaluations.append(evaluate(shifted, turns))
            passed = all(e["passed"] and e["turn_assignments"] == evaluations[0]["turn_assignments"]
                         for e in evaluations)
            return {"seconds": seconds, "phases": phases, "turns": turns,
                    "passed": passed, "evaluations": evaluations}

        report["cpu"] = run_pipeline("CPU")
        adapter, report["gpu_setup_seconds"] = timed(lambda: enable_directml(
            pipeline, model_path(str(ROOT)), profile_prefix=cache / "profile"))

        # Exercise variable batches, shorter windows, silence, and overlap masks.
        report["numerical_checks"] = []
        torch.manual_seed(42)
        for batch, samples in ((1, 16000), (3, 32000), (8, 160000)):
            waveforms = torch.from_numpy(data[:samples]).reshape(1, 1, -1).repeat(batch, 1, 1)
            masks = torch.randint(0, 2, (batch, 100)).float()
            masks[0] = 0
            for weights in (None, masks):
                expected = model(waveforms, weights=weights).numpy()
                actual = adapter(waveforms, weights=weights).numpy()
                matched = np.allclose(actual, expected, atol=2e-4, rtol=2e-3, equal_nan=True)
                report["numerical_checks"].append({"batch": batch, "samples": samples,
                                                    "masked": weights is not None, "passed": matched})
                assert matched, report["numerical_checks"]
        report["directml"] = run_pipeline("DirectML")
        assert adapter.accelerated, "GPU fell back to CPU"
        profile = json.loads(Path(adapter.session.end_profiling()).read_text())
        report["executed_providers"] = sorted({e.get("args", {}).get("provider") for e in profile if e.get("args", {}).get("provider")})
        report["identical_turns"] = report["cpu"]["turns"] == report["directml"]["turns"]
        report["passed"] = (report["cpu"]["passed"] and report["directml"]["passed"]
                            and report["identical_turns"] and "DmlExecutionProvider" in report["executed_providers"])
        output = cache / f"benchmark-{args.repeat}x-{torch.get_num_threads()}threads.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("cpu", "directml")}, indent=2))
        print(f"CPU: {report['cpu']['seconds']:.2f}s; DirectML: {report['directml']['seconds']:.2f}s")
        print(f"Report: {output}")
        assert report["passed"]


if __name__ == "__main__":
    main()
