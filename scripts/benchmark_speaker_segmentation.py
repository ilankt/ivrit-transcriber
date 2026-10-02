"""Validate speech-filter GPU acceleration on public/synthetic audio only."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torchaudio
from engine.diarization import _load_pipeline, model_path
from engine.speaker_segmentation import enable_segmentation_gpu


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = ROOT / "build/public-speaker-sample"
    waveform, rate = torchaudio.load(str(root / "sample.wav"))
    assert rate == 16000
    pipeline = _load_pipeline(model_path(str(ROOT)))
    model = pipeline._segmentation.model
    adapter = enable_segmentation_gpu(pipeline, model_path(str(ROOT)), "directml", root / "filters-profile")
    cases = {"public": waveform[:, :160000][None], "noise": torch.randn(1, 1, 160000),
             "silence": torch.zeros(1, 1, 160000), "quiet": waveform[:, :160000][None] * 1e-5}
    reports = []
    with torch.inference_mode():
        for name, audio in cases.items():
            for batch in (1, 8, 32):
                inputs = audio.repeat(batch, 1, 1)
                model.sincnet = adapter.original
                start = time.perf_counter()
                cpu = model(inputs).numpy()
                cpu_seconds = time.perf_counter() - start
                model.sincnet = adapter
                model(inputs)  # Exclude initial shape compilation from the warm timing.
                start = time.perf_counter()
                gpu = model(inputs).numpy()
                gpu_seconds = time.perf_counter() - start
                error = float(np.max(np.abs(np.exp(cpu) - np.exp(gpu))))
                matches = float(np.mean(cpu.argmax(-1) == gpu.argmax(-1)))
                assert adapter.accelerated
                assert error < 1e-3 and matches == 1.0, (name, batch, error, matches)
                reports.append(dict(case=name, batch=batch, cpu_seconds=cpu_seconds,
                                    gpu_seconds=gpu_seconds, probability_error=error, frame_matches=matches))
                print(json.dumps(reports[-1]), flush=True)
    profile = Path(adapter.session.end_profiling())
    events = json.loads(profile.read_text())
    providers = sorted({e.get("args", {}).get("provider") for e in events if e.get("args", {}).get("provider")})
    assert "DmlExecutionProvider" in providers
    (root / "segmentation-gpu-report.json").write_text(
        json.dumps({"checks": reports, "providers": providers}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
