"""Compare clustering on fixed public Pyannote fixtures, never user audio.

Download sample.wav and sample.rttm from the public pyannote-audio 3.3.2 tag's
pyannote/audio/sample directory to build/public-speaker-sample first.
Optionally add tests/data/dev00.wav, dev01.wav, and debug.development.rttm
(rename the latter to development.rttm). Requires the app's speaker model and
Windows DirectML dependencies. All fixtures use a 0-30 second evaluation region.
Additional optional fixtures: tests/data/trn03.wav and debug.train.rttm (as
training.rttm), tst00.wav, tst01.wav, and debug.test.rttm.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

import torch
import torchaudio
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from engine.diarization import _load_pipeline, model_path
from engine.speaker_directml import enable_directml


def evaluate_case(name, annotation_file):
    root = ROOT / "build/public-speaker-sample"
    reference = Annotation(uri=name)
    for index, line in enumerate((root / annotation_file).read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        if fields[1] != name:
            continue
        start, duration = map(float, fields[3:5])
        reference[Segment(start, start + duration), index] = fields[7]
    waveform, sample_rate = torchaudio.load(str(root / f"{name}.wav"))
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    pipeline = _load_pipeline(model_path(str(ROOT)))
    adapter = enable_directml(pipeline, model_path(str(ROOT)))
    artifacts = {}

    def hook(step, artifact, **kwargs):
        if artifact is not None and step in ("segmentation", "embeddings"):
            artifacts[step] = artifact

    results = []
    variants = (
        ("centroid", False, 12),
        ("centroid", True, 12),
        ("centroid", True, 2),
        ("average", False, 12),
        ("average", True, 12),
        ("average", False, 2),
        ("average", True, 2),
    )
    with torch.inference_mode():
        for method, constrained, minimum in variants:
            pipeline.clustering.method = method
            pipeline.clustering.constrained_assignment = constrained
            pipeline.clustering.min_cluster_size = minimum
            output = pipeline({"waveform": waveform, "sample_rate": 16000},
                              num_speakers=len(reference.labels()), hook=hook)
            details = DiarizationErrorRate(collar=0.0, skip_overlap=False)(
                reference, output, uem=Timeline([Segment(0, 30)]), detailed=True)
            results.append({"method": method, "constrained": constrained, "min_cluster_size": minimum, "metrics": details,
                            "detected_speakers": len(output.labels())})
            # Compare grouping with identical audio inference, rather than GPU variation.
            segmentations = artifacts["segmentation"]
            embeddings = artifacts["embeddings"]
            pipeline.get_segmentations = lambda file, hook=None: segmentations
            pipeline.get_embeddings = lambda *args, **kwargs: embeddings.copy()
        if not adapter.accelerated:
            raise RuntimeError("Public benchmark fell back from DirectML to CPU.")
    report = {"case": name, "source": "https://github.com/pyannote/pyannote-audio/tree/3.3.2",
              "audio_seconds": waveform.shape[-1] / 16000, "reference_speakers": len(reference.labels()),
              "results": results}
    print(json.dumps(report, indent=2))
    return report


def main():
    reports = [evaluate_case("sample", "sample.rttm")]
    root = ROOT / "build/public-speaker-sample"
    for name in ("dev00", "dev01"):
        if (root / f"{name}.wav").is_file():
            reports.append(evaluate_case(name, "development.rttm"))
    if (root / "trn03.wav").is_file():
        reports.append(evaluate_case("trn03", "training.rttm"))
    for name in ("tst00", "tst01"):
        if (root / f"{name}.wav").is_file():
            reports.append(evaluate_case(name, "debug.test.rttm"))
    (root / "clustering-comparison.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
