"""Exercise speaker-aware whisper.cpp exports using only Pyannote's public sample.

Requires build/public-speaker-sample/sample.wav and sample.rttm from Pyannote
3.3.2, the app's speaker model, and Models/ggml-large-v3.bin. This Windows
hardware check uses the installed DirectML and Vulkan paths.
"""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

from engine.diarization import diarize_chunks, label_segments, model_path, _speaker_for
from engine.whisper_cpp_runner import transcribe_chunk_whispercpp


def main():
    root = ROOT / "build/public-speaker-sample"
    # Reproduce the old assignment policy using the same real inference backend.
    with patch("engine.diarization._configure_speaker_assignment", lambda *args: None):
        old_turns = diarize_chunks([str(root / "sample.wav")], model_path(str(ROOT)),
                                  device="amd", num_speakers=2)
    statuses = []
    turns = diarize_chunks([str(root / "sample.wav")], model_path(str(ROOT)),
                          device="amd", num_speakers=2, status_callback=statuses.append)
    if not any("GPU (DirectML)" in status for status in statuses):
        raise RuntimeError("Public sample did not use the speaker GPU")
    if not any("finding speech — GPU + CPU (DirectML)" in status for status in statuses):
        raise RuntimeError("Public sample did not use GPU speech filters")
    _, encoded = transcribe_chunk_whispercpp(
        str(root / "sample.wav"), str(ROOT / "Models/ggml-large-v3.bin"),
        str(ROOT / "Binaries/whisper-cli.exe"), language="en", threads=8, word_timestamps=True)
    segments = list(map(json.loads, encoded))
    if not all("words" in s for s in segments):
        raise RuntimeError("Some public sample subtitles are missing word timestamps")
    before = [json.dumps({k: v for k, v in s.items() if k != "words"}) for s in segments]
    before_text, before_segments = label_segments(before, old_turns)
    after_text, after_segments = label_segments(encoded, turns)
    (root / "sentence-labels.txt").write_text(before_text, encoding="utf-8")
    (root / "word-labels.txt").write_text(after_text, encoding="utf-8")
    # Reference labels independently determine which speaker overlaps each word.
    reference = []
    for line in (root / "sample.rttm").read_text().splitlines():
        fields = line.split()
        if fields[1] == "sample":
            start, duration = map(float, fields[3:5])
            reference.append((start, start + duration, fields[7]))
    from itertools import permutations
    predicted_names = sorted({t[2] for t in turns})
    reference_names = sorted({t[2] for t in reference})
    def agreement(mapping, detected):
        return sum(max(0, min(b, d) - max(a, c))
                   for a, b, name in detected for c, d, ref in reference if mapping[name] == ref)
    mapping = max((dict(zip(predicted_names, names)) for names in permutations(reference_names)),
                  key=lambda m: agreement(m, turns))
    old_mapping = max((dict(zip(sorted({t[2] for t in old_turns}), names))
                       for names in permutations(reference_names)), key=lambda m: agreement(m, old_turns))
    counts = {"evaluated_words": 0, "sentence_label_errors": 0, "word_label_errors": 0}
    for s in segments:
        old = _speaker_for(s["start"], s["end"], old_turns)
        for w in s["words"]:
            expected = _speaker_for(w["start"], w["end"], reference)
            if expected == "Speaker unknown":
                continue
            new = _speaker_for(w["start"], w["end"], turns)
            counts["evaluated_words"] += 1
            counts["sentence_label_errors"] += old_mapping.get(old) != expected
            counts["word_label_errors"] += mapping.get(new) != expected
    before_content = "".join(json.loads(s)["text"] for s in before_segments)
    after_content = "".join(json.loads(s)["text"] for s in after_segments)
    assert before_content == after_content, "Word labeling lost transcript text"
    report = {"source": "https://github.com/pyannote/pyannote-audio/tree/3.3.2/pyannote/audio/sample",
              "original_subtitles": len(before_segments), "speaker_split_subtitles": len(after_segments),
              "text_preserved": True, **counts}
    (root / "word-label-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
