"""Keep speaker changes and multilingual text through whisper.cpp output parsing."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from engine.diarization import label_segments
from engine import whisper_cpp_runner as runner


def token(text, start=None, end=None, identifier=100):
    result = {"text": text, "id": identifier}
    if start is not None:
        result["offsets"] = {"from": start, "to": end}
    return result


def document(text, tokens):
    return {"model": {"multilingual": True}, "transcription": [
        {"text": text, "offsets": {"from": 0, "to": 5000}, "tokens": tokens}
    ]}


def conversation():
    return document(" Hello there. Good morning.", [
        token("[_BEG_]", 0, 0, 50365),
        token(" Hello", 0, 1000), token(" there", 1000, 1800), token(".", 1800, 2000),
        token(" Good", 2000, 3000), token(" morn", 3000, 3800),
        token("ing", 3800, 4900), token(".", 4900, 5000),
        token("[_TT_250]", 5000, 5000, 50615),
    ])


def test_subtitle_with_two_voices_is_split_at_whole_word_boundary():
    segments = runner.parse_json_segments(conversation())
    text, encoded = label_segments(list(map(json.dumps, segments)),
                                   [(0, 2, "Speaker 1"), (2, 5, "Speaker 2")])
    assert text == "Speaker 1: Hello there.\nSpeaker 2: Good morning."
    assert [(s["start"], s["end"]) for s in map(json.loads, encoded)] == [(0, 2), (2, 5)]
    assert len(segments[0]["words"]) == 4


def test_old_cli_split_utf8_tokens_are_reassembled_before_decoding():
    # A short synthetic word: split its first UTF-8 character between tokens.
    word = " \u05d0\u05d1"
    raw = word.encode("utf-8")
    fragments = [raw[:2].decode("utf-8", errors="surrogateescape"),
                 raw[2:].decode("utf-8", errors="surrogateescape")]
    segments = runner.parse_json_segments(document(word, [
        token(fragments[0], 100, 500), token(fragments[1], 500, 1000)]))
    assert segments[0]["words"] == [{"word": word, "start": .1, "end": 1.0}]
    assert segments[0]["text"] == word


@pytest.mark.parametrize("tokens", [
    [token(" Hello", 0, 1000)],  # Missing the second word.
    [token(" Hello", 0, 1000), token(" there.")],
    [token(" Hello", 0, 2000), token(" there.", 1000, 3000)],
    [token(" Hello", 0, 2000), token(" there.", 2000, 6000)],
    [token(" Hello", 0, 2000), token(" there.", 2000, float("nan"))],
])
def test_incomplete_or_invalid_word_timings_preserve_full_segment(tokens):
    segments = runner.parse_json_segments(document(" Hello there.", tokens))
    assert segments == [{"start": 0, "end": 5, "text": " Hello there."}]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("valid_json", [False, True])
def test_runner_requests_and_preserves_word_timings_with_srt_recovery(monkeypatch, enabled, valid_json):
    def popen(args, **kwargs):
        assert ("--output-json-full" in args) is enabled
        prefix = args[args.index("--output-file") + 1]
        Path(prefix + ".srt").write_text(
            "1\n00:00:00,000 --> 00:00:05,000\nHello there. Good morning.\n", encoding="utf-8")
        if enabled:
            Path(prefix + ".json").write_text(json.dumps(conversation()) if valid_json else "{", encoding="utf-8")
        return SimpleNamespace(stderr=io.BytesIO(), returncode=0, wait=Mock())
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    text, encoded = runner.transcribe_chunk_whispercpp("synthetic.wav", "model", "cli", word_timestamps=enabled)
    result = json.loads(encoded[0])
    assert text.strip() == "Hello there. Good morning."
    assert ("words" in result) is (enabled and valid_json)
