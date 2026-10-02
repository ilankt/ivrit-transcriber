"""Editable transcript documents; corrections stay separate from ASR results."""
import json
import math
from pathlib import Path

from core.jobs import TaskStatus
from core.storage import atomic_text_writer
from engine.checkpoint import merge_checkpoints_to_files


def review_path(job):
    return Path(job.record_path).with_name("review.json")


def transcript_rows(job):
    rows, offset = [], 0.0
    for task in job.tasks:
        if task.status == TaskStatus.DONE:
            for index, encoded in enumerate(task.srt_segments):
                segment = json.loads(encoded)
                rows.append(dict(id=f"{task.chunk_index}:{index}", start=offset + segment["start"],
                                 end=offset + segment["end"], text=segment["text"].strip(),
                                 speaker=segment.get("speaker", "")))
        offset += task.duration
    path = review_path(job)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved.get("version") != 1:
            raise ValueError("Unsupported transcript review format")
        edits = {row["id"]: row for row in saved["segments"]}
        rows = [edits.get(row["id"], row) for row in rows]
    validate_rows(rows)
    return rows


def validate_rows(rows):
    previous = -1.0
    for index, row in enumerate(rows, 1):
        start, end = row["start"], row["end"]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end)):
            raise ValueError(f"Row {index}: timestamps must be finite seconds.")
        if start < 0 or end <= start or start < previous:
            raise ValueError(f"Row {index}: use ordered, nonnegative starts and an end after the start.")
        if not isinstance(row["text"], str) or not isinstance(row["speaker"], str):
            raise ValueError(f"Row {index}: text and speaker must be strings.")
        previous = start


def save_review(job, rows):
    validate_rows(rows)
    with atomic_text_writer(review_path(job)) as stream:
        json.dump(dict(version=1, segments=rows), stream, ensure_ascii=False, indent=2)


def export_review(rows, output_dir, stem, output_format="both"):
    validate_rows(rows)
    text = "\n".join(f"{row['speaker']}: {row['text']}" if row['speaker'] else row['text'] for row in rows)
    return merge_checkpoints_to_files(output_dir, stem, checkpoints=[
        dict(text=text, srt_segments=[json.dumps(row, ensure_ascii=False) for row in rows], start_offset=0,
             duration=max((row["end"] for row in rows), default=0))], output_format=output_format)
