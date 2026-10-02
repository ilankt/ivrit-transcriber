"""Durable, local-only jobs. Transcripts survive cleanup of prepared audio."""
import json
import logging
import math
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from core.filenames import sanitize_output_stem
from core.jobs import Job, JobStatus, Task, TaskStatus
from core.settings import get_settings_path
from core.storage import atomic_text_writer


def source_signature(path):
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def save_job(job):
    if not job.record_path:
        return
    data = asdict(job)
    data["version"] = 1
    data["status"] = job.status.value
    for task in data["tasks"]:
        task["status"] = task["status"].value
    Path(job.record_path).parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_writer(job.record_path) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def job_checkpoints(job):
    offset = 0.0
    checkpoints = []
    for task in job.tasks:
        if task.status == TaskStatus.DONE:
            checkpoints.append(dict(chunk_index=task.chunk_index, text=task.text,
                                    srt_segments=task.srt_segments, duration=task.duration,
                                    start_offset=offset))
        offset += task.duration
    return checkpoints


def release_audio(job):
    """Delete only this job's owned cache, never paths supplied by a manifest."""
    if not job.record_path:
        return
    directory = Path(job.record_path).parent
    audio = directory / "audio"
    if any(path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) for path in (directory, audio)):
        raise ValueError("Refusing to remove a linked audio cache")
    if job.temp_dir and Path(job.temp_dir).resolve() != audio.resolve():
        raise ValueError("Audio cache is outside this job's directory")
    if audio.exists():
        shutil.rmtree(audio)
    job.temp_dir = None
    save_job(job)


class JobStore:
    def __init__(self, root=None):
        self.root = Path(root or Path(get_settings_path()).parent / "jobs")
        self.errors = []

    def create(self, source, settings=None, output_dir=None, output_stem=None):
        record = self.root / uuid4().hex / "job.json"
        job = Job(os.path.abspath(source), [], output_dir=output_dir,
                  custom_output_filename=output_stem,
                  record_path=str(record), temp_dir=str(record.parent / "audio"),
                  settings_snapshot=settings.model_dump() if settings is not None else None,
                  source_signature=source_signature(source),
                  created_at=datetime.now(timezone.utc).isoformat())
        save_job(job)
        return job

    def load(self):
        jobs = []
        self.errors = []
        for path in self.root.glob("*/job.json"):
            try:
                with path.open(encoding="utf-8") as stream:
                    data = json.load(stream)
                if data.pop("version") != 1:
                    raise ValueError("Unsupported job format")
                data["record_path"] = str(path)
                data["status"] = JobStatus(data["status"])
                tasks = []
                for item in data.pop("tasks"):
                    item["status"] = TaskStatus(item["status"])
                    task = Task(**item)
                    if task.status == TaskStatus.RUNNING:
                        task.status, task.progress = TaskStatus.PENDING, 0.0
                    tasks.append(task)
                job = Job(tasks=tasks, **data)
                if not isinstance(job.original_file_path, str) or not isinstance(job.created_at, str):
                    raise ValueError("Invalid source metadata")
                if any(value is not None and not isinstance(value, str) for value in
                       (job.output_dir, job.temp_dir, job.custom_output_filename)):
                    raise ValueError("Invalid job paths")
                if job.settings_snapshot is not None:
                    from core.settings import Settings
                    Settings.model_validate(job.settings_snapshot)
                for task in tasks:
                    if (not isinstance(task.chunk_path, str) or not isinstance(task.text, str)
                            or not isinstance(task.srt_segments, list)
                            or any(not isinstance(segment, str) for segment in task.srt_segments)
                            or not isinstance(task.duration, (int, float)) or not math.isfinite(task.duration)
                            or task.duration < 0):
                        raise ValueError("Invalid chunk data")
                    task.progress = 1.0 if task.status == TaskStatus.DONE else 0.0
                if [task.chunk_index for task in tasks] != list(range(len(tasks))):
                    raise ValueError("Invalid chunk order")
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.CANCELED
                    job.error_message = "Interrupted. Resume to continue unfinished chunks."
                job.update_progress()
                jobs.append(job)
            except (OSError, ValueError, TypeError, KeyError):
                logging.exception("Could not read job history: %s", path)
                self.errors.append(str(path))
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def unique_stem(self, source, output_dir, jobs):
        stem = sanitize_output_stem(Path(source).stem) or "transcript"
        reserved = {(os.path.normcase(os.path.abspath(j.output_dir or ".")),
                     (j.custom_output_filename or Path(j.original_file_path).stem).casefold())
                    for j in jobs}
        candidate, number = stem, 2
        while ((os.path.normcase(os.path.abspath(output_dir)), candidate.casefold()) in reserved
               or any((Path(output_dir) / f"{candidate}.{ext}").exists() for ext in ("txt", "srt"))):
            candidate = f"{stem[:180]} - {number}"
            number += 1
        return candidate
