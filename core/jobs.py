from dataclasses import dataclass, field
from enum import Enum


class JobStatus(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    CANCELED = "Canceled"
    DONE = "Done"
    ERROR = "Error"


class TaskStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    DONE = "Done"
    ERROR = "Error"


@dataclass
class Task:
    chunk_path: str
    chunk_index: int
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    text: str = ""
    srt_segments: list[str] = field(default_factory=list)
    error_message: str | None = None
    duration: float = 0.0


@dataclass
class Job:
    original_file_path: str
    tasks: list[Task]
    output_dir: str | None = None
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    error_message: str | None = None
    custom_output_filename: str | None = None
    temp_dir: str | None = None

    def update_progress(self):
        if not self.tasks:
            self.progress = 0.0
            return
        self.progress = sum(task.progress for task in self.tasks) / len(self.tasks)
