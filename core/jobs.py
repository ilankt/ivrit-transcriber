from enum import Enum
from typing import List

class JobStatus(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    PAUSED = "Paused"
    CANCELED = "Canceled"
    DONE = "Done"
    ERROR = "Error"

class TaskStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    DONE = "Done"
    ERROR = "Error"

class Task:
    def __init__(self, file_path: str, chunk_path: str, chunk_index: int):
        self.file_path = file_path
        self.chunk_path = chunk_path
        self.chunk_index = chunk_index
        self.status = TaskStatus.PENDING
        self.progress = 0.0  # 0.0 to 1.0
        self.text = ""
        self.srt_segments = []
        self.error_message = None
        self.duration = 0.0

class Job:
    def __init__(self, original_file_path: str, tasks: List[Task], output_dir: str = None):
        self.original_file_path = original_file_path
        self.output_dir = output_dir
        self.tasks = tasks
        self.status = JobStatus.QUEUED
        self.progress = 0.0 # 0.0 to 1.0
        self.error_message = None
        self.custom_output_filename = None  # Optional user-specified filename

    def update_progress(self):
        if not self.tasks:
            self.progress = 0.0
            return
        
        total_progress = sum(task.progress for task in self.tasks)
        self.progress = total_progress / len(self.tasks)

    def get_overall_status(self) -> JobStatus:
        if any(task.status == TaskStatus.ERROR for task in self.tasks):
            return JobStatus.ERROR
        if all(task.status == TaskStatus.DONE for task in self.tasks):
            return JobStatus.DONE
        if any(task.status == TaskStatus.RUNNING for task in self.tasks):
            return JobStatus.RUNNING
        if any(task.status == TaskStatus.PENDING for task in self.tasks):
            return JobStatus.QUEUED # Or some other intermediate state
        return JobStatus.QUEUED
