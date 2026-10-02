"""Measured, per-stage progress shared by speaker detection and transcription."""
import math
import time


class StageProgress:
    def __init__(self, steps=3, clock=time.monotonic):
        self.steps = steps
        self.clock = clock
        self.stage = None
        self.started = None
        self.completed = 0

    def exclude_duration(self, seconds):
        if self.started is not None:
            self.started += max(0, seconds)

    def update(self, stage, completed, total):
        now = self.clock()
        completed = max(0, min(completed, total)) if total > 0 else 0
        if stage != self.stage or completed < self.completed:
            self.stage, self.started = stage, now
        self.completed = completed
        fraction = completed / total if total > 0 else 0
        percent = min(100, int(100 * fraction))
        elapsed = max(0, now - self.started)
        if fraction >= 1:
            estimate = "Step complete"
        elif completed > 0 and elapsed >= 1:
            remaining = math.ceil(elapsed * (1 - fraction) / fraction)
            if remaining >= 3600:
                estimate = f"About {remaining // 3600}h {(remaining % 3600) // 60}m left in this step"
            elif remaining >= 60:
                estimate = f"About {remaining // 60}m {remaining % 60}s left in this step"
            else:
                estimate = f"About {remaining}s left in this step"
        else:
            estimate = "Estimating time remaining…"
        return percent, f"Step {stage}/{self.steps} — {percent}% of step — {estimate}"
