import os
import sys
import tempfile
import shutil
import time
import logging
from datetime import datetime
from PySide6.QtCore import QObject, Signal, QRunnable, Slot
from core.jobs import Job, Task, JobStatus, TaskStatus
from engine.model_loader import load_whisper_model, validate_model_path
from engine.transcriber import transcribe_chunk
from engine.merger import merge_srt_files, merge_txt_files
from engine.checkpoint import save_chunk_checkpoint, merge_checkpoints_to_files, cleanup_checkpoints


def get_base_path():
    """
    Get the base path for the application.
    When running as a PyInstaller executable, use the directory containing the exe.
    Otherwise, use the directory containing the main script.
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable - use exe's directory
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_model_path(model_type):
    """
    Get the model path for the specified model type.

    Args:
        model_type: "Fast" or "Accurate"

    Returns:
        Absolute path to the model directory
    """
    base_path = get_base_path()
    if model_type == "Fast":
        return os.path.join(base_path, 'Models', 'ivrit-large-v3-turbo-ct2')
    else:  # Accurate
        return os.path.join(base_path, 'Models', 'ivrit-large-v3-ct2')


class WorkerSignals(QObject):
    progress_updated = Signal(int, float)  # task_index, progress
    task_status_updated = Signal(int, TaskStatus, str)  # task_index, status, message
    job_status_updated = Signal(JobStatus, str)  # status, message
    eta_updated = Signal(str)  # eta_string
    finished = Signal()

class TranscriptionWorker(QRunnable):
    def __init__(self, job: Job, settings):
        super().__init__()
        self.job = job
        self.settings = settings
        self.signals = WorkerSignals()
        self.is_paused = False
        self.is_canceled = False
        self.start_time = 0
        self.processed_audio_duration = 0.0
        self.total_audio_duration = 0.0 # This needs to be calculated or passed from UI

    @Slot()
    def run(self):
        self.signals.job_status_updated.emit(JobStatus.RUNNING, "Starting job")
        model = None
        self.start_time = time.time()

        try:
            # Determine model path and beam size from settings
            model_path = get_model_path(self.settings.model_type)
            if self.settings.model_type == "Fast":
                beam_size = 1
            else:  # Accurate
                beam_size = 3

            logging.info(f"Starting transcription for {self.job.original_file_path} with model: {self.settings.model_type}")
            logging.info(f"Model path: {model_path}")

            # Validate model path before loading
            if not validate_model_path(model_path):
                self.signals.job_status_updated.emit(
                    JobStatus.ERROR,
                    f"Invalid model path: {model_path}. Required model files (model.bin, tokenizer.json, vocabulary.json) are missing."
                )
                logging.error(f"Model validation failed for path: {model_path}")
                return

            # Load model with device from settings
            model, error_message = load_whisper_model(
                model_path,
                self.settings.device,  # Use device from settings (auto/cpu/gpu)
                self.settings.compute_type,
                self.settings.threads
            )
            if error_message:
                self.signals.job_status_updated.emit(JobStatus.ERROR, f"Model loading failed: {error_message}")
                logging.error(f"Model loading failed for {self.job.original_file_path}: {error_message}")
                return

            # Ensure output directory exists
            os.makedirs(self.job.output_dir, exist_ok=True)

            # Calculate total audio duration for ETA
            self.total_audio_duration = sum(task.duration for task in self.job.tasks)
            self._emit_eta() # Emit initial ETA

            # Process tasks
            for i, task in enumerate(self.job.tasks):
                if self.is_canceled:
                    # Save any completed chunks before canceling
                    if hasattr(self.job, 'custom_output_filename') and self.job.custom_output_filename:
                        base_name = self.job.custom_output_filename
                    else:
                        base_name = os.path.splitext(os.path.basename(self.job.original_file_path))[0]
                    txt_path, srt_path = merge_checkpoints_to_files(self.job.output_dir, base_name, output_format=self.settings.output_format)
                    self._cleanup_checkpoint_files(base_name)
                    if txt_path or srt_path:
                        self.signals.job_status_updated.emit(
                            JobStatus.CANCELED,
                            "Job canceled. Partial results saved."
                        )
                        logging.info(f"Transcription canceled. Partial results saved to {txt_path} and {srt_path}")
                    else:
                        self.signals.job_status_updated.emit(JobStatus.CANCELED, "Job canceled")
                        logging.info(f"Transcription canceled for {self.job.original_file_path}")
                    return
                while self.is_paused:
                    time.sleep(0.1) # Sleep briefly to avoid busy-waiting

                self.signals.task_status_updated.emit(
                    i, TaskStatus.RUNNING,
                    f"Transcribing chunk {task.chunk_index + 1}/{len(self.job.tasks)} of {os.path.basename(self.job.original_file_path)}"
                )

                # Transcribe chunk
                text, srt_segments = transcribe_chunk(
                    task.chunk_path, model, "he", beam_size, self.settings.vad_enabled
                )

                if self.is_canceled: # Check for cancellation after transcription completes
                    # Current chunk was transcribed but not saved yet - save it first
                    task.text = text
                    task.srt_segments = srt_segments
                    task.status = TaskStatus.DONE

                    if hasattr(self.job, 'custom_output_filename') and self.job.custom_output_filename:
                        base_name = self.job.custom_output_filename
                    else:
                        base_name = os.path.splitext(os.path.basename(self.job.original_file_path))[0]
                    save_chunk_checkpoint(
                        self.job.output_dir, base_name, task.chunk_index,
                        text, srt_segments, task.duration
                    )

                    # Merge all completed chunks including this one
                    txt_path, srt_path = merge_checkpoints_to_files(self.job.output_dir, base_name, output_format=self.settings.output_format)
                    self._cleanup_checkpoint_files(base_name)
                    if txt_path or srt_path:
                        self.signals.job_status_updated.emit(
                            JobStatus.CANCELED,
                            "Job canceled. Partial results saved."
                        )
                        logging.info(f"Transcription canceled. Partial results saved to {txt_path} and {srt_path}")
                    else:
                        self.signals.job_status_updated.emit(JobStatus.CANCELED, "Job canceled during transcription")
                        logging.info(f"Transcription canceled for {self.job.original_file_path}")
                    return

                task.text = text
                task.srt_segments = srt_segments
                task.status = TaskStatus.DONE

                # Save checkpoint for this chunk
                if hasattr(self.job, 'custom_output_filename') and self.job.custom_output_filename:
                    base_name = self.job.custom_output_filename
                else:
                    base_name = os.path.splitext(os.path.basename(self.job.original_file_path))[0]
                save_chunk_checkpoint(
                    self.job.output_dir,
                    base_name,
                    task.chunk_index,
                    text,
                    srt_segments,
                    task.duration
                )
                logging.info(f"Saved checkpoint for chunk {task.chunk_index + 1}/{len(self.job.tasks)}")

                # Incrementally merge all completed chunks so far
                merge_checkpoints_to_files(self.job.output_dir, base_name, output_format=self.settings.output_format)
                logging.info(f"Merged {task.chunk_index + 1} completed chunks")

                self.signals.task_status_updated.emit(
                    i, TaskStatus.DONE,
                    f"Finished chunk {task.chunk_index + 1}/{len(self.job.tasks)} of {os.path.basename(self.job.original_file_path)}"
                )
                self.signals.progress_updated.emit(i, 1.0)  # 100% for this task

                self.processed_audio_duration += task.duration # Add processed chunk duration
                self._emit_eta()

            # All chunks completed successfully
            if hasattr(self.job, 'custom_output_filename') and self.job.custom_output_filename:
                base_name = self.job.custom_output_filename
            else:
                base_name = os.path.splitext(os.path.basename(self.job.original_file_path))[0]

            # Emit success status BEFORE attempting cleanup
            self.signals.job_status_updated.emit(JobStatus.DONE, "Job completed")
            total_time = time.time() - self.start_time
            logging.info(f"Transcription for {self.job.original_file_path} completed in {total_time:.2f} seconds.")

            # Clean up checkpoint files
            self._cleanup_checkpoint_files(base_name)

        except Exception as e:
            # Try to save partial results before failing
            try:
                if hasattr(self.job, 'custom_output_filename') and self.job.custom_output_filename:
                    base_name = self.job.custom_output_filename
                else:
                    base_name = os.path.splitext(os.path.basename(self.job.original_file_path))[0]
                txt_path, srt_path = merge_checkpoints_to_files(self.job.output_dir, base_name, output_format=self.settings.output_format)
                self._cleanup_checkpoint_files(base_name)
                if txt_path or srt_path:
                    logging.warning(f"Job failed, but partial results saved to {txt_path or srt_path}")
                    self.signals.job_status_updated.emit(
                        JobStatus.ERROR,
                        f"Error: {str(e)}\nPartial results saved."
                    )
                else:
                    self.signals.job_status_updated.emit(JobStatus.ERROR, str(e))
            except Exception as merge_error:
                logging.error(f"Failed to save partial results: {merge_error}")
                self.signals.job_status_updated.emit(JobStatus.ERROR, str(e))

            logging.error(f"Error during transcription for {self.job.original_file_path}: {e}")
        finally:
            # Clean up temporary directory
            if hasattr(self.job, 'temp_dir') and os.path.exists(self.job.temp_dir):
                shutil.rmtree(self.job.temp_dir)
            self.signals.finished.emit()

    def _cleanup_checkpoint_files(self, base_name):
        """Best-effort cleanup of checkpoint files."""
        try:
            cleanup_checkpoints(self.job.output_dir, base_name)
        except Exception as e:
            logging.warning(f"Could not clean up checkpoint files: {e}")

    def _emit_eta(self):
        elapsed_time = time.time() - self.start_time
        if self.processed_audio_duration > 0 and elapsed_time > 0:
            throughput = self.processed_audio_duration / elapsed_time # seconds of audio per second of real time
            remaining_audio_duration = self.total_audio_duration - self.processed_audio_duration
            if throughput > 0 and remaining_audio_duration > 0:
                remaining_time_seconds = remaining_audio_duration / throughput
                # Convert to hh:mm:ss format
                m, s = divmod(remaining_time_seconds, 60)
                h, m = divmod(m, 60)
                eta_string = f"~{int(h):02}:{int(m):02}:{int(s):02} remaining"
                self.signals.eta_updated.emit(eta_string)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def cancel(self):
        self.is_canceled = True
