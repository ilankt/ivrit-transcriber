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
from engine.whisper_cpp_runner import (
    get_whispercpp_binary_path, validate_whispercpp_binary,
    validate_ggml_model, transcribe_chunk_whispercpp,
)
from engine.merger import merge_srt_files, merge_txt_files
from engine.checkpoint import save_chunk_checkpoint, merge_checkpoints_to_files, cleanup_checkpoints


def get_base_path():
    """
    Get the base path for the application.
    When running as a PyInstaller executable, use the directory containing the exe.
    Otherwise, use the directory containing the main script.
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_model_path(model_type):
    """Get the CTranslate2 model path for faster-whisper."""
    base_path = get_base_path()
    if model_type == "Fast":
        return os.path.join(base_path, 'Models', 'ivrit-large-v3-turbo-ct2')
    else:
        return os.path.join(base_path, 'Models', 'ivrit-large-v3-ct2')


def get_ggml_model_path(model_type):
    """Get the GGML model path for whisper.cpp."""
    base_path = get_base_path()
    if model_type == "Fast":
        return os.path.join(base_path, 'Models', 'ggml-ivrit-large-v3-turbo.bin')
    else:
        return os.path.join(base_path, 'Models', 'ggml-ivrit-large-v3.bin')


def determine_engine(device: str) -> str:
    """
    Determine which transcription engine to use based on device setting.

    Returns: "faster-whisper" or "whisper-cpp"
    """
    if device == "amd":
        return "whisper-cpp"
    elif device == "auto":
        # Check if AMD GPU is available and no NVIDIA GPU
        from engine.gpu_detector import detect_cuda_gpu, detect_vulkan_gpu
        cuda_ok, _ = detect_cuda_gpu()
        if cuda_ok:
            return "faster-whisper"
        amd_ok, _ = detect_vulkan_gpu()
        if amd_ok:
            return "whisper-cpp"
        return "faster-whisper"  # CPU fallback
    else:
        return "faster-whisper"  # "cpu" or "nvidia"


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
        self.total_audio_duration = 0.0
        self._current_process = None  # For whisper.cpp subprocess cancellation

    def _get_base_name(self):
        if hasattr(self.job, 'custom_output_filename') and self.job.custom_output_filename:
            return self.job.custom_output_filename
        return os.path.splitext(os.path.basename(self.job.original_file_path))[0]

    @Slot()
    def run(self):
        self.signals.job_status_updated.emit(JobStatus.RUNNING, "Starting job")
        model = None
        self.start_time = time.time()

        engine = determine_engine(self.settings.device)

        try:
            beam_size = 1 if self.settings.model_type == "Fast" else 3

            logging.info(f"Starting transcription for {self.job.original_file_path}")
            logging.info(f"Engine: {engine}, Model type: {self.settings.model_type}, Device: {self.settings.device}")

            # Engine-specific setup
            if engine == "whisper-cpp":
                ggml_path = get_ggml_model_path(self.settings.model_type)
                binary_path = get_whispercpp_binary_path(get_base_path())

                if not binary_path or not validate_whispercpp_binary(binary_path):
                    self.signals.job_status_updated.emit(
                        JobStatus.ERROR,
                        "whisper-cli binary not found. Place whisper-cli.exe in the Binaries/ folder or add it to PATH."
                    )
                    return

                if not validate_ggml_model(ggml_path):
                    self.signals.job_status_updated.emit(
                        JobStatus.ERROR,
                        f"GGML model not found: {ggml_path}\nDownload the GGML model and place it in the Models/ folder."
                    )
                    return

                logging.info(f"whisper.cpp binary: {binary_path}")
                logging.info(f"GGML model: {ggml_path}")
            else:
                model_path = get_model_path(self.settings.model_type)
                logging.info(f"Model path: {model_path}")

                if not validate_model_path(model_path):
                    self.signals.job_status_updated.emit(
                        JobStatus.ERROR,
                        f"Invalid model path: {model_path}. Required model files (model.bin, tokenizer.json, vocabulary.json) are missing."
                    )
                    return

                device_for_loading = self.settings.device
                if device_for_loading == "nvidia":
                    device_for_loading = "gpu"

                model, error_message = load_whisper_model(
                    model_path, device_for_loading,
                    self.settings.compute_type, self.settings.threads
                )
                if error_message:
                    self.signals.job_status_updated.emit(JobStatus.ERROR, f"Model loading failed: {error_message}")
                    return

            # Ensure output directory exists
            os.makedirs(self.job.output_dir, exist_ok=True)

            # Calculate total audio duration for ETA
            self.total_audio_duration = sum(task.duration for task in self.job.tasks)
            self._emit_eta()

            # Process tasks
            for i, task in enumerate(self.job.tasks):
                if self.is_canceled:
                    self._save_and_report_cancel()
                    return

                while self.is_paused:
                    time.sleep(0.1)

                self.signals.task_status_updated.emit(
                    i, TaskStatus.RUNNING,
                    f"Transcribing chunk {task.chunk_index + 1}/{len(self.job.tasks)} of {os.path.basename(self.job.original_file_path)}"
                )

                # Transcribe chunk using the selected engine
                if engine == "whisper-cpp":
                    def progress_cb(pct):
                        self.signals.progress_updated.emit(i, pct / 100.0)

                    text, srt_segments = transcribe_chunk_whispercpp(
                        audio_path=task.chunk_path,
                        model_path=ggml_path,
                        binary_path=binary_path,
                        beam_size=beam_size,
                        vad_filter=self.settings.vad_enabled,
                        use_gpu=True,
                        progress_callback=progress_cb,
                    )
                else:
                    text, srt_segments = transcribe_chunk(
                        task.chunk_path, model, "he", beam_size, self.settings.vad_enabled
                    )

                if self.is_canceled:
                    task.text = text
                    task.srt_segments = srt_segments
                    task.status = TaskStatus.DONE
                    base_name = self._get_base_name()
                    save_chunk_checkpoint(
                        self.job.output_dir, base_name, task.chunk_index,
                        text, srt_segments, task.duration
                    )
                    self._save_and_report_cancel()
                    return

                task.text = text
                task.srt_segments = srt_segments
                task.status = TaskStatus.DONE

                base_name = self._get_base_name()
                save_chunk_checkpoint(
                    self.job.output_dir, base_name, task.chunk_index,
                    text, srt_segments, task.duration
                )
                logging.info(f"Saved checkpoint for chunk {task.chunk_index + 1}/{len(self.job.tasks)}")

                merge_checkpoints_to_files(self.job.output_dir, base_name, output_format=self.settings.output_format)
                logging.info(f"Merged {task.chunk_index + 1} completed chunks")

                self.signals.task_status_updated.emit(
                    i, TaskStatus.DONE,
                    f"Finished chunk {task.chunk_index + 1}/{len(self.job.tasks)} of {os.path.basename(self.job.original_file_path)}"
                )
                self.signals.progress_updated.emit(i, 1.0)

                self.processed_audio_duration += task.duration
                self._emit_eta()

            # All chunks completed
            base_name = self._get_base_name()
            self.signals.job_status_updated.emit(JobStatus.DONE, "Job completed")
            total_time = time.time() - self.start_time
            logging.info(f"Transcription for {self.job.original_file_path} completed in {total_time:.2f} seconds.")
            self._cleanup_checkpoint_files(base_name)

        except Exception as e:
            try:
                base_name = self._get_base_name()
                txt_path, srt_path = merge_checkpoints_to_files(self.job.output_dir, base_name, output_format=self.settings.output_format)
                self._cleanup_checkpoint_files(base_name)
                if txt_path or srt_path:
                    logging.warning(f"Job failed, but partial results saved to {txt_path or srt_path}")
                    self.signals.job_status_updated.emit(
                        JobStatus.ERROR, f"Error: {str(e)}\nPartial results saved."
                    )
                else:
                    self.signals.job_status_updated.emit(JobStatus.ERROR, str(e))
            except Exception as merge_error:
                logging.error(f"Failed to save partial results: {merge_error}")
                self.signals.job_status_updated.emit(JobStatus.ERROR, str(e))

            logging.error(f"Error during transcription for {self.job.original_file_path}: {e}")
        finally:
            if hasattr(self.job, 'temp_dir') and os.path.exists(self.job.temp_dir):
                shutil.rmtree(self.job.temp_dir)
            self.signals.finished.emit()

    def _save_and_report_cancel(self):
        base_name = self._get_base_name()
        txt_path, srt_path = merge_checkpoints_to_files(self.job.output_dir, base_name, output_format=self.settings.output_format)
        self._cleanup_checkpoint_files(base_name)
        if txt_path or srt_path:
            self.signals.job_status_updated.emit(JobStatus.CANCELED, "Job canceled. Partial results saved.")
            logging.info(f"Transcription canceled. Partial results saved.")
        else:
            self.signals.job_status_updated.emit(JobStatus.CANCELED, "Job canceled")
            logging.info(f"Transcription canceled for {self.job.original_file_path}")

    def _cleanup_checkpoint_files(self, base_name):
        try:
            cleanup_checkpoints(self.job.output_dir, base_name)
        except Exception as e:
            logging.warning(f"Could not clean up checkpoint files: {e}")

    def _emit_eta(self):
        elapsed_time = time.time() - self.start_time
        if self.processed_audio_duration > 0 and elapsed_time > 0:
            throughput = self.processed_audio_duration / elapsed_time
            remaining_audio_duration = self.total_audio_duration - self.processed_audio_duration
            if throughput > 0 and remaining_audio_duration > 0:
                remaining_time_seconds = remaining_audio_duration / throughput
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
        # Kill whisper.cpp subprocess immediately if running
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass
