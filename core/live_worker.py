"""
Live transcription worker thread.

Captures system audio via WASAPI loopback, buffers it, and feeds it to
faster-whisper for real-time Hebrew transcription. Always uses faster-whisper
(not whisper-cpp) because the model stays loaded in memory between buffers.

Features:
- Audio overlap between buffers to avoid cutting words at boundaries
- Context prompting (previous text fed to model for coherence)
- Word-by-word streaming to UI for live caption feel

Supports pyaudiowpatch (preferred for WASAPI loopback) and sounddevice (fallback).
"""
import os
import time
import logging
import threading
from datetime import datetime, timedelta
import numpy as np
from PySide6.QtCore import QThread, Signal

from engine.audio_capture import AudioBuffer, resample_to_16k_mono
from engine.model_loader import load_whisper_model, validate_model_path
from core.worker import get_model_path


# Minimum buffer duration before transcription (seconds)
BUFFER_DURATION_SEC = 3
# Poll interval while waiting for buffer to fill (seconds)
POLL_INTERVAL_SEC = 0.2
# Overlap duration between consecutive buffers (seconds)
OVERLAP_SEC = 1.0


class LiveTranscriptionWorker(QThread):
    """Worker thread for live audio capture and transcription."""

    # wall_time (str HH:MM:SS), words (list of str) — for word-by-word display
    words_ready = Signal(str, list)
    status_updated = Signal(str)
    error_occurred = Signal(str)
    audio_level = Signal(float)  # peak level 0.0-1.0
    session_finished = Signal()

    def __init__(self, device_index: int, device_sample_rate: int,
                 device_channels: int, settings, backend: str = 'sounddevice',
                 parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.device_sample_rate = device_sample_rate
        self.device_channels = device_channels
        self.settings = settings
        self.backend = backend
        self._stop_event = threading.Event()
        self._session_segments: list[tuple[str, str, str]] = []
        self._session_start_time: datetime | None = None

    def stop(self):
        """Signal the worker to stop after processing remaining audio."""
        self._stop_event.set()

    @property
    def session_segments(self) -> list[tuple[str, str, str]]:
        return list(self._session_segments)

    def run(self):
        model = None

        try:
            self.status_updated.emit("Loading model...")
            beam_size = 1

            model_path = get_model_path("Fast")
            if not validate_model_path(model_path):
                self.error_occurred.emit(
                    f"Invalid model path: {model_path}. Required model files are missing."
                )
                self.session_finished.emit()
                return

            device_for_loading = self.settings.device
            if device_for_loading in ("amd", "auto"):
                device_for_loading = "cpu"
            elif device_for_loading == "nvidia":
                device_for_loading = "gpu"

            model, error_message = load_whisper_model(
                model_path, device_for_loading,
                self.settings.compute_type, self.settings.threads
            )
            if error_message:
                self.error_occurred.emit(f"Model loading failed: {error_message}")
                self.session_finished.emit()
                return

            if self.backend == 'pyaudiowpatch':
                self._run_with_pyaudiowpatch(model, beam_size)
            else:
                self._run_with_sounddevice(model, beam_size)

        except Exception as e:
            logging.error(f"Live transcription error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.session_finished.emit()

    def _run_with_pyaudiowpatch(self, model, beam_size):
        """Capture audio using pyaudiowpatch (proper WASAPI loopback)."""
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        audio_buffer = AudioBuffer(self.device_sample_rate, self.device_channels)

        def audio_callback(in_data, frame_count, time_info, status_flags):
            audio = np.frombuffer(in_data, dtype=np.float32)
            if self.device_channels > 1:
                audio = audio.reshape(-1, self.device_channels)
            audio_buffer.write(audio)
            return (None, pyaudio.paContinue)

        try:
            self.status_updated.emit("Starting audio capture...")
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=self.device_channels,
                rate=self.device_sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024,
                stream_callback=audio_callback,
            )
            stream.start_stream()

            self._session_start_time = datetime.now()
            self.status_updated.emit("Recording...")

            try:
                self._capture_loop(audio_buffer, model, beam_size)
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            p.terminate()

        self.status_updated.emit("Session ended")

    def _run_with_sounddevice(self, model, beam_size):
        """Capture audio using sounddevice (fallback, limited loopback support)."""
        try:
            import sounddevice as sd
        except ImportError:
            self.error_occurred.emit(
                "No audio capture library available.\n"
                "Install pyaudiowpatch for WASAPI loopback:\n"
                "  pip install pyaudiowpatch\n\n"
                "Or install sounddevice as fallback:\n"
                "  pip install sounddevice"
            )
            return

        audio_buffer = AudioBuffer(self.device_sample_rate, self.device_channels)

        def audio_callback(indata, frames, time_info, status):
            if status:
                logging.warning(f"Audio capture status: {status}")
            audio_buffer.write(indata)

        self.status_updated.emit("Starting audio capture...")
        stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.device_sample_rate,
            channels=self.device_channels,
            dtype='float32',
            callback=audio_callback,
            blocksize=1024,
        )

        self._session_start_time = datetime.now()

        with stream:
            self.status_updated.emit("Recording...")
            self._capture_loop(audio_buffer, model, beam_size)

        self.status_updated.emit("Session ended")

    def _capture_loop(self, audio_buffer, model, beam_size):
        """Main capture/transcription loop with overlap and context."""
        elapsed_audio_time = 0.0
        overlap_audio: np.ndarray | None = None  # last OVERLAP_SEC of raw audio
        last_prompt = ""  # previous transcription for context

        overlap_samples = int(self.device_sample_rate * OVERLAP_SEC)

        while not self._stop_event.is_set():
            self.audio_level.emit(audio_buffer.peak_level)

            if audio_buffer.duration_seconds < BUFFER_DURATION_SEC:
                self._stop_event.wait(POLL_INTERVAL_SEC)
                continue

            raw_audio = audio_buffer.read_and_clear()
            if raw_audio is None:
                continue

            new_duration = len(raw_audio) / self.device_sample_rate

            # Prepend overlap from previous buffer
            if overlap_audio is not None:
                combined = np.concatenate([overlap_audio, raw_audio], axis=0)
            else:
                combined = raw_audio

            # Save overlap for next iteration (last OVERLAP_SEC of raw audio)
            if len(raw_audio) > overlap_samples:
                overlap_audio = raw_audio[-overlap_samples:]
            else:
                overlap_audio = raw_audio.copy()

            # Wall clock time for the NEW audio (excluding overlap)
            buffer_wall_start = self._session_start_time + timedelta(seconds=elapsed_audio_time)

            self.status_updated.emit("Transcribing...")
            segments = self._transcribe_buffer(combined, model, beam_size, last_prompt)

            # Determine how much overlap was prepended
            actual_overlap = len(combined) / self.device_sample_rate - new_duration

            # Filter and adjust: skip segments from overlap region
            new_segments = []
            for seg_start, seg_end, text in segments:
                # Adjust timestamps: subtract overlap to get time relative to new audio
                adj_start = seg_start - actual_overlap
                adj_end = seg_end - actual_overlap

                # Skip segments entirely in the overlap region
                if adj_end <= 0:
                    continue

                # Clamp start to 0 for segments that straddle the boundary
                adj_start = max(0.0, adj_start)
                new_segments.append((adj_start, adj_end, text))

            # Build word list and emit for live display
            all_words = []
            for _, _, text in new_segments:
                all_words.extend(text.split())

            if all_words:
                wall_time_str = buffer_wall_start.strftime("%H:%M:%S")
                self.words_ready.emit(wall_time_str, all_words)

                # Store full segments for session save
                full_text = " ".join(all_words)
                wall_end = buffer_wall_start + timedelta(seconds=new_duration)
                wall_end_str = wall_end.strftime("%H:%M:%S")
                self._session_segments.append((wall_time_str, wall_end_str, full_text))

                # Update context prompt for next buffer (last ~200 chars)
                last_prompt = full_text[-200:]

            elapsed_audio_time += new_duration
            self.status_updated.emit("Recording...")

        # Process remaining audio after stop
        remaining = audio_buffer.read_and_clear()
        if remaining is not None and len(remaining) > self.device_sample_rate:
            self.status_updated.emit("Processing remaining audio...")
            buffer_wall_start = self._session_start_time + timedelta(seconds=elapsed_audio_time)

            if overlap_audio is not None:
                combined = np.concatenate([overlap_audio, remaining], axis=0)
                actual_overlap = len(overlap_audio) / self.device_sample_rate
            else:
                combined = remaining
                actual_overlap = 0.0

            new_duration = len(remaining) / self.device_sample_rate
            segments = self._transcribe_buffer(combined, model, beam_size, last_prompt)

            all_words = []
            for seg_start, seg_end, text in segments:
                adj_end = seg_end - actual_overlap
                if adj_end <= 0:
                    continue
                all_words.extend(text.split())

            if all_words:
                wall_time_str = buffer_wall_start.strftime("%H:%M:%S")
                self.words_ready.emit(wall_time_str, all_words)
                full_text = " ".join(all_words)
                wall_end = buffer_wall_start + timedelta(seconds=new_duration)
                self._session_segments.append((wall_time_str, wall_end.strftime("%H:%M:%S"), full_text))

    def _transcribe_buffer(
        self, raw_audio: np.ndarray, model, beam_size: int, prompt: str = ""
    ) -> list[tuple[float, float, str]]:
        """
        Transcribe a raw audio buffer. Returns list of (start, end, text) tuples
        with timestamps relative to the buffer start (in seconds).
        """
        audio_16k = resample_to_16k_mono(raw_audio, self.device_sample_rate)

        # Skip near-silent buffers
        if np.max(np.abs(audio_16k)) < 0.01:
            return []

        segments = []

        try:
            kwargs = dict(
                language="he",
                beam_size=beam_size,
                vad_filter=self.settings.vad_enabled,
            )
            if prompt:
                kwargs["initial_prompt"] = prompt

            result_segments, info = model.transcribe(audio_16k, **kwargs)
            for seg in result_segments:
                if self._stop_event.is_set():
                    break
                text = seg.text.strip()
                if text:
                    segments.append((seg.start, seg.end, text))
        except Exception as e:
            logging.error(f"Transcription error: {e}")

        return segments


def save_live_session(segments: list[tuple[str, str, str]],
                      output_dir: str, base_name: str,
                      output_format: str = "both"):
    """
    Save accumulated live session segments to output files.

    Args:
        segments: List of (wall_start_str, wall_end_str, text)
        output_dir: Directory to save output files
        base_name: Base filename without extension
        output_format: "srt", "txt", or "both"
    """
    os.makedirs(output_dir, exist_ok=True)

    if not segments:
        return

    if output_format in ("txt", "both"):
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for _, _, text in segments:
                f.write(text + '\n')

    if output_format in ("srt", "both"):
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, (start_str, end_str, text) in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{start_str},000 --> {end_str},000\n")
                f.write(f"{text}\n\n")
