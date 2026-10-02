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
import logging
import threading
from datetime import datetime, timedelta
import numpy as np
from PySide6.QtCore import QThread, Signal

from engine.audio_capture import AudioBuffer, resample_to_16k_mono
from engine.model_loader import load_whisper_model, validate_model_path, resolve_model_path
from core.runtime import get_base_path
from core.storage import atomic_text_writer
from engine.checkpoint import _format_srt_time


# Minimum buffer duration before transcription (seconds)
BUFFER_DURATION_SEC = 2
# Poll interval while waiting for buffer to fill (seconds)
POLL_INTERVAL_SEC = 0.2
# Overlap duration between consecutive buffers (seconds)
OVERLAP_SEC = 0.5


class LiveTranscriptionWorker(QThread):
    """Worker thread for live audio capture and transcription."""

    # wall_time (str HH:MM:SS), words (list of str) — for word-by-word display
    words_ready = Signal(str, list)
    status_updated = Signal(str)
    error_occurred = Signal(str)
    backlog_updated = Signal(float)
    audio_level = Signal(float)  # peak level 0.0-1.0

    def __init__(self, device_index: int, device_sample_rate: int,
                 device_channels: int, settings, backend: str = 'sounddevice',
                 parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.device_sample_rate = device_sample_rate
        self.device_channels = device_channels
        self.settings = settings.model_copy(deep=True)
        self.backend = backend
        self._stop_event = threading.Event()
        self._capture_error = None
        self._audio_buffer = None
        self._session_segments: list[tuple[float, float, str]] = []
        self._session_start_time: datetime | None = None

    def stop(self):
        """Signal the worker to stop after processing remaining audio."""
        self._stop_event.set()

    @property
    def session_segments(self) -> list[tuple[float, float, str]]:
        return list(self._session_segments)

    @property
    def queued_seconds(self):
        return self._audio_buffer.duration_seconds if self._audio_buffer is not None else 0.0

    @property
    def capture_warning(self):
        return self._capture_error

    def run(self):
        cleanup_fn = None

        def stop_capture():
            nonlocal cleanup_fn
            if cleanup_fn is not None:
                cleanup, cleanup_fn = cleanup_fn, None
                try:
                    cleanup()
                except Exception:
                    logging.exception("Could not cleanly close audio capture")

        try:
            beam_size = 3

            # Start audio capture IMMEDIATELY so no audio is lost during model loading
            self.status_updated.emit("Starting audio capture...")
            audio_buffer, cleanup_fn = self._start_capture()
            self._audio_buffer = audio_buffer
            self._session_start_time = datetime.now()

            # Load model while audio accumulates in the buffer
            self.status_updated.emit("Loading model (recording audio)...")

            models_dir = getattr(self.settings, 'models_folder', None) or None
            model_path = resolve_model_path(self.settings.language, "faster-whisper", get_base_path(), models_dir)
            if not validate_model_path(model_path):
                self.error_occurred.emit(
                    f"Invalid model path: {model_path}. Required model files are missing."
                )
                return

            device_for_loading = self.settings.device
            if device_for_loading in ("amd", "metal", "auto"):
                device_for_loading = "cpu"
            elif device_for_loading == "nvidia":
                device_for_loading = "gpu"

            model, error_message = load_whisper_model(
                model_path, device_for_loading,
                self.settings.compute_type, self.settings.threads
            )
            if error_message:
                self.error_occurred.emit(f"Model loading failed: {error_message}")
                return

            # Model ready — start transcribing (buffer already has audio from loading period)
            self.status_updated.emit("Recording...")
            self._capture_loop(audio_buffer, model, beam_size, stop_capture)

            if self._capture_error or audio_buffer.overflowed:
                self.error_occurred.emit(self._capture_error or "Audio backlog reached its limit; recording stopped. Buffered audio was processed.")
            else:
                self.status_updated.emit("Session ended")

        except Exception as e:
            logging.error(f"Live transcription error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            stop_capture()

    def _start_capture(self):
        """Start audio capture and return (audio_buffer, cleanup_fn)."""
        if self.backend == 'pyaudiowpatch':
            return self._start_capture_pyaudiowpatch()
        else:
            return self._start_capture_sounddevice()

    def _start_capture_pyaudiowpatch(self):
        """Start capture using pyaudiowpatch. Returns (AudioBuffer, cleanup_fn)."""
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        audio_buffer = AudioBuffer(self.device_sample_rate, self.device_channels)

        def audio_callback(in_data, frame_count, time_info, status_flags):
            if self._stop_event.is_set():
                return (None, pyaudio.paComplete)
            audio = np.frombuffer(in_data, dtype=np.float32)
            if self.device_channels > 1:
                audio = audio.reshape(-1, self.device_channels)
            if status_flags or not audio_buffer.write(audio):
                self._capture_error = "Audio capture could not keep up. Recording stopped; accepted audio is being saved."
                self._stop_event.set()
                return (None, pyaudio.paComplete)
            return (None, pyaudio.paContinue)

        stream = None
        try:
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
        except Exception:
            try:
                if stream is not None:
                    stream.close()
            finally:
                p.terminate()
            raise

        def cleanup():
            try:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
            finally:
                p.terminate()

        return audio_buffer, cleanup

    def _start_capture_sounddevice(self):
        """Start capture using sounddevice. Returns (AudioBuffer, cleanup_fn)."""
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "No audio capture library available.\n"
                "Install pyaudiowpatch for WASAPI loopback:\n"
                "  pip install pyaudiowpatch\n\n"
                "Or install sounddevice as fallback:\n"
                "  pip install sounddevice"
            )

        audio_buffer = AudioBuffer(self.device_sample_rate, self.device_channels)

        def audio_callback(indata, frames, time_info, status):
            if self._stop_event.is_set():
                raise sd.CallbackStop
            if status:
                logging.warning(f"Audio capture status: {status}")
            if status or not audio_buffer.write(indata):
                self._capture_error = "Audio capture could not keep up. Recording stopped; accepted audio is being saved."
                self._stop_event.set()
                raise sd.CallbackStop

        stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.device_sample_rate,
            channels=self.device_channels,
            dtype='float32',
            callback=audio_callback,
            blocksize=1024,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise

        def cleanup():
            try:
                stream.stop()
            finally:
                stream.close()

        return audio_buffer, cleanup

    def _capture_loop(self, audio_buffer, model, beam_size, stop_capture=None):
        """Drain bounded batches in order, including accepted audio after Stop."""
        from engine.live_timestamps import LiveTimeline, subtitle_cues
        timeline = LiveTimeline()
        elapsed = 0.0
        overlap_audio = None
        last_prompt = ""
        capture_stopped = False
        overlap_samples = int(self.device_sample_rate * OVERLAP_SEC)

        while True:
            stopping = self._stop_event.is_set() or audio_buffer.overflowed
            if stopping and not capture_stopped:
                if stop_capture:
                    stop_capture()
                capture_stopped = True
            backlog = audio_buffer.duration_seconds
            self.backlog_updated.emit(backlog)
            self.audio_level.emit(audio_buffer.peak_level)
            if stopping and backlog <= 0:
                break
            if not stopping and backlog < BUFFER_DURATION_SEC:
                self._stop_event.wait(POLL_INTERVAL_SEC)
                continue
            raw_audio = audio_buffer.read_and_clear(max_seconds=10)
            if raw_audio is None:
                continue
            duration = len(raw_audio) / self.device_sample_rate
            overlap = len(overlap_audio) / self.device_sample_rate if overlap_audio is not None else 0.0
            combined = np.concatenate([overlap_audio, raw_audio], axis=0) if overlap_audio is not None else raw_audio
            # Copy the small tail so it cannot retain a large inference batch.
            overlap_audio = raw_audio[-overlap_samples:].copy()
            self.status_updated.emit("Processing remaining audio..." if stopping else "Transcribing...")
            words = self._transcribe_buffer(combined, model, beam_size, last_prompt)
            accepted = timeline.append(words, elapsed - overlap, elapsed, elapsed + duration)
            if accepted:
                text = " ".join(word[2] for word in accepted)
                wall_time = self._session_start_time + timedelta(seconds=accepted[0][0])
                self.words_ready.emit(wall_time.strftime("%H:%M:%S"), text.split())
                self._session_segments.extend(subtitle_cues(accepted))
                last_prompt = text[-200:]
            elapsed += duration
            if not stopping:
                self.status_updated.emit("Recording...")
        self.backlog_updated.emit(0.0)

    def _transcribe_buffer(
        self, raw_audio: np.ndarray, model, beam_size: int, prompt: str = ""
    ) -> list[tuple[float, float, str]]:
        """
        Transcribe a raw audio buffer. Returns list of (start, end, text) tuples
        with timestamps relative to the buffer start (in seconds).
        """
        audio_16k = resample_to_16k_mono(raw_audio, self.device_sample_rate)
        if not len(audio_16k):
            return []

        peak = float(np.max(np.abs(audio_16k)))
        logging.debug(f"Live buffer: {len(audio_16k)} samples, peak={peak:.4f}")

        # Skip near-silent buffers
        if peak < 0.001:
            logging.debug("Skipping silent buffer")
            return []

        segments = []

        try:
            kwargs = dict(
                language=self.settings.language,
                beam_size=beam_size,
                vad_filter=self.settings.vad_enabled,
                word_timestamps=True,
            )
            if prompt:
                kwargs["initial_prompt"] = prompt

            result_segments, _info = model.transcribe(audio_16k, **kwargs)
            for seg in result_segments:
                text = seg.text.strip()
                if text:
                    words = getattr(seg, "words", None)
                    if words:
                        segments.extend((word.start, word.end, word.word) for word in words if word.word.strip())
                    else:
                        segments.append((seg.start, seg.end, text))

            logging.debug(f"Transcribed {len(segments)} segments")
        except Exception as e:
            logging.error(f"Transcription error: {e}", exc_info=True)
            raise RuntimeError(f"Transcription error: {e}") from e

        return segments


def save_live_session(segments: list[tuple[float, float, str]],
                      output_dir: str, base_name: str,
                      output_format: str = "both"):
    """
    Save accumulated live session segments to output files.

    Args:
        segments: List of (start_seconds, end_seconds, text), relative to session start
        output_dir: Directory to save output files
        base_name: Base filename without extension
        output_format: "srt", "txt", or "both"
    """
    os.makedirs(output_dir, exist_ok=True)

    if not segments:
        return

    if output_format in ("txt", "both"):
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with atomic_text_writer(txt_path) as f:
            for _, _, text in segments:
                f.write(text + '\n')

    if output_format in ("srt", "both"):
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        with atomic_text_writer(srt_path) as f:
            for i, (start, end, text) in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
                f.write(f"{text}\n\n")
