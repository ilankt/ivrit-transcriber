"""
Live transcription worker thread.

Captures system audio via WASAPI loopback, buffers it, and feeds it to
faster-whisper or whisper.cpp for real-time Hebrew transcription.
"""
import os
import sys
import json
import time
import logging
import tempfile
import threading
import wave
import numpy as np
from PySide6.QtCore import QThread, Signal

from engine.audio_capture import AudioBuffer, resample_to_16k_mono
from engine.model_loader import load_whisper_model, validate_model_path
from engine.whisper_cpp_runner import (
    get_whispercpp_binary_path, validate_whispercpp_binary,
    validate_ggml_model, transcribe_chunk_whispercpp,
)
from core.worker import determine_engine, get_base_path, get_model_path, get_ggml_model_path


# Minimum buffer duration before transcription (seconds)
BUFFER_DURATION_SEC = 10
# Poll interval while waiting for buffer to fill (seconds)
POLL_INTERVAL_SEC = 0.5


class LiveTranscriptionWorker(QThread):
    """Worker thread for live audio capture and transcription."""

    segment_ready = Signal(float, float, str)  # abs_start, abs_end, text
    status_updated = Signal(str)
    error_occurred = Signal(str)
    audio_level = Signal(float)  # peak level 0.0-1.0
    session_finished = Signal()

    def __init__(self, device_index: int, device_sample_rate: int,
                 device_channels: int, settings, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.device_sample_rate = device_sample_rate
        self.device_channels = device_channels
        self.settings = settings
        self._stop_event = threading.Event()
        self._session_segments: list[tuple[float, float, str]] = []
        self._session_start_time: float = 0.0

    def stop(self):
        """Signal the worker to stop after processing remaining audio."""
        self._stop_event.set()

    @property
    def session_segments(self) -> list[tuple[float, float, str]]:
        return list(self._session_segments)

    def run(self):
        try:
            import sounddevice as sd
        except ImportError:
            self.error_occurred.emit("sounddevice package is not installed. Install with: pip install sounddevice")
            self.session_finished.emit()
            return

        engine = determine_engine(self.settings.device)
        model = None
        binary_path = None
        ggml_path = None

        try:
            # Load model
            self.status_updated.emit("Loading model...")
            beam_size = 1 if self.settings.model_type == "Fast" else 3

            if engine == "whisper-cpp":
                ggml_path = get_ggml_model_path(self.settings.model_type)
                binary_path = get_whispercpp_binary_path(get_base_path())
                if not binary_path or not validate_whispercpp_binary(binary_path):
                    self.error_occurred.emit(
                        "whisper-cli binary not found. Place whisper-cli.exe in the Binaries/ folder or add it to PATH."
                    )
                    self.session_finished.emit()
                    return
                if not validate_ggml_model(ggml_path):
                    self.error_occurred.emit(f"GGML model not found: {ggml_path}")
                    self.session_finished.emit()
                    return
            else:
                model_path = get_model_path(self.settings.model_type)
                if not validate_model_path(model_path):
                    self.error_occurred.emit(
                        f"Invalid model path: {model_path}. Required model files are missing."
                    )
                    self.session_finished.emit()
                    return

                device_for_loading = self.settings.device
                if device_for_loading == "nvidia":
                    device_for_loading = "gpu"

                model, error_message = load_whisper_model(
                    model_path, device_for_loading,
                    self.settings.compute_type, self.settings.threads
                )
                if error_message:
                    self.error_occurred.emit(f"Model loading failed: {error_message}")
                    self.session_finished.emit()
                    return

            # Set up audio capture
            self.status_updated.emit("Starting audio capture...")
            audio_buffer = AudioBuffer(self.device_sample_rate, self.device_channels)

            def audio_callback(indata, frames, time_info, status):
                if status:
                    logging.warning(f"Audio capture status: {status}")
                audio_buffer.write(indata)

            stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.device_sample_rate,
                channels=self.device_channels,
                dtype='float32',
                callback=audio_callback,
                blocksize=1024,
            )

            self._session_start_time = time.time()
            elapsed_audio_time = 0.0  # Tracks absolute position in session timeline

            with stream:
                self.status_updated.emit("Recording...")

                while not self._stop_event.is_set():
                    # Emit audio level for VU meter
                    self.audio_level.emit(audio_buffer.peak_level)

                    # Wait until buffer has enough audio
                    if audio_buffer.duration_seconds < BUFFER_DURATION_SEC:
                        self._stop_event.wait(POLL_INTERVAL_SEC)
                        continue

                    # Extract and transcribe buffer
                    raw_audio = audio_buffer.read_and_clear()
                    if raw_audio is None:
                        continue

                    buffer_duration = len(raw_audio) / self.device_sample_rate

                    self.status_updated.emit("Transcribing...")
                    segments = self._transcribe_buffer(
                        raw_audio, engine, model, beam_size,
                        binary_path, ggml_path
                    )

                    # Emit segments with absolute timestamps
                    for seg_start, seg_end, text in segments:
                        abs_start = elapsed_audio_time + seg_start
                        abs_end = elapsed_audio_time + seg_end
                        self._session_segments.append((abs_start, abs_end, text))
                        self.segment_ready.emit(abs_start, abs_end, text)

                    elapsed_audio_time += buffer_duration
                    self.status_updated.emit("Recording...")

                # Process remaining audio after stop
                remaining = audio_buffer.read_and_clear()
                if remaining is not None and len(remaining) > self.device_sample_rate:  # >1s
                    self.status_updated.emit("Processing remaining audio...")
                    buffer_duration = len(remaining) / self.device_sample_rate
                    segments = self._transcribe_buffer(
                        remaining, engine, model, beam_size,
                        binary_path, ggml_path
                    )
                    for seg_start, seg_end, text in segments:
                        abs_start = elapsed_audio_time + seg_start
                        abs_end = elapsed_audio_time + seg_end
                        self._session_segments.append((abs_start, abs_end, text))
                        self.segment_ready.emit(abs_start, abs_end, text)

            self.status_updated.emit("Session ended")

        except Exception as e:
            logging.error(f"Live transcription error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.session_finished.emit()

    def _transcribe_buffer(
        self, raw_audio: np.ndarray, engine: str, model,
        beam_size: int, binary_path: str | None, ggml_path: str | None
    ) -> list[tuple[float, float, str]]:
        """
        Transcribe a raw audio buffer. Returns list of (start, end, text) tuples
        with timestamps relative to the buffer start.
        """
        audio_16k = resample_to_16k_mono(raw_audio, self.device_sample_rate)

        # Skip near-silent buffers
        if np.max(np.abs(audio_16k)) < 0.01:
            return []

        segments = []

        try:
            if engine == "whisper-cpp":
                # whisper.cpp needs a WAV file
                segments = self._transcribe_whispercpp(
                    audio_16k, binary_path, ggml_path, beam_size
                )
            else:
                # faster-whisper accepts numpy arrays directly
                result_segments, info = model.transcribe(
                    audio_16k,
                    language="he",
                    beam_size=beam_size,
                    vad_filter=self.settings.vad_enabled,
                )
                for seg in result_segments:
                    if self._stop_event.is_set():
                        break
                    text = seg.text.strip()
                    if text:
                        segments.append((seg.start, seg.end, text))
        except Exception as e:
            logging.error(f"Transcription error: {e}")

        return segments

    def _transcribe_whispercpp(
        self, audio_16k: np.ndarray, binary_path: str,
        ggml_path: str, beam_size: int
    ) -> list[tuple[float, float, str]]:
        """Transcribe using whisper.cpp by writing a temp WAV file."""
        tmp_wav = None
        try:
            # Write temp WAV
            tmp_fd, tmp_wav = tempfile.mkstemp(suffix='.wav', prefix='ivrit_live_')
            os.close(tmp_fd)

            # Convert float32 to int16 for WAV
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            with wave.open(tmp_wav, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())

            text, srt_segments_json = transcribe_chunk_whispercpp(
                audio_path=tmp_wav,
                model_path=ggml_path,
                binary_path=binary_path,
                beam_size=beam_size,
                vad_filter=self.settings.vad_enabled,
                use_gpu=True,
            )

            segments = []
            for seg_json in srt_segments_json:
                seg = json.loads(seg_json)
                seg_text = seg['text'].strip()
                if seg_text:
                    segments.append((seg['start'], seg['end'], seg_text))
            return segments

        except Exception as e:
            logging.error(f"whisper.cpp live transcription error: {e}")
            return []
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass


def save_live_session(segments: list[tuple[float, float, str]],
                      output_dir: str, base_name: str,
                      output_format: str = "both"):
    """
    Save accumulated live session segments to output files.

    Args:
        segments: List of (start_seconds, end_seconds, text)
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
            for i, (start, end, text) in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
                f.write(f"{text}\n\n")


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
