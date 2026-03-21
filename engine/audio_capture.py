"""
Audio capture infrastructure for live transcription.

Provides WASAPI loopback device enumeration, thread-safe audio buffering,
and resampling to 16kHz mono float32 for whisper model input.

Prefers pyaudiowpatch for reliable WASAPI loopback on Windows,
falls back to sounddevice if unavailable.
"""
import logging
import threading
import numpy as np


def list_loopback_devices() -> list[dict]:
    """
    Query for WASAPI loopback devices.

    Tries pyaudiowpatch first (reliable WASAPI loopback support),
    falls back to sounddevice.

    Returns list of dicts with keys: index, name, channels, sample_rate, backend.
    Returns empty list if no suitable library or devices are found.
    """
    # Try pyaudiowpatch first - proper WASAPI loopback support
    try:
        import pyaudiowpatch as pyaudio
        devices = _list_devices_pyaudiowpatch(pyaudio)
        if devices:
            return devices
    except ImportError:
        pass

    # Fallback to sounddevice
    try:
        import sounddevice as sd
        return _list_devices_sounddevice(sd)
    except ImportError:
        pass

    return []


def _list_devices_pyaudiowpatch(pyaudio) -> list[dict]:
    """List WASAPI loopback devices using pyaudiowpatch."""
    devices = []
    p = pyaudio.PyAudio()
    try:
        # pyaudiowpatch provides a generator for loopback devices
        try:
            for dev in p.get_loopback_device_info_generator():
                devices.append({
                    'index': dev['index'],
                    'name': dev['name'],
                    'channels': dev['maxInputChannels'],
                    'sample_rate': int(dev['defaultSampleRate']),
                    'backend': 'pyaudiowpatch',
                })
        except OSError:
            # No WASAPI loopback devices available
            pass
    except Exception as e:
        logging.warning(f"pyaudiowpatch device enumeration failed: {e}")
    finally:
        p.terminate()

    return devices


def _list_devices_sounddevice(sd) -> list[dict]:
    """
    List WASAPI input devices using sounddevice (fallback).

    Note: sounddevice does NOT support true WASAPI loopback capture.
    Only devices that already appear as input devices (e.g. Stereo Mix,
    virtual audio cables) will work for system audio capture.
    """
    devices = []
    try:
        host_apis = sd.query_hostapis()
        wasapi_index = None
        for i, api in enumerate(host_apis):
            if 'wasapi' in api['name'].lower():
                wasapi_index = i
                break

        all_devices = sd.query_devices()
        for i, dev in enumerate(all_devices):
            if dev['max_input_channels'] > 0:
                is_wasapi = (wasapi_index is not None and dev['hostapi'] == wasapi_index)
                is_loopback = 'loopback' in dev['name'].lower()
                if is_wasapi or is_loopback:
                    devices.append({
                        'index': i,
                        'name': dev['name'],
                        'channels': dev['max_input_channels'],
                        'sample_rate': int(dev['default_samplerate']),
                        'backend': 'sounddevice',
                    })
    except Exception:
        pass

    return devices


class AudioBuffer:
    """Thread-safe audio accumulator for streaming capture."""

    def __init__(self, sample_rate: int, channels: int):
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._sample_rate = sample_rate
        self._channels = channels
        self._peak = 0.0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def write(self, data: np.ndarray):
        """Append audio chunk from callback. Must be fast."""
        with self._lock:
            self._chunks.append(data.copy())
            # Update peak level for VU meter
            peak = float(np.max(np.abs(data)))
            self._peak = max(self._peak, peak)

    def read_and_clear(self) -> np.ndarray | None:
        """Extract all accumulated audio and reset buffer. Returns None if empty."""
        with self._lock:
            if not self._chunks:
                return None
            audio = np.concatenate(self._chunks, axis=0)
            self._chunks.clear()
            self._peak = 0.0
            return audio

    @property
    def duration_seconds(self) -> float:
        """Current buffer duration in seconds."""
        with self._lock:
            if not self._chunks:
                return 0.0
            total_samples = sum(chunk.shape[0] for chunk in self._chunks)
            return total_samples / self._sample_rate

    @property
    def peak_level(self) -> float:
        """Current peak audio level (0.0-1.0) for VU meter."""
        with self._lock:
            return min(self._peak, 1.0)


def resample_to_16k_mono(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """
    Convert audio to 16kHz mono float32 for whisper model input.

    Args:
        audio: Input audio array, shape (samples,) or (samples, channels)
        source_rate: Source sample rate in Hz

    Returns:
        float32 numpy array at 16000 Hz, mono
    """
    # Convert to float32 if needed
    if audio.dtype != np.float32:
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)

    # Convert to mono if multi-channel
    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)
    elif audio.ndim > 1:
        audio = audio[:, 0]

    target_rate = 16000

    if source_rate == target_rate:
        return audio

    # Simple integer-ratio decimation for common cases (48kHz -> 16kHz = ratio 3)
    ratio = source_rate / target_rate
    if ratio == int(ratio) and ratio > 1:
        ratio_int = int(ratio)
        # Apply simple low-pass anti-aliasing filter (moving average)
        kernel = np.ones(ratio_int, dtype=np.float32) / ratio_int
        filtered = np.convolve(audio, kernel, mode='same')
        return filtered[::ratio_int].astype(np.float32)

    # General case: linear interpolation resampling
    duration = len(audio) / source_rate
    target_length = int(duration * target_rate)
    if target_length == 0:
        return np.array([], dtype=np.float32)

    x_old = np.linspace(0, duration, len(audio), endpoint=False)
    x_new = np.linspace(0, duration, target_length, endpoint=False)
    resampled = np.interp(x_new, x_old, audio)
    return resampled.astype(np.float32)
