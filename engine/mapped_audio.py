"""Disk-backed PCM for speaker inference without concatenating hours of RAM."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import wave

import numpy as np


@contextmanager
def mapped_audio(chunk_paths, checkpoint=lambda: None):
    frames = 0
    for path in chunk_paths:
        checkpoint()
        with wave.open(str(path), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                raise ValueError("Speaker detection requires 16 kHz mono PCM audio.")
            frames += audio.getnframes()
    if not frames:
        raise ValueError("No audio samples are available for speaker detection.")
    with tempfile.TemporaryDirectory(prefix="ivrit_speakers_") as directory:
        required = frames * 4
        if shutil.disk_usage(directory).free < required + 64 * 1024 * 1024:
            raise OSError(f"Speaker detection needs {required / 1024 ** 2:.0f} MB of temporary disk space.")
        mapped = np.memmap(Path(directory) / "audio.f32", mode="w+", dtype=np.float32, shape=(frames,))
        try:
            offset = 0
            for path in chunk_paths:
                with wave.open(str(path), "rb") as audio:
                    while True:
                        checkpoint()
                        block = audio.readframes(16000 * 10)
                        if not block:
                            break
                        pcm = np.frombuffer(block, dtype="<i2")
                        mapped[offset:offset + len(pcm)] = pcm.astype(np.float32) / 32768.0
                        offset += len(pcm)
            if offset != frames:
                raise ValueError("Audio ended before its declared duration.")
            mapped.flush()
            yield mapped
        finally:
            mapped._mmap.close()
