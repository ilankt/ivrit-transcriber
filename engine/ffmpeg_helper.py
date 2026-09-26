import ffmpeg
import json
import math
import shutil
import os
import subprocess
import sys

# On Windows, prevent console windows from flashing when running FFmpeg/FFprobe
_POPEN_EXTRA_KWARGS = {}
if sys.platform == 'win32':
    _POPEN_EXTRA_KWARGS['creationflags'] = subprocess.CREATE_NO_WINDOW


def _find_executable(name):
    resolved = shutil.which(name)
    if resolved:
        return resolved
    if sys.platform == 'darwin':
        for directory in ('/opt/homebrew/bin', '/usr/local/bin'):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return name


def _probe(path):
    args = [_find_executable('ffprobe'), '-v', 'error', '-show_format', '-show_streams', '-of', 'json', path]
    p = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        **_POPEN_EXTRA_KWARGS
    )
    out, err = p.communicate()
    if p.returncode != 0:
        raise ffmpeg.Error('ffprobe', out, err)
    return json.loads(out.decode('utf-8'))


def _select_audio_stream(probe):
    """Use the preferred playback track, or the first audio track if none is marked."""
    audio_streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
    if not audio_streams:
        raise ValueError('This file does not contain an audio track.')
    return next(
        (s for s in audio_streams if s.get('disposition', {}).get('default') == 1),
        audio_streams[0],
    )


def probe_media(path):
    try:
        probe = _probe(path)
        _select_audio_stream(probe)
        duration = float(probe.get('format', {}).get('duration', 0))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Could not determine a positive media duration.')
        is_video = any(
            s.get('codec_type') == 'video' and not s.get('disposition', {}).get('attached_pic')
            for s in probe['streams']
        )
        return duration, is_video
    except ffmpeg.Error as e:
        return None, e.stderr.decode('utf-8', errors='replace') if e.stderr else "Unknown FFmpeg error"
    except (OSError, ValueError) as e:
        return None, str(e)


def _run_ffmpeg(stream, overwrite_output=True):
    """Run an ffmpeg stream graph with CREATE_NO_WINDOW on Windows."""
    args = ffmpeg.compile(stream, cmd=_find_executable('ffmpeg'), overwrite_output=overwrite_output)
    p = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        **_POPEN_EXTRA_KWARGS
    )
    out, err = p.communicate()
    if p.returncode != 0:
        raise ffmpeg.Error('ffmpeg', out, err)
    return out, err


def extract_audio(input_path, out_wav_path, sample_rate=16000, mono=True):
    try:
        audio_stream = _select_audio_stream(_probe(input_path))
        stream = (ffmpeg
            .input(input_path)[str(audio_stream['index'])]
            .output(out_wav_path, acodec='pcm_s16le', ar=sample_rate, ac=1 if mono else 2))
        _run_ffmpeg(stream)
        return None
    except ffmpeg.Error as e:
        return e.stderr.decode('utf-8', errors='replace') if e.stderr else "Unknown FFmpeg error"
    except (OSError, ValueError) as e:
        return str(e)


def split_audio(in_wav_path, chunk_minutes, out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)
        stream = (ffmpeg
            .input(in_wav_path)['a:0']
            .output(os.path.join(out_dir, 'basename__part-%03d.wav'),
                    f='segment',
                    segment_time=chunk_minutes * 60,
                    c='copy', reset_timestamps=1))
        _run_ffmpeg(stream)

        chunk_paths = sorted([
            os.path.join(out_dir, f)
            for f in os.listdir(out_dir)
            if f.startswith('basename__part-') and f.endswith('.wav')
        ])
        return chunk_paths, None
    except ffmpeg.Error as e:
        return None, e.stderr.decode('utf-8') if e.stderr else "Unknown FFmpeg error"
