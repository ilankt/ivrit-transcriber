import ffmpeg
import json
import os
import subprocess
import sys

# On Windows, prevent console windows from flashing when running FFmpeg/FFprobe
_POPEN_EXTRA_KWARGS = {}
if sys.platform == 'win32':
    _POPEN_EXTRA_KWARGS['creationflags'] = subprocess.CREATE_NO_WINDOW


def probe_media(path):
    try:
        args = ['ffprobe', '-show_format', '-show_streams', '-of', 'json', path]
        p = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            **_POPEN_EXTRA_KWARGS
        )
        out, err = p.communicate()
        if p.returncode != 0:
            raise ffmpeg.Error('ffprobe', out, err)
        probe = json.loads(out.decode('utf-8'))
        duration = float(probe['format']['duration'])
        is_video = any(s['codec_type'] == 'video' for s in probe['streams'])
        return duration, is_video
    except ffmpeg.Error as e:
        return None, e.stderr.decode('utf-8') if e.stderr else "Unknown FFmpeg error"


def _run_ffmpeg(stream, overwrite_output=True):
    """Run an ffmpeg stream graph with CREATE_NO_WINDOW on Windows."""
    args = ffmpeg.compile(stream, overwrite_output=overwrite_output)
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
        stream = (ffmpeg
            .input(input_path)
            .output(out_wav_path, acodec='pcm_s16le', ar=sample_rate, ac=1 if mono else 2))
        _run_ffmpeg(stream)
        return None
    except ffmpeg.Error as e:
        return e.stderr.decode('utf-8') if e.stderr else "Unknown FFmpeg error"


def split_audio(in_wav_path, chunk_minutes, out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)
        stream = (ffmpeg
            .input(in_wav_path)
            .output(os.path.join(out_dir, 'basename__part-%03d.wav'),
                    f='segment',
                    segment_time=chunk_minutes * 60,
                    c='copy'))
        _run_ffmpeg(stream)

        chunk_paths = sorted([
            os.path.join(out_dir, f)
            for f in os.listdir(out_dir)
            if f.startswith('basename__part-') and f.endswith('.wav')
        ])
        return chunk_paths, None
    except ffmpeg.Error as e:
        return None, e.stderr.decode('utf-8') if e.stderr else "Unknown FFmpeg error"
