"""
whisper.cpp subprocess runner for AMD GPU (Vulkan) transcription.

Calls the whisper-cli binary as a subprocess and parses its output.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

_POPEN_EXTRA_KWARGS = {}
if sys.platform == 'win32':
    _POPEN_EXTRA_KWARGS['creationflags'] = subprocess.CREATE_NO_WINDOW


def get_whispercpp_binary_path(base_path: str) -> str | None:
    """Find the whisper-cli binary. Returns path or None."""
    exe_name = 'whisper-cli.exe' if sys.platform == 'win32' else 'whisper-cli'

    # Check bundled location first
    bundled = os.path.join(base_path, 'Binaries', exe_name)
    if os.path.isfile(bundled):
        return bundled

    # Check system PATH
    try:
        p = subprocess.Popen(
            ['whisper-cli', '--help'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            **_POPEN_EXTRA_KWARGS
        )
        p.communicate(timeout=5)
        if p.returncode is not None:
            return 'whisper-cli'  # Available on PATH
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return None


def validate_whispercpp_binary(binary_path: str) -> bool:
    """Check if the whisper-cli binary is functional."""
    try:
        p = subprocess.Popen(
            [binary_path, '--help'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            **_POPEN_EXTRA_KWARGS
        )
        p.communicate(timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def validate_ggml_model(model_path: str) -> bool:
    """Check if a GGML model file exists and has reasonable size (>10MB)."""
    if not os.path.isfile(model_path):
        return False
    return os.path.getsize(model_path) > 10 * 1024 * 1024


def parse_srt_content(srt_text: str) -> list[dict]:
    """
    Parse SRT formatted text into segment dicts.

    Returns:
        list of {"start": float, "end": float, "text": str}
    """
    segments = []
    blocks = re.split(r'\n\n+', srt_text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        # Line 0: sequence number
        # Line 1: timestamp range
        # Lines 2+: text
        timestamp_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
            lines[1]
        )
        if not timestamp_match:
            continue

        g = timestamp_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
        text = ' '.join(lines[2:]).strip()

        segments.append({"start": start, "end": end, "text": text})

    return segments


def transcribe_chunk_whispercpp(
    audio_path: str,
    model_path: str,
    binary_path: str,
    beam_size: int = 1,
    vad_filter: bool = True,
    use_gpu: bool = True,
    progress_callback=None,
) -> tuple[str, list[str], subprocess.Popen | None]:
    """
    Transcribe an audio chunk using whisper.cpp.

    Args:
        audio_path: Path to the audio file (WAV)
        model_path: Path to the GGML model file
        binary_path: Path to whisper-cli binary
        beam_size: Beam size for decoding
        vad_filter: Whether to use VAD (not directly supported, ignored)
        use_gpu: Whether to use GPU (Vulkan)
        progress_callback: Optional callable(int) for progress percentage

    Returns:
        (full_text, srt_segments_json_list, process_handle)
        srt_segments_json_list matches the format used by transcribe_chunk()
    """
    # Create a temp dir for output files
    tmp_dir = tempfile.mkdtemp(prefix='ivrit_wcpp_')
    output_prefix = os.path.join(tmp_dir, 'output')

    args = [
        binary_path,
        '--model', model_path,
        '--language', 'he',
        '--beam-size', str(beam_size),
        '--output-srt',
        '--output-txt',
        '--output-file', output_prefix,
        '--print-progress',
        '--file', audio_path,
    ]

    if not use_gpu:
        args.append('--no-gpu')

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_POPEN_EXTRA_KWARGS
    )

    # Read stderr for progress updates
    progress_pattern = re.compile(r'progress\s*=\s*(\d+)%')
    stderr_lines = []

    if process.stderr:
        for raw_line in iter(process.stderr.readline, b''):
            line = raw_line.decode('utf-8', errors='replace')
            stderr_lines.append(line)
            if progress_callback:
                match = progress_pattern.search(line)
                if match:
                    progress_callback(int(match.group(1)))

    process.wait()

    stderr_text = ''.join(stderr_lines)

    if process.returncode != 0:
        raise RuntimeError(f"whisper-cli failed (exit code {process.returncode}):\n{stderr_text[:500]}")

    # Check for errors in stderr even if exit code is 0
    if 'error:' in stderr_text.lower() and 'unknown argument' in stderr_text.lower():
        raise RuntimeError(f"whisper-cli argument error:\n{stderr_text[:500]}")

    # Parse output files
    srt_path = output_prefix + '.srt'
    txt_path = output_prefix + '.txt'

    full_text = ''
    srt_segments_json = []

    if os.path.isfile(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            full_text = f.read().strip()

    if os.path.isfile(srt_path):
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        segments = parse_srt_content(srt_content)
        # Convert to JSON format matching transcribe_chunk() output
        for seg in segments:
            srt_segments_json.append(json.dumps({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }))
        # If txt was empty, build it from SRT segments
        if not full_text:
            full_text = ' '.join(seg["text"] for seg in segments)

    # Clean up temp dir
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    return full_text, srt_segments_json
