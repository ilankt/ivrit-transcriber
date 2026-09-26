"""
whisper.cpp subprocess runner for AMD (Vulkan) and Apple Silicon (Metal).

Calls the whisper-cli binary as a subprocess and parses its output.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

_POPEN_EXTRA_KWARGS = {}
if sys.platform == 'win32':
    _POPEN_EXTRA_KWARGS['creationflags'] = subprocess.CREATE_NO_WINDOW


def _whispercpp_candidates(base_path: str) -> list[str]:
    exe_name = 'whisper-cli.exe' if sys.platform == 'win32' else 'whisper-cli'

    candidates = [os.path.join(base_path, 'Binaries', exe_name), shutil.which(exe_name)]
    if sys.platform == 'darwin':
        # Finder-launched apps do not inherit the shell's Homebrew PATH.
        for prefix in ('/opt/homebrew', '/usr/local'):
            candidates.append(f'{prefix}/bin/whisper-cli')
            # Also find installations whose executable has not been linked into bin.
            for formula in ('whisper.cpp', 'whisper-cpp'):
                candidates.append(f'{prefix}/opt/{formula}/bin/whisper-cli')
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def get_whispercpp_binary_path(base_path: str) -> str | None:
    """Find the first executable candidate without launching it."""
    candidates = _whispercpp_candidates(base_path)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def resolve_whispercpp_binary(base_path: str) -> tuple[str | None, str | None]:
    """Find a working CLI, retaining startup errors and trying alternative copies."""
    candidates = _whispercpp_candidates(base_path)
    failures = []
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        error = get_whispercpp_binary_error(candidate)
        if error is None:
            return candidate, None
        failures.append(f'{candidate}:\n{error}')
    if failures:
        return None, "whisper-cli was found but could not start:\n\n" + '\n\n'.join(failures)
    return None, "whisper-cli was not found. Searched:\n" + '\n'.join(candidates)


def whispercpp_setup_hint() -> str:
    if sys.platform == 'darwin':
        return (
            "Check whisper-cli --help in Terminal for startup errors.\n"
            "If whisper.cpp is not installed: brew install whisper.cpp\n"
            "The app checks Binaries/whisper-cli, PATH, and the standard Homebrew folders."
        )
    return "Place whisper-cli and its runtime libraries in Binaries/ or add it to PATH."


def get_whispercpp_binary_error(binary_path: str) -> str | None:
    """Return the actual startup failure, or None when the CLI can run."""
    # Homebrew initializes its Metal backend even for --help. A cold start can
    # compile GPU libraries before printing usage, so allow more time on Macs.
    timeout = 60 if sys.platform == 'darwin' else 10
    try:
        result = subprocess.run(
            [binary_path, '--help'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            **_POPEN_EXTRA_KWARGS,
        )
        if result.returncode == 0:
            return None
        output = (result.stderr or result.stdout or b'').decode('utf-8', errors='replace').strip()
        return f"Exited with code {result.returncode}.\n{output[-4000:]}".strip()
    except subprocess.TimeoutExpired as exc:
        output = (exc.stderr or exc.stdout or b'').decode('utf-8', errors='replace').strip()
        return f"Timed out after {timeout} seconds while running --help.\n{output[-4000:]}".strip()
    except OSError as exc:
        return str(exc)


def validate_whispercpp_binary(binary_path: str) -> bool:
    """Check if the whisper-cli binary is functional."""
    return get_whispercpp_binary_error(binary_path) is None


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
    language: str = "he",
    vad_filter: bool = True,
    use_gpu: bool = True,
    progress_callback=None,
    cancel_event=None,
    threads: int = 0,
    require_metal: bool = False,
) -> tuple[str, list[str]]:
    """
    Transcribe an audio chunk using whisper.cpp.

    Args:
        audio_path: Path to the audio file (WAV)
        model_path: Path to the GGML model file
        binary_path: Path to whisper-cli binary
        beam_size: Beam size for decoding
        vad_filter: Whether to use VAD (not directly supported, ignored)
        use_gpu: Whether to use GPU (Vulkan or Metal, depending on the binary)
        progress_callback: Optional callable(int) for progress percentage
        cancel_event: Optional threading.Event checked for cancellation

    Returns:
        (full_text, srt_segments_json_list)
        srt_segments_json_list matches the format used by transcribe_chunk()
    """
    # Create a temp dir for output files
    tmp_dir = tempfile.mkdtemp(prefix='ivrit_wcpp_')
    output_prefix = os.path.join(tmp_dir, 'output')

    args = [
        binary_path,
        '--model', model_path,
        '--language', language,
        '--beam-size', str(beam_size),
        '--output-srt',
        '--output-txt',
        '--output-file', output_prefix,
        '--print-progress',
        '--file', audio_path,
    ]

    if not use_gpu:
        args.append('--no-gpu')
    if threads > 0:
        args.extend(['--threads', str(threads)])

    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,  # Transcript is read from files; avoid a full pipe.
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
                if 'whisper_backend_init_gpu:' in line:
                    logging.info(line.strip())
                    if require_metal and ('no GPU found' in line or 'failed to initialize' in line):
                        process.terminate()
                        process.wait()
                        raise RuntimeError(
                            "Metal GPU initialization failed. " + whispercpp_setup_hint()
                        )
                if progress_callback:
                    match = progress_pattern.search(line)
                    if match:
                        progress_callback(int(match.group(1)))
                if cancel_event and cancel_event.is_set():
                    process.terminate()
                    process.wait()
                    raise InterruptedError("Transcription canceled")

        process.wait()

        stderr_text = ''.join(stderr_lines)

        if process.returncode != 0:
            logging.error(f"whisper-cli failed (exit code {process.returncode}). Full stderr:\n{stderr_text}")
            raise RuntimeError(f"whisper-cli failed (exit code {process.returncode}):\n{stderr_text[:2000]}")

        # Check for errors in stderr even if exit code is 0
        if 'error:' in stderr_text.lower() and 'unknown argument' in stderr_text.lower():
            raise RuntimeError(f"whisper-cli argument error:\n{stderr_text[:500]}")

        # Parse output files
        srt_path = output_prefix + '.srt'
        txt_path = output_prefix + '.txt'
        if not os.path.isfile(txt_path) and not os.path.isfile(srt_path):
            raise RuntimeError(f"whisper-cli produced no transcript files:\n{stderr_text[-2000:]}")

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

        return full_text, srt_segments_json
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
