"""
Checkpoint management for progressive saving of transcription results.
Allows recovery of partial results if transcription fails or is interrupted.
"""
import os
import json
import shutil
from core.storage import atomic_text_writer


def _format_srt_time(seconds):
    milliseconds = max(0, round(seconds * 1000))
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def _load_segment(segment_str):
    try:
        data = json.loads(segment_str)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def get_checkpoint_dir(output_dir, base_filename):
    """
    Get the checkpoint directory path for a job.

    Args:
        output_dir: Directory where final output files will be saved
        base_filename: Base name of the output file (without extension)

    Returns:
        Path to checkpoint directory
    """
    return os.path.join(output_dir, '.ivrit_checkpoint', base_filename)


def save_chunk_checkpoint(output_dir, base_filename, chunk_index, text, srt_segments, duration, start_offset=None):
    """
    Save a checkpoint file for a completed chunk.

    Args:
        output_dir: Output directory
        base_filename: Base name of the file being transcribed
        chunk_index: Index of the chunk (0-based)
        text: Transcribed text for this chunk
        srt_segments: List of SRT segment JSON strings for this chunk
        duration: Duration of the chunk in seconds

    Returns:
        Path to the saved checkpoint file
    """
    checkpoint_dir = get_checkpoint_dir(output_dir, base_filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_data = {
        'chunk_index': chunk_index,
        'text': text,
        'srt_segments': srt_segments,
        'duration': duration
    }
    if start_offset is not None:
        checkpoint_data['start_offset'] = start_offset

    checkpoint_path = os.path.join(checkpoint_dir, f'chunk_{chunk_index:03d}.json')
    with atomic_text_writer(checkpoint_path) as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    return checkpoint_path


def load_all_checkpoints(output_dir, base_filename):
    """
    Load all checkpoint files for a job, sorted by chunk index.

    Args:
        output_dir: Output directory
        base_filename: Base name of the file being transcribed

    Returns:
        List of checkpoint data dictionaries, sorted by chunk_index
    """
    checkpoint_dir = get_checkpoint_dir(output_dir, base_filename)
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('chunk_') and filename.endswith('.json'):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                checkpoints.append(checkpoint_data)

    # Sort by chunk index
    checkpoints.sort(key=lambda x: x['chunk_index'])
    return checkpoints


def cleanup_checkpoints(output_dir, base_filename):
    """
    Remove checkpoint directory after successful completion.

    Args:
        output_dir: Output directory
        base_filename: Base name of the file being transcribed
    """
    checkpoint_dir = get_checkpoint_dir(output_dir, base_filename)
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

        # Also remove parent .ivrit_checkpoint dir if empty
        parent_checkpoint_dir = os.path.join(output_dir, '.ivrit_checkpoint')
        if os.path.exists(parent_checkpoint_dir) and not os.listdir(parent_checkpoint_dir):
            os.rmdir(parent_checkpoint_dir)


def merge_checkpoints_to_files(output_dir, base_filename, checkpoints=None, output_format="srt"):
    """
    Merge checkpoint data into final output files.

    Args:
        output_dir: Output directory
        base_filename: Base name of the output files (without extension)
        checkpoints: Optional list of checkpoint data. If None, loads from disk.
        output_format: "srt", "txt", or "both"

    Returns:
        Tuple of (txt_path, srt_path) for the created files (None for skipped formats)
    """
    if checkpoints is None:
        checkpoints = load_all_checkpoints(output_dir, base_filename)

    if not checkpoints:
        return None, None

    merged_txt_path = None
    merged_srt_path = None

    # Merge text files
    if output_format in ("txt", "both"):
        merged_txt_path = os.path.join(output_dir, f"{base_filename}.txt")
        merged_text = [checkpoint['text'] for checkpoint in checkpoints]

        with atomic_text_writer(merged_txt_path) as f:
            f.write("\n".join(merged_text))

    # Merge SRT files
    if output_format in ("srt", "both"):
        merged_srt_path = os.path.join(output_dir, f"{base_filename}.srt")
        merged_srt_content = []
        time_offset_seconds = 0.0
        subtitle_index = 1

        for checkpoint in checkpoints:
            srt_segments = checkpoint['srt_segments']
            time_offset_seconds = checkpoint.get('start_offset', time_offset_seconds)

            for segment_str in srt_segments:
                data = _load_segment(segment_str)
                if not data:
                    continue

                try:
                    start_offset_s = data["start"] + time_offset_seconds
                    end_offset_s = data["end"] + time_offset_seconds
                    text = data["text"]
                    if data.get("speaker"):
                        text = f"{data['speaker']}: {text.strip()}"
                except (KeyError, TypeError):
                    continue

                merged_srt_content.append(str(subtitle_index))
                merged_srt_content.append(
                    f"{_format_srt_time(start_offset_s)} --> {_format_srt_time(end_offset_s)}"
                )
                merged_srt_content.append(text)
                merged_srt_content.append("")  # Empty line after each subtitle
                subtitle_index += 1

            # Silence belongs to the timeline too. New checkpoints also carry
            # absolute offsets so a skipped/failed chunk does not shift later text.
            if checkpoint.get('duration', 0) > 0:
                time_offset_seconds += checkpoint['duration']
            elif srt_segments:
                last_segment = _load_segment(srt_segments[-1])
                if last_segment and "end" in last_segment:
                    time_offset_seconds += last_segment["end"]
                else:
                    time_offset_seconds += checkpoint.get('duration', 60)

        with atomic_text_writer(merged_srt_path) as f:
            f.write("\n".join(merged_srt_content))

    return merged_txt_path, merged_srt_path
