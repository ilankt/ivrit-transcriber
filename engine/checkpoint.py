"""
Checkpoint management for progressive saving of transcription results.
Allows recovery of partial results if transcription fails or is interrupted.
"""
import os
import json
import shutil


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


def save_chunk_checkpoint(output_dir, base_filename, chunk_index, text, srt_segments, duration):
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

    checkpoint_path = os.path.join(checkpoint_dir, f'chunk_{chunk_index:03d}.json')
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
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
        merged_text = []
        for checkpoint in checkpoints:
            merged_text.append(checkpoint['text'])

        with open(merged_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(merged_text))

    # Merge SRT files
    if output_format in ("srt", "both"):
        merged_srt_path = os.path.join(output_dir, f"{base_filename}.srt")
        merged_srt_content = []
        time_offset_seconds = 0.0
        subtitle_index = 1

        for checkpoint in checkpoints:
            srt_segments = checkpoint['srt_segments']

            for segment_str in srt_segments:
                try:
                    data = json.loads(segment_str)
                    start_s = data["start"]
                    end_s = data["end"]
                    text = data["text"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

                # Apply offset
                start_offset_s = start_s + time_offset_seconds
                end_offset_s = end_s + time_offset_seconds

                # Format time to HH:MM:SS,ms
                start_h = int(start_offset_s // 3600)
                start_m = int((start_offset_s % 3600) // 60)
                start_s_frac = start_offset_s % 60

                end_h = int(end_offset_s // 3600)
                end_m = int((end_offset_s % 3600) // 60)
                end_s_frac = end_offset_s % 60

                merged_srt_content.append(str(subtitle_index))
                merged_srt_content.append(f"{start_h:02}:{start_m:02}:{start_s_frac:06.3f}".replace('.', ',') + " --> " + f"{end_h:02}:{end_m:02}:{end_s_frac:06.3f}".replace('.', ','))
                merged_srt_content.append(text)
                merged_srt_content.append("")  # Empty line after each subtitle
                subtitle_index += 1

            # Update time offset for the next checkpoint based on the last segment
            if srt_segments:
                try:
                    last_segment_data = json.loads(srt_segments[-1])
                    last_segment_end_s = last_segment_data["end"]
                    time_offset_seconds += last_segment_end_s
                except (json.JSONDecodeError, KeyError, TypeError, IndexError):
                    # If we can't parse the last segment, use duration as fallback
                    time_offset_seconds += checkpoint.get('duration', 60)

        with open(merged_srt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(merged_srt_content))

    return merged_txt_path, merged_srt_path
