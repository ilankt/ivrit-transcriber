import os
import json

def merge_srt_files(tasks, output_path):
    merged_srt_content = []
    time_offset_seconds = 0.0
    subtitle_index = 1

    for task in tasks:
        # Assuming srt_segments are in JSON format: {"start": float, "end": float, "text": str}
        # and need to be converted to proper SRT format
        for segment_str in task.srt_segments:
            try:
                data = json.loads(segment_str)
                start_s = data["start"]
                end_s = data["end"]
                text = data["text"]
            except (json.JSONDecodeError, KeyError, TypeError):
                # Skip malformed segments
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
            merged_srt_content.append("") # Empty line after each subtitle
            subtitle_index += 1
        
        # Update time offset for the next task
        if task.srt_segments:
            try:
                last_segment_data = json.loads(task.srt_segments[-1])
                last_segment_end_s = last_segment_data["end"]
                time_offset_seconds += last_segment_end_s
            except (json.JSONDecodeError, KeyError, TypeError, IndexError):
                # If we can't parse the last segment, don't update offset
                pass

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(merged_srt_content))

def merge_txt_files(tasks, output_path):
    merged_txt_content = []
    for task in tasks:
        merged_txt_content.append(task.text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(merged_txt_content))
