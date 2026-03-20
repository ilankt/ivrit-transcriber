from faster_whisper import WhisperModel
import os
import json

def transcribe_chunk(audio_path: str, model: WhisperModel, language: str, beam_size: int, vad_filter: bool):
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter
    )

    all_text = []
    srt_segments = []
    for segment in segments:
        all_text.append(segment.text)
        # Use JSON format to avoid issues with commas in transcribed text
        srt_segments.append(json.dumps({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        }))

    return " ".join(all_text), srt_segments
