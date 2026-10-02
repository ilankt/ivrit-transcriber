from faster_whisper import WhisperModel
import json

def transcribe_chunk(audio_path: str, model: WhisperModel, language: str, beam_size: int, vad_filter: bool, cancel_event=None, word_timestamps=False):
    segments, _info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
    )

    all_text = []
    srt_segments = []
    for segment in segments:
        if cancel_event and cancel_event.is_set():
            break
        all_text.append(segment.text)
        # Use JSON format to avoid issues with commas in transcribed text
        data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        }
        if word_timestamps and segment.words:
            data["words"] = [{"start": w.start, "end": w.end, "word": w.word}
                             for w in segment.words]
        srt_segments.append(json.dumps(data))

    return " ".join(all_text), srt_segments
