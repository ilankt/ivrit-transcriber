"""Reconcile timestamped live words across overlapping audio windows."""
import re


def _normalized(text):
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE).casefold()


class LiveTimeline:
    def __init__(self):
        self.recent = []

    def append(self, words, audio_start, new_start, audio_end):
        candidates = [(max(0.0, audio_start + start), min(audio_end, audio_start + end), text.strip())
                      for start, end, text in words if text.strip() and audio_start + end > new_start]
        candidates = [word for word in candidates if word[1] > word[0]]
        recent = [word for word in self.recent if word[1] > audio_start]
        # Compare only the repeated boundary prefix, with overlapping timestamps.
        for count in range(min(len(recent), len(candidates)), 0, -1):
            if all(_normalized(old[2]) == _normalized(new[2]) and
                   min(old[1], new[1]) > max(old[0], new[0])
                   for old, new in zip(recent[-count:], candidates[:count])):
                candidates = candidates[count:]
                break
        accepted = []
        last_end = self.recent[-1][1] if self.recent else 0.0
        for start, end, text in candidates:
            start = max(start, last_end)
            if end > start:
                accepted.append((start, end, text))
                last_end = end
        self.recent = (recent + accepted)[-40:]
        return accepted


def subtitle_cues(words, max_seconds=5.0, max_chars=84):
    """Keep speech timing while grouping words into readable subtitle cues."""
    cues = []
    for start, end, text in words:
        if cues and start - cues[-1][1] < .8 and end - cues[-1][0] <= max_seconds and len(cues[-1][2]) + len(text) < max_chars:
            previous = cues.pop()
            cues.append((previous[0], end, previous[2] + " " + text))
        else:
            cues.append((start, end, text))
    return cues
