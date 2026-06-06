"""Filename helpers shared by file and live transcription flows."""

_SAFE_STEM_CHARS = (" ", "-", "_")


def sanitize_output_stem(value: str, max_length: int = 200) -> str:
    """Return a filesystem-safe filename stem, without an extension."""
    cleaned = "".join(
        char for char in value.strip()
        if char.isalnum() or char in _SAFE_STEM_CHARS
    )
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_length]
