"""Filename helpers shared by file and live transcription flows."""

_SAFE_STEM_CHARS = (" ", "-", "_")


def sanitize_output_stem(value: str, max_length: int = 200) -> str:
    """Return a filesystem-safe filename stem, without an extension."""
    cleaned = "".join(
        char for char in value.strip()
        if char.isalnum() or char in _SAFE_STEM_CHARS
    )
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned[:max_length].rstrip()
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN", "CONOUT"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in "123456789\u00b9\u00b2\u00b3")
    if cleaned.upper() in reserved:
        cleaned = "_" + cleaned
    return cleaned
