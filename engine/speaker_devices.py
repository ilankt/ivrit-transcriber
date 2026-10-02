"""Speaker-device policy without importing optional inference libraries."""


def select_speaker_backend(requested, torch, platform):
    if requested == "cpu":
        return "cpu"
    if platform != "darwin" and requested in ("auto", "nvidia") and torch.cuda.is_available():
        return "cuda"
    if platform == "darwin" and requested in ("auto", "metal"):
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    if platform == "win32" and requested in ("auto", "amd"):
        return "directml"
    return "cpu"
