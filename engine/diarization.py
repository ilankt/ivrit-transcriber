"""Optional, local speaker diarization and timestamp-based transcript labeling."""
import importlib.metadata
import json
import os
from pathlib import Path
import wave

MODEL_REPO = "ivrit-ai/pyannote-speaker-diarization-3.1"
MODEL_FOLDER = "ivrit-pyannote-speaker-diarization-3.1"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}"
_READY_FILE = ".ivrit-ready.json"
_ASSETS = (
    (MODEL_REPO, "config.yaml", "config.yaml"),
    ("ivrit-ai/pyannote-segmentation-3.0", "pytorch_model.bin", "segmentation.bin"),
    ("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin", "embedding.bin"),
)


def model_path(base_path, models_folder=None):
    return os.path.join(models_folder or os.path.join(base_path, "Models"), MODEL_FOLDER)


def dependency_error():
    try:
        version = importlib.metadata.version("pyannote.audio")
        if version == "3.3.2":
            return None
    except (importlib.metadata.PackageNotFoundError, ValueError):
        pass
    return (
        "This Python environment is missing the speaker dependencies.\n"
        "Use Run Ivrit Transcriber.cmd in the project folder, or install into this environment:\n"
        "python -m pip install -r requirements-speakers.txt\n"
        "Then restart the app. A packaged app needs a build with speaker support."
    )


def model_available(path):
    """Only complete downloads are ready; also detect removed/truncated assets."""
    try:
        root = Path(path)
        manifest = json.loads((root / _READY_FILE).read_text(encoding="utf-8"))
        files = manifest["files"]
        return isinstance(files, dict) and bool(files) and "config.yaml" in files and all(
            (root / name).is_file() and (root / name).stat().st_size == size
            for name, size in files.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def download_model(path, token=None, progress=None, cancel_check=None):
    """Download all pipeline assets together; never store the access token."""
    from huggingface_hub import HfApi, hf_hub_download

    def check_cancel():
        if cancel_check and cancel_check():
            raise InterruptedError("Speaker model download canceled")

    check_cancel()
    # These are the public assets published/used by ivrit.ai's pipeline.
    # Do not let a stale saved token prevent downloading public models.
    access_token = token or False
    api = HfApi(token=access_token)
    manifest = {}
    revisions = {}
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    import shutil
    for index, (repo, name, local_name) in enumerate(_ASSETS):
        check_cancel()
        revision = api.model_info(repo).sha
        revisions[repo] = revision
        downloaded = hf_hub_download(
            repo, name, revision=revision, token=access_token,
        )
        destination = root / local_name
        temporary = root / (local_name + ".tmp")
        shutil.copyfile(downloaded, temporary)
        temporary.replace(destination)
        manifest[local_name] = destination.stat().st_size
        if progress:
            progress(int(100 * (index + 1) / len(_ASSETS)))
    check_cancel()
    if "config.yaml" not in manifest:
        raise RuntimeError("The speaker model download has no pipeline configuration.")
    temporary = root / (_READY_FILE + ".tmp")
    temporary.write_text(json.dumps({"revisions": revisions, "files": manifest}), encoding="utf-8")
    temporary.replace(root / _READY_FILE)


def _load_pipeline(path):
    """Load only local assets, without ivrit's buggy label-assignment wrapper."""
    import torch
    import yaml
    from pyannote.audio import Model
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from pyannote.audio.pipelines import SpeakerDiarization

    # Same narrow allowlist as ivrit.ai's RunPod image, rather than unrestricted loading.
    torch.serialization.add_safe_globals([Problem, Resolution, Specifications, torch.torch_version.TorchVersion])
    root = Path(path)
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    params = dict(config["pipeline"]["params"])
    params["segmentation"] = Model.from_pretrained(str(root / "segmentation.bin"))
    params["embedding"] = Model.from_pretrained(str(root / "embedding.bin"))
    pipeline = SpeakerDiarization(**params)
    pipeline.instantiate(config["params"])
    return pipeline


def diarize_chunks(chunk_paths, path, device="auto", num_speakers=0,
                   cancel_event=None, status_callback=None, pause_check=None):
    """Cluster the whole recording so labels are shared across every chunk.

    Input is the same decoded 16 kHz mono PCM used by transcription. Passing
    waveform data avoids a second decoder and Windows TorchCodec DLL issues.
    Cancellation waits for the next pipeline hook, so no abandoned inference
    thread keeps using audio or GPU resources after the worker finishes.
    """
    def checkpoint():
        import time
        while pause_check and pause_check():
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Speaker detection canceled")
            time.sleep(0.1)
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Speaker detection canceled")

    checkpoint()
    error = dependency_error()
    if error:
        raise RuntimeError(error)
    if not model_available(path):
        raise RuntimeError("Speaker model is missing or incomplete. Use Settings > Set Up Speakers.")

    if status_callback:
        status_callback("Loading speaker model...")
    # This feature is local-only, including pyannote's optional usage telemetry.
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    import numpy as np
    import torch

    checkpoint()
    pipeline = _load_pipeline(path)
    if pipeline is None:
        raise RuntimeError("Could not load the speaker model. Download it again in Settings.")
    use_cuda = device in ("auto", "nvidia") and torch.cuda.is_available()
    try:
        if use_cuda:
            pipeline.to(torch.device("cuda"))
        checkpoint()
        pieces = []
        for chunk_path in chunk_paths:
            checkpoint()
            with wave.open(chunk_path, "rb") as audio:
                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                    raise ValueError("Speaker detection requires 16 kHz mono PCM audio.")
                pieces.append(np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2"))
        waveform = np.concatenate(pieces).astype(np.float32) / 32768.0
        del pieces
        last_status = [None]

        def hook(step_name, step_artifact, file=None, total=None, completed=None):
            checkpoint()
            progress = f" ({min(completed, total)}/{total})" if total and completed is not None else ""
            message = f"Detecting speakers: {step_name}{progress}"
            if status_callback and message != last_status[0]:
                status_callback(message)
                last_status[0] = message

        kwargs = {"num_speakers": num_speakers} if num_speakers > 0 else {}
        with torch.inference_mode():
            result = pipeline({"waveform": torch.from_numpy(waveform).unsqueeze(0),
                               "sample_rate": 16000}, hook=hook, **kwargs)
        checkpoint()
        annotation = result
        turns = sorted((float(turn.start), float(turn.end), speaker)
                       for turn, _, speaker in annotation.itertracks(yield_label=True))
        labels = {}
        normalized = []
        for start, end, speaker in turns:
            if end <= start:
                continue
            labels.setdefault(speaker, f"Speaker {len(labels) + 1}")
            normalized.append((start, end, labels[speaker]))
        return normalized
    finally:
        del pipeline
        if use_cuda:
            torch.cuda.empty_cache()


def _speaker_for(start, end, turns):
    if end == start:
        return next((speaker for turn_start, turn_end, speaker in turns
                     if turn_start <= start < turn_end), "Speaker unknown")
    overlaps = {}
    for turn_start, turn_end, speaker in turns:
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap:
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
    return max(overlaps, key=overlaps.get) if overlaps else "Speaker unknown"


def label_segments(segments, turns, offset=0.0):
    """Label words when available, otherwise use the segment's dominant speaker."""
    labeled = []
    for encoded in segments:
        segment = json.loads(encoded)
        candidates = [turn for turn in turns
                      if turn[0] <= offset + segment["end"]
                      and turn[1] >= offset + segment["start"]]
        words = segment.get("words")
        # Retain exact segment text if word timing did not cover all of it.
        if not words or "".join(w["word"] for w in words).strip() != segment["text"].strip():
            segment["speaker"] = _speaker_for(
                offset + segment["start"], offset + segment["end"], candidates)
            labeled.append(segment)
            continue
        current = None
        for word in words:
            speaker = _speaker_for(offset + word["start"], offset + word["end"], candidates)
            if current is not None and current["speaker"] == speaker:
                current["end"] = word["end"]
                current["text"] += word["word"]
            else:
                current = {"start": word["start"], "end": word["end"],
                           "text": word["word"], "speaker": speaker}
                labeled.append(current)
    text = "\n".join(f"{s['speaker']}: {s['text'].strip()}" for s in labeled)
    return text, [json.dumps(s, ensure_ascii=False) for s in labeled]
