"""Accelerate WeSpeaker's convolutional encoder on Windows DirectML.

Feature extraction and weighted pooling stay in the original Pyannote model.
Only the expensive ResNet frame encoder is exported, preserving overlap masks
and clustering behavior. Imported lazily: CPU/CUDA installs need no ONNX tools.
"""
import hashlib
import logging
from pathlib import Path
import tempfile
import warnings

import numpy as np
import torch


class _FrameEncoder(torch.nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet

    def forward(self, features):
        return self.resnet.forward_frames(features)


def _export_encoder(model, directory):
    # Invalidate the derived model whenever the source weights change.
    root = Path(directory)
    with (root / "embedding.bin").open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()[:16]
    destination = root / f"embedding-directml-v1-{digest}.onnx"
    if destination.is_file():
        return destination
    with tempfile.NamedTemporaryFile(dir=root, suffix=".onnx", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                _FrameEncoder(model.resnet).eval(), (torch.zeros(1, 998, 80),),
                str(temporary_path), input_names=["features"], output_names=["frames"],
                dynamic_axes={"features": {0: "batch", 1: "time"},
                              "frames": {0: "batch", 3: "frames"}},
                opset_version=17, dynamo=False,
            )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


class DirectMLSpeakerModel(torch.nn.Module):
    """Retain the original model for exact CPU pooling and recovery on failure."""

    def __init__(self, model, session):
        super().__init__()
        self.model = model
        self.session = session
        self.accelerated = True

    @property
    def audio(self):
        return self.model.audio

    @property
    def dimension(self):
        return self.model.dimension

    def forward(self, waveforms, weights=None):
        if not self.accelerated:
            return self.model(waveforms, weights=weights)
        features = self.model.compute_fbank(waveforms)
        try:
            frames = self.session.run(None, {"features": features.cpu().numpy()})[0]
            if not np.isfinite(frames).all():
                raise RuntimeError("DirectML returned non-finite speaker features")
        except Exception:
            # Retry just this batch, rather than throwing away the whole job.
            logging.exception("DirectML speaker encoding failed; continuing on CPU")
            self.accelerated = False
            self.session = None
            return self.model.resnet(features, weights=weights)[1]
        return self.model.resnet.forward_embedding(torch.from_numpy(frames), weights=weights)[1]


def enable_directml(pipeline, directory, profile_prefix=None):
    """Install the adapter only after export/session setup succeeds.

    Returns the adapter so progress can report a subsequent CPU fallback.
    The default DirectX adapter is used (the tested PC has one RX 7600M XT).
    """
    import onnxruntime as ort

    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("Install onnxruntime-directml to enable AMD speaker acceleration")
    embedding = pipeline._embedding
    model = embedding.model_
    # This tiny-input probe belongs to the original Pyannote implementation.
    # Cache it before replacing the encoder to avoid compiling many GPU shapes.
    _ = embedding.min_num_samples
    exported = _export_encoder(model, directory)
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    session = ort.InferenceSession(
        str(exported), sess_options=options,
        providers=[("DmlExecutionProvider", {"device_id": 0})],
    )
    if "DmlExecutionProvider" not in session.get_providers():
        raise RuntimeError("DirectML could not initialize the GPU")
    session.disable_fallback()
    adapter = DirectMLSpeakerModel(model, session).eval()
    embedding.model_ = adapter
    return adapter
