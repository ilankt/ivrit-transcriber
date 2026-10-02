"""GPU speech filters with the original CPU recurrent speaker tracker.

DirectML's full LSTM export is slower than native CPU recurrence. Offloading
SincNet alone also provides a shared Metal path without requiring GPU LSTMs.
"""
import copy
import hashlib
import logging
from pathlib import Path
import tempfile

import numpy as np
import torch


class StableInstanceNorm(torch.nn.Module):
    """Explicit centered variance avoids GPU cancellation on silent channels."""
    def __init__(self, norm):
        super().__init__()
        self.register_buffer("weight", norm.weight.detach().clone()[None, :, None])
        self.register_buffer("bias", norm.bias.detach().clone()[None, :, None])
        self.eps = norm.eps

    def forward(self, values):
        centered = values - values.mean(dim=-1, keepdim=True)
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(variance + self.eps) * self.weight + self.bias


def _exportable_frontend(model):
    frontend = copy.deepcopy(model).eval()
    # Keep this small, amplitude-sensitive normalization identical on CPU.
    frontend.wav_norm1d = torch.nn.Identity()
    for index, norm in enumerate(frontend.norm1d):
        frontend.norm1d[index] = StableInstanceNorm(norm)
    return frontend


class AcceleratedSpeechFilters(torch.nn.Module):
    def __init__(self, original, session=None, encoder=None, device=None, normalize=None):
        super().__init__()
        self.original = original
        self.session = session
        self.encoder = encoder
        self.device = device
        self.normalize = normalize
        self.accelerated = True

    def num_frames(self, *args, **kwargs):
        return self.original.num_frames(*args, **kwargs)

    def receptive_field_size(self, *args, **kwargs):
        return self.original.receptive_field_size(*args, **kwargs)

    def receptive_field_center(self, *args, **kwargs):
        return self.original.receptive_field_center(*args, **kwargs)

    def forward(self, waveforms):
        if not self.accelerated:
            return self.original(waveforms)
        # Near-silence amplifies GPU rounding through successive normalizers.
        # Preserve the CPU model's decisions on these numerically sensitive rows.
        quiet = waveforms.abs().amax(dim=(1, 2)) < 1e-4
        if quiet.all():
            return self.original(waveforms)
        try:
            normalized = self.normalize(waveforms) if self.normalize is not None else waveforms
            if self.session is not None:
                features = self.session.run(None, {"audio": normalized.cpu().numpy()})[0]
                if not np.isfinite(features).all():
                    raise RuntimeError("GPU returned non-finite speech features")
                features = torch.from_numpy(features)
            else:
                features = self.encoder(normalized.to(self.device)).cpu()
            if not torch.isfinite(features).all():
                raise RuntimeError("GPU returned non-finite speech features")
            if quiet.any():
                features[quiet] = self.original(waveforms[quiet])
            return features
        except Exception:
            logging.exception("GPU speech filters failed; continuing detection on CPU")
            self.accelerated = False
            self.session = self.encoder = None
            return self.original(waveforms)


def enable_segmentation_gpu(pipeline, directory, backend, profile_prefix=None):
    model = pipeline._segmentation.model
    original = model.sincnet
    frontend = _exportable_frontend(original)
    if backend == "mps":
        device = torch.device("mps")
        adapter = AcceleratedSpeechFilters(original, encoder=frontend.to(device), device=device,
                                           normalize=original.wav_norm1d).eval()
    elif backend == "directml":
        import onnxruntime as ort
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DirectML is unavailable")
        root = Path(directory)
        with (root / "segmentation.bin").open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()[:16]
        destination = root / f"segmentation-filters-v2-{digest}.onnx"
        if not destination.is_file():
            with tempfile.NamedTemporaryFile(dir=root, suffix=".onnx", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                samples = int(model.specifications.duration * model.audio.sample_rate)
                with torch.inference_mode():
                    torch.onnx.export(
                        frontend, (torch.zeros(1, 1, samples),), str(temporary_path),
                        input_names=["audio"], output_names=["features"],
                        dynamic_axes={"audio": {0: "batch"}, "features": {0: "batch"}},
                        opset_version=17, dynamo=False)
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if profile_prefix:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        session = ort.InferenceSession(str(destination), sess_options=options,
                                      providers=[("DmlExecutionProvider", {"device_id": 0})])
        if "DmlExecutionProvider" not in session.get_providers():
            raise RuntimeError("DirectML could not initialize speech filters")
        session.disable_fallback()
        adapter = AcceleratedSpeechFilters(original, session=session, normalize=original.wav_norm1d).eval()
    else:
        raise ValueError(f"Unsupported speech filter backend: {backend}")
    model.sincnet = adapter
    return adapter
