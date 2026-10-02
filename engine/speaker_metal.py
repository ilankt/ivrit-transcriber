"""Apple Metal acceleration for the expensive WeSpeaker frame encoder.

Keep filterbank FFT and weighted pooling on CPU, using the same split validated
by the DirectML adapter. This avoids requiring MPS support for audio FFT/vmap
or the segmentation model's recurrent layers. Native Mac validation is still
required; unsupported operations recover on CPU at the batch boundary.
"""
import copy
import logging

import torch


class MetalSpeakerModel(torch.nn.Module):
    def __init__(self, model, encoder, device):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.device = device
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
            frames = self.encoder.forward_frames(features.to(self.device)).cpu()
            if not torch.isfinite(frames).all():
                raise RuntimeError("Metal returned non-finite speaker features")
        except Exception:
            logging.exception("Metal speaker encoding failed; continuing on CPU")
            self.accelerated = False
            self.encoder = None
            return self.model.resnet(features, weights=weights)[1]
        return self.model.resnet.forward_embedding(frames, weights=weights)[1]


def enable_metal(pipeline):
    embedding = pipeline._embedding
    _ = embedding.min_num_samples
    # Retain a CPU copy for the small pooling layer and recovery on failure.
    model = embedding.model_
    device = torch.device("mps")
    encoder = copy.deepcopy(model.resnet).eval().to(device)
    adapter = MetalSpeakerModel(model, encoder, device).eval()
    embedding.model_ = adapter
    return adapter
