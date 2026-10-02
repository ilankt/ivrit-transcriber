"""Recovery behavior without requiring a GPU on the test runner."""
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from engine.speaker_directml import DirectMLSpeakerModel
from engine.speaker_metal import MetalSpeakerModel


class Resnet(torch.nn.Module):
    def forward_frames(self, features):
        return features * 2

    def forward_embedding(self, frames, weights=None):
        return None, frames.sum(dim=-1) * weights

    def forward(self, features, weights=None):
        return self.forward_embedding(features * 2, weights)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet()

    def compute_fbank(self, waveform):
        return waveform + 1

    def forward(self, waveform, weights=None):
        return self.resnet(self.compute_fbank(waveform), weights)[1]


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_gpu_failure_retries_current_batch_and_remaining_batches_on_cpu(failure):
    session = Mock()
    if failure == "exception":
        session.run.side_effect = RuntimeError("device lost")
    else:
        session.run.return_value = [np.full((2, 3), np.nan, dtype=np.float32)]
    model = Model()
    adapter = DirectMLSpeakerModel(model, session)
    waveform = torch.ones(2, 3)
    weights = torch.tensor([1.0, 0.5])
    expected = model(waveform, weights=weights)
    assert torch.equal(adapter(waveform, weights=weights), expected)
    assert not adapter.accelerated and adapter.session is None
    assert torch.equal(adapter(waveform, weights=weights), expected)
    session.run.assert_called_once()


def test_gpu_encoder_preserves_original_features_and_weighted_pooling():
    model = Model()
    session = Mock()
    session.run.side_effect = lambda _, feed: [feed["features"] * 2]
    adapter = DirectMLSpeakerModel(model, session)
    waveform = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    weights = torch.tensor([1.0, 0.0])
    assert torch.equal(adapter(waveform, weights), model(waveform, weights))
    assert adapter.accelerated


def test_metal_adapter_preserves_pooling_with_cpu_test_device():
    model = Model()
    adapter = MetalSpeakerModel(model, Resnet(), torch.device("cpu"))
    waveform = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    weights = torch.tensor([1.0, 0.0])
    assert torch.equal(adapter(waveform, weights), model(waveform, weights))
    assert adapter.accelerated


@pytest.mark.parametrize("failure", ["unsupported", "nonfinite"])
def test_metal_failure_retries_batch_and_keeps_cpu_copy(failure):
    encoder = Mock()
    if failure == "unsupported":
        encoder.forward_frames.side_effect = NotImplementedError("MPS operator unsupported")
    else:
        encoder.forward_frames.return_value = torch.full((2, 3), float("nan"))
    model = Model()
    adapter = MetalSpeakerModel(model, encoder, torch.device("cpu"))
    waveform = torch.ones(2, 3)
    weights = torch.tensor([1.0, 0.5])
    expected = model(waveform, weights)
    assert torch.equal(adapter(waveform, weights), expected)
    assert torch.equal(adapter(waveform, weights), expected)
    assert not adapter.accelerated and adapter.encoder is None
    encoder.forward_frames.assert_called_once()
