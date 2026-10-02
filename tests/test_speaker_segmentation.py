from unittest.mock import Mock

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from engine.speaker_segmentation import AcceleratedSpeechFilters, StableInstanceNorm


@pytest.mark.parametrize("scale", [0, 1e-6, 1])
def test_centered_normalization_matches_original_on_silence_and_speech(scale):
    norm = torch.nn.InstanceNorm1d(3, affine=True).eval()
    values = torch.randn(4, 3, 100) * scale
    assert torch.allclose(StableInstanceNorm(norm)(values), norm(values), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("backend", ["directml", "metal"])
@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_speech_gpu_failure_recovers_current_and_future_batches(backend, failure):
    original = torch.nn.Identity()
    inputs = torch.randn(2, 1, 100)
    gpu = Mock(side_effect=RuntimeError("Device lost")) if failure == "exception" else Mock(return_value=torch.full_like(inputs, float("nan")))
    if backend == "directml":
        session = Mock()
        session.run.side_effect = RuntimeError("Device lost") if failure == "exception" else None
        session.run.return_value = [np.full(inputs.shape, np.nan, dtype=np.float32)]
        adapter = AcceleratedSpeechFilters(original, session=session)
    else:
        adapter = AcceleratedSpeechFilters(original, encoder=gpu, device=torch.device("cpu"))
    assert torch.equal(adapter(inputs), inputs)
    assert not adapter.accelerated
    assert torch.equal(adapter(inputs), inputs)
    assert adapter.session is None and adapter.encoder is None


def test_metal_speech_filters_preserve_outputs_and_metadata_on_test_device():
    original = torch.nn.Identity()
    original.num_frames = lambda n: n // 10
    original.receptive_field_size = lambda: 11
    original.receptive_field_center = lambda: 5
    adapter = AcceleratedSpeechFilters(original, encoder=torch.nn.Identity(), device=torch.device("cpu"))
    inputs = torch.randn(2, 1, 100)
    assert torch.equal(adapter(inputs), inputs)
    assert adapter.accelerated and adapter.num_frames(100) == 10
    assert adapter.receptive_field_size() == 11 and adapter.receptive_field_center() == 5


def test_near_silent_rows_use_exact_cpu_result_without_disabling_gpu():
    inputs = torch.ones(2, 1, 100)
    inputs[0] = 1e-6
    session = Mock()
    session.run.return_value = [(inputs + 1).numpy()]
    adapter = AcceleratedSpeechFilters(torch.nn.Identity(), session=session)
    output = adapter(inputs)
    assert torch.equal(output[0], inputs[0])
    assert torch.equal(output[1], inputs[1] + 1)
    assert adapter.accelerated
    session.reset_mock()
    assert torch.equal(adapter(inputs[:1]), inputs[:1])
    session.run.assert_not_called()
