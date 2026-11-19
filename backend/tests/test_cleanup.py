"""
"Clean up leftovers" (workers.cleanup): the separated track clears its own traces
from a residual. Synthetic signals only; no model, GPU or Redis needed.

Run from the repo root:  python -m pytest backend/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers import cleanup  # noqa: E402

SR = 16000


def harmonics(f0, seconds, amp, start=0.0, stop=None):
    """A buzzy harmonic tone (a stand-in for a cry or a voice), on only between start and stop."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(amp / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 6))
    gate = (t >= start) & (t < (stop if stop is not None else seconds))
    return (x * gate).astype(np.float32)


def band_energy(x, lo, hi, start=None, stop=None):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if start is not None:
        x = x[int(start * SR):int(stop * SR)]
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return spec[(freqs >= lo) & (freqs < hi)].sum()


def db(ratio):
    return 10 * np.log10(ratio + 1e-20)


def test_leftovers_fade_and_the_rest_stays():
    seconds = 6.0
    voice = harmonics(150, seconds, 0.2)                       # the rest of the recording, all the way through
    cry = harmonics(900, seconds, 0.3, start=2.0, stop=4.0)    # the separated sound, from 2 s to 4 s
    residual = voice + 0.3 * cry                               # what the model left: the voice plus a trace of the cry
    out = cleanup.clean_residual(torch.from_numpy(residual)[None], torch.from_numpy(cry)[None], SR)
    assert out.shape == (1, residual.shape[0]) and out.dtype == torch.float32
    out = out[0].numpy()

    # the cry's trace drops by well over 10 dB
    before = band_energy(residual, 850, 4600, 2.2, 3.8)
    after = band_energy(out, 850, 4600, 2.2, 3.8)
    assert db(after / before) < -10
    # where there is no cry, the voice is untouched (the first and last frames of a file are left out:
    # the analysis has edge effects of a fraction of a percent there)
    for start, stop in ((0.1, 1.8), (4.2, 5.9)):
        np.testing.assert_allclose(out[int(start * SR):int(stop * SR)], residual[int(start * SR):int(stop * SR)],
                                   atol=2e-3)
    # under the cry, the voice's own pitch range keeps most of its level
    assert db(band_energy(out, 100, 800, 2.2, 3.8) / band_energy(voice, 100, 800, 2.2, 3.8)) > -3


def test_stereo_keeps_its_shape_and_image():
    seconds = 3.0
    cry = harmonics(800, seconds, 0.3, start=1.0, stop=2.0)
    voice = harmonics(200, seconds, 0.2)
    left, right = voice + 0.3 * cry, 0.5 * voice + 0.15 * cry
    residual = torch.from_numpy(np.stack([left, right]))
    out = cleanup.clean_residual(residual, torch.from_numpy(np.stack([cry, 0.5 * cry])), SR)
    assert out.shape == residual.shape
    # one gain for both channels: the right channel stays half the left
    np.testing.assert_allclose(out[1].numpy(), 0.5 * out[0].numpy(), atol=1e-5)


def test_long_recordings_are_cleaned_in_blocks_without_seams():
    seconds = 75.0  # longer than one 30 s block plus its overlap
    voice = harmonics(180, seconds, 0.2)
    cry = harmonics(1000, seconds, 0.3, start=28.0, stop=33.0)  # straddles the first block boundary
    residual = voice + 0.3 * cry
    out = cleanup.clean_residual(torch.from_numpy(residual), torch.from_numpy(cry), SR)
    assert out.shape == (residual.shape[0],)  # a 1-D input comes back 1-D
    out = out.numpy()
    assert db(band_energy(out, 950, 5100, 28.2, 32.8) / band_energy(residual, 950, 5100, 28.2, 32.8)) < -10
    # no clicks or level changes at the block joins, where there is no cry
    for edge in (60.0,):
        a, b = int((edge - 1.5) * SR), int((edge + 1.5) * SR)
        np.testing.assert_allclose(out[a:b], residual[a:b], atol=2e-3)


def test_short_input_is_returned_as_is():
    x = torch.rand(2, 1000)
    out = cleanup.clean_residual(x, torch.rand(2, 1000), SR)
    assert torch.equal(out, x)
