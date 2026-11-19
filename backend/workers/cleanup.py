"""
"Clean up leftovers": an optional pass over "everything else" (Advanced settings).

SAM-Audio regenerates the sound it separates, and its "everything but this
sound" can keep traces of that sound. The separated track is a good map of where
the sound is, so every time-frequency bin of the residual R gets the gain

    g = |R|^2 / (|R|^2 + BETA * |T|^2)          (T: the separated track)

smoothed over 3x3 bins. Where the residual is stronger than the separated sound
it stays as it is; where the sound dominates, the residual fades. Parts of the
recording where the sound is absent are left alone. Another sound that overlaps
it in time and pitch can come out a little thinner, so this is opt-in.

Runs on the CPU in 30-second blocks with a 1-second crossfade, so long
recordings stay cheap (a fraction of a second per minute of audio).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

N_FFT = 2048
HOP = 512
BETA = 1.5
BLOCK_SECONDS = 30.0
OVERLAP_SECONDS = 1.0


def _as_2d(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.detach().to("cpu", torch.float32)
    return audio[None] if audio.dim() == 1 else audio


def _clean_block(residual: torch.Tensor, sound: torch.Tensor, beta: float) -> torch.Tensor:
    """``residual``: [channels, n]; ``sound``: [n], the separated track (mono)."""
    n = residual.shape[-1]
    if n < N_FFT:  # too short to analyse
        return residual.clone()
    window = torch.hann_window(N_FFT)
    sound_power = torch.stft(sound, N_FFT, HOP, window=window, return_complex=True).abs().pow(2)
    spec = torch.stft(residual, N_FFT, HOP, window=window, return_complex=True)
    power = spec.abs().pow(2).mean(dim=0)  # one set of gains for every channel keeps the stereo image
    gain = power / (power + beta * sound_power + 1e-12)
    gain = F.avg_pool2d(gain[None, None], kernel_size=3, stride=1, padding=1, count_include_pad=False)[0, 0]
    return torch.istft(spec * gain[None], N_FFT, HOP, window=window, length=n)


def clean_residual(residual: torch.Tensor, target: torch.Tensor, sample_rate: int,
                   beta: float = BETA) -> torch.Tensor:
    """
    ``residual`` ("everything else" or a track's "without") and ``target`` (the
    separated track), both [channels, samples] or [samples]. Returns a CPU float32
    tensor shaped like ``residual``. Every channel gets the same gains, so the
    stereo image is kept.
    """
    shape = residual.shape
    rest = _as_2d(residual)
    sound = _as_2d(target).mean(dim=0)
    n = rest.shape[-1]
    if sound.shape[-1] < n:
        sound = F.pad(sound, (0, n - sound.shape[-1]))
    sound = sound[:n]

    block = int(BLOCK_SECONDS * sample_rate)
    overlap = int(OVERLAP_SECONDS * sample_rate)
    if n <= block + overlap:
        return _clean_block(rest, sound, beta).reshape(shape)

    out = torch.zeros_like(rest)
    weight = torch.zeros(n)
    fade_in = torch.linspace(0.0, 1.0, overlap)
    start = 0
    while True:
        end = min(n, start + block + overlap)
        piece = _clean_block(rest[:, start:end], sound[start:end], beta)
        w = torch.ones(end - start)
        if start > 0:
            w[:overlap] = fade_in
        if end < n:
            w[-overlap:] = fade_in.flip(0)
        out[:, start:end] += piece * w
        weight[start:end] += w
        if end == n:
            break
        start += block
    return (out / weight.clamp_min(1e-6)).reshape(shape)
