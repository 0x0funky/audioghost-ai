"""
TorchCodec decoders shim.

Only implements what `sam-audio` imports:
  - AudioDecoder
  - VideoDecoder

Audio decoding is implemented via pydub+ffmpeg and returned as torch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class DecodedSamples:
    """Small wrapper matching TorchCodec's `.get_all_samples().data` pattern."""

    data: "Any"  # torch.Tensor, but avoid hard dependency at import-time


class AudioDecoder:
    """
    Minimal AudioDecoder compatible with SAM-Audio's usage.

    SAM-Audio calls:
        ad = AudioDecoder(path, sample_rate=..., num_channels=1)
        wav = ad.get_all_samples().data   # expected shape: (channels, samples)
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: Optional[int] = None,
        num_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        self.path = str(path)
        self.sample_rate = int(sample_rate) if sample_rate is not None else None
        self.num_channels = int(num_channels)

    def get_all_samples(self) -> DecodedSamples:
        import numpy as np
        import torch
        from pydub import AudioSegment

        seg = AudioSegment.from_file(self.path)

        if self.sample_rate is not None:
            seg = seg.set_frame_rate(self.sample_rate)

        if self.num_channels is not None:
            seg = seg.set_channels(self.num_channels)

        sample_width = int(seg.sample_width)  # bytes per sample
        raw = np.array(seg.get_array_of_samples())

        # pydub returns interleaved samples for multi-channel audio
        if self.num_channels and self.num_channels > 1:
            raw = raw.reshape((-1, self.num_channels)).T  # (channels, time)
        else:
            raw = raw.reshape((1, -1))

        # Normalize to [-1, 1]
        if sample_width == 1:
            # 8-bit audio is typically unsigned [0, 255]
            raw = raw.astype(np.int16) - 128
            max_val = 128.0
        else:
            max_val = float(1 << (8 * sample_width - 1))

        data = torch.from_numpy(raw.astype(np.float32) / max_val)
        return DecodedSamples(data=data)


class VideoDecoder:
    """Stub VideoDecoder. AudioGhost AI does not support video decoding in this MVP."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "VideoDecoder is not supported in AudioGhost's torchcodec shim. "
            "Use the official torchcodec package for video decoding."
        )


