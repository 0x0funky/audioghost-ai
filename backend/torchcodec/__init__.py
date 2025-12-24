"""
TorchCodec shim (audio-only) for AudioGhost AI.

Why this exists:
- Upstream `sam-audio` imports `torchcodec.decoders` at import time.
- Official `torchcodec` wheels may fail to load in some Docker/OS combinations
  (FFmpeg / libc / PyTorch ABI mismatch), which breaks *all* imports of
  `sam_audio` even if we only need audio-only inference.

What we do:
- Provide a minimal subset of the TorchCodec API required for SAM-Audio imports.
- Implement `AudioDecoder` using `pydub` (ffmpeg CLI) to decode audio into
  PyTorch tensors.
- Stub `VideoDecoder` (not needed for AudioGhost's audio-only MVP).

This keeps the project portable while preserving existing features.
"""

from __future__ import annotations

__all__ = ["decoders", "encoders", "samplers", "__version__"]

__version__ = "0.0.0-shim"

# Expose submodules that upstream may import
from . import decoders  # noqa: F401
from . import encoders  # noqa: F401
from . import samplers  # noqa: F401


