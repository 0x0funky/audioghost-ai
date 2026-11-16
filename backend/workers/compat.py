"""
Platform compatibility shims.

SAM-Audio imports Meta's perception_models package, whose transformer module
does ``from xformers.ops import AttentionBias, fmha`` at import time. xformers
has no wheels for Apple Silicon, and the code paths that call it (the PE vision
/ audio-frame encoders) are never built by AudioGhost's text-prompt engine.
When xformers is missing we register a tiny stand-in so the import succeeds;
anything that would actually call it raises a clear error instead.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types

logger = logging.getLogger(__name__)

_STUB_VERSION = "0.0.0+audioghost-stub"


def _unavailable(*_args, **_kwargs):
    raise RuntimeError(
        "xformers is not installed on this platform. AudioGhost's text prompting "
        "doesn't need it; this code path (SAM-Audio's vision encoders) is unsupported here."
    )


def install_xformers_stub() -> bool:
    """Register a minimal ``xformers`` module if the real one is absent. Returns True if stubbed."""
    if "xformers" in sys.modules or importlib.util.find_spec("xformers") is not None:
        return False

    class AttentionBias:  # used only in type annotations by perception_models
        pass

    root = types.ModuleType("xformers")
    root.__version__ = _STUB_VERSION
    ops = types.ModuleType("xformers.ops")
    fmha = types.ModuleType("xformers.ops.fmha")
    attn_bias = types.ModuleType("xformers.ops.fmha.attn_bias")

    attn_bias.AttentionBias = AttentionBias
    fmha.AttentionBias = AttentionBias
    fmha.attn_bias = attn_bias
    fmha.memory_efficient_attention = _unavailable
    ops.AttentionBias = AttentionBias
    ops.fmha = fmha
    ops.memory_efficient_attention = _unavailable
    root.ops = ops

    sys.modules.update({
        "xformers": root,
        "xformers.ops": ops,
        "xformers.ops.fmha": fmha,
        "xformers.ops.fmha.attn_bias": attn_bias,
    })
    logger.info("xformers not installed - using AudioGhost's stand-in (not needed for text prompts)")
    return True


def ensure_compat() -> None:
    install_xformers_stub()
