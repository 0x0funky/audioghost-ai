"""
SAM-Audio Lite engine for AudioGhost.

Everything the Celery worker (and the CLI in ``sam_audio_lite.py``) needs to run
text-prompted separation with Meta's SAM-Audio on consumer GPUs:

* **Lite loader** - builds the model *without* the vision encoder, rankers
  (ImageBind, CLAP, SAM-Audio-Judge) and span predictor, and loads the
  checkpoint memory-mapped straight into the target dtype. Nothing that text
  prompting never uses gets downloaded or allocated. It also talks to the Hub
  with plain ``hf_hub_download`` calls, so it does not depend on the
  ``ModelHubMixin`` signature that broke with newer ``huggingface_hub``.
* **Mixed precision done right** - the diffusion transformer runs in fp16
  (default; bf16/fp32 selectable) under autocast, while T5, the audio codec,
  the RoPE tables and the ODE state stay in float32.
* **Fast codec** - SAM-Audio runs DACVAE with cuDNN disabled, which sends its
  convolutions and LSTM to slow fallback kernels. We run the identical
  computation with cuDNN and fall back to the original path on cuDNN errors.
* **Overlap-add chunking** - equal-length, hop-aligned chunks with
  raised-cosine crossfades and one global noise sequence, so chunk boundaries
  are seamless and no audio is dropped. Chunks are batched through the
  transformer.
* Stereo, temporal anchors, ODE step presets, cooperative cancellation and
  per-step progress callbacks.

The functions that don't touch the model (chunk planning, windows, anchors)
are pure and unit-tested in ``backend/tests/test_sam_lite.py``.
"""
from __future__ import annotations

import gc
import importlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple
from unittest import mock

import warnings

import torch

from .compat import ensure_compat

logger = logging.getLogger(__name__)
ensure_compat()  # must run before anything imports sam_audio

# torch.backends.cudnn.flags() (used by SAM-Audio and by us) still sets TF32 via
# the legacy API, which makes PyTorch 2.9 print a deprecation notice per call.
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

from workers.model_files import MODEL_SIZES  # noqa: E402  (one list for API and worker)

#: Midpoint ODE steps per preset. Each step costs two transformer evaluations.
#: "balanced" matches Meta's default (midpoint, step_size = 2/32).
QUALITY_STEPS = {"fast": 8, "balanced": 16, "high": 32}

#: Overlap between neighbouring chunks (seconds); capped at 20% of the chunk.
DEFAULT_OVERLAP_SECONDS = 2.0

PRECISIONS = ("fp16", "bf16", "fp32")

#: Checkpoint keys that text prompting never needs.
_SKIPPED_PREFIXES = ("vision_encoder.",)
#: Keys that are legitimately absent from checkpoint.pt (loaded from the Hub).
_EXTERNAL_PREFIXES = ("text_encoder.",)


class SeparationCancelled(Exception):
    """Raised when ``should_cancel()`` returns True during separation."""


# Where the weights come from (Meta's gated repos or AudioGhost's public copies),
# shared with the API process, which must not import torch.
from workers.model_files import (  # noqa: E402,F401
    LOCAL_MODEL_DIR,
    MODEL_ORG,
    SHARED_REPO,
    cached as _cached,
    is_downloaded,
    model_repo_id,
    needs_token,
    weight_files,
)


# --------------------------------------------------------------------------- #
# Pure helpers (no model needed)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkPlan:
    """Equal-length, hop-aligned chunks covering ``[0, padded_length)``."""

    starts: Tuple[int, ...]
    length: int
    overlap: int
    padded_length: int

    @property
    def num_chunks(self) -> int:
        return len(self.starts)


def plan_chunks(num_samples: int, max_chunk: int, overlap: int, hop: int) -> ChunkPlan:
    """
    Split ``num_samples`` into equal-length chunks of at most ``max_chunk``.

    Chunk starts and lengths are multiples of ``hop`` (the codec frame size) so a
    single global noise tensor can be sliced per chunk. Instead of a short tail
    chunk (which the old code silently dropped when < 1 s), the chunk length is
    shrunk so every chunk is the same size - that also makes them batchable.
    The caller zero-pads the signal to ``padded_length`` (always >= num_samples).
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if hop <= 0:
        raise ValueError("hop must be positive")
    max_chunk = max(hop, (max_chunk // hop) * hop)
    single = math.ceil(num_samples / hop) * hop
    if single <= max_chunk:
        return ChunkPlan(starts=(0,), length=single, overlap=0, padded_length=single)

    overlap = (max(0, overlap) // hop) * hop
    overlap = min(overlap, (max_chunk // 2 // hop) * hop)
    step_max = max_chunk - overlap
    n = math.ceil((num_samples - overlap) / step_max)
    length = math.ceil((num_samples + (n - 1) * overlap) / n)
    length = math.ceil(length / hop) * hop
    step = length - overlap
    starts = tuple(k * step for k in range(n))
    padded = starts[-1] + length
    return ChunkPlan(starts=starts, length=length, overlap=overlap, padded_length=padded)


def crossfade_window(length: int, fade_in: int, fade_out: int) -> torch.Tensor:
    """Raised-cosine window; complementary fades sum to exactly 1 in an overlap."""
    w = torch.ones(length, dtype=torch.float32)
    if fade_in > 0:
        t = (torch.arange(fade_in, dtype=torch.float64) + 0.5) / fade_in
        w[:fade_in] = torch.sin(0.5 * math.pi * t).pow(2).float()
    if fade_out > 0:
        t = (torch.arange(fade_out, dtype=torch.float64) + 0.5) / fade_out
        w[length - fade_out:] = torch.cos(0.5 * math.pi * t).pow(2).float()
    return w


def overlap_add(chunks: Sequence[torch.Tensor], plan: ChunkPlan) -> torch.Tensor:
    """
    Blend chunk outputs (``[..., plan.length]`` each) into ``[..., padded_length]``.

    Weights are normalised by their running sum, so the result is exact even
    if fades don't line up perfectly.
    """
    if len(chunks) != plan.num_chunks:
        raise ValueError("number of chunks does not match the plan")
    lead_shape = chunks[0].shape[:-1]
    out = torch.zeros(*lead_shape, plan.padded_length, dtype=torch.float32)
    norm = torch.zeros(plan.padded_length, dtype=torch.float32)
    last = plan.num_chunks - 1
    for k, (start, chunk) in enumerate(zip(plan.starts, chunks)):
        w = crossfade_window(
            plan.length,
            plan.overlap if k > 0 else 0,
            plan.overlap if k < last else 0,
        )
        out[..., start:start + plan.length] += chunk.float() * w
        norm[start:start + plan.length] += w
    return out / norm.clamp_min(1e-8)


Anchor = Tuple[str, float, float]


def localize_anchors(
    anchors: Optional[Sequence[Anchor]], chunk_start_s: float, chunk_end_s: float
) -> List[Anchor]:
    """Clip global ``(token, start_s, end_s)`` anchors to one chunk, in chunk time."""
    local: List[Anchor] = []
    for token, start, end in anchors or ():
        s, e = max(float(start), chunk_start_s), min(float(end), chunk_end_s)
        if e > s:
            local.append((token, s - chunk_start_s, e - chunk_start_s))
    return local


def normalize_anchors(anchors) -> List[Anchor]:
    """
    Accept ``[["+", s, e], ...]`` or the legacy nested ``[[["+", s, e]]]`` form
    sent by older API versions, and drop malformed / empty spans.
    """
    if not anchors:
        return []
    if (
        len(anchors) == 1
        and isinstance(anchors[0], (list, tuple))
        and anchors[0]
        and isinstance(anchors[0][0], (list, tuple))
    ):
        anchors = anchors[0]
    result: List[Anchor] = []
    for item in anchors:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        token, start, end = item
        if token not in ("+", "-"):
            continue
        start, end = float(start), float(end)
        if end - start > 1e-3:
            result.append((token, max(0.0, start), end))
    return result


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


class _NoVisionEncoder(torch.nn.Module):
    """
    Stand-in for ``PerceptionEncoder``. ``SAMAudio._get_video_features`` only
    needs ``.dim`` to build the all-zero video features used for text prompts.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim

    def forward(self, *args, **kwargs):  # pragma: no cover - guard only
        raise RuntimeError(
            "AudioGhost lite mode does not load the vision encoder "
            "(visual prompting is not supported in lite mode)."
        )


@dataclass
class LiteModel:
    model: torch.nn.Module
    processor: object
    repo_id: str
    device: torch.device
    dit_dtype: torch.dtype
    codec_dtype: torch.dtype
    sample_rate: int
    hop_length: int
    latent_channels: int
    load_seconds: float = 0.0
    _text_cache: dict = field(default_factory=dict, repr=False)

    @property
    def precision(self) -> str:
        return {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(self.dit_dtype, "fp32")


def _hf_download(repo_id: str, filename: str, token: Optional[str]) -> str:
    from huggingface_hub import hf_hub_download

    if not needs_token(repo_id):
        token = False  # public repo: never send a (possibly stale) token

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    except Exception as err:  # translate the common Hub errors into advice
        name = type(err).__name__
        if name == "GatedRepoError" or " 403" in str(err):
            raise RuntimeError(
                f"Your HuggingFace account has no access to {repo_id} yet. Request access at "
                f"https://huggingface.co/{repo_id} (approval can take a while), then retry."
            ) from err
        if name in ("RepositoryNotFoundError", "HfHubHTTPError") and " 401" in str(err):
            raise RuntimeError(
                "HuggingFace rejected the token. Reconnect HuggingFace in the app."
            ) from err
        raise


def ensure_downloaded(
    repo_id: str,
    token: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    precision: str = "fp16",
) -> bool:
    """
    Make sure every file the model needs (``weight_files``) is on this computer,
    reporting download progress of the (multi-GB) weights.

    Returns True if something had to be downloaded.
    """
    import threading

    missing = [(repo, name) for repo, name in weight_files(repo_id, precision) if not _cached(repo, name)]
    if not missing:
        return False

    # huggingface_hub (incl. the Xet backend, which pre-allocates the file so
    # its size says nothing) reports progress through tqdm bars created by
    # file_download._get_progress_bar_context. Those bars are disabled when
    # stderr isn't a terminal (and then stop counting), so wrap update().
    counters: list = []
    hub_module = importlib.import_module("huggingface_hub.file_download")
    original_context = getattr(hub_module, "_get_progress_bar_context", None)

    class _Counting:
        def __init__(self, cm, entry):
            self._cm, self._entry = cm, entry

        def __enter__(self):
            bar = self._cm.__enter__()
            original_update, entry = bar.update, self._entry

            def update(n=1):
                entry["done"] += n or 0
                return original_update(n)

            bar.update = update
            return bar

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    def counting_context(*args, **kwargs):
        entry = {"total": kwargs.get("total") or 0, "done": kwargs.get("initial") or 0}
        counters.append(entry)
        return _Counting(original_context(*args, **kwargs), entry)

    if original_context is not None:
        hub_module._get_progress_bar_context = counting_context

    stop = threading.Event()
    started = time.monotonic()

    def watch():
        while not stop.wait(2.0):
            elapsed = int(time.monotonic() - started)
            big = max(counters, key=lambda c: c["total"], default={"total": 0, "done": 0})
            total, done = big["total"], big["done"]
            if total > 1e8 and done > 0:
                progress(
                    min(0.99, done / total),
                    f"Downloading {repo_id} ({done / 1e9:.1f} / {total / 1e9:.1f} GB, first run only)",
                )
            else:
                progress(0.0, f"Downloading {repo_id} - first run only ({elapsed // 60}m {elapsed % 60:02d}s)")

    if progress is not None:
        progress(0.0, f"Downloading {repo_id} - first run only")
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
    try:
        for repo, name in missing:
            _hf_download(repo, name, token)
    finally:
        stop.set()
        if original_context is not None:
            hub_module._get_progress_bar_context = original_context
    return True


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """
    ``AUDIOGHOST_DEVICE`` env var > explicit request > CUDA > CPU.

    Apple GPUs (MPS) are opt-in (``AUDIOGHOST_DEVICE=mps``, experimental): the
    v1.0 worker crashed there (PR #7), so macOS defaults to the CPU.
    """
    requested = os.environ.get("AUDIOGHOST_DEVICE") or requested
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _SkipWeightInit:
    """
    Turn ``torch.nn.init`` into no-ops while building modules whose weights are
    overwritten by the checkpoint anyway (random init of ~0.5-3B params takes
    10+ s on CPU). ``real()`` temporarily restores it for sub-models such as T5
    that load their own weights.
    """

    NAMES = ("uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_",
             "xavier_uniform_", "xavier_normal_", "orthogonal_", "constant_", "zeros_",
             "ones_", "eye_", "dirac_", "sparse_")

    def __init__(self):
        self.originals = {n: getattr(torch.nn.init, n) for n in self.NAMES if hasattr(torch.nn.init, n)}

    @staticmethod
    def _noop(tensor, *args, **kwargs):
        return tensor

    def __enter__(self):
        for name in self.originals:
            setattr(torch.nn.init, name, self._noop)
        return self

    def __exit__(self, *exc):
        for name, fn in self.originals.items():
            setattr(torch.nn.init, name, fn)
        return False

    def real(self):
        patcher = self

        class _Real:
            def __enter__(self_inner):
                self_inner.current = {n: getattr(torch.nn.init, n) for n in patcher.originals}
                for name, fn in patcher.originals.items():
                    setattr(torch.nn.init, name, fn)

            def __exit__(self_inner, *exc):
                for name, fn in self_inner.current.items():
                    setattr(torch.nn.init, name, fn)
                return False

        return _Real()


class _DefaultDtype:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __enter__(self):
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)

    def __exit__(self, *exc):
        torch.set_default_dtype(self.previous)
        return False


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _cached_or_download(repo_id: str, filename: str, token: Optional[str]) -> str:
    """Use a local copy without any network round-trip when possible."""
    return _cached(repo_id, filename) or _hf_download(repo_id, filename, token)


def _iter_weights(path: str):
    """``(name, tensor)`` pairs from a safetensors file or Meta's checkpoint.pt, lazily."""
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.startswith(_SKIPPED_PREFIXES):
                    yield key, f.get_tensor(key)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for key, value in state.items():
            if not key.startswith(_SKIPPED_PREFIXES):
                yield key, value


def _copy_weights(model: torch.nn.Module, paths: Sequence[str]) -> Tuple[List[str], List[str]]:
    """
    Copy the weights into the model one tensor at a time (the same cast as
    ``load_state_dict``, without holding a second copy of a multi-GB model in
    RAM). Returns ``(missing, unexpected)`` keys.
    """
    targets = model.state_dict(keep_vars=True)
    loaded, unexpected = set(), []
    with torch.no_grad():
        for path in paths:
            for key, value in _iter_weights(path):
                target = targets.get(key)
                if target is None:
                    unexpected.append(key)
                    continue
                if tuple(target.shape) != tuple(value.shape):
                    raise RuntimeError(f"{key}: checkpoint shape {tuple(value.shape)} != model {tuple(target.shape)}")
                target.copy_(value)
                loaded.add(key)
    return [k for k in targets if k not in loaded], unexpected


def load_lite_model(
    repo_id: str,
    device: Optional[torch.device | str] = None,
    precision: str = "fp16",
    token: Optional[str] = None,
) -> LiteModel:
    """
    Build a text-prompt-only SAM-Audio model.

    Compared to ``SAMAudio.from_pretrained`` + deleting modules afterwards this:
      * never constructs ImageBind / CLAP / SAM-Audio-Judge / PE-A-Frame
        (~25 GB of downloads that lite mode used to throw away),
      * never constructs the vision encoder and skips random weight init,
      * memory-maps the checkpoint and copies it straight into ``precision``,
      * keeps T5, the audio codec, RoPE tables and the ODE state in fp32,
      * avoids the ``proxies`` / ``resume_download`` TypeError (issue #9).

    ``precision`` only affects the diffusion transformer: ``fp16`` (default;
    same VRAM as bf16 but 3 more mantissa bits), ``bf16`` or ``fp32``.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    started = time.perf_counter()
    device = resolve_device(str(device) if device is not None else None)
    if device.type == "cpu":
        precision = "fp32"  # half-precision matmuls on CPU are slow
    dit_dtype = DTYPES[precision]
    codec_dtype = torch.float32

    from sam_audio import SAMAudioProcessor
    from sam_audio.model.config import SAMAudioConfig

    mm = importlib.import_module("sam_audio.model.model")

    files = weight_files(repo_id, precision)
    config_path = _cached_or_download(*files[0], token)
    weight_paths = [_cached_or_download(repo, name, token) for repo, name in files[1:]]
    with open(config_path, encoding="utf-8") as f:
        raw_config = json.load(f)
    raw_config.update(visual_ranker=None, text_ranker=None, span_predictor=None)
    cfg = SAMAudioConfig(**raw_config)

    skip_init = _SkipWeightInit()
    base_t5, base_codec = mm.T5TextEncoder, mm.DACVAE

    class _FP32TextEncoder(base_t5):
        def __init__(self, text_cfg):
            with _DefaultDtype(torch.float32), skip_init.real():
                super().__init__(text_cfg)

    class _FP32Codec(base_codec):
        def __init__(self, codec_cfg):
            with _DefaultDtype(torch.float32):
                super().__init__(codec_cfg)

    # Build the transformer directly in its final dtype (no fp32 copy in RAM).
    with mock.patch.object(mm, "PerceptionEncoder", _NoVisionEncoder), \
            mock.patch.object(mm, "T5TextEncoder", _FP32TextEncoder), \
            mock.patch.object(mm, "DACVAE", _FP32Codec), \
            skip_init, _DefaultDtype(dit_dtype):
        model = mm.SAMAudio(cfg)

    # SAMAudio overrides load_state_dict (and ignores strict=False), so the
    # weights are copied in directly and the result validated here.
    missing, unexpected = _copy_weights(model, weight_paths)
    missing = [k for k in missing if not k.startswith(_EXTERNAL_PREFIXES)]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the SAM-Audio code that is installed "
            f"(missing={missing[:5]}, unexpected={unexpected[:5]}). "
            "Re-install sam-audio at the commit pinned in README.md."
        )
    gc.collect()

    model.eval()
    fp32_buffers = _snapshot_fp32_buffers(model)
    for name, module in model.named_children():
        target = torch.float32 if name in ("audio_codec", "text_encoder") else dit_dtype
        module.to(device=device, dtype=target)
    _restore_buffers(model, fp32_buffers, device)
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"Non-finite weights after loading: {bad[:5]}")

    processor = SAMAudioProcessor(
        audio_hop_length=cfg.audio_codec.hop_length,
        audio_sampling_rate=cfg.audio_codec.sample_rate,
    )
    lite = LiteModel(
        model=model,
        processor=processor,
        repo_id=repo_id,
        device=device,
        dit_dtype=dit_dtype,
        codec_dtype=codec_dtype,
        sample_rate=cfg.audio_codec.sample_rate,
        hop_length=cfg.audio_codec.hop_length,
        latent_channels=cfg.transformer.out_channels // 2,  # target + residual are stacked
        load_seconds=time.perf_counter() - started,
    )
    logger.info(
        "Loaded %s (lite, %s) on %s in %.1fs", repo_id, lite.precision, device, lite.load_seconds
    )
    return lite


#: Non-persistent buffers that are consumed in fp32 by the model code. Casting
#: the module to bf16 would round them (RoPE rotation tables, timestep
#: frequencies) for no memory benefit, so they are kept in full precision.
_FP32_BUFFER_SUFFIXES = ("rope_embeddings.freqs_cis", "t_embedder.freqs")


def _snapshot_fp32_buffers(model: torch.nn.Module) -> dict:
    snapshot = {}
    for name, buf in model.named_buffers():
        if name.endswith(_FP32_BUFFER_SUFFIXES):
            if buf.dtype != torch.float32:
                logger.warning("Buffer %s was built in %s, not fp32", name, buf.dtype)
            snapshot[name] = buf.detach().float().clone()
    return snapshot


def _restore_buffers(model: torch.nn.Module, snapshot: dict, device: torch.device) -> None:
    for name, value in snapshot.items():
        module_name, _, buffer_name = name.rpartition(".")
        module = model.get_submodule(module_name)
        setattr(module, buffer_name, value.to(device))


def unload(lite: Optional[LiteModel]) -> None:
    if lite is None:
        return
    lite._text_cache.clear()
    lite.model.to("cpu")
    del lite.model
    gc.collect()
    empty_device_cache()


def empty_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.empty_cache()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


@dataclass
class SeparationOutput:
    target: Optional[torch.Tensor]  # [channels, samples] float32, CPU; None if not decoded
    residual: Optional[torch.Tensor]  # [channels, samples]; None if not decoded
    sample_rate: int
    chunks: int
    batch_size: int
    seed: int
    steps: int


def _text_features(lite: LiteModel, descriptions: Sequence[str]):
    key = tuple(descriptions)
    cached = lite._text_cache.get(key)
    if cached is None:
        features, mask = lite.model.text_encoder(list(descriptions))
        cached = (features.to(lite.dit_dtype), mask)
        if len(lite._text_cache) > 8:
            lite._text_cache.clear()
        lite._text_cache[key] = cached
    return cached


@torch.inference_mode()
def separate_batch(
    lite: LiteModel,
    audios: torch.Tensor,
    descriptions: Sequence[str],
    noise: torch.Tensor,
    steps: int = QUALITY_STEPS["balanced"],
    anchors: Optional[Sequence[Sequence[Anchor]]] = None,
    on_step: Optional[Callable[[], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    state_dtype: torch.dtype = torch.float32,
    decode_batch: int = 2,
    decode_residual: bool = True,
    decode_target: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Separate ``B`` equal-length mono clips.

    Args:
        audios: ``[B, samples]`` at ``lite.sample_rate``; samples % hop == 0.
        descriptions: one text prompt per clip.
        noise: ``[B, frames, 2 * latent_channels]`` starting point of the ODE.
        steps: midpoint steps (2 transformer evaluations each).
        anchors: optional per-clip anchors in clip-local seconds.
        state_dtype: dtype the ODE state is integrated in (fp32 by default).
        decode_residual: also decode the model's residual (everything but the
            target, generated by the model).
        decode_target: decode the target (skipped when only the residual is
            needed, e.g. a chain step of "everything else").

    Returns:
        ``(target, residual)``, each ``[B, samples]`` float32 on CPU, or None
        when not decoded.
    """
    model = lite.model
    device = lite.device
    batch_size, num_samples = audios.shape
    if num_samples % lite.hop_length:
        raise ValueError("clip length must be a multiple of the codec hop length")

    use_anchors = anchors is not None and any(len(a) for a in anchors)
    batch = lite.processor(
        descriptions=list(descriptions),
        audios=[a.unsqueeze(0) for a in audios],
        anchors=[list(a) for a in anchors] if use_anchors else None,
    ).to(device)

    features = codec_encode(lite, batch.audios.to(lite.codec_dtype)).transpose(1, 2)
    features = torch.cat([features, features], dim=2).to(lite.dit_dtype)
    text_features, text_mask = _text_features(lite, descriptions)
    forward_args = {
        "audio_features": features,
        "text_features": text_features,
        "text_mask": text_mask,
        "masked_video_features": model._get_video_features(None, features),
        "anchor_ids": batch.anchor_ids,
        "anchor_alignment": batch.anchor_alignment,
        "audio_pad_mask": batch.audio_pad_mask,
    }
    if noise.shape != features.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} != features {tuple(features.shape)}")

    # Half-precision weights + autocast: matmuls in fp16/bf16, while autocast
    # keeps softmax/norm-style ops in fp32.
    use_autocast = device.type == "cuda" and lite.dit_dtype != torch.float32

    def velocity(t: float, x: torch.Tensor) -> torch.Tensor:
        time_t = torch.full((batch_size,), t, device=device, dtype=lite.dit_dtype)
        with torch.autocast("cuda", dtype=lite.dit_dtype, enabled=use_autocast):
            out = model.forward(noisy_audio=x.to(lite.dit_dtype), time=time_t, **forward_args)
        return out.to(state_dtype)

    # Fixed-grid midpoint solver; identical to torchdiffeq's "midpoint" with
    # step_size=1/steps, but integrates in `state_dtype` and lets us report
    # progress / cancel between steps.
    x = noise.to(device=device, dtype=state_dtype)
    dt = 1.0 / steps
    for i in range(steps):
        if should_cancel is not None and should_cancel():
            raise SeparationCancelled()
        t0 = i * dt
        k1 = velocity(t0, x)
        x = x + dt * velocity(t0 + 0.5 * dt, x + (0.5 * dt) * k1)
        if on_step is not None:
            on_step()

    generated = x.transpose(1, 2)  # [B, 2C, T]: target channels, then residual
    channels = generated.shape[1] // 2
    parts = ([generated[:, :channels]] if decode_target else []) + (
        [generated[:, channels:]] if decode_residual else [])
    if not parts:
        raise ValueError("nothing to decode")
    latents = torch.stack(parts, dim=1).reshape(len(parts) * batch_size, channels, -1)
    latents = latents.to(lite.codec_dtype)
    decoded = []
    for i in range(0, latents.shape[0], max(1, decode_batch)):
        wav = codec_decode(lite, latents[i:i + decode_batch])
        decoded.append(wav[..., :num_samples].float().cpu())
        del wav
    wavs = torch.cat(decoded, dim=0).view(batch_size, len(parts), -1)
    target = wavs[:, 0].contiguous() if decode_target else None
    residual = wavs[:, -1].contiguous() if decode_residual else None
    return target, residual


def _lstm_block_forward(self, x):
    """
    ``dacvae.LSTMBlock.forward`` with a contiguous LSTM input, so cuDNN can run
    the whole sequence in one call instead of ~10k per-timestep kernels.
    Same math (the block belongs to DACVAE's watermarker, which stays intact);
    if cuDNN rejects the shape (very long chunks) it uses the native kernel.
    """
    x = x.permute(2, 0, 1).contiguous()
    try:
        y, _ = self.lstm(x)
    except RuntimeError as err:
        if "cudnn" not in str(err).lower():
            raise
        with torch.backends.cudnn.flags(enabled=False):
            y, _ = self.lstm(x)
    if self.skip:
        y = y + x
    return y.permute(1, 2, 0)


def _prepare_fast_codec(codec: torch.nn.Module) -> None:
    import types

    for module in codec.modules():
        if type(module).__name__ == "LSTMBlock" and hasattr(module, "lstm"):
            module.forward = types.MethodType(_lstm_block_forward, module)


def codec_encode(lite: LiteModel, wavs: torch.Tensor) -> torch.Tensor:
    """
    ``DACVAEEncoder.forward`` (waveform -> latent mean) one clip at a time, with
    cuDNN enabled when possible. Encoding clips one by one keeps peak memory
    flat when many chunks are batched through the transformer.
    """
    codec = lite.model.audio_codec
    fast = lite.device.type == "cuda" and os.environ.get("AUDIOGHOST_SAFE_CODEC") != "1"
    outputs = []
    for wav in wavs:
        wav = wav.unsqueeze(0)
        if fast:
            try:
                with torch.backends.cudnn.flags(enabled=True, benchmark=False):
                    z = codec.encoder(codec._pad(wav))
                    mean, _ = codec.quantizer.in_proj(z).chunk(2, dim=1)
                outputs.append(mean)
                continue
            except RuntimeError as err:
                if "cudnn" not in str(err).lower():
                    raise
                fast = False
        outputs.append(codec(wav))
    return torch.cat(outputs, dim=0)


def codec_decode(lite: LiteModel, latents: torch.Tensor) -> torch.Tensor:
    """
    SAM-Audio wraps the whole codec in ``cudnn.flags(enabled=False)``, which
    sends every convolution and the LSTM to slow fallback kernels. Decode with
    cuDNN on (identical computation) and fall back to the original if cuDNN
    errors out. Set AUDIOGHOST_SAFE_CODEC=1 to always use the original path.
    """
    codec = lite.model.audio_codec
    if lite.device.type != "cuda" or os.environ.get("AUDIOGHOST_SAFE_CODEC") == "1":
        return codec.decode(latents)
    if not getattr(codec, "_audioghost_fast", False):
        _prepare_fast_codec(codec)
        codec._audioghost_fast = True
    try:
        with torch.backends.cudnn.flags(enabled=True, benchmark=False):
            return codec.decoder(codec.quantizer.out_proj(latents))
    except RuntimeError as err:
        if "cudnn" not in str(err).lower():
            raise
        logger.warning("cuDNN decode failed (%s); using the original codec path", err)
        return codec.decode(latents)


def _is_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


#: Throughput-optimal transformer batch per model (RTX 4090, 25 s chunks):
#: small keeps scaling to 8, large peaks around 4.
_DEFAULT_BATCH = {"small": 8, "base": 6, "large": 4}
#: Codec decode needs ~0.095 GB per second of audio per decoded clip.
_DECODE_GB_PER_SECOND = 0.095


def _free_vram_gb(device: torch.device) -> float:
    free, _total = torch.cuda.mem_get_info(device)
    # Memory PyTorch has cached but not used is available to us too.
    free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return free / 1024**3


def suggest_decode_batch(lite: LiteModel, chunk_seconds: float) -> int:
    """Decode two clips at a time when there is room, otherwise one (same speed/clip)."""
    if lite.device.type != "cuda":
        return 1
    per_clip = _DECODE_GB_PER_SECOND * chunk_seconds
    return 2 if _free_vram_gb(lite.device) >= 2 * per_clip + 1.0 else 1


def suggest_batch_size(lite: LiteModel, chunk_seconds: float, requested: int = 0) -> int:
    """
    How many chunks to push through the transformer at once.

    ``requested`` > 0 wins. Transformer activations are small next to the
    codec decode, so this is mostly the per-model sweet spot, capped by free
    VRAM. Separation falls back to smaller batches on OOM anyway.
    """
    if requested and requested > 0:
        return int(requested)
    if lite.device.type != "cuda":
        return 1
    size = next((s for s in MODEL_SIZES if f"sam-audio-{s}" in lite.repo_id), "base")
    budget = (
        _free_vram_gb(lite.device)
        - suggest_decode_batch(lite, chunk_seconds) * _DECODE_GB_PER_SECOND * chunk_seconds
        - 0.75
    )
    per_item = 0.15 * max(1.0, chunk_seconds / 25.0) * (2.0 if lite.precision == "fp32" else 1.0)
    return int(max(1, min(_DEFAULT_BATCH[size], budget // per_item)))


def separate(
    lite: LiteModel,
    audio: torch.Tensor,
    description: str,
    *,
    chunk_seconds: float = 25.0,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    steps: int = QUALITY_STEPS["balanced"],
    anchors: Optional[Sequence[Anchor]] = None,
    batch_size: int = 0,
    seed: Optional[int] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    decode_residual: bool = True,
    decode_target: bool = True,
) -> SeparationOutput:
    """
    Separate ``description`` from ``audio`` (``[channels, samples]`` at
    ``lite.sample_rate``). Multi-channel input is processed per channel with a
    shared noise sequence, so stereo stays phase-coherent. With
    ``decode_residual=False`` only the target is decoded (``residual`` is None);
    with ``decode_target=False`` only the residual is (``target`` is None).
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.float().cpu()
    num_channels, num_samples = audio.shape
    sr, hop = lite.sample_rate, lite.hop_length

    max_chunk = int(max(1.0, chunk_seconds) * sr)
    overlap = int(min(overlap_seconds, 0.2 * chunk_seconds) * sr)
    plan = plan_chunks(num_samples, max_chunk, overlap, hop)
    padded = torch.nn.functional.pad(audio, (0, plan.padded_length - num_samples))

    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_frames = plan.padded_length // hop
    global_noise = torch.randn(
        total_frames, 2 * lite.latent_channels, generator=generator, dtype=torch.float32
    )

    # Work items: (chunk index, channel)
    items = [(k, ch) for k in range(plan.num_chunks) for ch in range(num_channels)]
    chunk_len_s = plan.length / sr
    batch = max(1, min(len(items), suggest_batch_size(lite, chunk_len_s, batch_size)))
    decode_batch = suggest_decode_batch(lite, chunk_len_s)

    results: dict = {}
    total_evals = max(1, math.ceil(len(items) / batch) * steps)
    done_evals = 0

    def report(msg: str):
        if progress is not None:
            progress(min(1.0, done_evals / total_evals), msg)

    index = 0
    while index < len(items):
        group = items[index:index + batch]
        clips = torch.stack(
            [padded[ch, plan.starts[k]:plan.starts[k] + plan.length] for k, ch in group]
        )
        noise = torch.stack(
            [
                global_noise[plan.starts[k] // hop:(plan.starts[k] + plan.length) // hop]
                for k, _ in group
            ]
        )
        group_anchors = [
            localize_anchors(anchors, plan.starts[k] / sr, (plan.starts[k] + plan.length) / sr)
            for k, _ in group
        ]
        chunk_ids = sorted({k for k, _ in group})
        label = (
            f"chunk {chunk_ids[0] + 1}/{plan.num_chunks}"
            if len(chunk_ids) == 1
            else f"chunks {chunk_ids[0] + 1}-{chunk_ids[-1] + 1}/{plan.num_chunks}"
        )

        def on_step():
            nonlocal done_evals
            done_evals += 1
            report(f"Separating {label}")

        evals_before = done_evals
        try:
            target, residual = separate_batch(
                lite,
                clips,
                [description] * len(group),
                noise,
                steps=steps,
                anchors=group_anchors,
                on_step=on_step,
                should_cancel=should_cancel,
                decode_batch=decode_batch,
                decode_residual=decode_residual,
                decode_target=decode_target,
            )
        except Exception as err:  # noqa: BLE001 - re-raised below
            if not _is_oom(err):
                raise
            done_evals = evals_before
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch == 1 and decode_batch == 1:
                raise RuntimeError(
                    "CUDA out of memory. Try the Small model, a shorter chunk "
                    "duration, or turn off stereo / High Quality mode."
                ) from err
            batch = max(1, batch // 2)
            decode_batch = 1
            total_evals = done_evals + math.ceil((len(items) - index) / batch) * steps
            logger.warning("OOM - retrying with batch size %d", batch)
            continue

        for j, key in enumerate(group):
            results[key] = (None if target is None else target[j], None if residual is None else residual[j])
        index += len(group)

    def assemble(which: int) -> torch.Tensor:
        channels = []
        for ch in range(num_channels):
            chunks = [results[(k, ch)][which] for k in range(plan.num_chunks)]
            channels.append(overlap_add(chunks, plan)[..., :num_samples])
        return torch.stack(channels)

    if progress is not None:
        progress(1.0, "Blending chunks")
    return SeparationOutput(
        target=assemble(0) if decode_target else None,
        residual=assemble(1) if decode_residual else None,
        sample_rate=sr,
        chunks=plan.num_chunks,
        batch_size=batch,
        seed=seed,
        steps=steps,
    )
