"""
Where the SAM-Audio weights come from, and whether they are on this computer.

No torch here: the API process asks these questions without loading PyTorch.

* Meta's checkpoints (``facebook/sam-audio-*``) are gated: an account with access
  and a token are needed to download them. Once they are in the Hugging Face
  cache they work offline, with no token.
* AudioGhost's copies (``audioghostai/sam-audio-*``, built by model_hub/) are
  public: no account at all. A repo per size holds the transformer in fp16
  (the default) and fp32; ``sam-audio-shared`` holds the audio codec.
"""
import os
from typing import List, Optional, Tuple

MODEL_SIZES = ("small", "base", "large")

#: The Hugging Face organization with AudioGhost's copies of the weights.
MODEL_ORG = os.environ.get("AUDIOGHOST_MODEL_ORG", "audioghostai")
SHARED_REPO = f"{MODEL_ORG}/sam-audio-shared"
#: A folder laid out like those repos (sam-audio-<size>/, sam-audio-shared/), for
#: offline installs and tests; files found there are used without a download.
LOCAL_MODEL_DIR = os.environ.get("AUDIOGHOST_MODEL_DIR") or None


def needs_token(repo_id: str) -> bool:
    """Meta's repos are gated; AudioGhost's copies are public."""
    return repo_id.startswith("facebook/")


def cached(repo_id: str, filename: str) -> Optional[str]:
    """A local copy of a model file (LOCAL_MODEL_DIR or the HF cache), without network."""
    if LOCAL_MODEL_DIR:
        local = os.path.join(LOCAL_MODEL_DIR, repo_id.split("/")[-1], filename)
        if os.path.isfile(local):
            return local
    from huggingface_hub import try_to_load_from_cache

    try:
        hit = try_to_load_from_cache(repo_id, filename)
    except Exception:  # noqa: BLE001 - malformed cache entries count as missing
        return None
    return hit if isinstance(hit, str) and os.path.exists(hit) else None


def model_repo_id(model_size: str, variant: str = "default") -> str:
    """
    Where a model comes from: Meta's own checkpoint when it is already in the
    Hugging Face cache (nothing new to download for people who have it),
    otherwise AudioGhost's public copy. Meta's ``-tv`` checkpoints (better target
    accuracy) have no copy yet and still need access to Meta's repo.
    """
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size {model_size!r}; expected one of {MODEL_SIZES}")
    if variant == "tv":
        return f"facebook/sam-audio-{model_size}-tv"
    meta = f"facebook/sam-audio-{model_size}"
    if cached(meta, "checkpoint.pt"):
        return meta
    return f"{MODEL_ORG}/sam-audio-{model_size}"


def stored_precision(precision: str) -> str:
    """The file a run precision loads: fp16 runs use the fp16 file, bf16/fp32 the fp32 values."""
    return "fp16" if precision == "fp16" else "fp32"


def weight_files(repo_id: str, precision: str = "fp16") -> List[Tuple[str, str]]:
    """``(repo, filename)`` pairs a model needs, config first."""
    if needs_token(repo_id):
        return [(repo_id, "config.json"), (repo_id, "checkpoint.pt")]
    return [(repo_id, "config.json"), (repo_id, f"model.{stored_precision(precision)}.safetensors"),
            (SHARED_REPO, "audio_codec.safetensors")]


def is_downloaded(repo_id: str, precision: str = "fp16") -> bool:
    """True if every file the model needs is already on this computer."""
    return all(cached(repo, name) for repo, name in weight_files(repo_id, precision))


def missing_requirement(model_size: str, variant: str, precision: str, *, token: Optional[str],
                        license_accepted: bool) -> Optional[str]:
    """
    Why a separation can't download its model yet ("license" or "token"), or None.
    Nothing is needed for a model that is already on this computer.
    """
    repo = model_repo_id(model_size, variant)
    if is_downloaded(repo, precision):
        return None
    if needs_token(repo):
        return None if token else "token"
    return None if license_accepted else "license"
