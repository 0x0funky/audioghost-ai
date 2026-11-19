"""
Build AudioGhost's copies of the SAM-Audio weights from Meta's checkpoints.

Phase 1 covers what text-prompted separation needs:

    sam-audio-{small,base,large}/
        config.json              Meta's config, unchanged
        model.fp16.safetensors   every tensor except the vision encoder and the audio codec,
                                 in fp16. AudioGhost's default: it loads into exactly the values
                                 Meta's fp32 checkpoint is cast to at load time.
        model.fp32.safetensors   the same tensors in fp32, Meta's values (bf16 / fp32 modes,
                                 fine-tuning)
    sam-audio-shared/
        audio_codec.safetensors  the DACVAE codec in fp32 (identical in all three sizes)

The checkpoints are read from the Hugging Face cache, so download them first with an
account that has access to facebook/sam-audio-*. A manifest with every file's size and
SHA-256 is written next to the output.

usage: python model_hub/build.py --out <folder> [--sizes small base large]
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

VISION_PREFIX = "vision_encoder."
CODEC_PREFIX = "audio_codec."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensors: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        t = tensors[key].contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(t.shape)).encode())
        digest.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()


def source(size: str):
    repo = f"facebook/sam-audio-{size}"
    checkpoint = Path(hf_hub_download(repo, "checkpoint.pt", local_files_only=True))
    config = Path(hf_hub_download(repo, "config.json", local_files_only=True))
    return repo, checkpoint.parent.name, checkpoint, config


def save(tensors: dict, path: Path, metadata: dict) -> dict:
    started = time.perf_counter()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path), metadata=metadata)
    entry = {"bytes": path.stat().st_size, "sha256": sha256(path), "tensors": len(tensors)}
    print(f"  {path.parent.name}/{path.name}: {entry['bytes'] / 2**30:.2f} GiB, "
          f"{len(tensors)} tensors ({time.perf_counter() - started:.0f}s)", flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", default=["small", "base", "large"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"made_by": "AudioGhost model_hub/build.py", "repos": {}}
    codec_digest = None

    for size in args.sizes:
        repo, revision, checkpoint, config = source(size)
        print(f"{repo}@{revision[:10]}", flush=True)
        folder = args.out / f"sam-audio-{size}"
        folder.mkdir(exist_ok=True)
        shutil.copyfile(config, folder / "config.json")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        base_meta = {"source": repo, "source_revision": revision, "made_by": manifest["made_by"]}

        codec = {k: v for k, v in state.items() if k.startswith(CODEC_PREFIX)}
        digest = tensor_digest(codec)
        if codec_digest is None:
            codec_digest = digest
            shared = args.out / "sam-audio-shared"
            shared.mkdir(exist_ok=True)
            entry = save(codec, shared / "audio_codec.safetensors", {**base_meta, "dtype": "float32"})
            entry["codec_sha256_of_tensors"] = digest
            manifest["repos"]["sam-audio-shared"] = {"audio_codec.safetensors": entry}
        elif digest != codec_digest:
            raise SystemExit(f"{repo}: the audio codec differs from the first size; it can't be shared")
        else:
            print("  audio codec: identical to the shared copy", flush=True)

        model = {k: v for k, v in state.items() if not k.startswith((VISION_PREFIX, CODEC_PREFIX))}
        files = {"config.json": {"bytes": (folder / "config.json").stat().st_size,
                                 "sha256": sha256(folder / "config.json")}}
        files["model.fp32.safetensors"] = save(model, folder / "model.fp32.safetensors", {**base_meta, "dtype": "float32"})
        half = {k: (v.to(torch.float16) if v.is_floating_point() else v) for k, v in model.items()}
        files["model.fp16.safetensors"] = save(half, folder / "model.fp16.safetensors", {**base_meta, "dtype": "float16"})
        manifest["repos"][f"sam-audio-{size}"] = {"source": f"{repo}@{revision}", **files}
        del state, codec, model, half

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
