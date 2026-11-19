"""
Publish the files model_hub/build.py made to AudioGhost's Hugging Face
organization: one public repo per folder, each with Meta's SAM License (the
license asks for a copy to go with the weights) and a model card that says
what was changed and lists every file's SHA-256.

usage: python model_hub/publish.py --built <folder> [--org audioghostai] [--dry-run]

Needs `hf auth login` with a token that can write to the organization.
"""
import argparse
import json
import shutil
from pathlib import Path

LICENSE = Path(__file__).resolve().parents[1] / "backend" / "licenses" / "SAM_LICENSE.txt"
TITLES = {"small": "Small", "base": "Base", "large": "Large"}

CARD_MODEL = """---
license: other
license_name: sam-license
license_link: LICENSE
base_model: {source_repo}
pipeline_tag: audio-to-audio
tags:
- audio
- source-separation
- sam-audio
- audioghost
---

# SAM-Audio {title} (AudioGhost copy)

An unofficial copy of Meta's [SAM-Audio {title}](https://huggingface.co/{source_repo}) weights, repackaged for
[AudioGhost](https://github.com/0x0funky/audioghost-ai) so that its users can download them without a Hugging Face
account. **Not made, endorsed or supported by Meta.** The audio codec it needs is in
[{org}/sam-audio-shared](https://huggingface.co/{org}/sam-audio-shared).

## What is different from Meta's checkpoint

- Only the parts text-prompted separation uses: the diffusion transformer and the small layers around it (anchor
  embedding, projections, video alignment). The vision encoder (used only for visual prompting) is left out, and the
  audio codec lives in the shared repo. The text encoder is the public `t5-base`, as in Meta's code.
- Safetensors instead of a pickled `checkpoint.pt`.
- `model.fp16.safetensors`: the same tensors cast to float16. This is AudioGhost's default, and it loads into exactly the
  values that Meta's float32 checkpoint is cast to when AudioGhost runs it in float16.
- `model.fp32.safetensors`: Meta's float32 values, unchanged (for bf16 / fp32 runs and fine-tuning).
- `config.json`: Meta's, unchanged.

Source: `{source_full}`.

## License

These weights are SAM Materials under Meta's **SAM License**, included here as [LICENSE](LICENSE). By downloading or
using them you agree to its terms, including the Trade Controls section. If you redistribute them, or anything derived
from them, you must do so under the same license and include a copy of it.

## Files

| File | Size | SHA-256 |
|---|---|---|
{rows}
"""

CARD_SHARED = """---
license: other
license_name: sam-license
license_link: LICENSE
pipeline_tag: audio-to-audio
tags:
- audio
- source-separation
- sam-audio
- audioghost
---

# SAM-Audio shared components (AudioGhost copy)

Parts of Meta's [SAM-Audio](https://github.com/facebookresearch/sam-audio) that every model size shares, repackaged for
[AudioGhost](https://github.com/0x0funky/audioghost-ai). **Not made, endorsed or supported by Meta.**

- `audio_codec.safetensors`: the DACVAE audio codec in float32. It is identical in Meta's Small, Base and Large
  checkpoints, so it is stored once here; the size repos ({org}/sam-audio-small, -base, -large) hold the transformers.

Source: `{source_full}`.

## License

These weights are SAM Materials under Meta's **SAM License**, included here as [LICENSE](LICENSE). By downloading or
using them you agree to its terms, including the Trade Controls section. If you redistribute them, or anything derived
from them, you must do so under the same license and include a copy of it.

## Files

| File | Size | SHA-256 |
|---|---|---|
{rows}
"""


def rows(files: dict) -> str:
    lines = []
    for name, entry in files.items():
        if not isinstance(entry, dict) or "sha256" not in entry:
            continue
        size = entry["bytes"]
        shown = f"{size / 2**30:.2f} GB" if size > 2**30 else f"{size / 2**20:.1f} MB" if size > 2**20 else f"{size} B"
        lines.append(f"| `{name}` | {size:,} bytes ({shown}) | `{entry['sha256']}` |")
    return "\n".join(lines)


def prepare(built: Path, org: str) -> list:
    manifest = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
    repos = manifest["repos"]
    first_source = next(v["source"] for k, v in repos.items() if k != "sam-audio-shared")
    folders = []
    for name, files in repos.items():
        folder = built / name
        shutil.copyfile(LICENSE, folder / "LICENSE")
        if name == "sam-audio-shared":
            card = CARD_SHARED.format(org=org, source_full=first_source, rows=rows(files))
        else:
            size = name.rsplit("-", 1)[-1]
            source_repo, _, _ = files["source"].partition("@")
            card = CARD_MODEL.format(title=TITLES[size], source_repo=source_repo, source_full=files["source"],
                                     org=org, rows=rows(files))
        (folder / "README.md").write_text(card, encoding="utf-8")
        folders.append((f"{org}/{name}", folder))
    return folders


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--built", required=True, type=Path)
    parser.add_argument("--org", default="audioghostai")
    parser.add_argument("--dry-run", action="store_true", help="write the cards and licenses, upload nothing")
    args = parser.parse_args()
    folders = prepare(args.built, args.org)
    for repo, folder in folders:
        print(f"{repo}: {', '.join(sorted(p.name for p in folder.iterdir()))}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    print("uploading as", api.whoami()["name"], flush=True)
    for repo, folder in folders:
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(folder), repo_type="model",
                          commit_message="AudioGhost copy of Meta's SAM-Audio weights (SAM License)")
        print(f"published https://huggingface.co/{repo}", flush=True)


if __name__ == "__main__":
    main()
