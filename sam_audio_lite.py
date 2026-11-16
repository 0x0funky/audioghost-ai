"""
SAM Audio Lite - command-line separation without the web UI.

Uses the same engine as the AudioGhost worker (backend/workers/sam_lite.py):
text-prompt-only SAM-Audio, fp16 transformer with fp32 codec/ODE state,
overlap-add chunking and batched chunks.

Examples (inside the audioghost conda env, from the repo root):

    python sam_audio_lite.py song.mp3 "singing voice"
    python sam_audio_lite.py interview.mp4 "background music" --model base --quality fast
    python sam_audio_lite.py live.flac "crowd noise" --stereo --start 12.5 --end 16 -o stems/

Writes "<name> [<prompt>].wav" (the described sound) and
"<name> [no <prompt>].wav" (everything else). Needs a HuggingFace token with
access to facebook/sam-audio-* (HF_TOKEN env var, `huggingface-cli login`, or
connect once in the web UI).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="audio or video file")
    parser.add_argument("prompt", help='what to separate, e.g. "vocals" or "a dog barking"')
    parser.add_argument("-o", "--out", default=".", help="output directory (default: current)")
    parser.add_argument("--model", default="base", choices=["small", "base", "large"])
    parser.add_argument("--tv", action="store_true", help="use the -tv checkpoint (better target accuracy)")
    parser.add_argument("--quality", default="balanced", choices=["fast", "balanced", "high"])
    parser.add_argument("--fp32", action="store_true", help="run the transformer in float32")
    parser.add_argument("--stereo", action="store_true", help="keep stereo (about 2x compute)")
    parser.add_argument("--chunk", type=float, default=25.0, help="max chunk length in seconds")
    parser.add_argument("--start", type=float, help="Temporal Lock: span where the sound is audible")
    parser.add_argument("--end", type=float)
    parser.add_argument("--seed", type=int, help="for reproducible results")
    parser.add_argument("--format", default="wav", choices=["wav", "flac", "mp3"])
    args = parser.parse_args()

    from settings import read_hf_token
    from workers import audio_io, sam_lite

    source = Path(args.input)
    if not source.is_file():
        parser.error(f"{source} does not exist")
    token = read_hf_token()
    repo = sam_lite.model_repo_id(args.model, "tv" if args.tv else "default")

    started = time.perf_counter()
    sam_lite.ensure_downloaded(repo, token, lambda f, m: print(f"\r{m}   ", end="", flush=True))
    lite = sam_lite.load_lite_model(repo, precision="fp32" if args.fp32 else "fp16", token=token)
    print(f"\rLoaded {repo} ({lite.precision}) on {lite.device} in {lite.load_seconds:.1f}s")

    audio = audio_io.load_audio(source, lite.sample_rate, stereo=args.stereo)
    anchors = None
    if args.start is not None and args.end is not None:
        anchors = sam_lite.normalize_anchors([["+", args.start, args.end]])

    def progress(fraction, message):
        print(f"\r[{fraction * 100:5.1f}%] {message}          ", end="", flush=True)

    result = sam_lite.separate(
        lite, audio, args.prompt,
        chunk_seconds=args.chunk,
        steps=sam_lite.QUALITY_STEPS[args.quality],
        anchors=anchors,
        seed=args.seed,
        progress=progress,
    )
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, wav in (("ghost", result.target), ("clean", result.residual)):
        work = out_dir / f".{source.stem}.{stem}.tmp.wav"
        audio_io.save_audio(work, wav, result.sample_rate)
        final = out_dir / audio_io.stem_filename(source.name, args.prompt, stem, args.format)
        audio_io.export_audio(work, final, args.format)  # 24-bit WAV/FLAC or 320k MP3
        work.unlink(missing_ok=True)
        print(f"  -> {final}")

    seconds = audio.shape[-1] / lite.sample_rate
    total = time.perf_counter() - started
    print(f"Done: {seconds:.1f}s of audio in {total:.1f}s (seed {result.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
