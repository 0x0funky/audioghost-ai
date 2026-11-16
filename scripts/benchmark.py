"""
Benchmark AudioGhost's SAM-Audio Lite engine on your own machine.

Examples (from the repo root, inside the audioghost env):

    python scripts/benchmark.py --file song.mp3 --prompt "singing voice"
    python scripts/benchmark.py --file song.mp3 --model large --quality fast --stereo
    python scripts/benchmark.py --file song.mp3 --model small --save out/   # keep the stems

Prints load time, processing time, realtime factor and peak VRAM as JSON.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True, help="audio or video file")
    parser.add_argument("--prompt", default="singing voice")
    parser.add_argument("--model", default="small", choices=["small", "base", "large"])
    parser.add_argument("--variant", default="default", choices=["default", "tv"])
    parser.add_argument("--quality", default="balanced", choices=["fast", "balanced", "high"])
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--chunk", type=float, default=25.0, help="max chunk length (s)")
    parser.add_argument("--batch", type=int, default=0, help="chunks per batch (0 = auto)")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runs", type=int, default=1, help="repeat to measure warm runs")
    parser.add_argument("--save", default=None, help="directory to write the stems to")
    args = parser.parse_args()

    import torch

    from settings import read_hf_token
    from workers import audio_io, sam_lite

    token = read_hf_token() or os.environ.get("HF_TOKEN")
    repo = sam_lite.model_repo_id(args.model, args.variant)
    sam_lite.ensure_downloaded(repo, token, lambda f, m: print(f"\r{m}", end="", flush=True))
    print()

    lite = sam_lite.load_lite_model(repo, precision=args.precision, token=token)
    vram_model = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

    audio = audio_io.load_audio(args.file, lite.sample_rate, stereo=args.stereo)
    duration = audio.shape[-1] / lite.sample_rate

    runs = []
    out = None
    for _ in range(max(1, args.runs)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = sam_lite.separate(
            lite, audio, args.prompt,
            chunk_seconds=args.chunk,
            steps=sam_lite.QUALITY_STEPS[args.quality],
            batch_size=args.batch,
            seed=args.seed,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        runs.append({"seconds": round(elapsed, 2), "realtime_x": round(duration / elapsed, 1),
                     "peak_vram_gb": round(peak, 2)})

    report = {
        "model": repo,
        "precision": lite.precision,
        "quality": args.quality,
        "steps": sam_lite.QUALITY_STEPS[args.quality],
        "device": str(lite.device),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "audio_seconds": round(duration, 1),
        "channels": int(audio.shape[0]),
        "chunks": out.chunks,
        "batch_size": out.batch_size,
        "load_seconds": round(lite.load_seconds, 1),
        "model_vram_gb": round(vram_model, 2),
        "runs": runs,
    }
    print(json.dumps(report, indent=2))

    if args.save:
        dest = Path(args.save)
        dest.mkdir(parents=True, exist_ok=True)
        name = Path(args.file).name
        audio_io.save_audio(dest / audio_io.stem_filename(name, args.prompt, "ghost", "wav"),
                            out.target, out.sample_rate)
        audio_io.save_audio(dest / audio_io.stem_filename(name, args.prompt, "clean", "wav"),
                            out.residual, out.sample_rate)
        print(f"Stems written to {dest}")


if __name__ == "__main__":
    main()
