"""
Audio / video I/O helpers shared by the worker and the API.

Decoding goes through TorchCodec (already required by SAM-Audio), which reads
both audio files and the audio track of video containers directly - no
intermediate WAV extraction step. Writing uses soundfile, so it does not depend
on torchaudio's backend du jour.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

AUDIO_EXPORT_FORMATS = ("wav", "flac", "mp3")

_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def probe_channels(path: str | Path) -> int:
    from torchcodec.decoders import AudioDecoder

    return int(AudioDecoder(str(path)).metadata.num_channels or 1)


def load_audio(path: str | Path, sample_rate: int, stereo: bool = False):
    """
    Decode ``path`` (audio or video) to a float32 tensor ``[channels, samples]``
    at ``sample_rate``. Mono unless ``stereo`` and the source has 2+ channels
    (surround is downmixed to stereo by FFmpeg).
    """
    import torch

    channels = 1
    if stereo:
        try:
            channels = 2 if probe_channels(path) >= 2 else 1
        except Exception:  # noqa: BLE001
            channels = 1
    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(str(path), sample_rate=sample_rate, num_channels=channels)
        wav = decoder.get_all_samples().data
    except Exception as torchcodec_error:  # noqa: BLE001
        wav = _load_with_ffmpeg(path, sample_rate, channels, torchcodec_error)
    wav = wav.float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[-1] == 0:
        raise ValueError("The file contains no audio samples.")
    return torch.nan_to_num(wav).contiguous()


def _load_with_ffmpeg(path, sample_rate: int, channels: int, original_error: Exception):
    """Fallback decoder through the ffmpeg CLI (raw float32 PCM on stdout)."""
    import torch

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"Could not decode {Path(path).name}: {original_error}") from original_error
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn",
        "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        message = proc.stderr.decode(errors="replace").strip() or str(original_error)
        raise RuntimeError(f"Could not decode {Path(path).name}: {message}")
    data = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, channels).T.copy()
    return torch.from_numpy(data)


def save_audio(path: str | Path, wav, sample_rate: int) -> None:
    """Write ``[channels, samples]`` as 32-bit float WAV (lossless, no clipping)."""
    import soundfile as sf

    data = wav.detach().float().cpu().numpy().T
    sf.write(str(path), data, sample_rate, subtype="FLOAT")


def export_audio(src: str | Path, dst: str | Path, fmt: str) -> Path:
    """Convert a working WAV to a download format (24-bit WAV/FLAC or 320k MP3)."""
    import soundfile as sf

    src, dst = Path(src), Path(dst)
    if fmt not in AUDIO_EXPORT_FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    if fmt in ("wav", "flac"):
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        np.clip(data, -1.0, 1.0, out=data)
        sf.write(str(tmp), data, sr, subtype="PCM_24", format=fmt.upper())
    else:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg is required for MP3 export.")
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
               "-c:a", "libmp3lame", "-b:a", "320k", str(tmp)]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace").strip() or "MP3 export failed")
    tmp.replace(dst)
    return dst


def merge_video_audio(video: str | Path, audio: str | Path, dst: str | Path) -> Path:
    """Replace the audio track of ``video`` without re-encoding the picture."""
    video, audio, dst = Path(video), Path(audio), Path(dst)
    if dst.exists() and dst.stat().st_mtime >= max(video.stat().st_mtime, audio.stat().st_mtime):
        return dst
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    # WebM only allows Opus/Vorbis audio; everything else gets AAC.
    codec = ["-c:a", "libopus", "-b:a", "192k"] if video.suffix.lower() == ".webm" \
        else ["-c:a", "aac", "-b:a", "256k"]
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-i", str(audio),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *codec, "-shortest", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip() or "FFmpeg merge failed")
    tmp.replace(dst)
    return dst


def safe_filename_part(text: Optional[str], max_len: int = 48, fallback: str = "audio") -> str:
    """Make ``text`` safe for file names on Windows/macOS/Linux."""
    text = _UNSAFE_CHARS.sub(" ", text or "").strip().strip(".")
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text or fallback


def stem_filename(original_name: Optional[str], description: str, stem: str, ext: str) -> str:
    """
    ``song.mp3`` + ``vocals`` -> ``song [vocals].wav`` / ``song [no vocals].wav``
    / ``song [original].wav``.
    """
    base = safe_filename_part(Path(original_name).stem if original_name else None)
    desc = safe_filename_part(description, max_len=40, fallback="sound")
    tag = {"ghost": desc, "clean": f"no {desc}", "original": "original"}.get(stem, stem)
    return f"{base} [{tag}].{ext}"
