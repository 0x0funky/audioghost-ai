"""
Sessions: one uploaded file, the sounds separated from it, and what is left.

Each session is a folder in SESSION_DIR:

    session.json            metadata (atomic writes, guarded by a Redis lock)
    source.<ext>            the upload as-is (videos: used for preview/export)
    original.wav            its audio at 48 kHz, mono or stereo
    t<id>.wav               one per take of a separated sound (32-bit float)
    t<id>.without.wav       the model's own "everything but this sound"
    rest.<rev>.wav          "everything else": every finished sound removed (two sounds or more)
    *.peaks.json            waveform overviews for the UI
    exports/                files built for downloads

A sound can have several takes (the same description run again: with another
seed, with hints, or on another part of the file). One finished take per sound
is the active one: it is what plays and what exports use. A take can cover only
a part of the file; outside it the take is silent and its "without" is the
original itself. Such a take stores only its part (its two files are as long as
the part). Whatever reads them, here or over HTTP, sees them at full length
through a ``Window``.

Everything played back is model output. SAM-Audio regenerates what it
separates, so its output does not line up with the original sample for sample,
and subtracting it leaves most of the sound in place. Instead "everything else"
is a chain: each new sound is removed by the model from what the earlier ones
left. With a single sound it is simply that sound's active "without" file.
Playback picks the best source for the mix (``mix_plan``): the original
when nothing is changed, the model's "without" when one sound is changed, the
chain plus the tracks otherwise. Files are never rewritten in place, only
replaced by new names, so a browser streaming one is never cut off (and Windows
never refuses a replace because the file is open). A name is never used twice
in a session either: the API serves these files as immutable, so a returning
name would play its old audio from the browser's cache. That is why the
"everything else" revision (``rest_rev``) only goes up, even after every track
has been removed.

Audio math streams through the files in blocks, so hour-long recordings don't
need gigabytes of RAM.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import struct
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np

from settings import RETENTION_HOURS, SESSION_DIR
from workers import queue_state

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48_000
SCHEMA_VERSION = 2
BLOCK_FRAMES = 10 * SAMPLE_RATE
MAX_GAIN = 4.0  # +12 dB
ACTIVE = ("queued", "running")
#: A rebuild of "everything else" for an hours-long file can take minutes.
REST_LOCK_SECONDS = 1800
CLEANUP_EVERY_SECONDS = 600
#: Neighbouring pieces of a mix that changes over time crossfade over this long.
CROSSFADE_SECONDS = 0.03
#: Older versions of "everything else" kept next to the current one, to compare.
MAX_REST_ALTS = 4
INTERRUPTED = "Interrupted: the worker stopped before this sound was finished. Try again."
INTERRUPTED_REBUILD = "Interrupted: the worker stopped before everything else was rebuilt. Try again."

_SESSION_ID = re.compile(r"^[0-9a-f]{16}$")
_TRACK_ID = re.compile(r"^t[0-9a-f]{8}$")
#: Files the API may serve from a session folder.
SERVABLE = re.compile(
    r"^(original\.wav|original\.peaks\.json|source\.[a-z0-9]{1,5}"
    r"|t[0-9a-f]{8}\.wav|t[0-9a-f]{8}\.without\.wav|t[0-9a-f]{8}\.peaks\.json|t[0-9a-f]{8}\.without\.peaks\.json"
    r"|rest\.\d+\.wav|rest\.\d+\.peaks\.json)$"
)


class SessionNotFound(KeyError):
    pass


class TrackNotFound(KeyError):
    pass


_REST_FILE = re.compile(r"^rest\.\d+\.wav$")
_TAKE_FILE = re.compile(r"^(t[0-9a-f]{8})(\.without)?\.wav$")


# --------------------------------------------------------------------------- #
# Paths and persistence
# --------------------------------------------------------------------------- #


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


def new_track_id() -> str:
    return "t" + uuid.uuid4().hex[:8]


def is_track_id(value: str) -> bool:
    return bool(_TRACK_ID.match(value or ""))


def session_dir(sid: str) -> Path:
    if not _SESSION_ID.match(sid or ""):
        raise SessionNotFound(sid)
    return SESSION_DIR / sid


def _replace(tmp: Path, dst: Path) -> None:
    # Windows refuses to replace a file another process has open for a moment.
    for attempt in range(40):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.025)


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as err:  # open elsewhere (Windows); cleanup retries later
        logger.debug("could not remove %s: %s", path, err)


def load(sid: str) -> dict:
    path = session_dir(sid) / "session.json"
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise SessionNotFound(sid) from None
        except (PermissionError, json.JSONDecodeError):
            if attempt == 19:
                raise
            time.sleep(0.025)
    raise SessionNotFound(sid)  # pragma: no cover


def _write(sid: str, data: dict) -> None:
    folder = session_dir(sid)
    tmp = folder / f"session.{uuid.uuid4().hex[:6]}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    _replace(tmp, folder / "session.json")


def update(sid: str, change: Callable[[dict], None]) -> dict:
    """Read-modify-write ``session.json`` under the session lock."""
    session_dir(sid)
    with queue_state.redis_lock(f"session:{sid}"):
        data = load(sid)
        change(data)
        data["updated_at"] = time.time()
        _write(sid, data)
        return data


def find_track(data: dict, tid: str) -> Optional[dict]:
    return next((t for t in data["tracks"] if t["id"] == tid), None)


# --------------------------------------------------------------------------- #
# Sounds and takes
# --------------------------------------------------------------------------- #


def sound_of(track: dict) -> str:
    """The sound a track is a take of. Tracks made before v1.4 are their own sound."""
    return track.get("sound") or track["id"]


def is_done(track: dict) -> bool:
    return track["status"] == "done" and bool(track.get("file"))


def takes_by_sound(data: dict) -> Dict[str, List[dict]]:
    """Each sound's finished takes, sounds in the order they were added."""
    out: Dict[str, List[dict]] = {}
    for track in data["tracks"]:
        takes = out.setdefault(sound_of(track), [])
        if is_done(track):
            takes.append(track)
    return {sound: takes for sound, takes in out.items() if takes}


def _pick(takes: Sequence[dict]) -> dict:
    """Of a sound's finished takes, the one that plays: the one marked active, else the newest."""
    marked = [t for t in takes if t.get("active", True)]
    return marked[-1] if marked else max(takes, key=lambda t: t.get("finished_at") or 0)


def active_takes(data: dict) -> List[dict]:
    """The finished take that plays for each sound."""
    return [_pick(takes) for takes in takes_by_sound(data).values()]


def _settle(data: dict) -> None:
    """Exactly one finished take per sound is marked active."""
    for takes in takes_by_sound(data).values():
        keep = _pick(takes)
        for take in takes:
            take["active"] = take is keep


def same_range(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a[0] - b[0]) < 1e-3 and abs(a[1] - b[1]) < 1e-3


def chain_need(data: dict, track: dict) -> str:
    """
    What a take that is about to finish means for "everything else":

    * ``"alias"``: it is the only sound, so "everything else" is its own "without";
    * ``"step"``: its sound is new to the chain: one more chain step;
    * ``"none"``: another take of a sound already in the chain, over the same range;
    * ``"rebuild"``: another take of a sound in the chain, over a different range.
    """
    rest = data.get("rest")
    sound = sound_of(track)
    alive = takes_by_sound(data)
    chain = [s for s in (rest or {}).get("tracks", []) if s in alive]
    if not [s for s in chain if s != sound]:
        return "alias"
    if sound not in chain:
        return "step"
    made_for = (rest.get("ranges") or {}).get(sound)
    return "none" if same_range(made_for, track.get("range")) else "rebuild"


def _alias(data: dict, take: dict) -> dict:
    """"Everything else" of a single sound: that take's own "without" file (no copy)."""
    return {"rev": next_rest_rev(data), "file": take["without"], "tracks": [sound_of(take)],
            "ranges": {sound_of(take): take.get("range")}, "seed": (take.get("settings") or {}).get("seed"),
            "created_at": time.time()}


def _drop_rest_files(folder: Path, *rests: Optional[dict]) -> None:
    """Delete chain files. A single sound's "everything else" is a take's "without": that stays."""
    for rest in rests:
        if rest and _REST_FILE.match(rest.get("file") or ""):
            _remove(folder / rest["file"])
            _remove(folder / rest["file"].replace(".wav", ".peaks.json"))


# --------------------------------------------------------------------------- #
# Audio helpers (numpy + soundfile, streamed in blocks)
# --------------------------------------------------------------------------- #


def _sf():
    import soundfile

    return soundfile


def _fit_channels(block: np.ndarray, channels: int) -> np.ndarray:
    if block.shape[1] == channels:
        return block
    if block.shape[1] == 1:
        return np.repeat(block, channels, axis=1)
    return block.mean(axis=1, keepdims=True)


def _read_block(f, frames: int, channels: int) -> np.ndarray:
    block = f.read(frames, dtype="float32", always_2d=True)
    if block.shape[0] < frames:  # shorter file: pad with silence
        block = np.pad(block, ((0, frames - block.shape[0]), (0, 0)))
    return _fit_channels(block, channels)


@dataclass(frozen=True)
class Window:
    """
    The file of a take that covers a part of the recording, seen at full length:
    the stored part from frame ``start`` on, and outside it silence (the track)
    or the audio of ``outside`` (its "without": the original).
    """

    path: Path
    start: int
    frames: int
    outside: Optional[Path] = None

    @property
    def name(self) -> str:
        return self.path.name


#: What the mixes read: a file, or a part-only take's file seen at full length.
Source = Union[Path, Window]


class _Reader:
    """Blocks of a ``Source`` from any position, padded with silence past its end."""

    def __init__(self, source: Source):
        sf = _sf()
        self.window = source if isinstance(source, Window) else None
        self.file = sf.SoundFile(str(self.window.path if self.window else source))
        self.outside = sf.SoundFile(str(self.window.outside)) if self.window and self.window.outside else None
        self.frames = self.window.frames if self.window else self.file.frames
        self.samplerate = self.file.samplerate
        self.channels = (self.outside or self.file).channels

    @staticmethod
    def _read(f, start: int, n: int, channels: int) -> np.ndarray:
        if n <= 0 or start >= f.frames:
            return np.zeros((max(0, n), channels), dtype=np.float32)
        if f.tell() != start:
            f.seek(start)
        return _read_block(f, n, channels)

    def read(self, position: int, n: int, channels: int) -> np.ndarray:
        window = self.window
        if window is None:
            return self._read(self.file, position, n, channels)
        if self.outside is not None:
            block = self._read(self.outside, position, n, channels)
        else:
            block = np.zeros((n, channels), dtype=np.float32)
        lo = max(position, window.start)
        hi = min(position + n, window.start + self.file.frames)
        if hi > lo:
            block[lo - position:hi - position] = self._read(self.file, lo - window.start, hi - lo, channels)
        return block

    def close(self) -> None:
        self.file.close()
        if self.outside is not None:
            self.outside.close()


def peaks_from_file(source: Source, bins: Optional[int] = None) -> dict:
    """Max |sample| per bin, all channels combined, as a waveform overview."""
    reader = _Reader(source)
    try:
        frames = reader.frames
        duration = frames / reader.samplerate
        if bins is None:
            bins = int(min(6000, max(400, duration * 25)))
        bins = max(1, min(bins, frames or 1))
        per_bin = max(1, math.ceil(frames / bins))
        out = np.zeros(bins, dtype=np.float32)
        position = 0
        step = BLOCK_FRAMES - BLOCK_FRAMES % per_bin or per_bin
        while position < frames:
            block = reader.read(position, min(step, frames - position), reader.channels)
            level = np.abs(block).max(axis=1)
            first = position // per_bin
            count = math.ceil(len(level) / per_bin)
            padded = np.pad(level, (0, count * per_bin - len(level)))
            maxima = padded.reshape(count, per_bin).max(axis=1)
            end = min(bins, first + count)
            out[first:end] = np.maximum(out[first:end], maxima[: end - first])
            position += len(block)
    finally:
        reader.close()
    return {"duration": round(duration, 3), "peaks": [round(float(v), 3) for v in np.clip(out, 0, 1)]}


def write_peaks(audio: Source, peaks_path: Path) -> None:
    tmp = peaks_path.with_name(peaks_path.name + ".tmp")
    tmp.write_text(json.dumps(peaks_from_file(audio)), encoding="utf-8")
    _replace(tmp, peaks_path)


def _combine(output: Path, sources: Sequence[tuple], frames: int, channels: int) -> None:
    """
    ``output`` = sum of ``gain * source`` over ``(source, gain)`` pairs, streamed.
    Mono sources are spread to both channels of a stereo output.
    """
    sf = _sf()
    tmp = output.with_name(output.stem + f".{uuid.uuid4().hex[:6]}.tmp.wav")
    readers = [(_Reader(source), float(gain)) for source, gain in sources]
    try:
        with sf.SoundFile(str(tmp), "w", SAMPLE_RATE, channels, subtype="FLOAT") as out:
            done = 0
            while done < frames:
                n = min(BLOCK_FRAMES, frames - done)
                acc = np.zeros((n, channels), dtype=np.float32)
                for reader, gain in readers:
                    acc += gain * reader.read(done, n, channels)
                out.write(acc)
                done += n
    finally:
        for reader, _ in readers:
            reader.close()
    _replace(tmp, output)


def part_frames(rng, total: int) -> tuple:
    """``(start, end)`` in frames for a take's range in seconds (the whole file when it has none)."""
    if not rng:
        return 0, total
    start = max(0, min(total, int(round(float(rng[0]) * SAMPLE_RATE))))
    end = max(start, min(total, int(round(float(rng[1]) * SAMPLE_RATE))))
    return start, end


def audio_source(data: dict, name: str) -> Source:
    """
    A session's audio file as the mixes read it: the file itself, or a ``Window``
    when it is the track or the "without" of a take that stores only its part.
    """
    folder = session_dir(data["id"])
    match = _TAKE_FILE.match(name or "")
    track = find_track(data, match.group(1)) if match else None
    if track is None or not track.get("part"):
        return folder / name
    outside = folder / "original.wav" if match.group(2) else None
    return Window(folder / name, int(track["part"][0]), int(data["source"]["frames"]), outside)


def source_channels(source: Source) -> int:
    reader = _Reader(source)
    try:
        return reader.channels
    finally:
        reader.close()


def read_tensor(data: dict, name: str, stereo: bool):
    """A session's audio file at full length, as a float32 tensor ``[channels, samples]`` (mono unless ``stereo``)."""
    import torch

    reader = _Reader(audio_source(data, name))
    try:
        channels = 2 if stereo and reader.channels >= 2 else 1
        out = np.empty((reader.frames, channels), dtype=np.float32)
        position = 0
        while position < reader.frames:
            n = min(BLOCK_FRAMES, reader.frames - position)
            out[position:position + n] = reader.read(position, n, channels)
            position += n
    finally:
        reader.close()
    return torch.from_numpy(np.ascontiguousarray(out.T))


def wav_header(frames: int, channels: int) -> bytes:
    """The header of a 32-bit float WAV file with this many frames."""
    size = min(frames * channels * 4, 0xFFFFFFFF - 64)
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 3, channels, SAMPLE_RATE, SAMPLE_RATE * channels * 4, channels * 4, 32)
    fact = struct.pack("<4sII", b"fact", 4, min(frames, 0xFFFFFFFF))
    body = b"WAVE" + fmt + fact + struct.pack("<4sI", b"data", size)
    return struct.pack("<4sI", b"RIFF", len(body) + size) + body


def window_size(source: Window) -> int:
    """Bytes of the WAV file a ``Window`` stands for."""
    channels = source_channels(source)
    return len(wav_header(source.frames, channels)) + source.frames * channels * 4


def window_bytes(source: Window, first: int, last: int) -> Iterator[bytes]:
    """Bytes ``first`` to ``last`` (inclusive) of the WAV file a ``Window`` stands for: what the browser streams."""
    reader = _Reader(source)
    try:
        channels = reader.channels
        header = wav_header(source.frames, channels)
        frame_bytes = channels * 4
        position = first
        if position < len(header):
            piece = header[position:last + 1]
            position += len(piece)
            yield piece
        while position <= last:
            offset = position - len(header)
            frame, skip = divmod(offset, frame_bytes)
            n = min(65536, source.frames - frame)
            if n <= 0:
                break
            data = reader.read(frame, n, channels).astype("<f4").tobytes()[skip:][: last + 1 - position]
            position += len(data)
            yield data
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #


def create(upload: Path, filename: str, is_video: bool) -> dict:
    """Move ``upload`` into a new session and decode its audio."""
    from workers import audio_io

    sid = new_session_id()
    folder = session_dir(sid)
    folder.mkdir(parents=True)
    try:
        ext = (Path(filename).suffix.lower() or ".bin")[:6]
        if not re.match(r"^\.[a-z0-9]{1,5}$", ext):
            ext = ".bin"
        source = folder / f"source{ext}"
        shutil.move(str(upload), str(source))
        wav = audio_io.load_audio(source, SAMPLE_RATE, stereo=True)
        audio_io.save_audio(folder / "original.wav", wav, SAMPLE_RATE)
        write_peaks(folder / "original.wav", folder / "original.peaks.json")
        now = time.time()
        data = {
            "id": sid,
            "version": SCHEMA_VERSION,
            "created_at": now,
            "updated_at": now,
            "source": {
                "filename": Path(filename).name or f"upload{ext}",
                "file": source.name,
                "is_video": bool(is_video),
                "duration": round(wav.shape[-1] / SAMPLE_RATE, 3),
                "frames": int(wav.shape[-1]),
                "sample_rate": SAMPLE_RATE,
                "channels": int(wav.shape[0]),
            },
            "tracks": [],
            "rest": None,
        }
        _write(sid, data)
        return data
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def list_sessions(limit: int = 50) -> List[dict]:
    out = []
    if not SESSION_DIR.exists():
        return out
    for folder in SESSION_DIR.iterdir():
        if not (folder.is_dir() and _SESSION_ID.match(folder.name)):
            continue
        try:
            data = load(folder.name)
        except Exception:  # noqa: BLE001 - half-written or foreign folder
            continue
        out.append(summary(data))
    out.sort(key=lambda s: s["updated_at"], reverse=True)
    return out[:limit]


def folder_size(sid: str) -> int:
    """Bytes a session takes on disk."""
    total = 0
    try:
        for root, _dirs, files in os.walk(session_dir(sid)):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    except SessionNotFound:
        pass
    return total


def summary(data: dict) -> dict:
    src = data["source"]
    sounds: Dict[str, dict] = {}
    for track in data["tracks"]:  # one line per sound: a take in progress, else the one that plays
        current = sounds.get(sound_of(track))
        if current is None or track["status"] in ACTIVE:
            sounds[sound_of(track)] = {"prompt": track["prompt"], "status": track["status"]}
    for take in active_takes(data):
        if sounds[sound_of(take)]["status"] not in ACTIVE:
            sounds[sound_of(take)] = {"prompt": take["prompt"], "status": "done"}
    return {
        "id": data["id"],
        "filename": src["filename"],
        "is_video": src["is_video"],
        "duration": src["duration"],
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "expires_at": expires_at(data),
        "size_bytes": folder_size(data["id"]),
        "tracks": list(sounds.values()),
    }


def expires_at(data: dict) -> Optional[float]:
    if RETENTION_HOURS <= 0:
        return None
    return data["updated_at"] + RETENTION_HOURS * 3600


def delete(sid: str) -> None:
    folder = session_dir(sid)
    try:
        data = load(sid)
        for job in {t.get("job_id") for t in data["tracks"] if t["status"] in ACTIVE}:
            if job:
                queue_state.request_cancel(job)
    except SessionNotFound:
        pass
    shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():  # a file still open somewhere; cleanup retries later
        (folder / "session.json").unlink(missing_ok=True)


def cleanup(max_age_hours: float) -> int:
    """
    Delete sessions untouched for ``max_age_hours`` (and broken leftovers).
    A running job updates its session every second or so, so a session this
    old with a track still queued or running was left behind by a worker that
    stopped: it goes too.
    """
    if max_age_hours <= 0 or not SESSION_DIR.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for folder in SESSION_DIR.iterdir():
        if not (folder.is_dir() and _SESSION_ID.match(folder.name)):
            continue
        try:
            data = load(folder.name)
            stale = data["updated_at"] < cutoff
        except SessionNotFound:
            stale = folder.stat().st_mtime < cutoff
        except Exception:  # noqa: BLE001
            continue
        if stale:
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1
    return removed


_last_cleanup = {"time": None}


def cleanup_if_due(max_age_hours: float) -> int:
    """``cleanup``, at most every 10 minutes (the API calls this on uploads and listings)."""
    now = time.monotonic()
    last = _last_cleanup["time"]
    if last is not None and now - last < CLEANUP_EVERY_SECONDS:
        return 0
    _last_cleanup["time"] = now
    return cleanup(max_age_hours)


def recover_interrupted(dead_jobs: Iterable[str] = (), all_active: bool = False) -> int:
    """
    Fail the tracks of jobs that can no longer finish: every track of a job
    that was running when the worker stopped. Celery acknowledges a job when it
    starts, so a restarted worker never resumes it, and its tracks would stay
    "running" forever. ``dead_jobs``: jobs known to have been running.
    ``all_active``: fail queued tracks too (the desktop app's queue lives in
    memory, so nothing survives a restart). Called when the worker starts;
    returns how many tracks were failed.
    """
    if not SESSION_DIR.exists():
        return 0
    failed = 0
    for folder in SESSION_DIR.iterdir():
        if not (folder.is_dir() and _SESSION_ID.match(folder.name)):
            continue
        try:
            data = load(folder.name)
        except Exception:  # noqa: BLE001
            continue
        jobs = set(dead_jobs) | {t.get("job_id") for t in data["tracks"] if t["status"] == "running"}
        jobs.discard(None)

        def dead(track, jobs=jobs):
            return track["status"] in ACTIVE and (all_active or track.get("job_id") in jobs)

        job = data.get("rest_job") or {}
        dead_rebuild = job.get("status") in ACTIVE and (
            all_active or job.get("status") == "running" or job.get("job_id") in jobs)
        if not any(dead(t) for t in data["tracks"]) and not dead_rebuild:
            continue
        count = {"n": 0}

        def change(d, dead=dead, count=count, dead_rebuild=dead_rebuild):
            now = time.time()
            for track in d["tracks"]:
                if dead(track):
                    track.update(status="failed", message="", error=INTERRUPTED, finished_at=now)
                    count["n"] += 1
            if dead_rebuild and d.get("rest_job"):
                d["rest_job"].update(status="failed", message="", error=INTERRUPTED_REBUILD)

        try:
            update(folder.name, change)
        except Exception as err:  # noqa: BLE001
            logger.warning("Could not update session %s: %s", folder.name, err)
            continue
        failed += count["n"]
    return failed


# --------------------------------------------------------------------------- #
# Tracks
# --------------------------------------------------------------------------- #


def add_tracks(sid: str, prompts: Sequence[dict], settings: dict, job_id: str) -> List[str]:
    """
    Queue one track per prompt; returns their ids. A prompt is ``{"text", "anchors"}`` plus, optionally:
    ``"sound"`` (the sound this is another take of; a new sound otherwise), ``"range"`` (``[start, end]``
    in seconds: only that part of the file is separated) and ``"seed"`` (replaces the settings' seed).
    """
    ids = [new_track_id() for _ in prompts]
    now = time.time()

    def change(data):
        known = {sound_of(t) for t in data["tracks"]}
        for tid, prompt in zip(ids, prompts):
            sound = prompt.get("sound") if prompt.get("sound") in known else tid
            own = dict(settings)
            if prompt.get("seed") is not None:
                own["seed"] = prompt["seed"]
            data["tracks"].append({
                "id": tid,
                "sound": sound,
                "active": False,  # a take plays once it is finished (see finish_track)
                "prompt": prompt["text"],
                "anchors": [list(a) for a in prompt.get("anchors") or []],
                "range": list(prompt["range"]) if prompt.get("range") else None,
                "status": "queued",
                "progress": 0.0,
                "message": "",
                "job_id": job_id,
                "settings": own,
                "file": None,
                "elapsed": None,
                "error": None,
                "created_at": now,
                "finished_at": None,
            })

    update(sid, change)
    return ids


def set_track(sid: str, tid: str, **fields) -> Optional[dict]:
    """Update a track's fields; returns the track, or None if it was removed."""
    found = {}

    def change(data):
        track = find_track(data, tid)
        if track is not None:
            track.update(fields)
            found["track"] = dict(track)

    update(sid, change)
    return found.get("track")


def drop_track(sid: str, tid: str) -> None:
    """
    Remove a track entry and its files (the caller rebuilds "everything else").
    Holds the rest lock, so a rebuild never sees a track's file vanish halfway.
    """
    folder = session_dir(sid)

    def change(data):
        data["tracks"] = [t for t in data["tracks"] if t["id"] != tid]

    with queue_state.redis_lock(f"rest:{sid}", timeout=REST_LOCK_SECONDS):
        update(sid, change)
        _remove(folder / f"{tid}.wav")
        _remove(folder / f"{tid}.peaks.json")
        _remove(folder / f"{tid}.without.wav")
        _remove(folder / f"{tid}.without.peaks.json")


def _match_stereo_image(original: Path, wav, start: int = 0):
    """
    A sound separated from the mono downmix of a stereo file comes out mono.
    Give it the left/right balance it has in the original: a least-squares gain
    per channel, scaled so the two average 1 (the mono sum stays exactly what
    the model produced). Otherwise "everything else" would keep half of a sound
    panned to one side, plus an inverted copy on the other side. Centred sounds
    stay mono.
    """
    import torch

    sf = _sf()
    mono = wav[0].detach().float().cpu().numpy()
    dots = np.zeros(2)
    energy = 0.0
    with sf.SoundFile(str(original)) as f:
        f.seek(min(start, f.frames))
        position = 0
        while position < len(mono):
            n = min(BLOCK_FRAMES, len(mono) - position)
            block = _read_block(f, n, 2).astype(np.float64)
            segment = mono[position:position + n].astype(np.float64)
            dots += block.T @ segment
            energy += float(segment @ segment)
            position += n
    if energy <= 1e-9:
        return wav
    gains = np.clip(dots / energy, 0.0, None)
    if gains.sum() <= 1e-9:
        return wav
    gains = 2.0 * gains / gains.sum()
    if abs(gains[0] - 1.0) < 0.05:
        return wav
    return torch.from_numpy(np.stack([gains[0] * mono, gains[1] * mono]).astype(np.float32))


def next_rest_rev(data: dict) -> int:
    """
    The next "everything else" revision, also recorded as ``data["rest_rev"]``. It
    only goes up within a session: ``rest.<rev>.wav`` is served as immutable, so
    after every track is removed and "everything else" starts over, ``rest.1.wav``
    must not come back with new audio (the browser would keep playing the old one).
    """
    rev = max(int(data.get("rest_rev") or 0), int((data.get("rest") or {}).get("rev") or 0)) + 1
    data["rest_rev"] = rev
    return rev


def finish_track(sid: str, tid: str, target, without, sample_rate: int, elapsed: float,
                 rest=None, chain_from: Optional[dict] = None, seed: Optional[int] = None,
                 part: Optional[Sequence[int]] = None) -> str:
    """
    Store a finished take: the track and the model's own "without this sound". With ``part``
    (``(start, end)`` in frames) the take covers only that part of the file and ``target`` and
    ``without`` hold just that part (see ``Window``); ``rest`` is always the whole file. The take becomes the
    one that plays for its sound in the same write that updates "everything else", so the UI never
    sees one without the other. What happens to "everything else" (see ``chain_need``):

    * the only sound: it is this take's own "without" file;
    * a sound new to the chain: ``rest``, the chain step's output. ``chain_from`` is the "everything
      else" that step started from: tracks removed from it meanwhile are put back, as a removal does;
    * another take of a sound already in the chain: unchanged.

    Returns ``"done"``; ``"rebuild"`` when the chain no longer matches the takes (the sound is missing
    from it, or was made over another range) and the caller has to rebuild it; ``"gone"`` if the track
    was removed meanwhile.
    """
    from workers import audio_io

    folder = session_dir(sid)
    data = load(sid)
    if find_track(data, tid) is None:
        return "gone"
    frames = data["source"]["frames"]
    if part is not None and (part[0] <= 0 and part[1] >= frames):
        part = None  # the "part" is the whole file
    if data["source"]["channels"] == 2 and target.shape[0] == 1:
        target = _match_stereo_image(folder / "original.wav", target, part[0] if part else 0)
    path = folder / f"{tid}.wav"
    peaks = folder / f"{tid}.peaks.json"
    without_path = folder / f"{tid}.without.wav"
    without_peaks = folder / f"{tid}.without.peaks.json"
    audio_io.save_audio(path, target, sample_rate)
    audio_io.save_audio(without_path, without, sample_rate)
    if part is None:
        write_peaks(path, peaks)
        write_peaks(without_path, without_peaks)
    else:  # the overviews show the whole file, like every other track's
        write_peaks(Window(path, int(part[0]), frames), peaks)
        write_peaks(Window(without_path, int(part[0]), frames, folder / "original.wav"), without_peaks)
    build = None
    if rest is not None:
        build = folder / f"rest-build-{uuid.uuid4().hex[:8]}.wav"
        audio_io.save_audio(build, rest, sample_rate)
    fields = dict(status="done", progress=1.0, message="", file=path.name, without=without_path.name,
                  elapsed=round(elapsed, 2), finished_at=time.time(),
                  part=[int(part[0]), int(part[1])] if part is not None else None)
    channels = data["source"]["channels"]
    state = {"build": build, "drop": []}

    def discard_take():
        for p in (path, peaks, without_path, without_peaks):
            _remove(p)

    def discard_build():
        if state["build"] is not None:
            _remove(state["build"])
            _remove(state["build"].with_suffix(".peaks.json"))

    with queue_state.redis_lock(f"rest:{sid}", timeout=REST_LOCK_SECONDS):
        current = load(sid)
        if find_track(current, tid) is None:
            discard_take()
            discard_build()
            return "gone"
        if build is not None:
            now = current.get("rest") or {}
            removed = [k for k in (chain_from or {}).get("tracks", []) if k not in now.get("tracks", [])]
            put_back = [folder / f"{k}.deleted.wav" for k in removed if (folder / f"{k}.deleted.wav").is_file()]
            if put_back:
                merged = folder / f"rest-build-{uuid.uuid4().hex[:8]}.wav"
                _combine(merged, [(build, 1.0)] + [(p, 1.0) for p in put_back], frames, channels)
                _remove(build)
                build = state["build"] = merged
            write_peaks(build, build.with_suffix(".peaks.json"))

        def change(d):
            track = find_track(d, tid)
            if track is None:
                state["gone"] = True
                return
            track.update(fields)
            if seed is not None:
                track.setdefault("settings", {})["seed"] = int(seed)
            sound = sound_of(track)
            for other in d["tracks"]:
                if sound_of(other) == sound and is_done(other):
                    other["active"] = other is track
            prev = d.get("rest")
            alive = takes_by_sound(d)
            chain = [s for s in (prev or {}).get("tracks", []) if s in alive]
            if not [s for s in chain if s != sound]:  # the only sound: its own "without"
                d["rest"] = _alias(d, track)
                state["drop"] += [prev] + list(d.get("rest_alts") or [])
                d["rest_alts"] = []
            elif sound not in chain:
                if build is None:
                    state["rebuild"] = True
                    return
                rev = next_rest_rev(d)
                _replace(build, folder / f"rest.{rev}.wav")
                _replace(build.with_suffix(".peaks.json"), folder / f"rest.{rev}.peaks.json")
                ranges = {s: (prev.get("ranges") or {}).get(s) for s in chain}
                ranges[sound] = track.get("range")
                d["rest"] = {"rev": rev, "file": f"rest.{rev}.wav", "tracks": chain + [sound], "ranges": ranges,
                             "seed": seed, "created_at": time.time()}
                state["used"] = True
                state["drop"] += [prev] + list(d.get("rest_alts") or [])
                d["rest_alts"] = []
            elif not same_range((prev.get("ranges") or {}).get(sound), track.get("range")):
                state["rebuild"] = True

        update(sid, change)
        if state.get("gone"):
            discard_take()
            discard_build()
            return "gone"
        if not state.get("used"):
            discard_build()
        _drop_rest_files(folder, *state["drop"])
    return "rebuild" if state.get("rebuild") else "done"


def remove_done_track(sid: str, tid: str) -> None:
    """
    Remove a finished take. While another take of its sound remains, "everything else" keeps that
    sound out. With the sound's last take gone, its sound goes back into "everything else": exactly
    when a single sound is left (that sound's own "without"), else as a fast approximation
    (everything else + the track).
    """
    folder = session_dir(sid)
    with queue_state.redis_lock(f"rest:{sid}", timeout=REST_LOCK_SECONDS):
        data = load(sid)
        track = find_track(data, tid)
        if track is None:
            return
        sound = sound_of(track)
        rest = data.get("rest")
        busy = any(t["status"] in ACTIVE for t in data["tracks"])
        alive = takes_by_sound({"tracks": [t for t in data["tracks"] if t["id"] != tid]})
        sound_gone = sound not in alive
        new = None
        if rest and alive:
            chain = [s for s in rest.get("tracks", []) if s in alive]
            ranges = {s: (rest.get("ranges") or {}).get(s) for s in chain}
            only = _pick(next(iter(alive.values()))) if len(alive) == 1 else None
            if only is not None and only.get("without"):
                new = rest if rest.get("file") == only["without"] else _alias(data, only)
            elif sound_gone and sound in rest.get("tracks", []) and track.get("file"):
                rev = next_rest_rev(data)
                out = folder / f"rest.{rev}.wav"
                _combine(out, [(audio_source(data, rest["file"]), 1.0), (audio_source(data, track["file"]), 1.0)],
                         data["source"]["frames"], data["source"]["channels"])
                write_peaks(out, folder / f"rest.{rev}.peaks.json")
                new = {"rev": rev, "file": out.name, "tracks": chain, "ranges": ranges,
                       "seed": rest.get("seed"), "created_at": time.time()}
            else:
                new = dict(rest, tracks=chain, ranges=ranges)
        changed = bool(rest) and (new is None or new["file"] != rest["file"])
        drop = []

        def change(d):
            d["tracks"] = [t for t in d["tracks"] if t["id"] != tid]
            d["rest"] = new
            _settle(d)
            if changed or sound_gone:  # the kept versions were made for another set of sounds
                drop.extend(d.get("rest_alts") or [])
                d["rest_alts"] = []
            # Keep counting even when "everything else" goes away (see next_rest_rev).
            d["rest_rev"] = max(int(d.get("rest_rev") or 0), int(data.get("rest_rev") or 0),
                                int((rest or {}).get("rev") or 0), int((new or {}).get("rev") or 0))

        update(sid, change)
        if changed:
            _drop_rest_files(folder, rest)
        _drop_rest_files(folder, *drop)
        target = folder / f"{tid}.wav"
        if busy and sound_gone and target.is_file():
            kept = folder / f"{sound}.deleted.wav"  # a running chain step puts it back
            if track.get("part"):  # at full length, like the chain it goes back into
                source = audio_source(data, track["file"])
                _combine(kept, [(source, 1.0)], data["source"]["frames"], source_channels(source))
                _remove(target)
            else:
                _replace(target, kept)
        else:
            _remove(target)
        _remove(folder / f"{tid}.without.wav")
        _remove(folder / f"{tid}.peaks.json")
        _remove(folder / f"{tid}.without.peaks.json")


def activate_take(sid: str, tid: str) -> None:
    """Make a finished take the one that plays for its sound."""
    folder = session_dir(sid)
    with queue_state.redis_lock(f"rest:{sid}", timeout=REST_LOCK_SECONDS):
        drop = []

        def change(d):
            track = find_track(d, tid)
            if track is None:
                raise TrackNotFound(tid)
            if not is_done(track):
                raise ValueError("Only a finished take can be chosen.")
            sound = sound_of(track)
            for other in d["tracks"]:
                if sound_of(other) == sound and is_done(other):
                    other["active"] = other is track
            rest = d.get("rest")
            alive = takes_by_sound(d)
            if rest and len(alive) == 1 and track.get("without") and rest.get("file") != track["without"]:
                drop.append(rest)  # the only sound: "everything else" follows the take
                d["rest"] = _alias(d, track)

        update(sid, change)
        _drop_rest_files(folder, *drop)


def rest_is_stale(data: dict) -> bool:
    """Whether "everything else" was made for other sounds or ranges than the takes that play now."""
    rest = data.get("rest")
    if not rest:
        return False
    takes = {sound_of(t): t for t in active_takes(data)}
    if set(rest.get("tracks", [])) != set(takes):
        return True
    ranges = rest.get("ranges") or {}
    return any(not same_range(ranges.get(sound), take.get("range")) for sound, take in takes.items())


def publish_rest(sid: str, audio, sample_rate: int, sounds: Sequence[str], seed: Optional[int],
                 keep_previous: bool = True) -> bool:
    """
    Publish a rebuilt "everything else" (the chain run over ``sounds`` in this order). The version it
    replaces is kept to compare with, when it was made for the same sounds and ranges. Returns False,
    and publishes nothing, if the sounds that play changed while the chain was running.
    """
    from workers import audio_io

    folder = session_dir(sid)
    build = folder / f"rest-build-{uuid.uuid4().hex[:8]}.wav"
    audio_io.save_audio(build, audio, sample_rate)
    write_peaks(build, build.with_suffix(".peaks.json"))
    with queue_state.redis_lock(f"rest:{sid}", timeout=REST_LOCK_SECONDS):
        state = {"drop": []}

        def change(d):
            takes = {sound_of(t): t for t in active_takes(d)}
            if set(takes) != set(sounds) or len(sounds) < 2:
                state["stale"] = True
                return
            prev = d.get("rest")
            alts = list(d.get("rest_alts") or [])
            comparable = bool(keep_previous and prev and _REST_FILE.match(prev.get("file") or "")
                              and not rest_is_stale(d))
            rev = next_rest_rev(d)
            _replace(build, folder / f"rest.{rev}.wav")
            _replace(build.with_suffix(".peaks.json"), folder / f"rest.{rev}.peaks.json")
            if comparable:
                alts.append({k: prev.get(k) for k in ("rev", "file", "tracks", "ranges", "seed", "created_at")})
            else:
                state["drop"] += [prev] + alts
                alts = []
            state["drop"] += alts[:-MAX_REST_ALTS]
            d["rest_alts"] = alts[-MAX_REST_ALTS:]
            d["rest"] = {"rev": rev, "file": f"rest.{rev}.wav", "tracks": list(sounds),
                         "ranges": {s: takes[s].get("range") for s in sounds}, "seed": seed,
                         "created_at": time.time()}

        update(sid, change)
        if state.get("stale"):
            _remove(build)
            _remove(build.with_suffix(".peaks.json"))
            return False
        _drop_rest_files(folder, *state["drop"])
    return True


def activate_rest(sid: str, rev: int) -> None:
    """Swap "everything else" with one of its kept versions."""
    with queue_state.redis_lock(f"rest:{sid}", timeout=REST_LOCK_SECONDS):

        def change(d):
            alts = list(d.get("rest_alts") or [])
            chosen = next((a for a in alts if a.get("rev") == rev), None)
            rest = d.get("rest")
            if chosen is None or not rest:
                raise TrackNotFound(f"rest.{rev}")
            alts = [a for a in alts if a is not chosen]
            alts.append({k: rest.get(k) for k in ("rev", "file", "tracks", "ranges", "seed", "created_at")})
            alts.sort(key=lambda a: a.get("rev") or 0)
            d["rest_alts"] = alts
            d["rest"] = dict(chosen)

        update(sid, change)


def remove_rest_alt(sid: str, rev: int) -> None:
    """Delete a kept version of "everything else" (not the current one)."""
    folder = session_dir(sid)
    with queue_state.redis_lock(f"rest:{sid}", timeout=REST_LOCK_SECONDS):
        drop = []

        def change(d):
            alts = list(d.get("rest_alts") or [])
            drop.extend(a for a in alts if a.get("rev") == rev)
            d["rest_alts"] = [a for a in alts if a.get("rev") != rev]

        update(sid, change)
        _drop_rest_files(folder, *drop)


def set_rest_job(sid: str, job: Optional[dict] = None, **fields) -> None:
    """The rebuild of "everything else" that is queued or running: replace it, patch it or clear it."""

    def change(d):
        if fields:
            if d.get("rest_job"):
                d["rest_job"].update(fields)
        elif job is None:
            d.pop("rest_job", None)
        else:
            d["rest_job"] = dict(job)

    update(sid, change)


def purge_deleted(sid: str) -> None:
    """Drop audio kept for chain steps once nothing is queued or running."""
    folder = session_dir(sid)
    try:
        data = load(sid)
    except SessionNotFound:
        return
    if any(t["status"] in ACTIVE for t in data["tracks"]):
        return
    for leftover in list(folder.glob("t*.deleted.wav")) + list(folder.glob("rest-build-*")):
        _remove(leftover)


# --------------------------------------------------------------------------- #
# Mixing and exports
# --------------------------------------------------------------------------- #


def _unity(value: float) -> bool:
    return abs(value - 1.0) < 1e-3


def mix_plan(data: dict, gains: Dict[str, float]) -> List[tuple]:
    """
    ``(path, gain)`` pairs for a mix (gains: linear, per track id and ``"rest"``;
    muted or soloed-out = 0). Only model output is played, so the source depends
    on what changed. Each sound plays its active take:

    * nothing changed: the original itself
    * one track changed: the model's own "without that sound" + the track at its gain
    * more: "everything else" (the chain) + every track at its gain
    """
    folder = session_dir(data["id"])
    done = active_takes(data)
    rest = data.get("rest")
    if not done or not rest:
        return [(folder / "original.wav", 1.0)]
    g = {t["id"]: min(max(0.0, float(gains.get(t["id"], 0.0))), MAX_GAIN) for t in done}
    g_rest = min(max(0.0, float(gains.get("rest", 0.0))), MAX_GAIN)
    changed = [t for t in done if not _unity(g[t["id"]])]
    if not changed and _unity(g_rest):
        return [(folder / "original.wav", 1.0)]
    if len(changed) == 1 and _unity(g_rest) and changed[0].get("without"):
        track = changed[0]
        plan = [(audio_source(data, track["without"]), 1.0)]
        if g[track["id"]] > 0:
            plan.append((audio_source(data, track["file"]), g[track["id"]]))
        return plan
    plan = [(audio_source(data, rest["file"]), g_rest)] if g_rest > 0 else []
    return plan + [(audio_source(data, t["file"]), g[t["id"]]) for t in done if g[t["id"]] > 0]


def mix_pieces(data: dict, gains: Dict[str, float], segments: Optional[Sequence[dict]]) -> List[tuple]:
    """
    A mix whose gains change over time, as ``(start frame, end frame, sources)`` pieces that cover the
    whole file. ``segments``: ``{"start", "end", "gains"}`` in seconds; the time outside them plays
    ``gains``. Neighbours with the same sources are merged.
    """
    frames = data["source"]["frames"]
    spans = []
    for segment in sorted(segments or [], key=lambda item: float(item["start"])):
        start = max(0, min(frames, int(round(float(segment["start"]) * SAMPLE_RATE))))
        end = max(0, min(frames, int(round(float(segment["end"]) * SAMPLE_RATE))))
        if spans:
            start = max(start, spans[-1][1])
        if end > start:
            spans.append((start, end, mix_plan(data, segment.get("gains") or {})))
    base = mix_plan(data, gains)
    pieces: List[tuple] = []
    position = 0
    for start, end, sources in spans + [(frames, frames, None)]:
        if start > position:
            pieces.append((position, start, base))
        if sources is not None:
            pieces.append((start, end, sources))
        position = max(position, end)
    merged: List[tuple] = []
    for piece in pieces:
        if merged and merged[-1][2] == piece[2]:
            merged[-1] = (merged[-1][0], piece[1], piece[2])
        else:
            merged.append(piece)
    return merged or [(0, frames, base)]


def _combine_pieces(output: Path, pieces: Sequence[tuple], frames: int, channels: int) -> None:
    """
    Like ``_combine``, with other ``(path, gain)`` sources per stretch of time: ``pieces`` are
    ``(start frame, end frame, sources)``, back to back. Neighbours crossfade over
    ``CROSSFADE_SECONDS`` centred on the boundary, with gains that add up to 1.
    """
    sf = _sf()
    fade = max(2, int(CROSSFADE_SECONDS * SAMPLE_RATE))
    tmp = output.with_name(output.stem + f".{uuid.uuid4().hex[:6]}.tmp.wav")
    readers: dict = {}

    def read(source: Source, start: int, n: int) -> np.ndarray:
        reader = readers.get(source)
        if reader is None:
            reader = readers[source] = _Reader(source)
        return reader.read(start, n, channels)

    last = len(pieces) - 1
    try:
        with sf.SoundFile(str(tmp), "w", SAMPLE_RATE, channels, subtype="FLOAT") as out:
            done = 0
            while done < frames:
                n = min(BLOCK_FRAMES, frames - done)
                acc = np.zeros((n, channels), dtype=np.float32)
                t = np.arange(done, done + n, dtype=np.float64)
                for index, (start, end, sources) in enumerate(pieces):
                    lo = start - fade / 2 if index else -math.inf
                    hi = end + fade / 2 if index < last else math.inf
                    if hi <= done or lo >= done + n or not sources:
                        continue
                    envelope = np.ones(n, dtype=np.float64)
                    if index:
                        envelope = np.minimum(envelope, np.clip((t - lo) / fade, 0.0, 1.0))
                    if index < last:
                        envelope = np.minimum(envelope, np.clip((hi - t) / fade, 0.0, 1.0))
                    block = np.zeros((n, channels), dtype=np.float32)
                    for path, gain in sources:
                        block += float(gain) * read(path, done, n)
                    acc += block * envelope[:, None].astype(np.float32)
                out.write(acc)
                done += n
    finally:
        for reader in readers.values():
            reader.close()
    _replace(tmp, output)


def _render_mix(data: dict, gains: Dict[str, float], segments: Optional[Sequence[dict]], output: Path) -> None:
    frames, channels = data["source"]["frames"], data["source"]["channels"]
    if not segments:
        sources = mix_plan(data, gains)
        if not sources:
            raise ValueError("Nothing to export: every track is muted.")
        _combine(output, sources, frames, channels)
        return
    pieces = mix_pieces(data, gains, segments)
    if not any(sources for _start, _end, sources in pieces):
        raise ValueError("Nothing to export: every track is muted.")
    _combine_pieces(output, pieces, frames, channels)


def mix_label(data: dict, gains: Dict[str, float], segments: Optional[Sequence[dict]] = None) -> str:
    """``vocals`` / ``no vocals`` / ``mix``, for file names."""
    if segments:
        return "mix"
    done = active_takes(data)
    audible = {t["id"] for t in done if gains.get(t["id"], 0) > 0}
    rest_on = bool(data.get("rest")) and gains.get("rest", 0) > 0
    unity = all(abs(gains.get(k, 1.0) - 1.0) < 1e-3 for k in list(audible) + (["rest"] if rest_on else []))
    if unity and len(audible) == 1 and not rest_on:
        return next(t["prompt"] for t in done if t["id"] in audible)
    if unity and rest_on and len(done) - len(audible) == 1:
        muted = next(t["prompt"] for t in done if t["id"] not in audible)
        return f"no {muted}"
    if unity and rest_on and not audible:
        return "everything else"
    return "mix"


def _base_name(data: dict) -> str:
    from workers import audio_io

    return audio_io.safe_filename_part(Path(data["source"]["filename"]).stem)


def _tag(text: str) -> str:
    from workers import audio_io

    return audio_io.safe_filename_part(text, max_len=40, fallback="sound")


def _exports_dir(sid: str) -> Path:
    folder = session_dir(sid) / "exports"
    folder.mkdir(exist_ok=True)
    cutoff = time.time() - 600  # keep the last few minutes of downloads only
    for old in folder.iterdir():
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass
    return folder


def export_mix(sid: str, gains: Dict[str, float], fmt: str, segments: Optional[Sequence[dict]] = None) -> tuple:
    """Render the mix to ``fmt``; returns ``(path, download name)``. ``segments``: see ``mix_pieces``."""
    from workers import audio_io

    data = load(sid)
    out = _exports_dir(sid)
    stamp = uuid.uuid4().hex[:8]
    wav = out / f"mix-{stamp}.f32.wav"  # not the export name: WAV exports are 24-bit
    _render_mix(data, gains, segments, wav)
    final = audio_io.export_audio(wav, out / f"mix-{stamp}.{fmt}", fmt)
    return final, f"{_base_name(data)} [{_tag(mix_label(data, gains, segments))}].{fmt}"


def export_tracks(sid: str, keys: Iterable[str], fmt: str) -> tuple:
    """ZIP of single tracks (``keys``: track ids and/or ``"rest"``)."""
    from workers import audio_io

    data = load(sid)
    folder = session_dir(sid)
    out = _exports_dir(sid)
    base = _base_name(data)
    entries = []
    used = set()
    for key in keys:
        if key == "rest" and data.get("rest"):
            source, label = audio_source(data, data["rest"]["file"]), "everything else"
        else:
            track = find_track(data, key)
            if track is None or track["status"] != "done" or not track.get("file"):
                continue
            source, label = audio_source(data, track["file"]), track["prompt"]
        src = source
        if isinstance(source, Window):  # a take of a part: export it at full length, so it lines up
            src = out / f"{source.path.stem}.full.wav"
            _combine(src, [(source, 1.0)], data["source"]["frames"], source_channels(source))
        name = f"{base} [{_tag(label)}].{fmt}"
        suffix = 2
        while name in used:  # two tracks with the same prompt
            name = f"{base} [{_tag(label)} {suffix}].{fmt}"
            suffix += 1
        used.add(name)
        exported = audio_io.export_audio(src, out / f"{Path(source.name).stem}.{fmt}", fmt)
        entries.append((exported, name))
    if not entries:
        raise ValueError("No finished tracks to export.")
    zip_path = out / f"tracks-{uuid.uuid4().hex[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path, name in entries:
            zf.write(path, arcname=name)
    return zip_path, f"{base} [tracks].zip"


def export_video(sid: str, gains: Dict[str, float], segments: Optional[Sequence[dict]] = None) -> tuple:
    """The source video with its audio replaced by the mix."""
    from workers import audio_io

    data = load(sid)
    if not data["source"]["is_video"]:
        raise ValueError("This session's source is not a video.")
    folder = session_dir(sid)
    out = _exports_dir(sid)
    stamp = uuid.uuid4().hex[:8]
    wav = out / f"mix-{stamp}.f32.wav"  # not the export name: WAV exports are 24-bit
    _render_mix(data, gains, segments, wav)
    ext = Path(data["source"]["file"]).suffix
    video = audio_io.merge_video_audio(folder / data["source"]["file"], wav, out / f"video-{stamp}{ext}")
    return video, f"{_base_name(data)} [{_tag(mix_label(data, gains, segments))}]{ext}"
