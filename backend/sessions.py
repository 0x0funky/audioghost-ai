"""
Sessions: one uploaded file, the sounds separated from it, and what is left.

Each session is a folder in SESSION_DIR:

    session.json            metadata (atomic writes, guarded by a Redis lock)
    source.<ext>            the upload as-is (videos: used for preview/export)
    original.wav            its audio at 48 kHz, mono or stereo
    t<id>.wav               one per separated sound (32-bit float)
    t<id>.without.wav       the model's own "everything but this sound"
    rest.<rev>.wav          "everything else": every finished sound removed
    *.peaks.json            waveform overviews for the UI
    exports/                files built for downloads

Everything played back is model output. SAM-Audio regenerates what it
separates, so its output does not line up with the original sample for sample,
and subtracting it leaves most of the sound in place. Instead "everything else"
is a chain: each new sound is removed by the model from what the earlier ones
left. Playback picks the best source for the mix (``mix_plan``): the original
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
import time
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from settings import RETENTION_HOURS, SESSION_DIR
from workers import queue_state

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48_000
SCHEMA_VERSION = 1
BLOCK_FRAMES = 10 * SAMPLE_RATE
MAX_GAIN = 4.0  # +12 dB
ACTIVE = ("queued", "running")
#: A rebuild of "everything else" for an hours-long file can take minutes.
REST_LOCK_SECONDS = 1800
CLEANUP_EVERY_SECONDS = 600
INTERRUPTED = "Interrupted: the worker stopped before this sound was finished. Try again."

_SESSION_ID = re.compile(r"^[0-9a-f]{16}$")
_TRACK_ID = re.compile(r"^t[0-9a-f]{8}$")
#: Files the API may serve from a session folder.
SERVABLE = re.compile(
    r"^(original\.wav|original\.peaks\.json|source\.[a-z0-9]{1,5}"
    r"|t[0-9a-f]{8}\.wav|t[0-9a-f]{8}\.without\.wav|t[0-9a-f]{8}\.peaks\.json"
    r"|rest\.\d+\.wav|rest\.\d+\.peaks\.json)$"
)


class SessionNotFound(KeyError):
    pass


class TrackNotFound(KeyError):
    pass


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


def peaks_from_file(path: Path, bins: Optional[int] = None) -> dict:
    """Max |sample| per bin, all channels combined, as a waveform overview."""
    sf = _sf()
    with sf.SoundFile(str(path)) as f:
        frames = f.frames
        duration = frames / f.samplerate
        if bins is None:
            bins = int(min(6000, max(400, duration * 25)))
        bins = max(1, min(bins, frames or 1))
        per_bin = max(1, math.ceil(frames / bins))
        out = np.zeros(bins, dtype=np.float32)
        position = 0
        while True:
            block = f.read(BLOCK_FRAMES - BLOCK_FRAMES % per_bin or per_bin, dtype="float32", always_2d=True)
            if not len(block):
                break
            level = np.abs(block).max(axis=1)
            first = position // per_bin
            count = math.ceil(len(level) / per_bin)
            padded = np.pad(level, (0, count * per_bin - len(level)))
            maxima = padded.reshape(count, per_bin).max(axis=1)
            end = min(bins, first + count)
            out[first:end] = np.maximum(out[first:end], maxima[: end - first])
            position += len(block)
    return {"duration": round(duration, 3), "peaks": [round(float(v), 3) for v in np.clip(out, 0, 1)]}


def write_peaks(audio_path: Path, peaks_path: Path) -> None:
    tmp = peaks_path.with_name(peaks_path.name + ".tmp")
    tmp.write_text(json.dumps(peaks_from_file(audio_path)), encoding="utf-8")
    _replace(tmp, peaks_path)


def _combine(output: Path, sources: Sequence[tuple], frames: int, channels: int) -> None:
    """
    ``output`` = sum of ``gain * source`` over ``(path, gain)`` pairs, streamed.
    Mono sources are spread to both channels of a stereo output.
    """
    sf = _sf()
    tmp = output.with_name(output.stem + f".{uuid.uuid4().hex[:6]}.tmp.wav")
    handles = [(sf.SoundFile(str(p)), float(g)) for p, g in sources]
    try:
        with sf.SoundFile(str(tmp), "w", SAMPLE_RATE, channels, subtype="FLOAT") as out:
            done = 0
            while done < frames:
                n = min(BLOCK_FRAMES, frames - done)
                acc = np.zeros((n, channels), dtype=np.float32)
                for f, gain in handles:
                    acc += gain * _read_block(f, n, channels)
                out.write(acc)
                done += n
    finally:
        for f, _ in handles:
            f.close()
    _replace(tmp, output)


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


def summary(data: dict) -> dict:
    src = data["source"]
    return {
        "id": data["id"],
        "filename": src["filename"],
        "is_video": src["is_video"],
        "duration": src["duration"],
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "expires_at": expires_at(data),
        "tracks": [{"prompt": t["prompt"], "status": t["status"]} for t in data["tracks"]],
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

        if not any(dead(t) for t in data["tracks"]):
            continue
        count = {"n": 0}

        def change(d, dead=dead, count=count):
            now = time.time()
            for track in d["tracks"]:
                if dead(track):
                    track.update(status="failed", message="", error=INTERRUPTED, finished_at=now)
                    count["n"] += 1

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
    """Queue one track per prompt (``{"text", "anchors"}``); returns their ids."""
    ids = [new_track_id() for _ in prompts]
    now = time.time()

    def change(data):
        for tid, prompt in zip(ids, prompts):
            data["tracks"].append({
                "id": tid,
                "prompt": prompt["text"],
                "anchors": [list(a) for a in prompt.get("anchors") or []],
                "status": "queued",
                "progress": 0.0,
                "message": "",
                "job_id": job_id,
                "settings": dict(settings),
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


def _match_stereo_image(original: Path, wav):
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


def finish_track(sid: str, tid: str, target, without, rest, sample_rate: int, elapsed: float,
                 chain_from: Optional[dict] = None) -> bool:
    """
    Store a separated sound: the track, the model's own "without this sound" and
    the new "everything else" (``rest``: the chain step's output). The track turns
    "done" in the same write that publishes the new "everything else", so the UI
    never sees one without the other. ``chain_from`` is the "everything else" the
    chain step started from: tracks removed from it meanwhile are put back, as a
    removal does. Returns False if the track was removed meanwhile.
    """
    from workers import audio_io

    folder = session_dir(sid)
    data = load(sid)
    if find_track(data, tid) is None:
        return False
    if data["source"]["channels"] == 2 and target.shape[0] == 1:
        target = _match_stereo_image(folder / "original.wav", target)
    path = folder / f"{tid}.wav"
    peaks = folder / f"{tid}.peaks.json"
    without_path = folder / f"{tid}.without.wav"
    build = folder / f"rest-build-{uuid.uuid4().hex[:8]}.wav"
    audio_io.save_audio(path, target, sample_rate)
    write_peaks(path, peaks)
    audio_io.save_audio(without_path, without, sample_rate)
    audio_io.save_audio(build, rest, sample_rate)
    fields = dict(status="done", progress=1.0, message="", file=path.name, without=without_path.name,
                  elapsed=round(elapsed, 2), finished_at=time.time())
    frames, channels = data["source"]["frames"], data["source"]["channels"]

    def discard(*extra):
        for p in (path, peaks, without_path) + extra:
            _remove(p)

    with queue_state.redis_lock(f"rest:{sid}", timeout=REST_LOCK_SECONDS):
        current = load(sid)
        if find_track(current, tid) is None:
            discard(build)
            return False
        now = current.get("rest") or {}
        removed = [k for k in (chain_from or {}).get("tracks", []) if k not in now.get("tracks", [])]
        put_back = [folder / f"{k}.deleted.wav" for k in removed if (folder / f"{k}.deleted.wav").is_file()]
        if put_back:
            merged = folder / f"rest-build-{uuid.uuid4().hex[:8]}.wav"
            _combine(merged, [(build, 1.0)] + [(p, 1.0) for p in put_back], frames, channels)
            _remove(build)
            build = merged
        write_peaks(build, build.with_suffix(".peaks.json"))
        state = {}

        def change(d):
            track = find_track(d, tid)
            if track is None:
                state["gone"] = True
                return
            prev = d.get("rest")
            rev = next_rest_rev(d)
            _replace(build, folder / f"rest.{rev}.wav")
            _replace(build.with_suffix(".peaks.json"), folder / f"rest.{rev}.peaks.json")
            kept = [k for k in (prev or {}).get("tracks", []) if find_track(d, k) is not None]
            d["rest"] = {"rev": rev, "file": f"rest.{rev}.wav", "tracks": kept + [tid]}
            state["old"] = prev
            track.update(fields)

        update(sid, change)
        if state.get("gone"):
            discard(build, build.with_suffix(".peaks.json"))
            return False
        old = state.get("old")
        if old:
            _remove(folder / old["file"])
            _remove(folder / old["file"].replace(".wav", ".peaks.json"))
    return True


def remove_done_track(sid: str, tid: str) -> None:
    """
    Remove a finished track. Its sound goes back into "everything else" (fast
    approximation: everything else + the track). With no finished tracks left
    there is no "everything else" any more.
    """
    folder = session_dir(sid)
    with queue_state.redis_lock(f"rest:{sid}", timeout=REST_LOCK_SECONDS):
        data = load(sid)
        track = find_track(data, tid)
        if track is None:
            return
        rest = data.get("rest")
        others = [t for t in data["tracks"] if t["id"] != tid and t["status"] == "done"]
        busy = any(t["status"] in ACTIVE for t in data["tracks"])
        new = None
        if rest and others:
            new = dict(rest)
            if tid in rest.get("tracks", []) and track.get("file"):
                rev = next_rest_rev(data)
                out = folder / f"rest.{rev}.wav"
                _combine(out, [(folder / rest["file"], 1.0), (folder / track["file"], 1.0)],
                         data["source"]["frames"], data["source"]["channels"])
                write_peaks(out, folder / f"rest.{rev}.peaks.json")
                new = {"rev": rev, "file": out.name, "tracks": [k for k in rest["tracks"] if k != tid]}

        def change(d):
            d["tracks"] = [t for t in d["tracks"] if t["id"] != tid]
            d["rest"] = new
            # Keep counting even when "everything else" goes away (see next_rest_rev).
            d["rest_rev"] = max(int(d.get("rest_rev") or 0), int(data.get("rest_rev") or 0),
                                int((rest or {}).get("rev") or 0))

        update(sid, change)
        if rest and (new is None or new["file"] != rest["file"]):
            _remove(folder / rest["file"])
            _remove(folder / rest["file"].replace(".wav", ".peaks.json"))
        target = folder / f"{tid}.wav"
        if busy and target.is_file():
            _replace(target, folder / f"{tid}.deleted.wav")  # a running chain step puts it back
        else:
            _remove(target)
        _remove(folder / f"{tid}.without.wav")
        _remove(folder / f"{tid}.peaks.json")


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
    on what changed:

    * nothing changed: the original itself
    * one track changed: the model's own "without that sound" + the track at its gain
    * more: "everything else" (the chain) + every track at its gain
    """
    folder = session_dir(data["id"])
    done = [t for t in data["tracks"] if t["status"] == "done" and t.get("file")]
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
        plan = [(folder / track["without"], 1.0)]
        if g[track["id"]] > 0:
            plan.append((folder / track["file"], g[track["id"]]))
        return plan
    plan = [(folder / rest["file"], g_rest)] if g_rest > 0 else []
    return plan + [(folder / t["file"], g[t["id"]]) for t in done if g[t["id"]] > 0]


def mix_label(data: dict, gains: Dict[str, float]) -> str:
    """``vocals`` / ``no vocals`` / ``mix``, for file names."""
    done = [t for t in data["tracks"] if t["status"] == "done"]
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


def export_mix(sid: str, gains: Dict[str, float], fmt: str) -> tuple:
    """Render the mix to ``fmt``; returns ``(path, download name)``."""
    from workers import audio_io

    data = load(sid)
    sources = mix_plan(data, gains)
    if not sources:
        raise ValueError("Nothing to export: every track is muted.")
    out = _exports_dir(sid)
    stamp = uuid.uuid4().hex[:8]
    wav = out / f"mix-{stamp}.f32.wav"  # not the export name: WAV exports are 24-bit
    _combine(wav, sources, data["source"]["frames"], data["source"]["channels"])
    final = audio_io.export_audio(wav, out / f"mix-{stamp}.{fmt}", fmt)
    return final, f"{_base_name(data)} [{_tag(mix_label(data, gains))}].{fmt}"


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
            src, label = folder / data["rest"]["file"], "everything else"
        else:
            track = find_track(data, key)
            if track is None or track["status"] != "done" or not track.get("file"):
                continue
            src, label = folder / track["file"], track["prompt"]
        name = f"{base} [{_tag(label)}].{fmt}"
        suffix = 2
        while name in used:  # two tracks with the same prompt
            name = f"{base} [{_tag(label)} {suffix}].{fmt}"
            suffix += 1
        used.add(name)
        exported = audio_io.export_audio(src, out / f"{Path(src).stem}.{fmt}", fmt)
        entries.append((exported, name))
    if not entries:
        raise ValueError("No finished tracks to export.")
    zip_path = out / f"tracks-{uuid.uuid4().hex[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path, name in entries:
            zf.write(path, arcname=name)
    return zip_path, f"{base} [tracks].zip"


def export_video(sid: str, gains: Dict[str, float]) -> tuple:
    """The source video with its audio replaced by the mix."""
    from workers import audio_io

    data = load(sid)
    if not data["source"]["is_video"]:
        raise ValueError("This session's source is not a video.")
    sources = mix_plan(data, gains)
    if not sources:
        raise ValueError("Nothing to export: every track is muted.")
    folder = session_dir(sid)
    out = _exports_dir(sid)
    stamp = uuid.uuid4().hex[:8]
    wav = out / f"mix-{stamp}.f32.wav"  # not the export name: WAV exports are 24-bit
    _combine(wav, sources, data["source"]["frames"], data["source"]["channels"])
    ext = Path(data["source"]["file"]).suffix
    video = audio_io.merge_video_audio(folder / data["source"]["file"], wav, out / f"video-{stamp}{ext}")
    return video, f"{_base_name(data)} [{_tag(mix_label(data, gains))}]{ext}"
