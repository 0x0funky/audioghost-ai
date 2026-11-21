"""
Sessions API: upload once, separate several sounds into tracks, mix them,
export what you hear.

    POST   /api/sessions                              upload a file -> session
    GET    /api/sessions                              sessions, newest first, with their sizes
    GET    /api/sessions/{id}                         session with tracks and file URLs
    DELETE /api/sessions/{id}
    POST   /api/sessions/{id}/separate                queue prompts -> one track (take) each
    DELETE /api/sessions/{id}/tracks/{track}          cancel (if running) or remove a take
    POST   /api/sessions/{id}/tracks/{track}/activate make a finished take the one that plays
    POST   /api/sessions/{id}/rest/rebuild            run the chain of "everything else" again
    DELETE /api/sessions/{id}/rest/job                cancel that, or dismiss its error
    POST   /api/sessions/{id}/rest/{rev}/activate     go back to a kept version of "everything else"
    DELETE /api/sessions/{id}/rest/{rev}              delete a kept version
    GET    /api/sessions/{id}/files/{name}            audio, peaks and the source video
    POST   /api/sessions/{id}/export                  mix / tracks ZIP / video with the mix

The v1.1 endpoints (/api/separate, /api/tasks) keep working.
"""
import random
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import model_license
import sessions
from api.separate import _is_video, _save_upload
from settings import RETENTION_HOURS, SESSION_DIR, read_hf_token
from workers import dispatch, queue_state

router = APIRouter()

MODEL_SIZES = ("small", "base", "large")
VARIANTS = ("default", "tv")
QUALITIES = ("fast", "balanced", "high")
MEDIA_TYPES = {
    ".wav": "audio/wav", ".json": "application/json", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".m4a": "audio/mp4", ".aac": "audio/aac", ".ogg": "audio/ogg", ".opus": "audio/ogg",
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo", ".mpeg": "video/mpeg", ".zip": "application/zip",
}
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}
MAX_SEED = 2**31 - 1
#: The shortest part of a file that can be separated on its own.
MIN_PART_SECONDS = 1.0
ENGLISH_ONLY = ('Describe the sound in English, for example "a dog barking". '
                "SAM-Audio only understands English descriptions.")


class PromptSpec(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    #: ("+" sound is here | "-" not here, start s, end s)
    anchors: List[Tuple[Literal["+", "-"], float, float]] = Field(default_factory=list, max_length=16)
    #: A track of the sound this is another take of. A new sound when missing.
    retake_of: Optional[str] = None


class ModelSettings(BaseModel):
    model_size: str = "base"
    quality: str = "balanced"
    variant: str = "default"
    stereo: bool = True  # only applies to stereo sources
    use_float32: bool = False
    chunk_duration: float = 25.0
    #: The same seed with the same settings repeats a result. Random when missing.
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)


class SeparateRequest(ModelSettings):
    prompts: List[PromptSpec] = Field(min_length=1, max_length=8)
    cleanup: bool = False  # "Clean up leftovers": the track clears its traces from everything else
    #: Separate only this part of the file (start s, end s). The whole file when missing.
    range: Optional[Tuple[float, float]] = None


class RebuildRequest(ModelSettings):
    #: Sound ids, the first removed first. The current order when empty.
    order: List[str] = Field(default_factory=list, max_length=16)


class Segment(BaseModel):
    start: float
    end: float
    gains: Dict[str, float] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    kind: Literal["mix", "tracks", "video"] = "mix"
    format: Literal["wav", "flac", "mp3"] = "wav"
    #: Linear gain per track id (and "rest"), as heard: muted/soloed-out = 0.
    gains: Dict[str, float] = Field(default_factory=dict)
    #: Parts of the file that play other gains ("only here" changes). Outside them: ``gains``.
    segments: List[Segment] = Field(default_factory=list, max_length=400)
    #: For kind == "tracks": track ids and/or "rest".
    tracks: List[str] = Field(default_factory=list)


def english_only(text: str) -> bool:
    """Whether every letter of ``text`` is a Latin one (SAM-Audio's text encoder is English only)."""
    return all(not ch.isalpha() or "LATIN" in unicodedata.name(ch, "") for ch in text)


def new_seed() -> int:
    return random.SystemRandom().randrange(MAX_SEED + 1)


def _load(sid: str) -> dict:
    try:
        return sessions.load(sid)
    except sessions.SessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found (it may have been deleted).") from None


def present(data: dict) -> dict:
    """The stored session plus URLs, which take plays for each sound, and queue positions."""
    sid = data["id"]
    files = f"/api/sessions/{sid}/files/"
    source = dict(data["source"])
    source["audio_url"] = files + "original.wav"
    source["peaks_url"] = files + "original.peaks.json"
    source["video_url"] = files + source["file"] if source["is_video"] else None
    playing = {track["id"] for track in sessions.active_takes(data)}
    tracks = []
    for track in data["tracks"]:
        item = {k: v for k, v in track.items() if k != "job_id"}
        item["sound"] = sessions.sound_of(track)
        item["active"] = track["id"] in playing
        item["range"] = track.get("range")
        item["seed"] = (track.get("settings") or {}).get("seed")
        item["audio_url"] = files + track["file"] if track.get("file") else None
        item["peaks_url"] = files + f"{track['id']}.peaks.json" if track.get("file") else None
        item["without_url"] = files + track["without"] if track.get("without") else None
        item["jobs_ahead"] = queue_state.jobs_ahead(track["job_id"]) if track["status"] == "queued" else None
        tracks.append(item)

    def urls(rest: dict) -> dict:
        return {"rev": rest.get("rev"), "seed": rest.get("seed"), "created_at": rest.get("created_at"),
                "audio_url": files + rest["file"], "peaks_url": files + rest["file"].replace(".wav", ".peaks.json")}

    rest = data.get("rest")
    if rest:
        rest = dict(rest, **urls(rest))
        rest["alts"] = [urls(alt) for alt in data.get("rest_alts") or []]
        rest["stale"] = sessions.rest_is_stale(data)
    job = data.get("rest_job")
    return {
        "id": sid,
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "expires_at": sessions.expires_at(data),
        "source": source,
        "tracks": tracks,
        "rest": rest,
        "rest_job": {k: job.get(k) for k in ("status", "progress", "message", "error")} if job else None,
    }


def _model_settings(request: ModelSettings) -> dict:
    """Validated model settings, or an HTTP error naming what still has to happen first."""
    if request.model_size not in MODEL_SIZES:
        raise HTTPException(status_code=400, detail=f"model_size must be one of {MODEL_SIZES}")
    if request.quality not in QUALITIES:
        raise HTTPException(status_code=400, detail=f"quality must be one of {QUALITIES}")
    if request.variant not in VARIANTS:
        raise HTTPException(status_code=400, detail=f"variant must be one of {VARIANTS}")
    from workers import model_files

    need = model_files.missing_requirement(
        request.model_size, request.variant, "fp32" if request.use_float32 else "fp16",
        token=read_hf_token(), license_accepted=model_license.accepted(),
    )
    if need == "token":
        raise HTTPException(
            status_code=401,
            detail="Meta's precision (-tv) checkpoints need a Hugging Face account with access to them: connect it first.",
        )
    if need == "license":
        raise HTTPException(
            status_code=409,
            detail="Accept the SAM-Audio license first: AudioGhost downloads the model the first time it is used.",
        )
    return {
        "model_size": request.model_size,
        "quality": request.quality,
        "variant": request.variant,
        "stereo": request.stereo,
        "use_float32": request.use_float32,
        "chunk_duration": max(5.0, min(60.0, request.chunk_duration)),
        "seed": request.seed,
    }


def clean_prompt(text: str) -> str:
    """One line, single spaces; raises the HTTP error for an empty or non-English description."""
    text = " ".join(text.split())
    if not text:
        raise HTTPException(status_code=400, detail="Prompts must not be empty.")
    if not english_only(text):
        raise HTTPException(status_code=400, detail=ENGLISH_ONLY)
    return text


# Handlers that touch files or Redis are plain functions: FastAPI runs them in
# its thread pool, so a slow disk never stalls the polls and audio streams.


@router.post("", status_code=201)
async def create_session(file: UploadFile = File(...)):
    upload = await _save_upload(file, uuid.uuid4().hex)
    try:
        data = await run_in_threadpool(
            sessions.create, upload, file.filename or upload.name, _is_video(file)
        )
    except Exception as err:  # noqa: BLE001 - undecodable files land here
        upload.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read the audio in this file: {err}") from None
    await run_in_threadpool(sessions.cleanup_if_due, RETENTION_HOURS)
    return present(data)


@router.get("")
def list_sessions(limit: int = 200):
    sessions.cleanup_if_due(RETENTION_HOURS)
    items = sessions.list_sessions(max(1, min(limit, 1000)))
    return {
        "sessions": items,
        "total_bytes": sum(item["size_bytes"] for item in items),
        "folder": str(SESSION_DIR),
        #: Hours after which untouched sessions are deleted; 0 = never.
        "retention_hours": RETENTION_HOURS,
    }


@router.get("/{sid}")
def get_session(sid: str):
    return present(_load(sid))


@router.delete("/{sid}")
def delete_session(sid: str):
    _load(sid)
    sessions.delete(sid)
    return {"deleted": sid}


@router.post("/{sid}/separate", status_code=202)
def separate(sid: str, request: SeparateRequest):
    from workers.tasks import separate_session_task

    data = _load(sid)
    options = _model_settings(request)
    options["cleanup"] = request.cleanup

    duration = data["source"]["duration"]
    part = None
    if request.range is not None:
        start, end = sorted((max(0.0, min(duration, v)) for v in request.range))
        if end - start < MIN_PART_SECONDS:
            raise HTTPException(status_code=400, detail=f"Select at least {MIN_PART_SECONDS:g} second of the file.")
        if start > 0.01 or end < duration - 0.01:  # (nearly) all of it is just the whole file
            part = [round(start, 3), round(end, 3)]
    prompts = []
    for spec in request.prompts:
        text = clean_prompt(spec.text)
        anchors = []
        for kind, start, end in spec.anchors:
            start, end = max(0.0, min(start, end)), min(duration, max(start, end))
            if end - start >= 0.05:
                anchors.append([kind, round(start, 3), round(end, 3)])
        sound = None
        if spec.retake_of:
            original = sessions.find_track(data, spec.retake_of)
            if original is None:
                raise HTTPException(status_code=404, detail="The track to try again no longer exists.")
            sound = sessions.sound_of(original)
        prompts.append({"text": text, "anchors": anchors, "sound": sound, "range": part,
                        "seed": request.seed if request.seed is not None else new_seed()})

    job_id = str(uuid.uuid4())
    track_ids = sessions.add_tracks(sid, prompts, options, job_id)
    queue_state.enqueue(job_id)
    dispatch.submit(separate_session_task, args=[sid, track_ids, options], task_id=job_id)
    return {"job_id": job_id, "track_ids": track_ids, "session": present(sessions.load(sid))}


def _worker_runs(job_id: Optional[str]) -> bool:
    """Whether a live worker is running this job right now."""
    return bool(job_id) and queue_state.worker_state() is not None and queue_state.running_task() == job_id


@router.delete("/{sid}/tracks/{tid}")
def remove_track(sid: str, tid: str):
    """
    Queued or finished: removed. Running: cancelled (the worker drops it), or
    removed right away when no worker is running it any more (it stopped).
    """
    data = _load(sid)
    track = sessions.find_track(data, tid)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    if track["status"] == "running" and _worker_runs(track.get("job_id")):
        queue_state.request_cancel(f"{track['job_id']}:{tid}")
        return {"cancelling": tid}
    if track["status"] == "done":
        sessions.remove_done_track(sid, tid)  # its sound's last take: back into "everything else"
    else:
        sessions.drop_track(sid, tid)
    return {"removed": tid}


@router.post("/{sid}/tracks/{tid}/activate")
def activate_track(sid: str, tid: str):
    """Make a finished take the one that plays (and exports) for its sound."""
    _load(sid)
    try:
        sessions.activate_take(sid, tid)
    except sessions.TrackNotFound:
        raise HTTPException(status_code=404, detail="Track not found") from None
    except ValueError as err:
        raise HTTPException(status_code=409, detail=str(err)) from None
    return present(sessions.load(sid))


@router.post("/{sid}/rest/rebuild", status_code=202)
def rebuild_rest(sid: str, request: RebuildRequest):
    """
    Run the chain of "everything else" again, with another seed and, if asked, another order: the
    sound removed first comes out cleanest. The version it replaces is kept to compare with.
    """
    from workers.tasks import rebuild_rest_task

    data = _load(sid)
    if len(sessions.active_takes(data)) < 2:
        raise HTTPException(status_code=409, detail="With a single sound, try that sound again instead: "
                                                    "everything else is its own \"without\".")
    job = data.get("rest_job") or {}
    if job.get("status") in sessions.ACTIVE or any(t["status"] in sessions.ACTIVE for t in data["tracks"]):
        raise HTTPException(status_code=409, detail="Wait until the running separation is finished.")
    options = _model_settings(request)
    options["order"] = [sound for sound in request.order if sessions.is_track_id(sound)]
    if options["seed"] is None:
        options["seed"] = new_seed()
    job_id = str(uuid.uuid4())
    sessions.set_rest_job(sid, {"job_id": job_id, "status": "queued", "progress": 0.0, "message": "", "error": None})
    queue_state.enqueue(job_id)
    dispatch.submit(rebuild_rest_task, args=[sid, options], task_id=job_id)
    return {"job_id": job_id, "session": present(sessions.load(sid))}


@router.delete("/{sid}/rest/job")
def cancel_rebuild(sid: str):
    """Cancel the rebuild of "everything else" (or dismiss its error)."""
    data = _load(sid)
    job = data.get("rest_job") or {}
    if job.get("status") == "running" and _worker_runs(job.get("job_id")):
        queue_state.request_cancel(job["job_id"])
        return {"cancelling": True, "session": present(data)}
    if job.get("status") == "queued":
        queue_state.request_cancel(job.get("job_id"))
    sessions.set_rest_job(sid)
    return {"cancelling": False, "session": present(sessions.load(sid))}


@router.post("/{sid}/rest/{rev}/activate")
def activate_rest(sid: str, rev: int):
    _load(sid)
    try:
        sessions.activate_rest(sid, rev)
    except sessions.TrackNotFound:
        raise HTTPException(status_code=404, detail="That version no longer exists.") from None
    return present(sessions.load(sid))


@router.delete("/{sid}/rest/{rev}")
def delete_rest_version(sid: str, rev: int):
    _load(sid)
    sessions.remove_rest_alt(sid, rev)
    return present(sessions.load(sid))


def _missing_overview(path: Path) -> None:
    """Sessions from before v1.4 have no waveform overview of a take's "without": make it on demand."""
    audio = path.with_name(path.name.replace(".peaks.json", ".wav"))
    if path.name.endswith(".peaks.json") and not path.exists() and audio.is_file():
        sessions.write_peaks(audio, path)


def _byte_range(header: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """``(first, last)`` for a single ``Range: bytes=...`` header; None for the whole file or a range that makes no sense."""
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    first, _, last = header[6:].strip().partition("-")
    try:
        if not first:  # the last N bytes
            return max(0, size - int(last)), size - 1
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    except ValueError:
        return None
    return (start, end) if 0 <= start <= end else None


def _serve_window(request: Request, source: "sessions.Window") -> Response:
    """
    A part-only take's file, streamed at full length (see ``sessions.Window``): the browser
    plays it in sync with every other track, and seeks in it with Range requests.
    """
    size = sessions.window_size(source)
    headers = dict(IMMUTABLE, **{"Accept-Ranges": "bytes"})
    wanted = _byte_range(request.headers.get("range"), size)
    if request.headers.get("range") and (wanted is None or wanted[0] >= size):
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    first, last = wanted or (0, size - 1)
    headers["Content-Length"] = str(last - first + 1)
    if wanted:
        headers["Content-Range"] = f"bytes {first}-{last}/{size}"
    return StreamingResponse(sessions.window_bytes(source, first, last), status_code=206 if wanted else 200,
                             media_type="audio/wav", headers=headers)


@router.get("/{sid}/files/{name}")
def get_file(sid: str, name: str, request: Request):
    if not sessions.SERVABLE.match(name):
        raise HTTPException(status_code=404, detail="Not found")
    try:
        path = sessions.session_dir(sid) / name
    except sessions.SessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found") from None
    if not path.is_file():
        _missing_overview(path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    if sessions._TAKE_FILE.match(name):
        try:
            source = sessions.audio_source(sessions.load(sid), name)
        except sessions.SessionNotFound:
            raise HTTPException(status_code=404, detail="Session not found") from None
        if isinstance(source, sessions.Window):
            return _serve_window(request, source)
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                        headers=IMMUTABLE)


@router.post("/{sid}/export")
def export(sid: str, request: ExportRequest):
    _load(sid)
    gains = {key: max(0.0, float(value)) for key, value in request.gains.items()}
    segments = [{"start": s.start, "end": s.end,
                 "gains": {key: max(0.0, float(value)) for key, value in s.gains.items()}}
                for s in request.segments if s.end > s.start]
    try:
        if request.kind == "mix":
            path, name = sessions.export_mix(sid, gains, request.format, segments)
        elif request.kind == "tracks":
            path, name = sessions.export_tracks(sid, request.tracks, request.format)
        else:
            path, name = sessions.export_video(sid, gains, segments)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from None
    except RuntimeError as err:  # FFmpeg missing / failed
        raise HTTPException(status_code=500, detail=str(err)) from None
    return FileResponse(path, media_type=MEDIA_TYPES.get(Path(name).suffix.lower(), "application/octet-stream"),
                        filename=name)
