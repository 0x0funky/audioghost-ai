"""
Sessions API (v1.2): upload once, separate several sounds into tracks, mix
them, export what you hear.

    POST   /api/sessions                      upload a file -> session
    GET    /api/sessions                      recent sessions
    GET    /api/sessions/{id}                 session with tracks and file URLs
    DELETE /api/sessions/{id}
    POST   /api/sessions/{id}/separate        queue prompts -> one track each
    DELETE /api/sessions/{id}/tracks/{track}  cancel (if running) or remove
    GET    /api/sessions/{id}/files/{name}    audio, peaks and the source video
    POST   /api/sessions/{id}/export          mix / tracks ZIP / video with the mix

The v1.1 endpoints (/api/separate, /api/tasks) keep working.
"""
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import sessions
from api.separate import _is_video, _save_upload
from settings import RETENTION_HOURS, read_hf_token
from workers import queue_state

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


class PromptSpec(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    #: ("+" sound is here | "-" not here, start s, end s)
    anchors: List[Tuple[Literal["+", "-"], float, float]] = Field(default_factory=list, max_length=16)


class SeparateRequest(BaseModel):
    prompts: List[PromptSpec] = Field(min_length=1, max_length=8)
    model_size: str = "base"
    quality: str = "balanced"
    variant: str = "default"
    stereo: bool = True  # only applies to stereo sources
    use_float32: bool = False
    chunk_duration: float = 25.0
    seed: Optional[int] = None


class ExportRequest(BaseModel):
    kind: Literal["mix", "tracks", "video"] = "mix"
    format: Literal["wav", "flac", "mp3"] = "wav"
    #: Linear gain per track id (and "rest"), as heard: muted/soloed-out = 0.
    gains: Dict[str, float] = Field(default_factory=dict)
    #: For kind == "tracks": track ids and/or "rest".
    tracks: List[str] = Field(default_factory=list)


def _load(sid: str) -> dict:
    try:
        return sessions.load(sid)
    except sessions.SessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found (it may have expired).") from None


def present(data: dict) -> dict:
    """The stored session plus URLs, expiry and queue positions."""
    sid = data["id"]
    files = f"/api/sessions/{sid}/files/"
    source = dict(data["source"])
    source["audio_url"] = files + "original.wav"
    source["peaks_url"] = files + "original.peaks.json"
    source["video_url"] = files + source["file"] if source["is_video"] else None
    tracks = []
    for track in data["tracks"]:
        item = {k: v for k, v in track.items() if k != "job_id"}
        item["audio_url"] = files + track["file"] if track.get("file") else None
        item["peaks_url"] = files + f"{track['id']}.peaks.json" if track.get("file") else None
        item["without_url"] = files + track["without"] if track.get("without") else None
        item["jobs_ahead"] = queue_state.jobs_ahead(track["job_id"]) if track["status"] == "queued" else None
        tracks.append(item)
    rest = data.get("rest")
    if rest:
        rest = dict(rest)
        rest["audio_url"] = files + rest["file"]
        rest["peaks_url"] = files + rest["file"].replace(".wav", ".peaks.json")
    return {
        "id": sid,
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "expires_at": sessions.expires_at(data),
        "source": source,
        "tracks": tracks,
        "rest": rest,
    }


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
def list_sessions(limit: int = 30):
    sessions.cleanup_if_due(RETENTION_HOURS)
    return {"sessions": sessions.list_sessions(max(1, min(limit, 200)))}


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
    if request.model_size not in MODEL_SIZES:
        raise HTTPException(status_code=400, detail=f"model_size must be one of {MODEL_SIZES}")
    if request.quality not in QUALITIES:
        raise HTTPException(status_code=400, detail=f"quality must be one of {QUALITIES}")
    if request.variant not in VARIANTS:
        raise HTTPException(status_code=400, detail=f"variant must be one of {VARIANTS}")
    if not read_hf_token():
        raise HTTPException(
            status_code=401,
            detail="Connect your Hugging Face account first: the SAM-Audio models are gated.",
        )

    duration = data["source"]["duration"]
    prompts = []
    for spec in request.prompts:
        text = " ".join(spec.text.split())
        if not text:
            raise HTTPException(status_code=400, detail="Prompts must not be empty.")
        anchors = []
        for kind, start, end in spec.anchors:
            start, end = max(0.0, min(start, end)), min(duration, max(start, end))
            if end - start >= 0.05:
                anchors.append([kind, round(start, 3), round(end, 3)])
        prompts.append({"text": text, "anchors": anchors})

    options = {
        "model_size": request.model_size,
        "quality": request.quality,
        "variant": request.variant,
        "stereo": request.stereo,
        "use_float32": request.use_float32,
        "chunk_duration": max(5.0, min(60.0, request.chunk_duration)),
        "seed": request.seed,
    }
    job_id = str(uuid.uuid4())
    track_ids = sessions.add_tracks(sid, prompts, options, job_id)
    queue_state.enqueue(job_id)
    separate_session_task.apply_async(args=[sid, track_ids, options], task_id=job_id)
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
        sessions.remove_done_track(sid, tid)  # its sound goes back into "everything else"
    else:
        sessions.drop_track(sid, tid)
    return {"removed": tid}


@router.get("/{sid}/files/{name}")
async def get_file(sid: str, name: str):
    if not sessions.SERVABLE.match(name):
        raise HTTPException(status_code=404, detail="Not found")
    try:
        path = sessions.session_dir(sid) / name
    except sessions.SessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                        headers=IMMUTABLE)


@router.post("/{sid}/export")
def export(sid: str, request: ExportRequest):
    _load(sid)
    gains = {key: max(0.0, float(value)) for key, value in request.gains.items()}
    try:
        if request.kind == "mix":
            path, name = sessions.export_mix(sid, gains, request.format)
        elif request.kind == "tracks":
            path, name = sessions.export_tracks(sid, request.tracks, request.format)
        else:
            path, name = sessions.export_video(sid, gains)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from None
    except RuntimeError as err:  # FFmpeg missing / failed
        raise HTTPException(status_code=500, detail=str(err)) from None
    return FileResponse(path, media_type=MEDIA_TYPES.get(Path(name).suffix.lower(), "application/octet-stream"),
                        filename=name)
