"""
Tasks API - Task Status and Results
"""
import zipfile
from pathlib import Path
from typing import List, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from settings import OUTPUT_DIR
from workers import queue_state
from workers.audio_io import (
    AUDIO_EXPORT_FORMATS,
    export_audio,
    merge_video_audio,
    stem_filename,
)
from workers.celery_app import celery_app

router = APIRouter()

AUDIO_STEMS = ("original", "ghost", "clean")
VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mpeg": "video/mpeg",
}
AUDIO_MEDIA_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg"}


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed, cancelled
    progress: int  # 0-100
    message: Optional[str] = None
    result: Optional[dict] = None
    queue_position: Optional[int] = None


def _completed_result(task_id: str) -> dict:
    result = AsyncResult(task_id, app=celery_app)
    try:
        ok = result.state == "SUCCESS" and isinstance(result.result, dict)
    except Exception:  # noqa: BLE001 - malformed backend data
        ok = False
    if not ok:
        raise HTTPException(status_code=404, detail="Task not completed (or expired)")
    return result.result


def _existing(path_str: Optional[str], what: str) -> Path:
    path = Path(path_str or "")
    if not path_str or not path.is_file():
        raise HTTPException(status_code=404, detail=f"{what} not found (results expire after 24h)")
    return path


@router.get("/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a separation task"""
    result = AsyncResult(task_id, app=celery_app)
    try:
        state = result.state
        info = result.info
    except Exception as err:  # noqa: BLE001 - malformed backend data
        return TaskStatus(task_id=task_id, status="failed", progress=0,
                          message=f"Could not read task state: {err}")

    if state == "PENDING":
        ahead = queue_state.jobs_ahead(task_id)
        if ahead:
            message = f"Waiting in queue - {ahead} job{'s' if ahead != 1 else ''} ahead of yours"
        else:
            message = "Waiting for the worker to pick up the task..."
        return TaskStatus(task_id=task_id, status="pending", progress=0, message=message,
                          queue_position=ahead)

    if state in ("PROGRESS", "STARTED"):
        info = info if isinstance(info, dict) else {}
        return TaskStatus(
            task_id=task_id,
            status="processing",
            progress=int(info.get("progress", 1)),
            message=info.get("message", "Starting..."),
        )

    if state == "SUCCESS":
        return TaskStatus(task_id=task_id, status="completed", progress=100,
                          message="Task completed successfully", result=info)

    if state == "FAILURE":
        return TaskStatus(task_id=task_id, status="failed", progress=0, message=str(info))

    if state in ("REVOKED", "CANCELLED"):
        return TaskStatus(task_id=task_id, status="cancelled", progress=0, message="Cancelled")

    return TaskStatus(task_id=task_id, status=state.lower(), progress=0,
                      message=f"Task state: {state}")


@router.get("/{task_id}/download/{file_type}")
async def download_result(
    task_id: str,
    file_type: str,
    format: Optional[str] = Query(None, description="wav | flac | mp3 (attachment)"),
):
    """
    Get a result file.

    - **file_type**: "original", "ghost", "clean" or "video"
    - **format**: omit to stream the working file (used by the players); set
      ``wav`` (24-bit), ``flac`` or ``mp3`` (320k) to download it with a
      readable file name such as ``song [vocals].wav``.
    """
    if file_type not in (*AUDIO_STEMS, "video"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    data = _completed_result(task_id)

    if file_type == "video":
        video = _existing(data.get("video_path"), "Video file")
        ext = video.suffix.lower()
        name = stem_filename(data.get("original_filename"), data.get("description", ""),
                             "original", ext.lstrip("."))
        return FileResponse(video, media_type=VIDEO_MEDIA_TYPES.get(ext, "video/mp4"),
                            filename=name if format else None)

    source = _existing(data.get(f"{file_type}_path"), "Audio file")
    if not format:
        return FileResponse(source, media_type="audio/wav")

    fmt = format.lower()
    if fmt not in AUDIO_EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"format must be one of {AUDIO_EXPORT_FORMATS}")
    exported = OUTPUT_DIR / f"{task_id}.{file_type}.export.{fmt}"
    try:
        await run_in_threadpool(export_audio, source, exported, fmt)
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=str(err))
    name = stem_filename(data.get("original_filename"), data.get("description", ""), file_type, fmt)
    return FileResponse(exported, media_type=AUDIO_MEDIA_TYPES[fmt], filename=name)


@router.get("/{task_id}/download-all")
async def download_all(task_id: str, format: str = Query("wav")):
    """Zip the isolated and residual stems (plus the original) in one download."""
    fmt = format.lower()
    if fmt not in AUDIO_EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"format must be one of {AUDIO_EXPORT_FORMATS}")
    data = _completed_result(task_id)
    original_name = data.get("original_filename")
    description = data.get("description", "")

    def build_zip() -> Path:
        zip_path = OUTPUT_DIR / f"{task_id}.stems.{fmt}.zip"
        entries = []
        for stem in ("ghost", "clean", "original"):
            source = _existing(data.get(f"{stem}_path"), f"{stem} stem")
            exported = export_audio(source, OUTPUT_DIR / f"{task_id}.{stem}.export.{fmt}", fmt)
            entries.append((exported, stem_filename(original_name, description, stem, fmt)))
        if zip_path.exists() and all(zip_path.stat().st_mtime >= p.stat().st_mtime for p, _ in entries):
            return zip_path
        tmp = zip_path.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            for path, arcname in entries:
                zf.write(path, arcname)
        tmp.replace(zip_path)
        return zip_path

    try:
        zip_path = await run_in_threadpool(build_zip)
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=str(err))
    base = stem_filename(original_name, description, "ghost", "zip").replace(".zip", " stems.zip")
    return FileResponse(zip_path, media_type="application/zip", filename=base)


@router.get("/{task_id}/download-video-with-audio/{audio_type}")
async def download_video_with_audio(task_id: str, audio_type: str):
    """
    Download the video with its audio track replaced (picture is not re-encoded).

    - **audio_type**: "original", "ghost", or "clean"
    """
    if audio_type not in AUDIO_STEMS:
        raise HTTPException(status_code=400, detail="audio_type must be original, ghost or clean")
    data = _completed_result(task_id)
    video = _existing(data.get("video_path"), "Video file")
    audio = _existing(data.get(f"{audio_type}_path"), f"Audio '{audio_type}'")
    ext = video.suffix.lower()
    merged = OUTPUT_DIR / f"{task_id}.{audio_type}.merged{ext}"
    try:
        await run_in_threadpool(merge_video_audio, video, audio, merged)
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {err}")
    name = stem_filename(data.get("original_filename"), data.get("description", ""),
                         audio_type, ext.lstrip("."))
    return FileResponse(merged, media_type=VIDEO_MEDIA_TYPES.get(ext, "video/mp4"), filename=name)


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a queued or running task. Running tasks stop at the next diffusion
    step (the solo-pool worker can't be interrupted from outside).
    """
    queue_state.request_cancel(task_id)
    AsyncResult(task_id, app=celery_app).revoke()
    return {"success": True, "message": "Cancellation requested"}


@router.get("/", response_model=List[TaskStatus])
async def list_recent_tasks(limit: int = 10):
    """Task history is not persisted yet (planned)."""
    return []
