"""
Separation API - Audio/Video Separation Endpoints
"""
import json
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from settings import MAX_UPLOAD_MB, RETENTION_HOURS, UPLOAD_DIR
from workers import dispatch, queue_state
from workers.tasks import cleanup_old_files, separate_audio_task

router = APIRouter()

VIDEO_TYPES = ["video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/mpeg",
               "video/x-matroska"]
VIDEO_EXTENSIONS = [".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".m4v"]
MODEL_SIZES = ("small", "base", "large")
MODEL_VARIANTS = ("default", "tv")
QUALITY_PRESETS = ("fast", "balanced", "high")
MODES = ("extract", "remove")
MAX_DESCRIPTION_CHARS = 300
_CHUNK = 1024 * 1024


class SeparationResponse(BaseModel):
    task_id: str
    status: str
    message: str


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _validate_choice(name: str, value: str, choices) -> str:
    value = (value or "").strip().lower()
    if value not in choices:
        raise HTTPException(status_code=400, detail=f"{name} must be one of {', '.join(choices)}")
    return value


async def _save_upload(file: UploadFile, task_id: str) -> Path:
    """Stream the upload to disk instead of reading it into memory."""
    extension = Path(file.filename or "").suffix.lower() or ".bin"
    upload_path = UPLOAD_DIR / f"{task_id}{extension}"
    limit = int(MAX_UPLOAD_MB * 1024 * 1024) if MAX_UPLOAD_MB > 0 else None
    written = 0
    try:
        with open(upload_path, "wb") as f:
            while chunk := await file.read(_CHUNK):
                written += len(chunk)
                if limit and written > limit:
                    raise HTTPException(
                        status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB:.0f} MB"
                    )
                f.write(chunk)
    except BaseException:
        upload_path.unlink(missing_ok=True)
        raise
    if written == 0:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    return upload_path


def _is_video(file: UploadFile) -> bool:
    extension = Path(file.filename or "").suffix.lower()
    return (file.content_type in VIDEO_TYPES) or extension in VIDEO_EXTENSIONS


def _submit(upload_path: Path, description: str, options: dict) -> str:
    task_id = str(uuid.uuid4())
    queue_state.enqueue(task_id)
    dispatch.submit(
        separate_audio_task,
        kwargs={"audio_path": str(upload_path), "description": description, **options},
        task_id=task_id,
    )
    return task_id


@router.post("/", response_model=SeparationResponse)
async def create_separation_task(
    file: UploadFile = File(...),
    description: str = Form(...),
    mode: str = Form("extract"),
    start_time: Optional[float] = Form(None),
    end_time: Optional[float] = Form(None),
    model_size: str = Form("base"),
    chunk_duration: float = Form(25.0),
    use_float32: str = Form("false"),
    quality: str = Form("balanced"),
    stereo: str = Form("false"),
    model_variant: str = Form("default"),
    seed: Optional[int] = Form(None),
):
    """
    Create a new audio/video separation task.

    - **description**: what to separate, e.g. "vocals", "a dog barking"
    - **mode**: "extract" or "remove" (both stems are always produced)
    - **start_time / end_time**: optional span where the sound is audible (Temporal Lock)
    - **model_size**: small | base | large
    - **model_variant**: default | tv (Meta's "-tv" checkpoints, more accurate targets)
    - **quality**: fast (8 ODE steps) | balanced (16, Meta default) | high (32)
    - **chunk_duration**: max chunk length in seconds (5-60)
    - **use_float32**: run the transformer in float32 instead of bfloat16
    - **stereo**: keep stereo (processes each channel, ~2x compute)
    - **seed**: optional, for reproducible results
    """
    description = (description or "").strip()
    if not description:
        raise HTTPException(status_code=400, detail="Description must not be empty")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise HTTPException(status_code=400, detail="Description is too long")
    options = {
        "mode": _validate_choice("mode", mode, MODES),
        "model_size": _validate_choice("model_size", model_size, MODEL_SIZES),
        "model_variant": _validate_choice("model_variant", model_variant, MODEL_VARIANTS),
        "quality": _validate_choice("quality", quality, QUALITY_PRESETS),
        "chunk_duration": max(5.0, min(60.0, float(chunk_duration))),
        "use_float32": _as_bool(use_float32),
        "stereo": _as_bool(stereo),
        "is_video": _is_video(file),
        "original_filename": Path(file.filename or "audio").name,
        "seed": seed,
    }
    anchors = None
    if start_time is not None and end_time is not None and end_time - start_time > 0.05:
        anchors = [["+", max(0.0, float(start_time)), float(end_time)]]
    options["anchors"] = anchors

    cleanup_old_files(RETENTION_HOURS)
    upload_path = await _save_upload(file, uuid.uuid4().hex)
    task_id = _submit(upload_path, description, options)
    return SeparationResponse(task_id=task_id, status="pending", message="Task submitted successfully")


@router.post("/batch", response_model=List[SeparationResponse])
async def create_batch_separation(
    file: UploadFile = File(...),
    descriptions: str = Form(...),  # JSON array, e.g. ["vocals", "drums"]
    mode: str = Form("extract"),
    model_size: str = Form("base"),
    chunk_duration: float = Form(25.0),
    use_float32: str = Form("false"),
    quality: str = Form("balanced"),
    stereo: str = Form("false"),
    model_variant: str = Form("default"),
):
    """Queue one separation per description for the same file (one upload)."""
    try:
        desc_list = [str(d).strip() for d in json.loads(descriptions) if str(d).strip()]
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="descriptions must be a JSON array of strings")
    if not desc_list:
        raise HTTPException(status_code=400, detail="No descriptions given")
    options = {
        "mode": _validate_choice("mode", mode, MODES),
        "model_size": _validate_choice("model_size", model_size, MODEL_SIZES),
        "model_variant": _validate_choice("model_variant", model_variant, MODEL_VARIANTS),
        "quality": _validate_choice("quality", quality, QUALITY_PRESETS),
        "chunk_duration": max(5.0, min(60.0, float(chunk_duration))),
        "use_float32": _as_bool(use_float32),
        "stereo": _as_bool(stereo),
        "is_video": _is_video(file),
        "original_filename": Path(file.filename or "audio").name,
    }
    cleanup_old_files(RETENTION_HOURS)
    upload_path = await _save_upload(file, uuid.uuid4().hex)
    return [
        SeparationResponse(
            task_id=_submit(upload_path, desc, options),
            status="pending",
            message=f"Task for '{desc}' submitted",
        )
        for desc in desc_list
    ]
