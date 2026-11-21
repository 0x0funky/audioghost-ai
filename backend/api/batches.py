"""
Batches API: apply one recipe to many files.

    POST   /api/batches                        create a batch from a recipe
    GET    /api/batches                        batches, newest first
    GET    /api/batches/{id}                   the batch with each file's progress
    POST   /api/batches/{id}/items             add a file: upload -> session -> queued
    GET    /api/batches/{id}/items/{session}/export   one file's result
    POST   /api/batches/{id}/export            ZIP of every finished result
    DELETE /api/batches/{id}                   forget the batch (?sessions=true deletes its files too)
"""
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import batches
import sessions
from api.separate import _is_video, _save_upload
from api.sessions import MAX_SEED, MEDIA_TYPES, ModelSettings, _model_settings, clean_prompt, new_seed
from workers import dispatch, queue_state

router = APIRouter()


class RecipePrompt(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    #: Linear gain of this sound in the result, as heard: 0 = removed, 1 = as recorded.
    gain: float = Field(default=1.0, ge=0.0, le=4.0)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)


class BatchRequest(ModelSettings):
    name: str = Field(default="", max_length=120)
    prompts: List[RecipePrompt] = Field(min_length=1, max_length=8)
    rest_gain: float = Field(default=1.0, ge=0.0, le=4.0)
    cleanup: bool = False
    kind: Literal["mix", "tracks", "video"] = "mix"
    format: Literal["wav", "flac", "mp3"] = "wav"


def _load(bid: str) -> dict:
    try:
        return batches.load(bid)
    except batches.BatchNotFound:
        raise HTTPException(status_code=404, detail="Batch not found.") from None


@router.post("", status_code=201)
def create_batch(request: BatchRequest):
    settings = _model_settings(request)
    settings["cleanup"] = request.cleanup
    recipe = {
        "prompts": [{"text": clean_prompt(p.text), "gain": p.gain, "seed": p.seed} for p in request.prompts],
        "rest_gain": request.rest_gain,
        "settings": settings,
        "kind": request.kind,
        "format": request.format,
    }
    return batches.present(batches.create(request.name, recipe))


@router.get("")
def list_batches():
    return {"batches": batches.list_batches()}


@router.get("/{bid}")
def get_batch(bid: str):
    return batches.present(_load(bid))


@router.post("/{bid}/items", status_code=201)
async def add_file(bid: str, file: UploadFile = File(...)):
    """Upload one more file: it becomes a session and its sounds are queued right away."""
    from workers.tasks import separate_session_task

    batch = _load(bid)
    upload = await _save_upload(file, uuid.uuid4().hex)
    try:
        data = await run_in_threadpool(sessions.create, upload, file.filename or upload.name, _is_video(file))
    except Exception as err:  # noqa: BLE001 - undecodable files land here
        upload.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read the audio in this file: {err}") from None

    def queue() -> dict:
        recipe = batch["recipe"]
        options = dict(recipe["settings"])
        prompts = [{"text": p["text"], "anchors": [], "seed": p["seed"] if p.get("seed") is not None else new_seed()}
                   for p in recipe["prompts"]]
        job_id = str(uuid.uuid4())
        track_ids = sessions.add_tracks(data["id"], prompts, options, job_id)
        queue_state.enqueue(job_id)
        dispatch.submit(separate_session_task, args=[data["id"], track_ids, options], task_id=job_id)
        return batches.add_item(bid, data["id"], data["source"]["filename"], track_ids)

    return batches.present(await run_in_threadpool(queue))


def _send(path: Path, name: str) -> FileResponse:
    return FileResponse(path, media_type=MEDIA_TYPES.get(Path(name).suffix.lower(), "application/octet-stream"),
                        filename=name)


@router.get("/{bid}/items/{sid}/export")
def export_file(bid: str, sid: str):
    batch = _load(bid)
    try:
        return _send(*batches.export_item(batch, sid))
    except sessions.SessionNotFound:
        raise HTTPException(status_code=404, detail="This file's session was deleted.") from None
    except ValueError as err:
        raise HTTPException(status_code=409, detail=str(err)) from None
    except RuntimeError as err:  # FFmpeg missing / failed
        raise HTTPException(status_code=500, detail=str(err)) from None


@router.post("/{bid}/export")
def export_batch(bid: str):
    _load(bid)
    try:
        return _send(*batches.export_all(bid))
    except ValueError as err:
        raise HTTPException(status_code=409, detail=str(err)) from None
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=str(err)) from None


@router.delete("/{bid}")
def delete_batch(bid: str, sessions: bool = False):  # noqa: A002 - the query parameter's name
    _load(bid)
    batches.delete(bid, with_sessions=sessions)
    return {"deleted": bid}
