"""
Separation API - Audio Separation Endpoints
"""
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from workers.celery_app import celery_app
from workers.tasks import separate_audio_task

router = APIRouter()

UPLOAD_DIR = Path("uploads")


class SeparationRequest(BaseModel):
    description: str
    mode: str = "extract"  # "extract" or "remove"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    model_size: str = "base"  # "small", "base", "large"


class SeparationResponse(BaseModel):
    task_id: str
    status: str
    message: str


@router.post("/", response_model=SeparationResponse)
async def create_separation_task(
    file: UploadFile = File(...),
    description: str = Form(...),
    mode: str = Form("extract"),
    start_time: Optional[float] = Form(None),
    end_time: Optional[float] = Form(None),
    model_size: str = Form("base"),
    chunk_duration: float = Form(25.0)
):
    """
    Create a new audio separation task
    
    - **file**: Audio file to process
    - **description**: Text prompt describing the sound to separate (e.g., "singing voice", "background music")
    - **mode**: "extract" to isolate the sound, "remove" to remove it
    - **start_time**: Optional start time for temporal prompting
    - **end_time**: Optional end time for temporal prompting
    - **model_size**: SAM Audio model size (small/base/large)
    - **chunk_duration**: Audio chunk duration in seconds (5-60, default 25)
    """
    
    # Validate chunk_duration
    chunk_duration = max(5.0, min(60.0, chunk_duration))
    
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/wav", "audio/mp3", "audio/x-wav", "audio/flac"]
    if file.content_type and file.content_type not in allowed_types:
        # Be lenient with content type checking
        pass
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix or ".mp3"
    upload_path = UPLOAD_DIR / f"{task_id}{file_extension}"
    
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Build anchors for temporal prompting
    anchors = None
    if start_time is not None and end_time is not None:
        anchors = [[["+", start_time, end_time]]]
    
    # Submit Celery task
    celery_task = separate_audio_task.apply_async(
        args=[
            str(upload_path),
            description,
            mode,
            anchors,
            model_size,
            chunk_duration
        ],
        task_id=task_id
    )
    
    return SeparationResponse(
        task_id=task_id,
        status="pending",
        message="Task submitted successfully"
    )


@router.post("/batch", response_model=List[SeparationResponse])
async def create_batch_separation(
    file: UploadFile = File(...),
    descriptions: str = Form(...),  # JSON array of descriptions
    mode: str = Form("extract")
):
    """
    Create multiple separation tasks for the same audio file
    Useful for separating multiple stems at once
    """
    import json
    
    try:
        desc_list = json.loads(descriptions)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid descriptions format")
    
    # Save file once
    base_task_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix or ".mp3"
    upload_path = UPLOAD_DIR / f"{base_task_id}{file_extension}"
    
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    responses = []
    for i, desc in enumerate(desc_list):
        task_id = f"{base_task_id}-{i}"
        
        separate_audio_task.apply_async(
            args=[str(upload_path), desc, mode, None, "small"],
            task_id=task_id
        )
        
        responses.append(SeparationResponse(
            task_id=task_id,
            status="pending",
            message=f"Task for '{desc}' submitted"
        ))
    
    return responses
