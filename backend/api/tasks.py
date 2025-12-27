"""
Tasks API - Task Status and Results
"""
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from celery.result import AsyncResult

from workers.celery_app import celery_app

router = APIRouter()

OUTPUT_DIR = Path("outputs")


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: Optional[str] = None
    result: Optional[dict] = None


class TaskResult(BaseModel):
    original_url: str
    ghost_url: str  # Separated target
    clean_url: str  # Residual


@router.get("/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a separation task"""
    
    result = AsyncResult(task_id, app=celery_app)
    
    if result.state == "PENDING":
        return TaskStatus(
            task_id=task_id,
            status="pending",
            progress=0,
            message="Task is waiting to be processed"
        )
    
    elif result.state == "PROGRESS":
        info = result.info or {}
        return TaskStatus(
            task_id=task_id,
            status="processing",
            progress=info.get("progress", 0),
            message=info.get("message", "Processing...")
        )
    
    elif result.state == "SUCCESS":
        return TaskStatus(
            task_id=task_id,
            status="completed",
            progress=100,
            message="Task completed successfully",
            result=result.result
        )
    
    elif result.state == "FAILURE":
        return TaskStatus(
            task_id=task_id,
            status="failed",
            progress=0,
            message=str(result.info)
        )
    
    else:
        return TaskStatus(
            task_id=task_id,
            status=result.state.lower(),
            progress=0,
            message=f"Task state: {result.state}"
        )


@router.get("/{task_id}/download/{file_type}")
async def download_result(task_id: str, file_type: str):
    """
    Download processed audio or video file
    
    - **file_type**: "original", "ghost", "clean", or "video"
    """
    
    if file_type not in ["original", "ghost", "clean", "video"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    result = AsyncResult(task_id, app=celery_app)
    
    if result.state != "SUCCESS":
        raise HTTPException(status_code=404, detail="Task not completed")
    
    # Handle video file separately
    if file_type == "video":
        video_path = result.result.get("video_path")
        if not video_path:
            raise HTTPException(status_code=404, detail="No video file for this task")
        
        file_path = Path(video_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Determine media type based on extension
        extension = file_path.suffix.lower()
        media_types = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska"
        }
        media_type = media_types.get(extension, "video/mp4")
        
        return FileResponse(
            path=file_path,
            filename=f"{task_id}_video{extension}",
            media_type=media_type
        )
    
    # Handle audio files
    file_path = Path(result.result.get(f"{file_type}_path", ""))
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=f"{task_id}_{file_type}.wav",
        media_type="audio/wav"
    )


@router.get("/{task_id}/download-video-with-audio/{audio_type}")
async def download_video_with_audio(task_id: str, audio_type: str):
    """
    Download video with merged audio track
    
    - **audio_type**: "original", "ghost", or "clean"
    """
    import subprocess
    import tempfile
    import os
    
    if audio_type not in ["original", "ghost", "clean"]:
        raise HTTPException(status_code=400, detail="Invalid audio type. Use 'original', 'ghost', or 'clean'")
    
    result = AsyncResult(task_id, app=celery_app)
    
    if result.state != "SUCCESS":
        raise HTTPException(status_code=404, detail="Task not completed")
    
    # Get video path
    video_path = result.result.get("video_path")
    if not video_path:
        raise HTTPException(status_code=404, detail="No video file for this task")
    
    video_file = Path(video_path)
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Get audio path
    audio_path = Path(result.result.get(f"{audio_type}_path", ""))
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file '{audio_type}' not found")
    
    # Create output file in the same directory as video
    output_dir = video_file.parent
    extension = video_file.suffix.lower()
    output_filename = f"{task_id}_{audio_type}_merged{extension}"
    output_path = output_dir / output_filename
    
    # Use FFmpeg to merge video and audio
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_file),
            "-i", str(audio_path),
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-c:a", "aac",   # Encode audio to AAC
            "-b:a", "192k",  # Audio bitrate
            "-map", "0:v:0", # Use video from first input
            "-map", "1:a:0", # Use audio from second input
            "-shortest",     # Match shortest stream
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr.decode()}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="FFmpeg not found. Please install FFmpeg.")
    
    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Failed to create merged video")
    
    # Determine media type
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska"
    }
    media_type = media_types.get(extension, "video/mp4")
    
    # Map audio type to display name
    audio_labels = {
        "original": "original",
        "ghost": "isolated",
        "clean": "without_isolated"
    }
    
    return FileResponse(
        path=output_path,
        filename=f"{task_id}_{audio_labels[audio_type]}_video{extension}",
        media_type=media_type
    )


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending or running task"""
    
    result = AsyncResult(task_id, app=celery_app)
    result.revoke(terminate=True)
    
    return {"success": True, "message": "Task cancelled"}


@router.get("/", response_model=List[TaskStatus])
async def list_recent_tasks(limit: int = 10):
    """List recent tasks (simplified - in production would use database)"""
    # Note: This is a simplified implementation
    # In production, you would store task metadata in a database
    
    return []
