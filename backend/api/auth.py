"""
Authentication API - HuggingFace Token Management
"""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from config import settings

router = APIRouter()


class TokenRequest(BaseModel):
    token: str

class AuthStatus(BaseModel):
    authenticated: bool
    model_downloaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None


def get_saved_token() -> Optional[str]:
    """Get saved HuggingFace token (from env var or file)"""
    return settings.get_hf_token()


def save_token(token: str):
    """Save HuggingFace token"""
    settings.save_hf_token(token)


def check_model_downloaded() -> bool:
    """Check if SAM Audio model is downloaded"""
    # Check for common model files in checkpoints directory
    model_files = list(settings.CHECKPOINTS_DIR.glob("*.safetensors")) + \
                  list(settings.CHECKPOINTS_DIR.glob("*.bin"))
    return len(model_files) > 0


def get_device_info() -> str:
    """Get current device information for display."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
        return "CPU"
    except Exception:
        return "Unknown"


@router.get("/status", response_model=AuthStatus)
async def get_auth_status():
    """Check authentication and model status"""
    token = get_saved_token()
    authenticated = False
    
    if token:
        try:
            api = HfApi(token=token)
            api.whoami()
            authenticated = True
        except Exception:
            authenticated = False
    
    return AuthStatus(
        authenticated=authenticated,
        model_downloaded=check_model_downloaded(),
        model_name="facebook/sam-audio-large" if check_model_downloaded() else None,
        device=get_device_info()
    )


@router.post("/login")
async def login(request: TokenRequest):
    """Validate and save HuggingFace token"""
    try:
        # Validate token
        api = HfApi(token=request.token)
        user_info = api.whoami()
        
        # Check if user has access to SAM Audio
        try:
            api.model_info("facebook/sam-audio-large", token=request.token)
        except HfHubHTTPError as e:
            if "403" in str(e) or "401" in str(e):
                raise HTTPException(
                    status_code=403,
                    detail="You need to request access to facebook/sam-audio-large on HuggingFace first"
                )
            raise
        
        # Save token
        save_token(request.token)
        
        return {
            "success": True,
            "username": user_info.get("name", "Unknown"),
            "message": "Successfully authenticated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


@router.post("/download-model")
async def download_model():
    """Download SAM Audio model"""
    token = get_saved_token()
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Note: In production, this should be a background task
        # For MVP, we'll use the HuggingFace auto-download feature
        # which downloads on first use
        
        return {
            "success": True,
            "message": "Model will be downloaded automatically on first use"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/logout")
async def logout():
    """Clear saved token"""
    settings.clear_hf_token()
    return {"success": True, "message": "Logged out"}
