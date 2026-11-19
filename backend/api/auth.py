"""
Authentication API - HuggingFace Token Management
"""
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import HfApi
from pydantic import BaseModel

import model_license
from settings import TOKEN_FILE, read_hf_token

router = APIRouter()

ACCESS_CHECK_REPO = "facebook/sam-audio-large"


class TokenRequest(BaseModel):
    token: str


class AuthStatus(BaseModel):
    authenticated: bool
    model_downloaded: bool
    model_name: Optional[str] = None
    downloaded_models: List[str] = []
    username: Optional[str] = None
    #: Meta's SAM License, accepted in the app before the first download of AudioGhost's copies.
    license_accepted: bool = False


def get_saved_token() -> Optional[str]:
    """Get saved HuggingFace token"""
    return read_hf_token()


def save_token(token: str):
    """Save HuggingFace token (readable by the current user only where supported)."""
    TOKEN_FILE.write_text(token, encoding="utf-8")
    try:
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass


def downloaded_models() -> List[str]:
    """
    Models that are ready on this computer (no network): Meta's checkpoints
    (``facebook/sam-audio-<size>[-tv]``) or AudioGhost's copies
    (``audioghostai/sam-audio-<size>``, default fp16 files).
    """
    from workers import model_files

    found = []
    for size in model_files.MODEL_SIZES:
        for repo in (f"facebook/sam-audio-{size}", f"{model_files.MODEL_ORG}/sam-audio-{size}",
                     f"facebook/sam-audio-{size}-tv"):
            if model_files.is_downloaded(repo):
                found.append(repo)
    return found


def check_model_downloaded() -> bool:
    """True if at least one SAM-Audio checkpoint is cached locally."""
    return bool(downloaded_models())


@router.get("/status", response_model=AuthStatus)
async def get_auth_status():
    """Check authentication and model status"""
    token = get_saved_token()
    username = None
    authenticated = False
    if token:
        try:
            info = await run_in_threadpool(HfApi(token=token).whoami)
            username = info.get("name")
            authenticated = True
        except Exception:  # noqa: BLE001
            authenticated = False

    models = await run_in_threadpool(downloaded_models)
    return AuthStatus(
        authenticated=authenticated,
        model_downloaded=bool(models),
        model_name=models[0] if models else None,
        downloaded_models=models,
        username=username,
        license_accepted=model_license.accepted(),
    )


@router.post("/login")
async def login(request: TokenRequest):
    """Validate and save HuggingFace token"""
    token = request.token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token must not be empty")
    api = HfApi(token=token)
    try:
        user_info = await run_in_threadpool(api.whoami)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    try:
        # model_info() works on gated repos without access; auth_check() doesn't.
        await run_in_threadpool(api.auth_check, ACCESS_CHECK_REPO)
    except Exception as e:  # noqa: BLE001
        text = str(e)
        if type(e).__name__ == "GatedRepoError" or "403" in text or "401" in text \
                or "gated" in text.lower():
            raise HTTPException(
                status_code=403,
                detail=(f"You need to request access to {ACCESS_CHECK_REPO} on HuggingFace first: "
                        f"https://huggingface.co/{ACCESS_CHECK_REPO}"),
            )
        raise HTTPException(status_code=502, detail=f"Could not reach HuggingFace: {e}")

    save_token(token)
    return {
        "success": True,
        "username": user_info.get("name", "Unknown"),
        "message": "Successfully authenticated",
    }


@router.post("/download-model")
async def download_model():
    """Models download automatically on first use (progress shows in the task)."""
    if not get_saved_token():
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"success": True, "message": "Model will be downloaded automatically on first use"}


@router.post("/logout")
async def logout():
    """Forget the token. An empty file (not none) also stops the fallback to
    HF_TOKEN or the Hugging Face CLI's saved token."""
    TOKEN_FILE.write_text("", encoding="utf-8")
    return {"success": True, "message": "Logged out"}
