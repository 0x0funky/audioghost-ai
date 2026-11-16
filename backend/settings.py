"""
Shared paths and settings for the API process and the Celery worker.

Everything is resolved relative to this file, so the backend works no matter
which directory uvicorn / celery are started from. Override via env vars.
"""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.environ.get("AUDIOGHOST_UPLOAD_DIR", BACKEND_DIR / "uploads"))
OUTPUT_DIR = Path(os.environ.get("AUDIOGHOST_OUTPUT_DIR", BACKEND_DIR / "outputs"))
TOKEN_FILE = BACKEND_DIR / ".hf_token"

REDIS_URL = os.environ.get("AUDIOGHOST_REDIS_URL", "redis://localhost:6379/0")

#: Uploads and results older than this are deleted (0 = keep forever).
#: Matches Celery's result_expires, after which the UI can't reach them anyway.
RETENTION_HOURS = float(os.environ.get("AUDIOGHOST_RETENTION_HOURS", "24"))

#: Largest accepted upload in MB (0 = unlimited).
MAX_UPLOAD_MB = float(os.environ.get("AUDIOGHOST_MAX_UPLOAD_MB", "2048"))

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "AUDIOGHOST_FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
#: `next dev` moves to 3001, 3002, ... when 3000 is taken, so accept any local
#: port 3000-3099 as well (still localhost-only).
FRONTEND_ORIGIN_REGEX = os.environ.get(
    "AUDIOGHOST_FRONTEND_ORIGIN_REGEX", r"^http://(localhost|127\.0\.0\.1):30\d{2}$"
)

for _directory in (UPLOAD_DIR, OUTPUT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def read_hf_token():
    """Token saved by the UI, falling back to the standard HF_TOKEN env var."""
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token
    return os.environ.get("HF_TOKEN") or None
