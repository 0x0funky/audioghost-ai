"""
AudioGhost AI - FastAPI Backend
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import auth, separate, tasks
from settings import FRONTEND_ORIGIN_REGEX, FRONTEND_ORIGINS, OUTPUT_DIR, RETENTION_HOURS

VERSION = "1.1.0"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from workers.tasks import cleanup_old_files

    removed = await run_in_threadpool(cleanup_old_files, RETENTION_HOURS)
    if removed:
        print(f"[AudioGhost] Removed {removed} files older than {RETENTION_HOURS:g}h")
    yield


app = FastAPI(
    title="AudioGhost AI",
    description="AI-Powered Audio Separation Tool",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_origin_regex=FRONTEND_ORIGIN_REGEX or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Direct file access (kept for backwards compatibility)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(separate.router, prefix="/api/separate", tags=["Separation"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])


@app.get("/")
async def root():
    return {"name": "AudioGhost AI", "version": VERSION, "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/system/status", tags=["System"])
async def system_status():
    """Is Redis up, is a worker listening? Used by the UI to explain problems."""
    from workers import queue_state
    from workers.celery_app import celery_app

    redis_ok = await run_in_threadpool(queue_state.redis_ok)
    running = queue_state.running_task() if redis_ok else None
    worker_ok = bool(running)
    if redis_ok and not worker_ok:
        try:
            replies = await run_in_threadpool(celery_app.control.ping, timeout=1.0)
            worker_ok = bool(replies)
        except Exception:  # noqa: BLE001
            worker_ok = False
    return {
        "version": VERSION,
        "redis": redis_ok,
        "worker": worker_ok,
        "busy": bool(running),
    }
