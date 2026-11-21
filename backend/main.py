"""
AudioGhost AI - FastAPI Backend
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import auth, batches, models, separate, sessions, tasks
from settings import (FRONTEND_ORIGIN_REGEX, FRONTEND_ORIGINS, INPROCESS, LEGACY_RETENTION_HOURS, OUTPUT_DIR,
                      RETENTION_HOURS, STATIC_DIR)

VERSION = "1.4.0"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import sessions as session_store
    from workers.tasks import cleanup_old_files

    removed = await run_in_threadpool(cleanup_old_files, LEGACY_RETENTION_HOURS)
    removed_sessions = await run_in_threadpool(session_store.cleanup, RETENTION_HOURS)
    if removed or removed_sessions:
        print(f"[AudioGhost] Removed {removed} old files and {removed_sessions} sessions")
    if INPROCESS:
        # Desktop app: this process is also the worker. Same start-up as a Celery
        # worker: fail interrupted tracks, start the heartbeat, warm up.
        from workers import dispatch
        from workers import tasks as worker_tasks

        worker_tasks._start_heartbeat()
        worker_tasks._warm_up()
    yield
    if INPROCESS:
        dispatch.shutdown()
        worker_tasks._stop_heartbeat()


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
    expose_headers=["Content-Disposition"],
)

# Direct file access (kept for backwards compatibility)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(separate.router, prefix="/api/separate", tags=["Separation"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(batches.router, prefix="/api/batches", tags=["Batches"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])


if not STATIC_DIR:
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
    info = (queue_state.worker_info() or {}) if redis_ok else {}
    heartbeat = info.get("state")
    worker_ok = bool(running or heartbeat)
    if redis_ok and not worker_ok and not INPROCESS:
        # Fallback for workers that don't send heartbeats (older versions).
        try:
            replies = await run_in_threadpool(celery_app.control.ping, timeout=1.0)
            worker_ok = bool(replies)
        except Exception:  # noqa: BLE001
            worker_ok = False
    return {
        "version": VERSION,
        "redis": redis_ok,
        "worker": worker_ok,
        "worker_starting": heartbeat == "starting",
        "busy": bool(running),
        # From the worker's heartbeat: device, GPU name, memory, loaded model.
        "device": {k: info[k] for k in ("device", "gpu", "vram_total_gb", "vram_used_gb", "model")
                   if k in info} or None,
    }


if STATIC_DIR:
    # Desktop app: the exported web UI, on the same origin as the API.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
