"""
AudioGhost AI - FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import auth, separate, tasks
from config import settings

app = FastAPI(
    title="AudioGhost AI",
    description="AI-Powered Audio Separation Tool",
    version="1.0.0"
)

# CORS Configuration (configurable via CORS_ORIGINS env var)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for downloads
app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="outputs")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(separate.router, prefix="/api/separate", tags=["Separation"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])


@app.get("/")
async def root():
    return {
        "name": "AudioGhost AI",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
