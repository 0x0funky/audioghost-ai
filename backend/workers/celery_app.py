"""
Celery Application Configuration
"""
from celery import Celery

# Import settings (handles environment variables)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

celery_app = Celery(
    "audioghost",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["workers.tasks"]
)

# Celery Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.TASK_TIME_LIMIT,
    worker_prefetch_multiplier=1,  # Process one task at a time (GPU memory)
    result_expires=settings.RESULT_EXPIRES,
    broker_connection_retry_on_startup=True,  # Celery 6.0 compatibility
)
