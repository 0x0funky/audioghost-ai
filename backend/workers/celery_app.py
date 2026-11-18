"""
Celery Application Configuration
"""
from celery import Celery

from settings import INPROCESS, REDIS_URL

# The desktop app runs jobs in the API process (workers.dispatch), so its broker
# and result store live in memory; task progress is read from the same process.
celery_app = Celery(
    "audioghost",
    broker="memory://" if INPROCESS else REDIS_URL,
    backend="cache+memory://" if INPROCESS else REDIS_URL,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # one GPU job at a time
    result_expires=86400,  # results (and the files behind them) live for 24 hours
    broker_connection_retry_on_startup=True,
)
