"""
Small Redis-backed bookkeeping shared by the API and the worker:

* queue order, so waiting users can see how many jobs are ahead of them,
* which task is running (the solo-pool worker can't answer pings while busy),
* cooperative cancellation flags (``revoke(terminate=True)`` does nothing to a
  task that is already running in a ``--pool=solo`` worker).

Every helper fails soft: if Redis is unreachable, the app still works, it just
loses these extras.
"""
import logging
import time
from typing import Optional

from settings import REDIS_URL

logger = logging.getLogger(__name__)

QUEUE_KEY = "audioghost:queue"
RUNNING_KEY = "audioghost:running"
CANCEL_PREFIX = "audioghost:cancel:"
RUNNING_TTL_SECONDS = 3600
CANCEL_TTL_SECONDS = 3600

_client = None


def _redis():
    global _client
    if _client is None:
        import redis

        _client = redis.Redis.from_url(REDIS_URL, socket_timeout=2, socket_connect_timeout=2)
    return _client


def enqueue(task_id: str) -> None:
    try:
        _redis().zadd(QUEUE_KEY, {task_id: time.time()})
    except Exception as err:  # noqa: BLE001
        logger.debug("enqueue bookkeeping failed: %s", err)


def mark_started(task_id: str) -> None:
    try:
        client = _redis()
        client.zrem(QUEUE_KEY, task_id)
        client.set(RUNNING_KEY, task_id, ex=RUNNING_TTL_SECONDS)
    except Exception as err:  # noqa: BLE001
        logger.debug("start bookkeeping failed: %s", err)


def touch_running(task_id: str) -> None:
    try:
        _redis().set(RUNNING_KEY, task_id, ex=RUNNING_TTL_SECONDS)
    except Exception:  # noqa: BLE001
        pass


def mark_finished(task_id: str) -> None:
    try:
        client = _redis()
        client.zrem(QUEUE_KEY, task_id)
        if (client.get(RUNNING_KEY) or b"").decode() == task_id:
            client.delete(RUNNING_KEY)
        client.delete(CANCEL_PREFIX + task_id)
    except Exception as err:  # noqa: BLE001
        logger.debug("finish bookkeeping failed: %s", err)


def running_task() -> Optional[str]:
    try:
        value = _redis().get(RUNNING_KEY)
        return value.decode() if value else None
    except Exception:  # noqa: BLE001
        return None


def jobs_ahead(task_id: str) -> Optional[int]:
    """Number of jobs that will run before ``task_id`` (None if unknown)."""
    try:
        client = _redis()
        # Drop entries left behind by crashed workers (older than a day).
        client.zremrangebyscore(QUEUE_KEY, 0, time.time() - 86400)
        rank = client.zrank(QUEUE_KEY, task_id)
        if rank is None:
            return None
        running = client.get(RUNNING_KEY)
        return int(rank) + (1 if running else 0)
    except Exception:  # noqa: BLE001
        return None


def request_cancel(task_id: str) -> None:
    try:
        client = _redis()
        client.set(CANCEL_PREFIX + task_id, "1", ex=CANCEL_TTL_SECONDS)
        client.zrem(QUEUE_KEY, task_id)
    except Exception as err:  # noqa: BLE001
        logger.debug("cancel bookkeeping failed: %s", err)


def is_cancelled(task_id: str) -> bool:
    try:
        return bool(_redis().exists(CANCEL_PREFIX + task_id))
    except Exception:  # noqa: BLE001
        return False


def redis_ok() -> bool:
    try:
        return bool(_redis().ping())
    except Exception:  # noqa: BLE001
        return False
