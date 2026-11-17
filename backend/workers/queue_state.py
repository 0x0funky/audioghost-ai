"""
Small Redis-backed bookkeeping shared by the API and the worker:

* queue order, so waiting users can see how many jobs are ahead of them,
* which task is running (the solo-pool worker can't answer pings while busy),
* a worker heartbeat, refreshed by a background thread, so the UI knows the
  worker is alive while warm-up or a job keeps it from answering pings,
* cooperative cancellation flags (``revoke(terminate=True)`` does nothing to a
  task that is already running in a ``--pool=solo`` worker).

Every helper fails soft: if Redis is unreachable, the app still works, it just
loses these extras.
"""
import contextlib
import json
import logging
import time
from typing import Optional

from settings import REDIS_URL

logger = logging.getLogger(__name__)

QUEUE_KEY = "audioghost:queue"
RUNNING_KEY = "audioghost:running"
CANCEL_PREFIX = "audioghost:cancel:"
LOCK_PREFIX = "audioghost:lock:"
HEARTBEAT_KEY = "audioghost:worker"
RUNNING_TTL_SECONDS = 3600
CANCEL_TTL_SECONDS = 3600
#: The worker refreshes its heartbeat every 15 s; a worker that died is noticed
#: within a minute.
HEARTBEAT_TTL_SECONDS = 60

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


def worker_heartbeat(state: str, info: Optional[dict] = None) -> None:
    """
    Record that a worker is alive: "starting" during warm-up, then "ready".
    ``info`` (GPU name, memory, loaded model) is shown in the UI's status chip.
    """
    try:
        payload = json.dumps({"state": state, **(info or {})})
        _redis().set(HEARTBEAT_KEY, payload, ex=HEARTBEAT_TTL_SECONDS)
    except Exception as err:  # noqa: BLE001
        logger.debug("heartbeat failed: %s", err)


def _heartbeat() -> Optional[dict]:
    try:
        value = _redis().get(HEARTBEAT_KEY)
    except Exception:  # noqa: BLE001
        return None
    if not value:
        return None
    text = value.decode(errors="replace")
    if not text.startswith("{"):
        return {"state": text}  # v1.1.1 workers send a bare state
    try:
        return json.loads(text)
    except ValueError:
        return None


def worker_info() -> Optional[dict]:
    """The worker's last heartbeat (state plus whatever info it sent), or None."""
    return _heartbeat()


def worker_gone() -> None:
    try:
        _redis().delete(HEARTBEAT_KEY)
    except Exception:  # noqa: BLE001
        pass


def worker_state() -> Optional[str]:
    """"starting", "ready", or None if no worker sent a heartbeat recently."""
    beat = _heartbeat()
    return beat.get("state") if beat else None


@contextlib.contextmanager
def redis_lock(name: str, timeout: float = 30.0):
    """
    A lock shared by the API and the worker (both edit session files).
    Degrades to no lock if Redis is unavailable rather than failing the call.
    """
    lock = None
    try:
        lock = _redis().lock(LOCK_PREFIX + name, timeout=timeout, blocking_timeout=timeout)
        if not lock.acquire():
            logger.warning("Could not lock %s within %.0f s; going ahead without it", name, timeout)
            lock = None
    except Exception as err:  # noqa: BLE001
        logger.debug("lock %s unavailable: %s", name, err)
        lock = None
    try:
        yield
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:  # noqa: BLE001 - expired locks raise on release
                pass


def clear_running() -> None:
    """Forget the running job (a worker that just started is running nothing)."""
    try:
        _redis().delete(RUNNING_KEY)
    except Exception:  # noqa: BLE001
        pass


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
