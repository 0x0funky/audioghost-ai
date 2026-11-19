"""
Start a separation job.

The web version hands it to the Celery worker through Redis. The desktop app
(``AUDIOGHOST_INPROCESS=1``) runs it on a background thread of the API process
instead, through Celery's local ``apply`` so the task code is the same. One job
runs at a time in both cases, like the ``--pool=solo`` worker.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from settings import INPROCESS

logger = logging.getLogger(__name__)

_executor: Optional[ThreadPoolExecutor] = None


def _run(task, args: list, kwargs: dict, task_id: Optional[str]) -> None:
    try:
        result = task.apply(args=args, kwargs=kwargs, task_id=task_id)
        if result.failed():
            logger.error("Job %s failed: %s", task_id, result.result)
    except Exception:  # noqa: BLE001 - a crashed job must not stop the executor
        logger.exception("Job %s crashed", task_id)


def submit(task, args: Optional[Sequence] = None, kwargs: Optional[dict] = None,
           task_id: Optional[str] = None) -> None:
    """Queue ``task`` with ``args`` / ``kwargs`` under ``task_id``."""
    if not INPROCESS:
        call = {"task_id": task_id}
        if args is not None:
            call["args"] = list(args)
        if kwargs is not None:
            call["kwargs"] = dict(kwargs)
        task.apply_async(**call)
        return
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audioghost-job")
    _executor.submit(_run, task, list(args or []), dict(kwargs or {}), task_id)


def shutdown() -> None:
    """Stop taking jobs (the desktop app is closing); a running job is abandoned."""
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
