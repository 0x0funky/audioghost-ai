"""
Desktop mode (AUDIOGHOST_INPROCESS=1): the in-memory stand-in for Redis, jobs
run on a background thread, and a restart fails everything that was queued.
No model, GPU or Redis needed.

Run from the repo root:  python -m pytest backend/tests -q
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sessions  # noqa: E402
from workers import dispatch, queue_state  # noqa: E402
from workers.memory_store import MemoryRedis  # noqa: E402


def test_values_expire_and_come_back_as_bytes(monkeypatch):
    store = MemoryRedis()
    store.set("a", "1", ex=60)
    store.set("b", b"2")
    assert store.get("a") == b"1" and store.get("b") == b"2"
    assert store.exists("a", "b", "c") == 2
    clock = {"now": time.monotonic()}
    monkeypatch.setattr("workers.memory_store.time.monotonic", lambda: clock["now"])
    clock["now"] += 61
    assert store.get("a") is None and store.get("b") == b"2"
    assert store.delete("a", "b") == 1 and store.get("b") is None


def test_sorted_set_ranks_like_redis():
    store = MemoryRedis()
    store.zadd("q", {"late": 3.0})
    store.zadd("q", {"early": 1.0, "middle": 2.0})
    assert [store.zrank("q", m) for m in ("early", "middle", "late")] == [0, 1, 2]
    assert store.zrank("q", "missing") is None
    assert store.zrem("q", "middle") == 1
    assert store.zremrangebyscore("q", 0, 1.5) == 1
    assert store.zrank("q", "late") == 0


def test_lock_is_exclusive_and_times_out():
    store = MemoryRedis()
    first = store.lock("session:x", timeout=30, blocking_timeout=1)
    assert first.acquire()
    second = store.lock("session:x", timeout=30, blocking_timeout=0.05)
    assert not second.acquire()
    other = store.lock("session:y", timeout=30, blocking_timeout=0.05)
    assert other.acquire()
    first.release()
    assert second.acquire()


def test_queue_state_uses_memory_store_in_desktop_mode(monkeypatch):
    monkeypatch.setattr(queue_state, "INPROCESS", True)
    monkeypatch.setattr(queue_state, "_client", None)
    queue_state.enqueue("job-1")
    queue_state.enqueue("job-2")
    assert queue_state.redis_ok()
    assert queue_state.jobs_ahead("job-2") == 1
    queue_state.mark_started("job-1")
    assert queue_state.running_task() == "job-1"
    assert queue_state.jobs_ahead("job-2") == 1  # job-1 is running
    queue_state.request_cancel("job-2")
    assert queue_state.is_cancelled("job-2")
    with queue_state.redis_lock("session:abc", timeout=1):
        pass
    assert isinstance(queue_state._client, MemoryRedis)


def test_desktop_jobs_run_on_a_background_thread(monkeypatch):
    monkeypatch.setattr(dispatch, "INPROCESS", True)
    monkeypatch.setattr(dispatch, "_executor", None)
    ran = {}
    done = threading.Event()

    class Result:
        def failed(self):
            return False

    class Task:
        def apply(self, args, kwargs, task_id):
            ran.update(args=args, kwargs=kwargs, task_id=task_id, thread=threading.current_thread().name)
            done.set()
            return Result()

        def apply_async(self, **_):
            raise AssertionError("desktop mode must not use the Celery broker")

    dispatch.submit(Task(), args=["s1", ["t1"], {}], task_id="job-9")
    assert done.wait(5)
    assert ran["args"] == ["s1", ["t1"], {}] and ran["kwargs"] == {} and ran["task_id"] == "job-9"
    assert ran["thread"].startswith("audioghost-job")
    dispatch.shutdown()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSION_DIR", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    monkeypatch.setattr(queue_state, "_client", None)
    monkeypatch.setattr(queue_state, "_redis", lambda: (_ for _ in ()).throw(ConnectionError("no redis")))
    return tmp_path


def test_restart_fails_queued_tracks_too(store):
    sr = sessions.SAMPLE_RATE
    upload = store / "clip.wav"
    sf.write(str(upload), np.zeros((sr, 1), dtype=np.float32), sr, subtype="FLOAT")
    sid = sessions.create(upload, "clip.wav", is_video=False)["id"]
    sessions.add_tracks(sid, [{"text": "a"}, {"text": "b"}], {}, "job-1")
    # The web worker resumes queued jobs from Redis: only a running job is lost.
    assert sessions.recover_interrupted(set()) == 0
    # The desktop app's queue lived in memory: everything active is gone.
    assert sessions.recover_interrupted(set(), all_active=True) == 2
    assert {t["status"] for t in sessions.load(sid)["tracks"]} == {"failed"}
