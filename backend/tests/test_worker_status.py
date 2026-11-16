"""
Worker heartbeat bookkeeping and the /api/system/status worker check.
No Redis needed: a small in-memory stand-in replaces the client.

Run from the repo root:  python -m pytest backend/tests -q
"""
import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers import queue_state  # noqa: E402


class FakeRedis:
    """The handful of Redis commands queue_state uses, with expiry."""

    def __init__(self):
        self.data = {}

    def _live(self, key):
        item = self.data.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and time.monotonic() >= expires:
            del self.data[key]
            return None
        return value

    def ping(self):
        return True

    def set(self, key, value, ex=None):
        if isinstance(value, str):
            value = value.encode()
        self.data[key] = (value, None if ex is None else time.monotonic() + ex)

    def get(self, key):
        return self._live(key)

    def delete(self, key):
        self.data.pop(key, None)

    def exists(self, key):
        return int(self._live(key) is not None)

    def zadd(self, key, mapping):
        pass

    def zrem(self, key, member):
        pass


@pytest.fixture
def fake_redis(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(queue_state, "_client", client)
    return client


def test_heartbeat_states(fake_redis):
    assert queue_state.worker_state() is None
    queue_state.worker_heartbeat("starting")
    assert queue_state.worker_state() == "starting"
    queue_state.worker_heartbeat("ready")
    assert queue_state.worker_state() == "ready"
    queue_state.worker_gone()
    assert queue_state.worker_state() is None


def test_heartbeat_expires(fake_redis, monkeypatch):
    monkeypatch.setattr(queue_state, "HEARTBEAT_TTL_SECONDS", 0.05)
    queue_state.worker_heartbeat("ready")
    assert queue_state.worker_state() == "ready"
    time.sleep(0.08)
    assert queue_state.worker_state() is None


def test_heartbeat_fails_soft_without_redis(monkeypatch):
    class Down:
        def __getattr__(self, _name):
            def fail(*_args, **_kwargs):
                raise ConnectionError("redis is down")
            return fail

    monkeypatch.setattr(queue_state, "_client", Down())
    queue_state.worker_heartbeat("ready")  # must not raise
    queue_state.worker_gone()
    assert queue_state.worker_state() is None


def _status(monkeypatch, *, ping_replies):
    import main
    from workers.celery_app import celery_app

    calls = []

    def fake_ping(timeout=1.0):
        calls.append(timeout)
        return ping_replies

    monkeypatch.setattr(celery_app.control, "ping", fake_ping)
    return asyncio.run(main.system_status()), calls


def test_status_uses_heartbeat_without_pinging(fake_redis, monkeypatch):
    queue_state.worker_heartbeat("starting")
    status, pings = _status(monkeypatch, ping_replies=[])
    assert status["worker"] is True
    assert status["worker_starting"] is True
    assert pings == []  # a fresh heartbeat is enough


def test_status_busy_worker_counts_as_up(fake_redis, monkeypatch):
    queue_state.mark_started("job-1")
    status, _ = _status(monkeypatch, ping_replies=[])
    assert status["worker"] is True
    assert status["busy"] is True


def test_status_falls_back_to_ping(fake_redis, monkeypatch):
    status, pings = _status(monkeypatch, ping_replies=[{"celery@host": {"ok": "pong"}}])
    assert status["worker"] is True
    assert pings, "no heartbeat, so the status check pings the worker"

    status, _ = _status(monkeypatch, ping_replies=[])
    assert status["worker"] is False
    assert status["worker_starting"] is False
