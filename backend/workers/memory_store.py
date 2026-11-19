"""
In-memory stand-in for the few Redis commands ``queue_state`` uses.

The desktop app runs the API and the separation in one process
(``AUDIOGHOST_INPROCESS=1``), so the shared bookkeeping (queue order, running
job, heartbeat, cancel flags, locks) only has to be shared between threads.
Values are bytes, like redis-py returns them.
"""
import threading
import time
from typing import Dict, Optional, Tuple


class MemoryLock:
    """Non-reentrant named lock with an expiry, like ``redis.lock.Lock``."""

    def __init__(self, store: "MemoryRedis", name: str, timeout: Optional[float],
                 blocking_timeout: Optional[float]):
        self._store = store
        self._name = name
        self._timeout = timeout
        self._blocking_timeout = blocking_timeout
        self._token: Optional[object] = None

    def acquire(self) -> bool:
        wait = self._blocking_timeout if self._blocking_timeout is not None else float("inf")
        deadline = time.monotonic() + wait
        token = object()
        while True:
            with self._store._mutex:
                now = time.monotonic()
                holder = self._store._locks.get(self._name)
                if holder is None or holder[1] <= now:
                    expires = now + self._timeout if self._timeout else float("inf")
                    self._store._locks[self._name] = (token, expires)
                    self._token = token
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

    def release(self) -> None:
        with self._store._mutex:
            holder = self._store._locks.get(self._name)
            if holder is not None and holder[0] is self._token:
                del self._store._locks[self._name]
        self._token = None


class MemoryRedis:
    """Thread-safe subset of the redis-py client API."""

    def __init__(self):
        self._mutex = threading.RLock()
        self._values: Dict[str, Tuple[bytes, float]] = {}
        self._zsets: Dict[str, Dict[str, float]] = {}
        self._locks: Dict[str, Tuple[object, float]] = {}

    @staticmethod
    def _bytes(value) -> bytes:
        return value if isinstance(value, bytes) else str(value).encode()

    def _value(self, key: str) -> Optional[bytes]:
        item = self._values.get(key)
        if item is None:
            return None
        value, expires = item
        if expires and expires <= time.monotonic():
            del self._values[key]
            return None
        return value

    def ping(self) -> bool:
        return True

    def set(self, key: str, value, ex: Optional[float] = None) -> bool:
        with self._mutex:
            self._values[key] = (self._bytes(value), time.monotonic() + ex if ex else 0.0)
        return True

    def get(self, key: str) -> Optional[bytes]:
        with self._mutex:
            return self._value(key)

    def delete(self, *keys: str) -> int:
        with self._mutex:
            removed = 0
            for key in keys:
                removed += int(self._values.pop(key, None) is not None)
                removed += int(self._zsets.pop(key, None) is not None)
            return removed

    def exists(self, *keys: str) -> int:
        with self._mutex:
            return sum(1 for key in keys if self._value(key) is not None or key in self._zsets)

    def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        with self._mutex:
            zset = self._zsets.setdefault(key, {})
            added = sum(1 for member in mapping if str(member) not in zset)
            zset.update({str(member): float(score) for member, score in mapping.items()})
            return added

    def zrem(self, key: str, *members: str) -> int:
        with self._mutex:
            zset = self._zsets.get(key, {})
            return sum(1 for member in members if zset.pop(str(member), None) is not None)

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        with self._mutex:
            zset = self._zsets.get(key, {})
            gone = [member for member, score in zset.items() if min_score <= score <= max_score]
            for member in gone:
                del zset[member]
            return len(gone)

    def zrank(self, key: str, member: str) -> Optional[int]:
        with self._mutex:
            zset = self._zsets.get(key, {})
            if str(member) not in zset:
                return None
            order = sorted(zset.items(), key=lambda item: (item[1], item[0]))
            return [name for name, _ in order].index(str(member))

    def lock(self, name: str, timeout: Optional[float] = None,
             blocking_timeout: Optional[float] = None) -> MemoryLock:
        return MemoryLock(self, name, timeout, blocking_timeout)
