"""Single active producer lock and conflict manager (GS-08, #10197).

Ensures only one producer owns the simulator session at any given time,
preventing collisions with real launch monitors or multiple control clients.
Never terminates or kills unrelated processes on conflict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import threading


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_to_iso(dt: datetime) -> str:
    return dt.isoformat()


class ProducerConflictError(RuntimeError):
    """Raised when another producer holds an active session lease."""

    def __init__(
        self,
        active_producer_id: str,
        session_id: str,
        expires_at_utc: str,
        message: str | None = None,
    ) -> None:
        self.active_producer_id = active_producer_id
        self.session_id = session_id
        self.expires_at_utc = expires_at_utc
        msg = (
            message
            or f"Active producer '{active_producer_id}' holds session '{session_id}' until {expires_at_utc}"
        )
        super().__init__(msg)


@dataclass(frozen=True)
class ProducerLock:
    """Immutable lease token granting exclusive session ownership."""

    producer_id: str
    session_id: str
    lease_ttl_seconds: float
    acquired_at_utc: str
    expires_at_utc: str

    def is_expired(self, now: datetime | None = None) -> bool:
        current = now or _utc_now()
        exp = datetime.fromisoformat(self.expires_at_utc)
        return current >= exp


class ProducerLockManager:
    """Thread-safe lease manager for simulator sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_lock: ProducerLock | None = None

    def is_locked(self) -> bool:
        with self._lock:
            if self._current_lock is None:
                return False
            if self._current_lock.is_expired():
                self._current_lock = None
                return False
            return True

    def current_producer(self) -> str | None:
        with self._lock:
            if self._current_lock is None or self._current_lock.is_expired():
                self._current_lock = None
                return None
            return self._current_lock.producer_id

    def acquire(
        self,
        producer_id: str,
        session_id: str,
        ttl_seconds: float = 60.0,
    ) -> ProducerLock:
        """Acquire exclusive session lock for producer.

        Raises:
            ValueError: If producer_id or session_id is empty, or ttl_seconds <= 0.
            ProducerConflictError: If another active producer holds an unexpired lock.
        """
        if not producer_id or not str(producer_id).strip():
            raise ValueError("producer_id must be a non-empty string")
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id must be a non-empty string")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        now = _utc_now()
        expires = now + timedelta(seconds=ttl_seconds)
        new_lock = ProducerLock(
            producer_id=str(producer_id),
            session_id=str(session_id),
            lease_ttl_seconds=float(ttl_seconds),
            acquired_at_utc=_utc_to_iso(now),
            expires_at_utc=_utc_to_iso(expires),
        )

        with self._lock:
            if self._current_lock is not None and not self._current_lock.is_expired(
                now
            ):
                if self._current_lock.producer_id != producer_id:
                    raise ProducerConflictError(
                        active_producer_id=self._current_lock.producer_id,
                        session_id=self._current_lock.session_id,
                        expires_at_utc=self._current_lock.expires_at_utc,
                    )
            # Either unheld, expired, or held by same producer
            self._current_lock = new_lock
            return new_lock

    def renew(
        self, lock: ProducerLock, extension_seconds: float = 60.0
    ) -> ProducerLock:
        """Extend TTL of an existing lock held by the same producer."""
        if extension_seconds <= 0:
            raise ValueError("extension_seconds must be positive")

        now = _utc_now()
        with self._lock:
            if self._current_lock is not None and not self._current_lock.is_expired(
                now
            ):
                if self._current_lock.producer_id != lock.producer_id:
                    raise ProducerConflictError(
                        active_producer_id=self._current_lock.producer_id,
                        session_id=self._current_lock.session_id,
                        expires_at_utc=self._current_lock.expires_at_utc,
                    )
            expires = now + timedelta(seconds=extension_seconds)
            renewed = ProducerLock(
                producer_id=lock.producer_id,
                session_id=lock.session_id,
                lease_ttl_seconds=extension_seconds,
                acquired_at_utc=lock.acquired_at_utc,
                expires_at_utc=_utc_to_iso(expires),
            )
            self._current_lock = renewed
            return renewed

    def release(self, lock: ProducerLock) -> None:
        """Voluntarily release session lock."""
        with self._lock:
            if self._current_lock is None or self._current_lock.is_expired():
                self._current_lock = None
                return
            if self._current_lock.producer_id != lock.producer_id:
                raise ValueError("Cannot release lock held by another producer")
            if self._current_lock.expires_at_utc != lock.expires_at_utc:
                raise ValueError("Cannot release stale lock token")
            self._current_lock = None
