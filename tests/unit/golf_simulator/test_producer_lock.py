"""Unit tests for single-producer simulator session lock and conflict detection (GS-08, #10197).

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

import time
from unittest.mock import patch
import pytest

from src.shared.python.golf_simulator.producer_lock import (
    ProducerConflictError,
    ProducerLock,
    ProducerLockManager,
)

pytestmark = pytest.mark.unit


def test_acquire_and_release_lock() -> None:
    manager = ProducerLockManager()
    assert not manager.is_locked()
    assert manager.current_producer() is None

    lock = manager.acquire(
        producer_id="producer-model-1", session_id="session-42", ttl_seconds=10.0
    )
    assert isinstance(lock, ProducerLock)
    assert lock.producer_id == "producer-model-1"
    assert lock.session_id == "session-42"
    assert manager.is_locked()
    assert manager.current_producer() == "producer-model-1"

    manager.release(lock)
    assert not manager.is_locked()
    assert manager.current_producer() is None


def test_conflict_detection_when_held() -> None:
    manager = ProducerLockManager()
    manager.acquire(
        producer_id="launch-monitor-lm1", session_id="session-42", ttl_seconds=60.0
    )

    # Attempt by a different producer must fail with ProducerConflictError
    with pytest.raises(ProducerConflictError) as exc_info:
        manager.acquire(
            producer_id="model-producer-2", session_id="session-42", ttl_seconds=10.0
        )

    err = exc_info.value
    assert err.active_producer_id == "launch-monitor-lm1"
    assert err.session_id == "session-42"
    assert "Active producer 'launch-monitor-lm1' holds session" in str(err)


def test_no_process_termination_on_conflict() -> None:
    manager = ProducerLockManager()
    manager.acquire(
        producer_id="real-launch-monitor", session_id="session-42", ttl_seconds=60.0
    )

    # Ensure no kill or termination primitives are ever called on conflict
    with patch("os.kill") as mock_kill, patch("subprocess.run") as mock_subproc:
        with pytest.raises(ProducerConflictError):
            manager.acquire(producer_id="competing-agent", session_id="session-42")

        mock_kill.assert_not_called()
        mock_subproc.assert_not_called()


def test_same_producer_reacquire_or_renew() -> None:
    manager = ProducerLockManager()
    lock1 = manager.acquire(
        producer_id="producer-model-1", session_id="session-42", ttl_seconds=10.0
    )

    renewed = manager.renew(lock1, extension_seconds=20.0)
    assert renewed.producer_id == "producer-model-1"
    assert renewed.session_id == "session-42"
    assert renewed.expires_at_utc >= lock1.expires_at_utc


def test_expired_lock_allows_new_producer() -> None:
    manager = ProducerLockManager()
    manager.acquire(producer_id="producer-1", session_id="session-42", ttl_seconds=0.05)
    time.sleep(0.06)

    # Lock is expired; new producer should acquire without conflict
    lock2 = manager.acquire(
        producer_id="producer-2", session_id="session-42", ttl_seconds=10.0
    )
    assert lock2.producer_id == "producer-2"
    assert manager.current_producer() == "producer-2"


def test_cannot_release_with_mismatched_producer() -> None:
    manager = ProducerLockManager()
    lock1 = manager.acquire(
        producer_id="producer-1", session_id="session-42", ttl_seconds=10.0
    )

    fake_lock = ProducerLock(
        producer_id="imposter",
        session_id="session-42",
        lease_ttl_seconds=10.0,
        acquired_at_utc=lock1.acquired_at_utc,
        expires_at_utc=lock1.expires_at_utc,
    )
    with pytest.raises(
        ValueError, match="Cannot release lock held by another producer"
    ):
        manager.release(fake_lock)


def test_cannot_release_with_stale_lock_after_renewal() -> None:
    manager = ProducerLockManager()
    lock1 = manager.acquire(
        producer_id="producer-1", session_id="session-42", ttl_seconds=10.0
    )
    lock2 = manager.renew(lock1, extension_seconds=30.0)

    # Releasing the stale lock1 token must raise ValueError and keep lock2 active
    with pytest.raises(ValueError, match="Cannot release stale lock token"):
        manager.release(lock1)

    assert manager.is_locked()
    assert manager.current_producer() == "producer-1"

    # Releasing active lock2 succeeds
    manager.release(lock2)
    assert not manager.is_locked()
