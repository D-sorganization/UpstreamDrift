"""Tests for the deterministic wait helper (issue #7156)."""

from __future__ import annotations

import threading
import time

import pytest

from tests.support.waiting import wait_until

pytestmark = pytest.mark.unit


def test_returns_immediately_when_predicate_already_true() -> None:
    start = time.monotonic()
    wait_until(lambda: True, timeout=1.0)
    assert time.monotonic() - start < 0.5


def test_returns_once_predicate_flips_from_another_thread() -> None:
    flag = {"ready": False}

    def flip() -> None:
        time.sleep(0.02)
        flag["ready"] = True

    threading.Thread(target=flip, daemon=True).start()
    wait_until(lambda: flag["ready"], timeout=2.0)
    assert flag["ready"] is True


def test_raises_with_descriptive_message_on_timeout() -> None:
    with pytest.raises(AssertionError, match="never true"):
        wait_until(lambda: False, timeout=0.05, message="never true")


def test_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError):
        wait_until(lambda: True, timeout=0.0)
