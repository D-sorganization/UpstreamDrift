"""Deterministic waiting helpers for concurrent tests (issue #7156).

Bare ``time.sleep(<small>)`` to synchronise a triggering action with its
assertion encodes a false assumption about scheduler latency: it passes locally
and fails intermittently on loaded CI runners (and doubly so under
``pytest -n auto``). Each intermittent failure burns a re-run and erodes trust
in red builds, which then hides real regressions.

Prefer, in order:

1. a ``threading.Event`` set from the callback under test, then
   ``assert event.wait(timeout)``;
2. otherwise ``wait_until(predicate, timeout)`` — a bounded poll that returns as
   soon as the condition holds and raises a self-describing ``AssertionError``
   on timeout.

The short ``sleep`` *inside* ``wait_until``'s poll loop is fine: it is bounded
and self-terminating, never an open-loop bet on timing.
"""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = ["wait_until"]


def wait_until(
    predicate: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.01,
    message: str | None = None,
) -> None:
    """Poll *predicate* until it returns truthy or *timeout* elapses.

    Args:
        predicate: Zero-arg callable evaluated repeatedly; waiting ends as soon
            as it returns a truthy value.
        timeout: Maximum seconds to wait before failing.
        interval: Seconds between polls (bounded, self-terminating).
        message: Optional context included in the timeout ``AssertionError``.

    Raises:
        AssertionError: if *predicate* never becomes truthy within *timeout*.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if interval <= 0:
        raise ValueError("interval must be positive")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    # One final check after the deadline to avoid a lost race on a slow tick.
    if predicate():
        return
    detail = f": {message}" if message else f": {predicate!r}"
    raise AssertionError(f"condition not met within {timeout}s{detail}")
