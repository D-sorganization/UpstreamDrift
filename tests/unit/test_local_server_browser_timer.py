"""Tests for the auto-open-browser timer in the local server (#6924)."""

from __future__ import annotations

import pytest

local_server = pytest.importorskip("src.api.local_server")


def test_schedule_browser_open_returns_daemon_timer() -> None:
    """The browser-open timer must be a daemon so a failed bind exits cleanly.

    Regression for #6924: a non-daemon Timer kept the process alive for the
    full delay when the server failed to start.
    """
    timer = local_server._schedule_browser_open("127.0.0.1", 8000, delay=0.0)
    try:
        assert timer.daemon is True
    finally:
        timer.cancel()


def test_schedule_browser_open_respects_suppression(monkeypatch) -> None:
    """When the browser is suppressed, the scheduled callback opens nothing."""
    opened: list[str] = []

    monkeypatch.setattr(
        "src.shared.python.config.environment.is_browser_suppressed",
        lambda: True,
    )
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url: opened.append(url),
    )

    timer = local_server._schedule_browser_open("127.0.0.1", 8000, delay=0.0)
    timer.join(timeout=2.0)
    assert opened == []


def test_schedule_browser_open_opens_when_not_suppressed(monkeypatch) -> None:
    """When not suppressed, the timer opens the local server URL."""
    opened: list[str] = []

    monkeypatch.setattr(
        "src.shared.python.config.environment.is_browser_suppressed",
        lambda: False,
    )
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url: opened.append(url),
    )

    timer = local_server._schedule_browser_open("127.0.0.1", 8123, delay=0.0)
    timer.join(timeout=2.0)
    assert opened == ["http://127.0.0.1:8123"]
