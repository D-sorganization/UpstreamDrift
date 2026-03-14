"""Tests for integration fixture engine probe status handling."""

from __future__ import annotations

import pytest

from tests.fixtures.fixtures_lib import (
    EngineProbeStatus,
    _probe_engine_instance,
)


def test_probe_engine_instance_marks_missing_dependency() -> None:
    """Unavailable optional engines should be marked as missing."""
    result = _probe_engine_instance("Drake", lambda: False, lambda: object())

    assert result.available is False
    assert result.engine is None
    assert result.status is EngineProbeStatus.MISSING
    assert result.error is None


def test_probe_engine_instance_marks_broken_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed-but-broken engines should be distinguished from missing ones."""
    monkeypatch.delenv("UPSTREAM_DRIFT_STRICT_ENGINE_PROBES", raising=False)

    def _broken_loader() -> object:
        raise RuntimeError("loader exploded")

    result = _probe_engine_instance("Pinocchio", lambda: True, _broken_loader)

    assert result.available is False
    assert result.engine is None
    assert result.status is EngineProbeStatus.BROKEN
    assert "loader exploded" in (result.error or "")


def test_probe_engine_instance_can_fail_fast_for_broken_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict CI lanes should be able to fail on broken expected engines."""
    monkeypatch.setenv("UPSTREAM_DRIFT_STRICT_ENGINE_PROBES", "true")

    def _broken_loader() -> object:
        raise RuntimeError("strict failure")

    with pytest.raises(pytest.fail.Exception, match="strict failure"):
        _probe_engine_instance("MuJoCo", lambda: True, _broken_loader)


def test_probe_engine_instance_marks_ready_engine() -> None:
    """Healthy engines should be marked ready."""
    engine = object()

    result = _probe_engine_instance("MuJoCo", lambda: True, lambda: engine)

    assert result.available is True
    assert result.engine is engine
    assert result.status is EngineProbeStatus.READY
    assert result.error is None
