"""Tests for the PreflightDialog data model (epic #5968, Phase 4.1)."""

from __future__ import annotations

import pytest

from src.shared.python.ux.preflight import (
    PreflightCheck,
    PreflightError,
    Severity,
    run_preflight,
)

pytestmark = pytest.mark.unit


# ---- PreflightCheck construction ------------------------------------


def test_preflight_check_constructs_with_all_fields():
    check = PreflightCheck(
        id="engine_ready",
        label="Engine is ready",
        severity=Severity.BLOCK,
        why="MuJoCo Python bindings are not importable.",
        fix_action="Install: pip install mujoco",
        passed=False,
    )
    assert check.id == "engine_ready"
    assert check.severity is Severity.BLOCK
    assert check.passed is False


def test_preflight_check_id_must_be_snake_case():
    with pytest.raises((ValueError, PreflightError)):
        PreflightCheck(
            id="Engine Ready",
            label="x",
            severity=Severity.INFO,
            why="x",
            fix_action=None,
            passed=True,
        )


def test_preflight_check_is_frozen():
    check = PreflightCheck(
        id="x",
        label="x",
        severity=Severity.INFO,
        why="x",
        fix_action=None,
        passed=True,
    )
    with pytest.raises((AttributeError, TypeError)):
        check.passed = False  # type: ignore[misc]


def test_preflight_check_label_and_why_required():
    with pytest.raises((ValueError, PreflightError)):
        PreflightCheck(
            id="x",
            label="",
            severity=Severity.INFO,
            why="x",
            fix_action=None,
            passed=True,
        )
    with pytest.raises((ValueError, PreflightError)):
        PreflightCheck(
            id="x",
            label="x",
            severity=Severity.INFO,
            why="",
            fix_action=None,
            passed=True,
        )


# ---- run_preflight aggregator ---------------------------------------


def _make(passed: bool, severity: Severity, ident: str = "c") -> PreflightCheck:
    return PreflightCheck(
        id=ident,
        label=ident,
        severity=severity,
        why="reason",
        fix_action=None,
        passed=passed,
    )


def test_run_preflight_all_passing():
    result = run_preflight(
        [
            _make(True, Severity.INFO, "a"),
            _make(True, Severity.WARN, "b"),
            _make(True, Severity.BLOCK, "c"),
        ]
    )
    assert result.can_proceed() is True
    assert result.blocking_failures() == ()
    assert result.warning_failures() == ()


def test_run_preflight_failing_block_prevents_proceed():
    result = run_preflight(
        [
            _make(True, Severity.INFO, "a"),
            _make(False, Severity.BLOCK, "b"),
        ]
    )
    assert result.can_proceed() is False
    assert tuple(c.id for c in result.blocking_failures()) == ("b",)


def test_run_preflight_failing_warn_allows_proceed():
    result = run_preflight(
        [
            _make(False, Severity.WARN, "a"),
        ]
    )
    assert result.can_proceed() is True
    assert tuple(c.id for c in result.warning_failures()) == ("a",)


def test_run_preflight_with_override_allows_block_to_proceed():
    result = run_preflight(
        [_make(False, Severity.BLOCK, "a")],
        override_reason="acknowledged: testing locally with stub engine",
    )
    assert result.can_proceed() is True
    assert result.was_overridden is True


def test_run_preflight_rejects_empty_override_reason():
    with pytest.raises((ValueError, PreflightError)):
        run_preflight(
            [_make(False, Severity.BLOCK, "a")],
            override_reason="",
        )


def test_run_preflight_rejects_short_override_reason():
    # A typed override must be substantive, not a single character.
    with pytest.raises((ValueError, PreflightError)):
        run_preflight(
            [_make(False, Severity.BLOCK, "a")],
            override_reason="ok",
        )


def test_run_preflight_rejects_duplicate_ids():
    with pytest.raises((ValueError, PreflightError)):
        run_preflight(
            [_make(True, Severity.INFO, "a"), _make(True, Severity.INFO, "a")]
        )


def test_preflight_result_summary_lists_failures():
    result = run_preflight(
        [
            _make(False, Severity.BLOCK, "engine_ready"),
            _make(False, Severity.WARN, "timestep_unusual"),
            _make(True, Severity.INFO, "ok"),
        ]
    )
    summary = result.summary()
    assert "engine_ready" in summary
    assert "timestep_unusual" in summary
    assert "BLOCK" in summary
    assert "WARN" in summary


def test_preflight_error_subclasses_value_error():
    assert issubclass(PreflightError, ValueError)
