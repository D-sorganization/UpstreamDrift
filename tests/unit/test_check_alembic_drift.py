"""Tests for the Alembic drift-check helper script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_alembic_drift.py"


@pytest.fixture()
def check_alembic_drift() -> ModuleType:
    """Import the drift-check script as a module."""
    spec = importlib.util.spec_from_file_location("check_alembic_drift", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_check_alembic_drift_returns_success_when_models_match_migrations(
    check_alembic_drift: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clean autogenerate check exits successfully."""
    cfg = Mock()

    with (
        patch.object(check_alembic_drift, "build_alembic_config", return_value=cfg),
        patch("alembic.command.check") as command_check,
    ):
        result = check_alembic_drift.check_alembic_drift()

    assert result == 0
    command_check.assert_called_once_with(cfg)
    assert "No Alembic drift detected" in capsys.readouterr().out


def test_check_alembic_drift_reports_failure_to_stderr(
    check_alembic_drift: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Detected drift exits non-zero and tells contributors how to fix it."""
    cfg = Mock()

    with (
        patch.object(check_alembic_drift, "build_alembic_config", return_value=cfg),
        patch("alembic.command.check", side_effect=RuntimeError("new op detected")),
    ):
        result = check_alembic_drift.check_alembic_drift()

    assert result == 1
    assert "Alembic drift detected" in capsys.readouterr().err


def test_main_uses_check_result(
    check_alembic_drift: ModuleType,
) -> None:
    """The CLI entrypoint returns the drift-check result code."""
    with patch.object(check_alembic_drift, "check_alembic_drift", return_value=7):
        assert check_alembic_drift.main() == 7
