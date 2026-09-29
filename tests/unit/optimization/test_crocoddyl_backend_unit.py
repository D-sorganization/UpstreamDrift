"""Unit tests for the Crocoddyl backend's degradation paths (B4/#8399).

This tree's conftest installs a spec-less ``pinocchio`` MagicMock, so the
mock-tolerant probe reports the stack as unavailable here. Real solves and
the subprocess stack-health probe run in
``tests/integration/optimization/test_crocoddyl_swing_live.py``.
"""

from __future__ import annotations

import pytest

from src.shared.python.optimization.crocoddyl_backend import (
    CrocoddylNotAvailableError,
    crocoddyl_available,
    crocoddyl_stack_healthy,
    require_crocoddyl,
    solve_swing_ddp,
)

pytestmark = pytest.mark.unit


def test_mocked_stack_counts_as_unavailable() -> None:
    assert crocoddyl_available() is False


def test_require_raises_with_install_hint() -> None:
    with pytest.raises(CrocoddylNotAvailableError, match="conda-forge"):
        require_crocoddyl()


def test_health_probe_reports_unavailable_without_crashing() -> None:
    healthy, reason = crocoddyl_stack_healthy()
    assert healthy is False
    assert reason


def test_solve_raises_when_unavailable() -> None:
    with pytest.raises(CrocoddylNotAvailableError):
        solve_swing_ddp(horizon=5)


def test_argument_validation_precedes_dependency_check() -> None:
    with pytest.raises(ValueError, match="horizon"):
        solve_swing_ddp(horizon=1)
    with pytest.raises(ValueError, match="dt"):
        solve_swing_ddp(dt=0.0)
    with pytest.raises(ValueError, match="target_speed"):
        solve_swing_ddp(target_speed=-1.0)
