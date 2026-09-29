"""Live Crocoddyl tests for the DDP swing backend (B4/#8399).

Runs correctly on both healthy and broken installs: on a consistent
binary stack (conda-forge) the real FDDP solve must converge with
clubhead-speed progress; on a mixed PyPI-wheel stack (bundled
libpinocchio duplication) the subprocess health probe must diagnose the
condition and the solve must degrade with conda-forge guidance instead
of crashing the process.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


CROCODDYL_AVAILABLE = _available("crocoddyl") and _available("pinocchio")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_crocoddyl,
    pytest.mark.skipif(not CROCODDYL_AVAILABLE, reason="crocoddyl not installed"),
]

from src.shared.python.optimization.crocoddyl_backend import (  # noqa: E402
    crocoddyl_stack_healthy,
    solve_swing_ddp,
)


def test_health_probe_never_crashes_the_host() -> None:
    healthy, reason = crocoddyl_stack_healthy()
    assert isinstance(healthy, bool)
    if not healthy:
        assert "conda-forge" in reason


def test_solve_converges_or_degrades_with_diagnosis() -> None:
    healthy, _ = crocoddyl_stack_healthy()
    result = solve_swing_ddp(horizon=30, dt=0.02, max_iterations=150)
    if healthy:
        assert result.success is True
        assert result.xs.shape[0] == 31
        assert result.us.shape[0] == 30
        assert np.all(np.isfinite(result.xs))
        # From rest toward a 45 m/s target the solver must make real
        # clubhead-speed progress.
        assert result.terminal_speed > 1.0
    else:
        assert result.success is False
        assert "conda-forge" in result.message
        assert result.iterations == 0
