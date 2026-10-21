"""Pure tests for lift Simbody kinematics helpers (no OpenSim runtime)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.engines.physics_engines.opensim.python.msk_lift_simbody_kinematics import (
    mapped_acceleration_from_speeds,
)


def _speeds_t_squared(t: float) -> np.ndarray:
    """Generalized speed u(t) = t² (e.g. q(t) = t³/3, unmapped coordinate)."""
    return np.array([t * t], dtype=float)


def test_endpoint_t_zero_uses_analytic_not_clamped_secant() -> None:
    """Codex counterexample: clipped window at t=0 must not return u(Δt)/Δt."""
    t_min, t_max, sample_dt = 0.0, 1.0, 0.02
    analytic = np.array([0.0])
    udot = mapped_acceleration_from_speeds(
        0.0,
        t_min,
        t_max,
        sample_dt,
        _speeds_t_squared,
        analytic,
    )
    assert udot[0] == 0.0
    assert udot[0] != 0.01


def test_endpoint_t_max_uses_analytic() -> None:
    t_min, t_max, sample_dt = 0.0, 1.0, 0.02
    analytic = np.array([2.0])
    udot = mapped_acceleration_from_speeds(
        1.0,
        t_min,
        t_max,
        sample_dt,
        _speeds_t_squared,
        analytic,
    )
    np.testing.assert_allclose(udot, analytic)


def test_interior_central_difference_matches_quadratic_acceleration() -> None:
    t_min, t_max, sample_dt = 0.0, 1.0, 0.02
    t = 0.5
    analytic = np.array([2.0 * t])
    udot = mapped_acceleration_from_speeds(
        t,
        t_min,
        t_max,
        sample_dt,
        _speeds_t_squared,
        analytic,
    )
    np.testing.assert_allclose(udot[0], 1.0, rtol=0.0, atol=1e-12)


def _speeds_cubic(t: float) -> np.ndarray:
    return np.array([t**3], dtype=float)


def test_interior_fd_error_scales_with_dt_squared() -> None:
    """Centre difference is second-order for nonlinear mapped speeds."""
    t_min, t_max = 0.0, 1.0
    t = 0.4
    analytic = np.array([3.0 * t * t])
    err_coarse = abs(
        mapped_acceleration_from_speeds(
            t, t_min, t_max, 0.1, _speeds_cubic, analytic
        )[0]
        - analytic[0]
    )
    err_fine = abs(
        mapped_acceleration_from_speeds(
            t, t_min, t_max, 0.02, _speeds_cubic, analytic
        )[0]
        - analytic[0]
    )
    assert err_fine < err_coarse
    assert err_coarse > 1e-4
    assert err_coarse / err_fine > 5.0
