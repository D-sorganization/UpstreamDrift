"""Coefficient-ordering parity for the MuJoCo polynomial-torque evaluator.

Regression coverage for issue #7688: the MuJoCo evaluators must lay out
``theta`` ascending in power — ``theta[:, k]`` is the coefficient of
``t^k`` (column 0 = constant term) — matching the canonical convention
used by Drake / Pinocchio / OpenSim. A reversed ordering silently flips
the torque-vs-time profile when the same flat ``theta`` is shared across
engines.

This module imports only the pure-numpy ``_evaluate_polynomial`` and does
NOT require the ``mujoco`` package, so the ordering contract is verified
on every CI runner regardless of whether MuJoCo is installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.mujoco.python.motion_matching.torque_driver import (
    _evaluate_polynomial,
)

pytestmark = [pytest.mark.unit]


def _reference(theta: np.ndarray, t: float) -> np.ndarray:
    """Non-Horner reference: ``sum_k theta[:, k] * t^k`` (column 0 = t^0)."""
    powers = np.array([t**k for k in range(theta.shape[1])], dtype=np.float64)
    return theta @ powers


def test_column_zero_is_constant_term() -> None:
    """``theta[:, 0]`` is the ``t^0`` coefficient (independent of ``t``)."""
    theta = np.zeros((2, 7), dtype=np.float64)
    theta[:, 0] = [3.0, -5.0]
    for t in (0.0, 0.1, 0.7, 2.5):
        got = _evaluate_polynomial(theta, t)
        np.testing.assert_allclose(got, [3.0, -5.0], atol=1e-12)


def test_only_t1_column_is_linear() -> None:
    """A pure ``theta[:, 1]`` column yields ``coef * t`` (the ``t^1`` term)."""
    theta = np.zeros((1, 7), dtype=np.float64)
    theta[0, 1] = 4.0
    for t in (0.0, 0.25, 1.5):
        got = _evaluate_polynomial(theta, t)[0]
        assert got == pytest.approx(4.0 * t, abs=1e-12)


def test_top_column_is_t6() -> None:
    """The last column (index 6) is the ``t^6`` coefficient."""
    theta = np.zeros((1, 7), dtype=np.float64)
    theta[0, 6] = 2.0
    t = 0.3
    assert _evaluate_polynomial(theta, t)[0] == pytest.approx(2.0 * t**6, abs=1e-12)


def test_matches_ascending_power_reference() -> None:
    """Horner evaluation equals the explicit ``sum_k theta[:, k] * t^k``."""
    rng = np.random.default_rng(7688)
    for _ in range(25):
        n_joints = int(rng.integers(1, 6))
        theta = rng.uniform(-3.0, 3.0, size=(n_joints, 7)).astype(np.float64)
        t = float(rng.uniform(0.0, 0.5))
        np.testing.assert_allclose(
            _evaluate_polynomial(theta, t),
            _reference(theta, t),
            rtol=1e-12,
            atol=1e-12,
        )


def test_handcomputed_full_polynomial() -> None:
    """A fully-populated row matches the ascending-power hand calculation."""
    theta = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], dtype=np.float64)
    t = 0.1
    expected = (
        1.0 + 2.0 * t + 3.0 * t**2 + 4.0 * t**3 + 5.0 * t**4 + 6.0 * t**5 + 7.0 * t**6
    )
    assert _evaluate_polynomial(theta, t)[0] == pytest.approx(expected, abs=1e-12)


def test_simscape_to_canonical_conversion_parity_at_zero_and_nonzero_times() -> None:
    """Simscape native theta is highest-power-first; canonical torque driver is lowest-power-first.

    Converting via theta[:, ::-1] must yield exact physical torque parity at t=0
    and throughout the continuous trajectory.
    """
    from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape

    # Arbitrary 3-joint Bernstein control points
    controls = np.array(
        [
            [-40.94, -44.95, -48.81, -52.54, -56.16, -59.70, -63.18],
            [10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0],
            [-5.0, 0.0, 5.0, 10.0, 5.0, 0.0, -5.0],
        ],
        dtype=np.float64,
    )
    duration_s = 0.70

    # Highest-power-first native Simscape format [t^6, ..., t^0]
    simscape_theta = bernstein_to_simscape(controls, duration_s=duration_s)
    # Canonical lowest-power-first format [t^0, ..., t^6]
    canonical_theta = simscape_theta[:, ::-1]

    # At t=0, physical torque must equal control point c_0
    np.testing.assert_allclose(
        _evaluate_polynomial(canonical_theta, 0.0),
        controls[:, 0],
        atol=1e-10,
    )

    # At t=0.35s (midpoint, s=0.5)
    weights = np.array([1, 6, 15, 20, 15, 6, 1]) * (0.5**6)
    expected_mid = np.sum(controls * weights, axis=1)
    np.testing.assert_allclose(
        _evaluate_polynomial(canonical_theta, 0.35),
        expected_mid,
        atol=1e-10,
    )
