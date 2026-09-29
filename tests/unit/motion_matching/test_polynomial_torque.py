"""Shared polynomial torque evaluator coverage for cross-engine users."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    evaluate_torque_polynomial as evaluate_drake_torque,
)
from src.engines.physics_engines.mujoco.python.motion_matching.torque_driver import (
    PolynomialTorqueDriver,
    _evaluate_polynomial as evaluate_mujoco_torque,
)
from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
    evaluate_polynomial_torque as evaluate_opensim_torque,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
    evaluate_polynomial_torque as evaluate_pinocchio_torque,
)
from src.shared.python.motion_matching.polynomial_torque import (
    COEFFS_PER_JOINT,
    POLY_DEGREE,
    evaluate_polynomial_torque,
)
from src.shared.python.motion_matching.validate_theta import (
    COEFFS_PER_JOINT as VALIDATOR_COEFFS_PER_JOINT,
)

pytestmark = pytest.mark.unit


def test_lowest_power_first_a_to_g_contract() -> None:
    """Columns ``[A..G]`` map to ``t**0`` through ``t**6``."""
    coeffs = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [-3.0, 0.5, -0.25, 0.125, -0.0625, 0.03125, -0.015625],
        ],
        dtype=np.float64,
    )
    t = 0.4

    expected = np.array(
        [sum(row[k] * t**k for k in range(COEFFS_PER_JOINT)) for row in coeffs],
        dtype=np.float64,
    )

    assert COEFFS_PER_JOINT == VALIDATOR_COEFFS_PER_JOINT == 7
    assert POLY_DEGREE == 6
    np.testing.assert_allclose(
        evaluate_polynomial_torque(coeffs, t),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_rejects_non_matrix_coefficients_and_non_finite_time() -> None:
    with pytest.raises(ValueError, match="2D"):
        evaluate_polynomial_torque(np.zeros(COEFFS_PER_JOINT), 0.0)
    with pytest.raises(ValueError, match="7 columns"):
        evaluate_polynomial_torque(np.zeros((2, COEFFS_PER_JOINT - 1)), 0.0)
    with pytest.raises(ValueError, match="finite"):
        evaluate_polynomial_torque(np.zeros((1, COEFFS_PER_JOINT)), np.inf)


def test_engine_evaluators_match_shared_helper() -> None:
    rng = np.random.default_rng(7728)
    coeffs = rng.normal(size=(4, COEFFS_PER_JOINT)).astype(np.float64)
    t = 0.37
    expected = evaluate_polynomial_torque(coeffs, t)

    assert evaluate_pinocchio_torque is evaluate_polynomial_torque
    assert evaluate_opensim_torque is evaluate_polynomial_torque
    np.testing.assert_allclose(evaluate_mujoco_torque(coeffs, t), expected)
    np.testing.assert_allclose(
        evaluate_drake_torque(coeffs.reshape(-1), t, coeffs.shape[0]),
        expected,
    )


def test_mujoco_driver_evaluate_delegates_to_shared_helper() -> None:
    class _Model:
        nu = 2

    coeffs = np.array(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0],
            [0.5, 0.25, 0.125, 0.0, -0.125, -0.25, -0.5],
        ],
        dtype=np.float64,
    )
    driver = PolynomialTorqueDriver(_Model(), coeffs, t0=0.25, clip_to_ctrlrange=False)
    t = 0.75

    np.testing.assert_allclose(
        driver.evaluate(t),
        evaluate_polynomial_torque(coeffs, t - 0.25),
        rtol=1e-12,
        atol=1e-12,
    )
