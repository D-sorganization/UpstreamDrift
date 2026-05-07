"""Pure-unit tests for prescribed OpenSim polynomial controller boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.motion_matching.prescribed_controller import (
    PrescribedControllerUnavailableError,
    build_prescribed_polynomial_controller,
    build_prescribed_polynomial_torque_plan,
)
from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    evaluate_polynomial_torque,
)

pytestmark = [pytest.mark.unit]


def _theta_fixture(n_actuators: int = 3) -> np.ndarray:
    return np.arange(n_actuators * COEFFS_PER_JOINT, dtype=np.float64) / 100.0


def test_plan_samples_polynomial_torques_like_callback_evaluator() -> None:
    actuator_names = ("hip_flexion", "knee_angle", "ankle_angle")
    time_grid = np.array([0.0, 0.05, 0.2, 0.5], dtype=np.float64)
    theta = _theta_fixture(len(actuator_names))

    plan = build_prescribed_polynomial_torque_plan(
        theta=theta,
        time_grid=time_grid,
        actuator_names=actuator_names,
    )

    coeffs = theta.reshape(len(actuator_names), COEFFS_PER_JOINT)
    expected_tau = np.vstack(
        [evaluate_polynomial_torque(coeffs, float(t)) for t in time_grid]
    )
    np.testing.assert_allclose(plan.sampled_tau, expected_tau, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(plan.time_grid, time_grid)
    assert plan.actuator_names == actuator_names


@pytest.mark.parametrize(
    ("time_grid", "match"),
    [
        (np.array([], dtype=np.float64), "time_grid must contain"),
        (np.array([[0.0, 0.1]], dtype=np.float64), "time_grid must be 1-D"),
        (np.array([0.0, np.nan], dtype=np.float64), "time_grid must be finite"),
        (np.array([0.0, 0.0], dtype=np.float64), "strictly increasing"),
    ],
)
def test_plan_rejects_invalid_time_grid(time_grid: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_prescribed_polynomial_torque_plan(
            theta=_theta_fixture(2),
            time_grid=time_grid,
            actuator_names=("hip", "knee"),
        )


@pytest.mark.parametrize(
    ("theta", "match"),
    [
        (np.zeros((2, COEFFS_PER_JOINT + 1), dtype=np.float64), "7 coefficients"),
        (np.zeros(2 * COEFFS_PER_JOINT + 1, dtype=np.float64), "divisible by 7"),
        (
            np.array([0.0, np.inf] + [0.0] * 12, dtype=np.float64),
            "theta must be finite",
        ),
    ],
)
def test_plan_rejects_invalid_theta_shape_or_values(
    theta: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_prescribed_polynomial_torque_plan(
            theta=theta,
            time_grid=np.array([0.0, 0.1], dtype=np.float64),
            actuator_names=("hip", "knee"),
        )


@pytest.mark.parametrize(
    ("actuator_names", "match"),
    [
        (("hip",), "actuator name count"),
        (("hip", ""), "non-empty strings"),
        (("hip", "hip"), "unique"),
    ],
)
def test_plan_rejects_invalid_actuator_names(
    actuator_names: tuple[str, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_prescribed_polynomial_torque_plan(
            theta=_theta_fixture(2),
            time_grid=np.array([0.0, 0.1], dtype=np.float64),
            actuator_names=actuator_names,
        )


def test_prescribed_controller_constructor_unavailable_is_typed() -> None:
    opensim_module = SimpleNamespace()

    with pytest.raises(
        PrescribedControllerUnavailableError, match="PrescribedController"
    ):
        build_prescribed_polynomial_controller(
            theta=_theta_fixture(1),
            time_grid=np.array([0.0, 0.1], dtype=np.float64),
            actuator_names=("hip",),
            opensim_module=opensim_module,
        )


def test_prescribed_controller_constructor_unavailable_can_fallback() -> None:
    opensim_module = SimpleNamespace()
    theta = _theta_fixture(1)
    time_grid = np.array([0.0, 0.1], dtype=np.float64)

    controller, plan = build_prescribed_polynomial_controller(
        theta=theta,
        time_grid=time_grid,
        actuator_names=("hip",),
        opensim_module=opensim_module,
        unavailable="fallback",
    )

    assert controller is None
    expected = np.vstack(
        [
            evaluate_polynomial_torque(theta.reshape(1, COEFFS_PER_JOINT), float(t))
            for t in time_grid
        ]
    )
    np.testing.assert_allclose(plan.sampled_tau, expected)
