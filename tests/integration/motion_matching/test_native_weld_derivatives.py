"""Finite weld pose derivatives must remain correct away from closure (#10260)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_model import (
    FullBodyPinocchioModel,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]


@pytest.mark.parametrize("offset", [0.0, 0.1, -0.4])
def test_finite_weld_error_jacobian_matches_directional_differences(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]], offset: float
) -> None:
    plant, pose = plant_state
    pose["LWInputX"] += offset
    pose["LWInputY"] -= offset / 2.0
    pose = dict(reversed(list(pose.items())))
    result = plant.closure_position_linearization(pose)
    assert result.names == tuple(pose)
    assert not result.position.flags.writeable
    assert not result.jacobian.flags.writeable
    saved = result.jacobian.copy()
    rng = np.random.default_rng(10260)
    for _ in range(5):
        direction = rng.normal(size=len(pose))
        direction /= np.linalg.norm(direction)
        residuals = []
        for sign in (-1.0, 1.0):
            trial = {
                name: value + sign * 1e-6 * direction[index]
                for index, (name, value) in enumerate(pose.items())
            }
            residuals.append(plant.closure_residuals(trial)[0])
        observed = (residuals[1] - residuals[0]) / 2e-6
        np.testing.assert_allclose(
            result.jacobian @ direction, observed, rtol=2e-6, atol=2e-7
        )
    np.testing.assert_array_equal(result.jacobian, saved)


def test_acceleration_residual_keeps_velocity_constraint_jacobian(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]],
) -> None:
    plant, pose = plant_state
    pose["LWInputX"] += 0.3
    zero = dict.fromkeys(pose, 0.0)
    linear = plant.closure_trajectory_linearization(
        pose, zero, zero, finite_difference_step=1e-6
    )
    rng = np.random.default_rng(10260)
    direction = rng.normal(size=len(pose))
    samples = []
    for sign in (-1.0, 1.0):
        acceleration = {
            name: sign * 1e-4 * direction[index] for index, name in enumerate(pose)
        }
        sample = plant.closure_trajectory_residuals(pose, zero, acceleration)
        samples.append(np.concatenate(sample))
    observed = (samples[1] - samples[0]) / 2e-4
    np.testing.assert_allclose(linear.da @ direction, observed, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_weld_state_rejected(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]], invalid: float
) -> None:
    plant, pose = plant_state
    pose["LWInputX"] = invalid
    with pytest.raises(ValueError):
        plant.closure_position_linearization(pose)


@pytest.mark.parametrize("method", ["position", "trajectory"])
def test_weld_log_branch_has_no_claimed_derivative(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]], method: str
) -> None:
    plant, pose = plant_state
    pose["LWInputX"] += np.pi
    zero = dict.fromkeys(pose, 0.0)
    with pytest.raises(ValueError, match="log.*branch"):
        if method == "position":
            plant.closure_position_linearization(pose)
        else:
            plant.closure_trajectory_linearization(
                pose, zero, zero, finite_difference_step=1e-6
            )


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_position_derivative_on_each_side_of_log_branch(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]], side: float
) -> None:
    plant, pose = plant_state
    pose["LWInputX"] += np.pi + side * 1e-4
    linear = plant.closure_position_linearization(pose)
    residuals = []
    for sign in (-1.0, 1.0):
        trial = {**pose, "LWInputX": pose["LWInputX"] + sign * 1e-7}
        residuals.append(plant.closure_residuals(trial)[0])
    observed = (residuals[1] - residuals[0]) / 2e-7
    index = linear.names.index("LWInputX")
    np.testing.assert_allclose(linear.jacobian[:, index], observed, atol=2e-6)


def test_trajectory_difference_step_must_not_cross_log_branch(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]],
) -> None:
    plant, pose = plant_state
    pose["LWInputX"] += np.pi - 1e-5
    zero = dict.fromkeys(pose, 0.0)
    with pytest.raises(ValueError, match="log.*branch"):
        plant.closure_trajectory_linearization(
            pose, zero, zero, finite_difference_step=1e-4
        )
