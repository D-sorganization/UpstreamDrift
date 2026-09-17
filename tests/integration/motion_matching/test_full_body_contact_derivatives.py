"""Independent full-plant derivative checks for the Crocoddyl boundary (#10255)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_model import (
    FullBodyPinocchioModel,
)
from src.shared.python.motion_matching.contact_law import GroundPlane

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def plant_state() -> tuple[FullBodyPinocchioModel, dict[str, float]]:
    """Use an archived closed-grip pose, without claiming fit qualification."""
    pin = pytest.importorskip("pinocchio")
    if not isinstance(getattr(pin, "__version__", None), str):
        pytest.skip("A real Pinocchio runtime is required")
    folder = ROOT / "docs/development/full_body_models"
    spec = json.loads((folder / "full_body_spec_v1.json").read_text())
    candidate = json.loads(
        (folder / "evidence/native_candidates/returned81_candidate.json").read_text()
    )
    plant = FullBodyPinocchioModel(spec)
    pose = dict.fromkeys(spec["coordinate_order"], 0.0)
    pose.update(zip(candidate["coordinate_names"], candidate["q0"], strict=True))
    # Tilt legs away from straight-leg singular configurations.
    pose["knee_angle_r"] = 0.15
    pose["knee_angle_l"] = 0.18
    return plant, pose


def _contact_plane(plant: FullBodyPinocchioModel, pose: dict[str, float]) -> None:
    samples = plant.contact_forces(pose, dict.fromkeys(pose, 0.0))
    lowest = min(float(sample.contact_point_m[2]) for sample in samples.values())
    plant.ground_plane = GroundPlane((0.0, 0.0, 1.0), lowest + 0.007)


def _directional_difference(
    plant: FullBodyPinocchioModel,
    state: list[dict[str, float]],
    argument: int,
    direction: np.ndarray,
) -> np.ndarray:
    step = 1e-6
    values = []
    for sign in (-1.0, 1.0):
        trial = [dict(vector) for vector in state]
        for index, name in enumerate(state[0]):
            trial[argument][name] += sign * step * direction[index]
        acceleration = plant.accelerations(*trial)
        values.append(np.array([acceleration[name] for name in state[0]]))
    return (values[1] - values[0]) / (2.0 * step)


@pytest.mark.parametrize("contact", [False, True])
@pytest.mark.parametrize("argument", [0, 1, 2], ids=["pose", "velocity", "effort"])
def test_full_plant_derivative_matches_independent_rollout(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]],
    contact: bool,
    argument: int,
) -> None:
    plant, pose = plant_state
    if contact:
        _contact_plane(plant, pose)
    else:
        plant.ground_plane = GroundPlane((0.0, 0.0, 1.0), -10.0)
    # Reordered inputs must not be confused with Pinocchio's tree order.
    pose = dict(reversed(list(pose.items())))
    rates = dict.fromkeys(pose, 0.0)
    rates["TranslationInputX"] = 0.012
    rates["TranslationInputZ"] = -0.02
    rates["HipInputZ"] = 0.23
    rates["knee_angle_r"] = 0.31
    rates["ankle_angle_l"] = -0.18
    efforts = dict.fromkeys(pose, 0.0)
    derivatives = plant.acceleration_derivatives(pose, rates, efforts)
    assert derivatives.names == tuple(pose)
    rng = np.random.default_rng(10255)
    for _ in range(3):
        direction = rng.normal(size=len(pose))
        direction /= np.linalg.norm(direction)
        measured = _directional_difference(
            plant, [pose, rates, efforts], argument, direction
        )
        matrix = (derivatives.dq, derivatives.dv, derivatives.deffort)[argument]
        np.testing.assert_allclose(matrix @ direction, measured, rtol=2e-5, atol=2e-5)


def test_derivatives_are_detached_from_subsequent_calls(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]],
) -> None:
    plant, pose = plant_state
    _contact_plane(plant, pose)
    zero = dict.fromkeys(pose, 0.0)
    result = plant.acceleration_derivatives(pose, zero, zero)
    saved = tuple(matrix.copy() for matrix in result[1:])
    changed = {**pose, "TranslationInputZ": pose["TranslationInputZ"] + 0.002}
    plant.acceleration_derivatives(changed, zero, zero)
    for matrix, original in zip(result[1:], saved, strict=True):
        assert not matrix.flags.writeable
        np.testing.assert_array_equal(matrix, original)


def test_contact_query_preserves_constraint_cache_and_reports_inactive_branch(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]],
) -> None:
    plant, pose = plant_state
    _contact_plane(plant, pose)
    zero = dict.fromkeys(pose, 0.0)
    plant.accelerations(pose, zero, zero)
    before = plant.closure_errors()
    changed = {**pose, "LWInputX": pose["LWInputX"] + 0.1}
    active = plant.contact_effort_derivatives(changed, zero)
    after = plant.closure_errors()
    assert active.differentiable
    assert np.linalg.norm(active.dq) > 1.0
    for first, second in zip(before, after, strict=True):
        np.testing.assert_array_equal(first, second)
    plant.ground_plane = GroundPlane((0.0, 0.0, 1.0), -10.0)
    inactive = plant.contact_effort_derivatives(pose, zero)
    assert inactive.names == tuple(pose)
    assert inactive.differentiable
    np.testing.assert_array_equal(inactive.dq, np.zeros_like(inactive.dq))
    np.testing.assert_array_equal(inactive.dv, np.zeros_like(inactive.dv))


def test_contact_boundary_is_not_claimed_differentiable(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]],
) -> None:
    plant, pose = plant_state
    zero = dict.fromkeys(pose, 0.0)
    samples = plant.contact_forces(pose, zero)
    height = min(float(sample.contact_point_m[2]) for sample in samples.values())
    plant.ground_plane = GroundPlane((0.0, 0.0, 1.0), height)
    result = plant.contact_effort_derivatives(pose, zero)
    assert not result.differentiable
    for matrix in (result.dq, result.dv):
        assert np.isfinite(matrix).all()
        assert not matrix.flags.writeable


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_invalid_state_rejected(
    plant_state: tuple[FullBodyPinocchioModel, dict[str, float]], invalid: Any
) -> None:
    plant, pose = plant_state
    zero = dict.fromkeys(pose, 0.0)
    pose["TranslationInputX"] = invalid
    with pytest.raises(ValueError, match="finite"):
        plant.acceleration_derivatives(pose, zero, zero)
