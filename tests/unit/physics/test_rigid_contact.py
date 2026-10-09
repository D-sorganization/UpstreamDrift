"""Independent physical contracts for finite-inertia two-body impact."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.physics.rigid_contact import (
    RigidContactInput,
    solve_rigid_contact,
)

pytestmark = pytest.mark.unit


def _state(**changes) -> RigidContactInput:
    state = RigidContactInput(
        club_velocity_mps=np.array([45.0, 0.0, 0.0]),
        club_angular_velocity_rad_s=np.zeros(3),
        ball_velocity_mps=np.zeros(3),
        ball_angular_velocity_rad_s=np.zeros(3),
        normal=np.array([1.0, 0.0, 0.0]),
        club_lever_m=np.array([0.04, 0.0, 0.0]),
        club_inertia_world_kg_m2=np.diag([0.00012, 0.00018, 0.00020]),
        club_mass_kg=0.2,
        restitution=0.78,
        friction_coefficient=0.4,
    )
    return replace(state, **changes)


def _total_angular(state: RigidContactInput, club_v, club_w, ball_v, ball_w):
    ball_center = state.club_lever_m + GOLF_BALL_RADIUS_M * state.normal
    ball_inertia = 0.4 * GOLF_BALL_MASS_KG * GOLF_BALL_RADIUS_M**2
    return (
        state.club_inertia_world_kg_m2 @ club_w
        + ball_inertia * ball_w
        + np.cross(ball_center, GOLF_BALL_MASS_KG * ball_v)
    )


@pytest.mark.parametrize("offset_m", [0.0, -0.02, 0.02])
def test_linear_angular_momentum_and_restitution(offset_m: float) -> None:
    state = _state(club_lever_m=np.array([0.04, offset_m, 0.0]))
    result = solve_rigid_contact(state)
    before_linear = (
        state.club_mass_kg * state.club_velocity_mps
        + GOLF_BALL_MASS_KG * state.ball_velocity_mps
    )
    after_linear = (
        state.club_mass_kg * result.club_velocity_mps
        + GOLF_BALL_MASS_KG * result.ball_velocity_mps
    )
    np.testing.assert_allclose(after_linear, before_linear, atol=1e-10)
    np.testing.assert_allclose(
        _total_angular(
            state,
            result.club_velocity_mps,
            result.club_angular_velocity_rad_s,
            result.ball_velocity_mps,
            result.ball_angular_velocity_rad_s,
        ),
        _total_angular(
            state,
            state.club_velocity_mps,
            state.club_angular_velocity_rad_s,
            state.ball_velocity_mps,
            state.ball_angular_velocity_rad_s,
        ),
        atol=1e-10,
    )
    assert result.post_relative_normal_mps == pytest.approx(
        -state.restitution * result.pre_relative_normal_mps, abs=1e-8
    )
    assert result.post_kinetic_energy_j <= result.pre_kinetic_energy_j + 1e-8
    assert float(result.ball_impulse_n_s @ state.normal) >= 0


def test_central_frictionless_impact_matches_reduced_mass() -> None:
    state = _state(friction_coefficient=0.0)
    result = solve_rigid_contact(state)
    expected = (1 + state.restitution) * 45.0 / (1 / 0.2 + 1 / GOLF_BALL_MASS_KG)
    np.testing.assert_allclose(
        result.ball_impulse_n_s, [expected, 0.0, 0.0], atol=1e-10
    )
    np.testing.assert_allclose(result.ball_angular_velocity_rad_s, 0.0, atol=1e-10)


def test_sliding_solution_honors_friction_cone_and_energy() -> None:
    state = _state(
        club_velocity_mps=np.array([45.0, 15.0, 0.0]),
        club_lever_m=np.array([0.04, 0.02, 0.01]),
        friction_coefficient=0.05,
    )
    result = solve_rigid_contact(state)
    normal_impulse = float(result.ball_impulse_n_s @ state.normal)
    tangent_impulse = np.linalg.norm(
        result.ball_impulse_n_s - normal_impulse * state.normal
    )
    assert tangent_impulse <= state.friction_coefficient * normal_impulse + 1e-8
    assert result.post_kinetic_energy_j <= result.pre_kinetic_energy_j + 1e-8
    assert result.post_relative_normal_mps == pytest.approx(
        -0.78 * result.pre_relative_normal_mps, abs=1e-8
    )


def test_sticking_eliminates_tangential_contact_slip() -> None:
    state = _state(
        club_velocity_mps=np.array([45.0, 3.0, 1.0]),
        club_lever_m=np.array([0.04, 0.01, -0.01]),
        friction_coefficient=1.0,
    )
    result = solve_rigid_contact(state)
    rb = -GOLF_BALL_RADIUS_M * state.normal
    relative = (
        result.club_velocity_mps
        + np.cross(result.club_angular_velocity_rad_s, state.club_lever_m)
        - result.ball_velocity_mps
        - np.cross(result.ball_angular_velocity_rad_s, rb)
    )
    assert result.friction_regime == "stick"
    np.testing.assert_allclose(
        relative - (relative @ state.normal) * state.normal, 0.0, atol=1e-9
    )


def test_rotating_all_inputs_rotates_all_outputs() -> None:
    state = _state(
        club_velocity_mps=np.array([45.0, 2.0, -1.0]),
        club_angular_velocity_rad_s=np.array([1.0, -2.0, 3.0]),
        club_lever_m=np.array([0.04, 0.02, 0.01]),
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = replace(
        state,
        club_velocity_mps=rotation @ state.club_velocity_mps,
        club_angular_velocity_rad_s=rotation @ state.club_angular_velocity_rad_s,
        ball_velocity_mps=rotation @ state.ball_velocity_mps,
        ball_angular_velocity_rad_s=rotation @ state.ball_angular_velocity_rad_s,
        normal=rotation @ state.normal,
        club_lever_m=rotation @ state.club_lever_m,
        club_inertia_world_kg_m2=rotation @ state.club_inertia_world_kg_m2 @ rotation.T,
    )
    original_result = solve_rigid_contact(state)
    rotated_result = solve_rigid_contact(rotated)
    for name in (
        "club_velocity_mps",
        "club_angular_velocity_rad_s",
        "ball_velocity_mps",
        "ball_angular_velocity_rad_s",
        "ball_impulse_n_s",
    ):
        np.testing.assert_allclose(
            getattr(rotated_result, name),
            rotation @ getattr(original_result, name),
            atol=1e-8,
        )


def test_invalid_inertia_and_separating_contact_rejected() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        solve_rigid_contact(_state(club_inertia_world_kg_m2=np.diag([1.0, 1.0, -1.0])))
    with pytest.raises(ValueError, match="approaching"):
        solve_rigid_contact(_state(club_velocity_mps=np.array([-1.0, 0.0, 0.0])))
