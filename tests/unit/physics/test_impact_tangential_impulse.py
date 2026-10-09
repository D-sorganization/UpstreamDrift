"""Finite-club tangential impact conservation and contact kinematics."""

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.physics.impact_model import (
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
)

pytestmark = pytest.mark.unit


def _state(*, pre_spin: np.ndarray | None = None) -> PreImpactState:
    return PreImpactState(
        clubhead_velocity=np.array([45.0, 3.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        ball_position=np.array([GOLF_BALL_RADIUS_M, 0.0, 0.0]),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.zeros(3) if pre_spin is None else pre_spin,
        clubhead_mass=0.2,
    )


def _slip(
    club_velocity: np.ndarray, ball_velocity: np.ndarray, spin: np.ndarray
) -> np.ndarray:
    n = np.array([1.0, 0.0, 0.0])
    relative = club_velocity - ball_velocity + GOLF_BALL_RADIUS_M * np.cross(spin, n)
    return relative - np.dot(relative, n) * n


def test_uncapped_contact_reaches_no_slip_with_finite_club_mass() -> None:
    state = _state()
    post = RigidBodyImpactModel().solve(
        state, ImpactParameters(cor=0.8, friction_coefficient=10.0)
    )
    np.testing.assert_allclose(
        _slip(post.clubhead_velocity, post.ball_velocity, post.ball_angular_velocity),
        np.zeros(3),
        atol=1e-11,
    )
    before = (
        GOLF_BALL_MASS_KG * state.ball_velocity
        + state.clubhead_mass * state.clubhead_velocity
    )
    after = (
        GOLF_BALL_MASS_KG * post.ball_velocity
        + state.clubhead_mass * post.clubhead_velocity
    )
    np.testing.assert_allclose(after, before, atol=1e-12)
    impulse = GOLF_BALL_MASS_KG * (post.ball_velocity - state.ball_velocity)
    np.testing.assert_allclose(
        post.ball_angular_velocity - state.ball_angular_velocity,
        GOLF_BALL_RADIUS_M
        / GOLF_BALL_MOMENT_OF_INERTIA_KG_M2
        * np.cross(impulse, np.array([1.0, 0.0, 0.0])),
        atol=1e-10,
    )
    assert 0 < post.ball_velocity[1] < state.clubhead_velocity[1]


def test_coulomb_cap_and_zero_friction_recovery() -> None:
    state = _state()
    no_friction = RigidBodyImpactModel().solve(
        state, ImpactParameters(cor=0.8, friction_coefficient=0.0)
    )
    np.testing.assert_allclose(no_friction.ball_velocity[1:], np.zeros(2))
    np.testing.assert_allclose(
        no_friction.clubhead_velocity[1:], state.clubhead_velocity[1:]
    )
    np.testing.assert_allclose(
        no_friction.ball_angular_velocity, state.ball_angular_velocity
    )

    mu = 0.01
    capped = RigidBodyImpactModel().solve(
        state, ImpactParameters(cor=0.8, friction_coefficient=mu)
    )
    normal_impulse = GOLF_BALL_MASS_KG * capped.ball_velocity[0]
    tangent_impulse = GOLF_BALL_MASS_KG * capped.ball_velocity[1]
    assert tangent_impulse == pytest.approx(mu * normal_impulse, rel=1e-12)
    assert (
        np.linalg.norm(
            _slip(
                capped.clubhead_velocity,
                capped.ball_velocity,
                capped.ball_angular_velocity,
            )
        )
        > 0
    )


def test_preexisting_spin_reverses_tangential_friction() -> None:
    # Sufficient backspin overdrives the contact surface past the club.
    state = _state(pre_spin=np.array([0.0, 0.0, -200.0]))
    post = RigidBodyImpactModel().solve(
        state, ImpactParameters(cor=0.8, friction_coefficient=10.0)
    )
    assert post.ball_velocity[1] < 0.0
    np.testing.assert_allclose(
        _slip(post.clubhead_velocity, post.ball_velocity, post.ball_angular_velocity),
        np.zeros(3),
        atol=1e-11,
    )


def test_separating_contact_is_rejected_instead_of_attracting_ball() -> None:
    state = _state()
    state.clubhead_velocity = np.array([-1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="approaching"):
        RigidBodyImpactModel().solve(state, ImpactParameters())
