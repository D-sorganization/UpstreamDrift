"""Independent vector invariants for the translating central-contact model."""

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG as MASS,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2 as INERTIA,
    GOLF_BALL_RADIUS_M as RADIUS,
)
from src.shared.python.physics.impact_model import (
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
)

pytestmark = pytest.mark.unit


def _state():
    normal = np.array([2.0, -1.0, 3.0]) / np.sqrt(14)
    return PreImpactState(
        clubhead_velocity=np.array([30.0, 8.0, 15.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=normal,
        ball_position=RADIUS * normal,
        ball_velocity=np.array([2.0, -1.0, 3.0]),
        ball_angular_velocity=np.array([80.0, -60.0, 100.0]),
        clubhead_mass=0.27,
    )


@pytest.mark.parametrize("friction", [0.0, 0.001, 0.4, 10.0])
def test_vector_impulse_preserves_momentum_and_dissipates_energy(friction):
    state = _state()
    post = RigidBodyImpactModel().solve(
        state, ImpactParameters(cor=0.8, friction_coefficient=friction)
    )
    before_p = (
        MASS * state.ball_velocity + state.clubhead_mass * state.clubhead_velocity
    )
    after_p = MASS * post.ball_velocity + state.clubhead_mass * post.clubhead_velocity
    np.testing.assert_allclose(after_p, before_p, rtol=0, atol=1e-13)
    # Club point mass at the contact origin; ball center is R*n from it.
    before_l = (
        np.cross(state.ball_position, MASS * state.ball_velocity)
        + INERTIA * state.ball_angular_velocity
    )
    after_l = (
        np.cross(state.ball_position, MASS * post.ball_velocity)
        + INERTIA * post.ball_angular_velocity
    )
    np.testing.assert_allclose(after_l, before_l, rtol=0, atol=1e-14)

    def kinetic(ball_v, club_v, spin):
        return 0.5 * (
            MASS * np.dot(ball_v, ball_v)
            + state.clubhead_mass * np.dot(club_v, club_v)
            + INERTIA * np.dot(spin, spin)
        )

    assert (
        kinetic(post.ball_velocity, post.clubhead_velocity, post.ball_angular_velocity)
        <= kinetic(
            state.ball_velocity, state.clubhead_velocity, state.ball_angular_velocity
        )
        + 1e-12
    )
    n = state.clubhead_orientation
    before_n = np.dot(state.clubhead_velocity - state.ball_velocity, n)
    after_n = np.dot(post.clubhead_velocity - post.ball_velocity, n)
    assert after_n == pytest.approx(-0.8 * before_n, abs=1e-12)
    if friction == 10:
        slip = (
            post.clubhead_velocity
            - post.ball_velocity
            + RADIUS * np.cross(post.ball_angular_velocity, n)
        )
        slip -= np.dot(slip, n) * n
        np.testing.assert_allclose(slip, np.zeros(3), atol=1e-12)


def test_impact_is_covariant_under_rigid_rotation():
    state = _state()
    params = ImpactParameters(cor=0.8, friction_coefficient=0.4)
    post = RigidBodyImpactModel().solve(state, params)
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotated = _state()
    for field in (
        "clubhead_velocity",
        "clubhead_orientation",
        "ball_position",
        "ball_velocity",
        "ball_angular_velocity",
    ):
        setattr(rotated, field, rotation @ getattr(state, field))
    result = RigidBodyImpactModel().solve(rotated, params)
    for field in ("clubhead_velocity", "ball_velocity", "ball_angular_velocity"):
        np.testing.assert_allclose(
            getattr(result, field), rotation @ getattr(post, field), atol=1e-12
        )


def test_existing_no_slip_spin_requires_no_tangential_impulse():
    state = _state()
    n = state.clubhead_orientation
    tangent = state.clubhead_velocity - state.ball_velocity
    tangent -= np.dot(tangent, n) * n
    state.ball_angular_velocity = np.cross(tangent, n) / RADIUS + 10 * n
    post = RigidBodyImpactModel().solve(
        state, ImpactParameters(cor=0.8, friction_coefficient=0.4)
    )
    impulse = MASS * (post.ball_velocity - state.ball_velocity)
    np.testing.assert_allclose(
        impulse - np.dot(impulse, n) * n, np.zeros(3), atol=1e-13
    )
    np.testing.assert_allclose(
        post.ball_angular_velocity, state.ball_angular_velocity, atol=1e-12
    )
