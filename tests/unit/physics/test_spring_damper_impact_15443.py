"""Spring-damper impact model reference tests (epic #15443, F5).

The model used to return smash 0.833 and zero spin: the default damping gave a
damping ratio of about 2.6 (over-damped, COR near 0) and no tangential
impulse. These tests pin the repaired behaviour against the rigid-body model
and the sourced driver smash band.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.physics_constants import GOLF_BALL_MASS_KG
from src.shared.python.physics.impact_model import (
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
)
from src.shared.python.physics.impact_model.models import (
    kelvin_voigt_damping_for_cor,
)

pytestmark = pytest.mark.unit

# Driver strike, TrackMan PGA Tour driver smash ~1.48 (bounded below 1.52 in
# docs/physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md).
SMASH_BAND = (1.40, 1.52)
CLUB_SPEED_M_S = 45.0
LOFT_DEG = 10.5
CLUB_MASS_KG = 0.200


def _strike(loft_deg: float = LOFT_DEG) -> PreImpactState:
    loft = np.radians(loft_deg)
    return PreImpactState(
        clubhead_velocity=np.array([CLUB_SPEED_M_S, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=np.array([np.cos(loft), 0.0, np.sin(loft)]),
        ball_position=np.zeros(3),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.zeros(3),
        clubhead_mass=CLUB_MASS_KG,
    )


def test_driver_smash_in_sourced_band() -> None:
    post = SpringDamperImpactModel().solve(_strike(), ImpactParameters())
    smash = float(np.linalg.norm(post.ball_velocity)) / CLUB_SPEED_M_S
    assert SMASH_BAND[0] <= smash <= SMASH_BAND[1], smash


def test_ball_speed_matches_rigid_body_for_same_cor() -> None:
    params = ImpactParameters()
    spring = SpringDamperImpactModel().solve(_strike(), params)
    rigid = RigidBodyImpactModel().solve(_strike(), params)
    np.testing.assert_allclose(
        np.linalg.norm(spring.ball_velocity),
        np.linalg.norm(rigid.ball_velocity),
        rtol=0.02,
    )


def test_spin_positive_for_positive_loft_and_zero_for_flat_face() -> None:
    model = SpringDamperImpactModel()
    lofted = model.solve(_strike(), ImpactParameters())
    assert np.linalg.norm(lofted.ball_angular_velocity) > 1.0
    assert lofted.ball_angular_velocity[1] < 0.0  # backspin for +x launch
    flat = model.solve(_strike(0.0), ImpactParameters())
    assert np.linalg.norm(flat.ball_angular_velocity) < 1e-6


def test_spin_matches_rigid_body_friction_impulse() -> None:
    params = ImpactParameters()
    spring = SpringDamperImpactModel().solve(_strike(), params)
    rigid = RigidBodyImpactModel().solve(_strike(), params)
    np.testing.assert_allclose(
        spring.ball_angular_velocity, rigid.ball_angular_velocity, rtol=0.05
    )


def test_momentum_conserved_along_normal() -> None:
    pre = _strike()
    post = SpringDamperImpactModel().solve(pre, ImpactParameters())
    p_pre = pre.clubhead_mass * pre.clubhead_velocity
    p_post = (
        pre.clubhead_mass * post.clubhead_velocity
        + GOLF_BALL_MASS_KG * post.ball_velocity
    )
    np.testing.assert_allclose(p_post, p_pre, rtol=1e-3, atol=1e-9)


@pytest.mark.parametrize("cor", [0.2, 0.5, 0.83])
def test_damping_derivation_reproduces_cor(cor: float) -> None:
    """Closed-form zeta(e) gives back e for the linear oscillator."""
    stiffness, mass = 1e6, 0.037
    damping = kelvin_voigt_damping_for_cor(stiffness, mass, cor)
    zeta = damping / (2.0 * np.sqrt(stiffness * mass))
    assert np.exp(-np.pi * zeta / np.sqrt(1.0 - zeta**2)) == pytest.approx(cor)


def test_damping_derivation_rejects_bad_cor() -> None:
    with pytest.raises(ValueError):
        kelvin_voigt_damping_for_cor(1e6, 0.037, 1.0)
    with pytest.raises(ValueError):
        kelvin_voigt_damping_for_cor(1e6, 0.037, 0.0)
    with pytest.raises(ValueError):
        kelvin_voigt_damping_for_cor(-1.0, 0.037, 0.5)
