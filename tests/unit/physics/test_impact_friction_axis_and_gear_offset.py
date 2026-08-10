"""Regression tests for two impact-model defects (ported from Tools PR #4114).

1. Friction spin axis: the Coulomb-friction spin contribution must use
   ``tangent_dir x n`` (torque of the friction impulse about the ball
   center at the contact point ``-R n``), not ``n x tangent_dir``. The
   old axis spun a lofted strike toward topspin and contradicted the
   pre-existing-spin slip-reduction cap.

2. ``ImpactSolverAPI.solve_with_gear_effect`` must carry
   ``impact_offset`` into the base solve so the MOI effective-mass
   reduction applies; it was previously dropped when building
   ``PreImpactState``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.physics.impact_model import (
    ImpactParameters,
    ImpactSolverAPI,
    PreImpactState,
    RigidBodyImpactModel,
)
from src.shared.python.physics.impact_model.models import SPHERE_ROLLING_CAP_FACTOR

pytestmark = pytest.mark.unit


class TestFrictionSpinAxis:
    """The friction impulse must generate backspin, not topspin."""

    def test_lofted_strike_backspin_axis_and_cap(self) -> None:
        """Lofted (oblique) strike: backspin about -Y, capped at 2/7 limit.

        Club moves +X, face normal tilted up by loft in the XZ plane.
        The tangential face motion drags the ball's contact surface
        downward-forward, so the friction torque axis is
        ``tangent x n = -Y`` (backspin for a target-bound +X shot).
        """
        loft = np.radians(10.5)
        n = np.array([np.cos(loft), 0.0, np.sin(loft)])
        pre = PreImpactState(
            clubhead_velocity=np.array([50.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=n,
            ball_position=np.array([GOLF_BALL_RADIUS_M, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.2,
        )
        # Absurd friction so the rolling-without-slip cap binds.
        params = ImpactParameters(cor=0.83, friction_coefficient=10.0)
        post = RigidBodyImpactModel().solve(pre, params)

        assert post.ball_angular_velocity[1] < 0.0
        assert abs(post.ball_angular_velocity[0]) < 1e-12
        assert abs(post.ball_angular_velocity[2]) < 1e-12

        # Cap: J_f = m v_t (2/7) → omega = (5/7) v_t / R for a solid sphere.
        v_rel = pre.clubhead_velocity
        v_t = float(np.linalg.norm(v_rel - np.dot(v_rel, n) * n))
        i_ball = (2.0 / 5.0) * GOLF_BALL_MASS_KG * GOLF_BALL_RADIUS_M**2
        cap_spin = (
            (GOLF_BALL_MASS_KG * v_t * SPHERE_ROLLING_CAP_FACTOR)
            * GOLF_BALL_RADIUS_M
            / i_ball
        )
        assert float(np.linalg.norm(post.ball_angular_velocity)) == pytest.approx(
            cap_spin, rel=1e-9
        )

    def test_axis_consistent_with_slip_reduction(self) -> None:
        """Pre-existing spin ALONG the generated axis must reduce the
        additional friction spin (the cap logic and axis agree in sign)."""
        loft = np.radians(10.5)
        n = np.array([np.cos(loft), 0.0, np.sin(loft)])

        def solve(pre_spin: np.ndarray) -> np.ndarray:
            pre = PreImpactState(
                clubhead_velocity=np.array([50.0, 0.0, 0.0]),
                clubhead_angular_velocity=np.zeros(3),
                clubhead_orientation=n,
                ball_position=np.array([GOLF_BALL_RADIUS_M, 0.0, 0.0]),
                ball_velocity=np.zeros(3),
                ball_angular_velocity=pre_spin.copy(),
                clubhead_mass=0.2,
            )
            params = ImpactParameters(cor=0.83, friction_coefficient=10.0)
            return RigidBodyImpactModel().solve(pre, params).ball_angular_velocity

        added_no_spin = solve(np.zeros(3))
        # Backspin axis is -Y; pre-existing backspin reduces sliding.
        pre_spin = np.array([0.0, -100.0, 0.0])
        added_with_spin = solve(pre_spin) - pre_spin

        assert float(np.linalg.norm(added_with_spin)) < float(
            np.linalg.norm(added_no_spin)
        )


class TestGearEffectOffsetCarried:
    """solve_with_gear_effect must apply the MOI effective-mass reduction."""

    def test_off_center_gear_effect_ball_speed_below_center(self) -> None:
        solver = ImpactSolverAPI()
        v_club = np.array([45.0, 0.0, 0.0])
        normal = np.array([1.0, 0.0, 0.0])

        center = solver.solve_impact(0.0, v_club, normal, record=False)
        toe = solver.solve_with_gear_effect(
            0.0, v_club, normal, np.array([0.02, 0.0]), record=False
        )

        v_center = float(np.linalg.norm(center.ball_velocity))
        v_toe = float(np.linalg.norm(toe.ball_velocity))
        assert v_toe < v_center

    def test_recorded_pre_state_preserves_offset(self) -> None:
        solver = ImpactSolverAPI()
        offset = np.array([0.015, -0.005])
        solver.solve_with_gear_effect(
            0.0,
            np.array([45.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            offset,
        )

        event = solver.recorder.get_all_events()[-1]
        assert event.pre_state.impact_offset is not None
        np.testing.assert_allclose(event.pre_state.impact_offset, offset)

    def test_solve_impact_offset_matches_direct_model_solve(self) -> None:
        """solve_impact(impact_offset=...) equals a direct model solve with
        the same offset in the pre-state."""
        solver = ImpactSolverAPI()
        offset = np.array([0.02, 0.0])
        via_api = solver.solve_impact(
            0.0,
            np.array([45.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            record=False,
            impact_offset=offset,
        )

        pre = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.2,
            impact_offset=offset,
        )
        direct = RigidBodyImpactModel().solve(pre, solver.params)

        np.testing.assert_allclose(via_api.ball_velocity, direct.ball_velocity)
