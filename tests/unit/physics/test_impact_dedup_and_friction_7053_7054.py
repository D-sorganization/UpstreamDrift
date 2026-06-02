"""Tests for impact-model de-duplication and friction-cap provenance.

Covers:
- #7053: the flat ``_impact_physics`` / ``_impact_recorder`` modules are thin
  re-export shims over the canonical ``impact_model`` package. Each public
  symbol must be the *same object* as its canonical definition (identity, not
  just equality), proving there is a single source of truth.
- #7054: the sphere rolling-without-slip friction cap is the analytic 2/7
  factor (not 0.4). An analytic friction-spin test pins the generated spin to
  the rolling-without-slip ceiling.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)

M_BALL = float(GOLF_BALL_MASS_KG)
R_BALL = float(GOLF_BALL_RADIUS_M)
I_BALL = float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)


# ---------------------------------------------------------------------------
# #7053 — single source of truth (shim identity)
# ---------------------------------------------------------------------------


class TestImpactPhysicsShimIdentity:
    """``_impact_physics`` re-exports must BE the canonical objects."""

    @pytest.mark.unit
    def test_rigid_body_is_canonical(self) -> None:
        from src.shared.python.physics import _impact_physics
        from src.shared.python.physics.impact_model import models

        assert _impact_physics.RigidBodyImpactModel is models.RigidBodyImpactModel

    @pytest.mark.unit
    def test_all_physics_symbols_are_canonical(self) -> None:
        from src.shared.python.physics import _impact_physics
        from src.shared.python.physics.impact_model import models, types, utils

        expected = {
            "RigidBodyImpactModel": models.RigidBodyImpactModel,
            "SpringDamperImpactModel": models.SpringDamperImpactModel,
            "FiniteTimeImpactModel": models.FiniteTimeImpactModel,
            "ImpactModel": models.ImpactModel,
            "create_impact_model": models.create_impact_model,
            "ImpactModelType": types.ImpactModelType,
            "ImpactParameters": types.ImpactParameters,
            "PreImpactState": types.PreImpactState,
            "PostImpactState": types.PostImpactState,
            "compute_gear_effect_spin": utils.compute_gear_effect_spin,
            "validate_energy_balance": utils.validate_energy_balance,
        }
        for name, canonical in expected.items():
            assert getattr(_impact_physics, name) is canonical, name

    @pytest.mark.unit
    def test_recorder_symbols_are_canonical(self) -> None:
        from src.shared.python.physics import _impact_recorder
        from src.shared.python.physics.impact_model import solver, types

        assert _impact_recorder.ImpactRecorder is solver.ImpactRecorder
        assert _impact_recorder.ImpactSolverAPI is solver.ImpactSolverAPI
        assert _impact_recorder.ImpactEvent is types.ImpactEvent

    @pytest.mark.unit
    def test_single_class_definition_in_source(self) -> None:
        """AC: grep for the class body returns exactly one definition."""
        from pathlib import Path

        physics_dir = Path(__file__).parents[3] / "src/shared/python/physics"
        hits = 0
        for path in physics_dir.rglob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("class RigidBodyImpactModel"):
                    hits += 1
        assert hits == 1


# ---------------------------------------------------------------------------
# #7054 — analytic friction-spin / rolling-without-slip cap (2/7)
# ---------------------------------------------------------------------------


class TestSphereRollingCapFactor:
    """The rolling cap factor must be the analytic 2/7 sphere value."""

    @pytest.mark.unit
    def test_constant_value_is_two_sevenths(self) -> None:
        from src.shared.python.physics.impact_model.models import (
            SPHERE_ROLLING_CAP_FACTOR,
        )

        assert pytest.approx(2.0 / 7.0) == SPHERE_ROLLING_CAP_FACTOR

    @pytest.mark.unit
    def test_friction_spin_capped_at_rolling_without_slip(self) -> None:
        """With huge friction the spin saturates at the rolling-without-slip
        ceiling.

        For a glancing impact with tangential approach speed v_t (no
        pre-existing spin) and a friction coefficient large enough to saturate
        the cap, the friction impulse equals J_f = m * v_t * (2/7), producing
        a spin magnitude omega = J_f * R / I = m * v_t * (2/7) * R / I.
        For a uniform sphere I = (2/5) m R^2, so omega = (5/7) * v_t / R.
        """
        from src.shared.python.physics.impact_model import (
            ImpactParameters,
            PreImpactState,
            RigidBodyImpactModel,
        )

        v_club = np.array([45.0, 6.0, 0.0])  # +Y tangential component
        params = ImpactParameters(cor=0.78, friction_coefficient=10.0)  # saturate
        state = PreImpactState(
            clubhead_velocity=v_club,
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.array([R_BALL, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.2,
        )

        post = RigidBodyImpactModel().solve(state, params)

        # v_tangent is the +Y component of v_rel (ball at rest); n is +X.
        v_t = 6.0
        # Rolling-without-slip ceiling for a uniform solid sphere.
        expected_omega = (5.0 / 7.0) * v_t / R_BALL
        # spin_axis = n x tangent_dir = +X x +Y = +Z.
        omega_z = float(post.ball_angular_velocity[2])
        assert omega_z == pytest.approx(expected_omega, rel=1e-9)
