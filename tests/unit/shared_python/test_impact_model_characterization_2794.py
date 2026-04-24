"""Characterization tests for the impact model — Issue #2794 / #2700.

PR #2774 (origin branch `fix/2700-impact-model-critical-fixes`) attempted
to correct four physics defects flagged in issue #2700:

1. Gear-effect spin-axis sign in
   ``RigidBodyImpactModel._compute_friction_spin`` — currently
   ``np.cross(n, tangent_dir)``; the derivation in #2700 argues for
   ``np.cross(tangent_dir, n)``.
2. Angular-momentum conservation: the clubhead angular velocity is
   copied verbatim into ``PostImpactState`` with no friction-impulse
   reaction torque (Newton's third law).
3. Dynamic loft / lie / MOI: ``PreImpactState`` carries ``clubhead_loft``
   and ``clubhead_lie`` but the solver never consumes them; the face
   normal comes straight from ``clubhead_orientation``.
4. Energy loss factor: ``validate_energy_balance`` uses ``1 − COR²``
   instead of the Newton-impact form ``μ·(1 − e²)/(1 + μ)²`` with
   ``μ = m_club / m_ball``.

PR #2774 was closed unmerged (189 files, dirty, broken CI). Rather than
re-asserting unverified physics, this module **pins the current
behavior** so that any future correction is an intentional, reviewed
change with a textbook citation attached. When the physics is re-derived
and cited, flip each ``assert`` to the corrected expectation and replace
the TRACKED comment with the reference.

References to chase when rebuilding (issue #2700 body):

* Cochran & Stobbs, *Search for the Perfect Swing* (1968) — classical
  gear-effect derivation.
* Penner, "The physics of golf: The optimum loft of a driver,"
  *Am. J. Phys.* 69, 563 (2001).
* Cross, "Impact of a ball with a bat or racket," *Am. J. Phys.* 67, 692
  (1999) — for the Newton impact energy-loss formula.

DO NOT "fix" these tests by flipping signs without a citation; see
issue #2794.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics.impact_model import (
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
    compute_gear_effect_spin,
    validate_energy_balance,
)


def _make_pre_state(
    *,
    clubhead_velocity: np.ndarray | None = None,
    ball_velocity: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
    impact_offset: np.ndarray | None = None,
) -> PreImpactState:
    return PreImpactState(
        clubhead_velocity=(
            clubhead_velocity
            if clubhead_velocity is not None
            else np.array([45.0, 0.0, 0.0])
        ),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=(
            orientation if orientation is not None else np.array([1.0, 0.0, 0.0])
        ),
        ball_position=np.zeros(3),
        ball_velocity=(ball_velocity if ball_velocity is not None else np.zeros(3)),
        ball_angular_velocity=np.zeros(3),
        clubhead_mass=0.200,
        impact_offset=impact_offset,
    )


class TestGearEffectSignCharacterization:
    """Pin the gear-effect sign convention used by the current solver.

    TRACKED(#2794): verify against Cochran & Stobbs gear-effect derivation
    before changing any of the signs below.
    """

    def test_toe_strike_horizontal_spin_sign(self) -> None:
        """A +x (toe-side) impact offset currently produces negative
        horizontal (vertical-axis) spin under the default convention
        (see ``compute_gear_effect_spin`` lines 55–56)."""
        offset = np.array([0.02, 0.0])  # 20 mm toe-side
        v_club = np.array([45.0, 0.0, 0.0])
        face_normal = np.array([1.0, 0.0, 0.0])

        spin = compute_gear_effect_spin(offset, v_club, face_normal, gear_factor=0.5)

        # Vertical (z) component stores horizontal-axis spin in the
        # current implementation. Toe-side offset gives a negative z spin.
        assert spin[2] < 0.0, (
            "Current impl returns negative z spin for toe strike; "
            "issue #2700 claims this sign is inverted vs. gear-effect "
            "measurement. Do not flip without a citation."
        )

    def test_heel_strike_horizontal_spin_sign(self) -> None:
        offset = np.array([-0.02, 0.0])  # heel-side
        v_club = np.array([45.0, 0.0, 0.0])
        face_normal = np.array([1.0, 0.0, 0.0])

        spin = compute_gear_effect_spin(offset, v_club, face_normal, gear_factor=0.5)
        assert spin[2] > 0.0

    def test_high_strike_vertical_spin_sign(self) -> None:
        offset = np.array([0.0, 0.01])  # high on face
        v_club = np.array([45.0, 0.0, 0.0])
        face_normal = np.array([1.0, 0.0, 0.0])

        spin = compute_gear_effect_spin(offset, v_club, face_normal, gear_factor=0.5)
        # Non-zero horizontal-axis component expected.
        assert np.linalg.norm(spin[:2]) > 0.0

    def test_centered_strike_has_no_gear_spin(self) -> None:
        offset = np.zeros(2)
        v_club = np.array([45.0, 0.0, 0.0])
        face_normal = np.array([1.0, 0.0, 0.0])

        spin = compute_gear_effect_spin(offset, v_club, face_normal)
        assert np.allclose(spin, 0.0)


class TestFrictionSpinCrossProductCharacterization:
    """Pin the friction-spin spin-axis convention in
    ``RigidBodyImpactModel._compute_friction_spin``.

    TRACKED(#2794): Issue #2700 derives the torque as ``τ = r × F`` with
    ``r = −R·n`` and ``F = −j·tangent_dir``; this would give a spin axis
    proportional to ``tangent_dir × n`` (opposite of the current
    ``np.cross(n, tangent_dir)``). Confirm against a textbook before
    changing.
    """

    def test_oblique_impact_produces_spin_along_expected_axis(self) -> None:
        # Clubhead moves mostly along +x with a small +z component so
        # the relative velocity has a tangential piece perpendicular to n.
        pre = _make_pre_state(
            clubhead_velocity=np.array([45.0, 0.0, 3.0]),
            orientation=np.array([1.0, 0.0, 0.0]),
        )
        params = ImpactParameters()
        model = RigidBodyImpactModel()

        post = model.solve(pre, params)
        spin = post.ball_angular_velocity

        # Non-trivial spin produced.
        assert np.linalg.norm(spin) > 0.0
        # With n = +x and tangent_dir along +z, the corrected torque
        # τ = r × F = (-R·n) × (j·tangent_dir) gives spin_axis = tangent_dir × n
        # = z × x = +y. Fix from #2794 (citing #2700 derivation).
        assert spin[1] > 0.0, (
            "Corrected convention: spin_axis = tangent_dir × n gives "
            "positive-y for upward club motion (τ = r × F derivation)."
        )


class TestAngularMomentumConservation:
    """Verify that the rigid-body solver applies the friction-impulse
    reaction torque to the clubhead for off-center impacts (#2700 point 2,
    fixed in #2794).
    """

    def test_center_strike_clubhead_angular_velocity_unchanged(self) -> None:
        """Center strike with no impact offset → no friction torque on club."""
        pre_omega = np.array([1.0, 2.0, 3.0])
        pre = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 2.0]),
            clubhead_angular_velocity=pre_omega.copy(),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
        )
        post = RigidBodyImpactModel().solve(pre, ImpactParameters())
        # No impact_offset → moment arm is zero → club omega unchanged.
        assert np.allclose(post.clubhead_angular_velocity, pre_omega)

    def test_off_center_strike_modifies_clubhead_angular_velocity(self) -> None:
        """Off-center strike → friction reaction torque changes club omega."""
        pre_omega = np.zeros(3)
        pre = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 2.0]),
            clubhead_angular_velocity=pre_omega.copy(),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            impact_offset=np.array([0.02, 0.0]),  # 20 mm toe-side
        )
        post = RigidBodyImpactModel().solve(pre, ImpactParameters())
        # Friction reaction torque must shift club omega away from zero.
        assert not np.allclose(post.clubhead_angular_velocity, pre_omega), (
            "Off-center friction impulse must react on club (Newton's 3rd law)."
        )


class TestLoftAndLieUnusedKnownDefect:
    """Pin the fact that ``clubhead_loft`` and ``clubhead_lie`` flow
    into ``PreImpactState`` but are ignored by the rigid-body solver
    (issue #2700 point 3).
    """

    def test_changing_loft_does_not_affect_launch(self) -> None:
        base = _make_pre_state()
        base.clubhead_loft = np.radians(9.0)
        flat = _make_pre_state()
        flat.clubhead_loft = np.radians(18.0)

        params = ImpactParameters()
        model = RigidBodyImpactModel()
        v_base = model.solve(base, params).ball_velocity
        v_flat = model.solve(flat, params).ball_velocity

        # TRACKED(#2794): dynamic loft should shift launch direction by
        # ±8° per #2700. When that coupling is added, this assertion
        # must flip to assert a non-trivial difference.
        assert np.allclose(v_base, v_flat)

    def test_changing_lie_does_not_affect_launch(self) -> None:
        base = _make_pre_state()
        base.clubhead_lie = np.radians(55.0)
        upright = _make_pre_state()
        upright.clubhead_lie = np.radians(65.0)

        params = ImpactParameters()
        model = RigidBodyImpactModel()
        v_base = model.solve(base, params).ball_velocity
        v_upr = model.solve(upright, params).ball_velocity

        assert np.allclose(v_base, v_upr)


class TestEnergyLossFormulaKnownDefect:
    """Pin the *current* 1 − COR² energy-loss expectation reported by
    ``validate_energy_balance``. Issue #2700 point 4 calls this wrong.
    """

    def test_expected_loss_factor_uses_one_minus_cor_squared(self) -> None:
        pre = _make_pre_state()
        params = ImpactParameters(cor=0.83)
        post = RigidBodyImpactModel().solve(pre, params)

        result = validate_energy_balance(pre, post, params)

        # TRACKED(#2794): Newton-impact formula is
        #   ΔKE/KE_pre = μ·(1 − e²)/(1 + μ)²
        # with μ = m_club / m_ball ≈ 0.200 / 0.0459 ≈ 4.36. The current
        # code reports 1 − e² instead. Replace with the correct form
        # when the derivation has a citation (Cross 1999 or Penner
        # 2001).
        assert result["expected_loss_factor"] == pytest.approx(1 - 0.83**2)


class TestSolveProducesFiniteState:
    """Smoke test — regardless of the open physics questions, the
    rigid-body solver must return a finite, well-formed PostImpactState.
    """

    def test_finite_outputs_for_default_driver_strike(self) -> None:
        pre = _make_pre_state()
        post = RigidBodyImpactModel().solve(pre, ImpactParameters())

        assert np.all(np.isfinite(post.ball_velocity))
        assert np.all(np.isfinite(post.ball_angular_velocity))
        assert np.all(np.isfinite(post.clubhead_velocity))
        assert np.all(np.isfinite(post.clubhead_angular_velocity))
        assert np.isfinite(post.energy_transfer)
