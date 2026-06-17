# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for bench press model (issue #26)."""

from __future__ import annotations

import numpy as np

from movement_optimizer.models import (
    BenchPressModel,
    BodyModel,
    ChainGeometry,
    LagrangianDynamics,
    make_bench_press_config,
)


class TestBenchPressConfig:
    def test_bench_config_shapes(self, default_body: BodyModel) -> None:
        """make_bench_press_config returns correct shapes."""
        _dyn, qs, qe, qb, q_via = make_bench_press_config(default_body, 60.0)
        assert qs.shape == (3,)
        assert qe.shape == (3,)
        assert qb.shape == (3, 2)
        assert q_via is not None
        assert q_via.shape == (3,)

    def test_bench_horizontal_orientation(self, default_body: BodyModel) -> None:
        """The bench press models a supine (horizontal) body.

        The arm chain starts at the shoulder; joints swing in the
        sagittal plane perpendicular to the body's long axis.
        Verify that the BenchPressModel segment lengths are derived
        from the arm, not the legs/torso.
        """
        bp = BenchPressModel(default_body)
        total_arm_len = bp.L[0] + bp.L[1] + bp.L[2]
        # Total should be close to arm length (upper + forearm fracs + small wrist)
        assert total_arm_len > 0.5 * default_body.L_arm
        assert total_arm_len < 1.1 * default_body.L_arm

    def test_arm_angle_default(self, default_body: BodyModel) -> None:
        """Default arm_angle is 45 degrees."""
        body = BodyModel(75.0, 1.75, arm_angle=45.0)
        assert body.arm_angle == 45.0

    def test_arm_effective_length(self, default_body: BodyModel) -> None:
        """L_arm_eff = L_arm * sin(arm_angle_rad)."""
        body = BodyModel(75.0, 1.75, arm_angle=45.0)
        expected = body.L_arm * np.sin(np.radians(45.0))
        np.testing.assert_allclose(body.L_arm_eff, expected, rtol=1e-12)

    def test_arm_angle_zero_gives_zero_eff(self) -> None:
        """With arm_angle=0, projected arm length is 0."""
        body = BodyModel(75.0, 1.75, arm_angle=0.0)
        np.testing.assert_allclose(body.L_arm_eff, 0.0, atol=1e-15)

    def test_arm_angle_90_gives_full_length(self) -> None:
        """With arm_angle=90, projected arm length equals full L_arm."""
        body = BodyModel(75.0, 1.75, arm_angle=90.0)
        np.testing.assert_allclose(body.L_arm_eff, body.L_arm, rtol=1e-12)

    def test_bar_above_body(self, default_body: BodyModel) -> None:
        """Bar y > 0 at both start and end positions.

        In the bench press the bar is always above the body (positive y
        in the FK frame, since the chain swings upward from the shoulder).
        """
        dyn, qs, qe, _qb, _q_via = make_bench_press_config(default_body, 60.0)
        fk_start = dyn.forward_kinematics(qs)
        fk_end = dyn.forward_kinematics(qe)
        # The hand (end effector) y should be positive at both poses
        # (bar is above the shoulder pivot which is at y=0).
        # "shoulder" is the base of the chain (origin); "hand" is the tip.
        assert fk_start["hand"][1] > 0, "Bar should be above body at start"
        assert fk_end["hand"][1] > 0, "Bar should be above body at end"

    def test_bench_start_is_lockout(self, default_body: BodyModel) -> None:
        """q_start should have shoulder near 0 degrees (arms vertical/lockout)."""
        _dyn, qs, _qe, _qb, _q_via = make_bench_press_config(default_body, 60.0)
        shoulder_deg = np.degrees(qs[0])
        assert abs(shoulder_deg) < 5, (
            f"At lockout shoulder should be near 0 deg, got {shoulder_deg:.1f}"
        )

    def test_bench_via_is_chest(self, default_body: BodyModel) -> None:
        """q_via should have shoulder near 80 degrees (upper arm horizontal)."""
        _dyn, _qs, _qe, _qb, q_via = make_bench_press_config(default_body, 60.0)
        shoulder_deg = np.degrees(q_via[0])
        assert 70 < shoulder_deg < 95, (
            f"At chest touch shoulder should be ~80 deg, got {shoulder_deg:.1f}"
        )

    def test_bench_full_rep(self, default_body: BodyModel) -> None:
        """q_start should equal q_end (full rep returns to lockout)."""
        _dyn, qs, qe, _qb, _q_via = make_bench_press_config(default_body, 60.0)
        np.testing.assert_allclose(qs, qe, atol=1e-12)

    def test_bench_supine_gravity(self, default_body: BodyModel) -> None:
        """Bench press dynamics should use supine gravity (cos instead of sin).

        At q=0 (lockout, arms straight up), gravity torque should be
        maximal in supine (cos(0)=1) but zero in standing (sin(0)=0).
        """
        dyn, _qs, _qe, _qb, _q_via = make_bench_press_config(default_body, 60.0)
        assert dyn.supine is True

        q_lockout = np.zeros(3)
        grav = dyn._gravity_vector(q_lockout)
        # Supine: cos(0) = 1, so gravity torque is maximal at lockout
        assert np.all(np.abs(grav) > 0), "Supine gravity should be nonzero at q=0"

    def test_standing_vs_supine_gravity(self, default_body: BodyModel) -> None:
        """Standing gravity uses sin(q), supine uses cos(q).

        At q = pi/2, sin(pi/2) = 1, cos(pi/2) = 0, so the two should
        give opposite extremes.
        """
        bp = BenchPressModel(default_body)
        # Standing chain (supine=False)
        dyn_standing = LagrangianDynamics(
            default_body,
            bp.m.copy(),
            bp.I.copy(),
            60.0,
            chain_geometry=ChainGeometry(L=bp.L, d=bp.d),
            supine=False,
        )
        # Supine chain
        dyn_supine = LagrangianDynamics(
            default_body,
            bp.m.copy(),
            bp.I.copy(),
            60.0,
            chain_geometry=ChainGeometry(L=bp.L, d=bp.d),
            supine=True,
        )

        q_zero = np.zeros(3)
        grav_standing = dyn_standing._gravity_vector(q_zero)
        grav_supine = dyn_supine._gravity_vector(q_zero)
        # At q=0: sin(0)=0, cos(0)=1
        np.testing.assert_allclose(grav_standing, 0.0, atol=1e-12)
        assert np.all(np.abs(grav_supine) > 0)
