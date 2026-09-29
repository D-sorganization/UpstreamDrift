# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for spinal stress module (L5/S1 compression and shear)."""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.models import BodyModel, make_squat_config
from movement_optimizer.spine_loads import (
    NIOSH_COMPRESSION_LIMIT,
    spinal_compression,
    spinal_shear,
)


@pytest.fixture()
def default_body() -> BodyModel:
    return BodyModel(75.0, 1.75)


@pytest.fixture()
def squat_dyn(default_body: BodyModel):
    dyn, _qs, _qe, _qb = make_squat_config(default_body, 60.0)
    return dyn


class TestStandingCompression:
    """At standing (q=0, qd=0, qdd=0) compression should equal gravity on mass above L5."""

    def test_standing_compression_equals_gravity(self, default_body: BodyModel, squat_dyn) -> None:
        q = np.zeros(3)
        qd = np.zeros(3)
        qdd = np.zeros(3)
        bar_mass = 60.0

        comp = spinal_compression(q, qd, qdd, default_body, bar_mass, "squat")

        # At standing (torso_angle=0), cos(0)=1, so compression = full gravity
        m_above = default_body.m_squat[2]  # torso+head+arms
        expected = (m_above + bar_mass) * default_body.g
        np.testing.assert_allclose(comp, expected, rtol=1e-6)

    def test_standing_compression_no_bar(self, default_body: BodyModel, squat_dyn) -> None:
        q = np.zeros(3)
        qd = np.zeros(3)
        qdd = np.zeros(3)

        comp = spinal_compression(q, qd, qdd, default_body, 0.0, "squat")

        m_above = default_body.m_squat[2]
        expected = m_above * default_body.g
        np.testing.assert_allclose(comp, expected, rtol=1e-6)


class TestStandingShear:
    """At standing (spine vertical), shear should be near zero."""

    def test_standing_shear_near_zero(self, default_body: BodyModel, squat_dyn) -> None:
        q = np.zeros(3)
        qd = np.zeros(3)
        qdd = np.zeros(3)

        shear = spinal_shear(q, qd, qdd, default_body, 60.0, "squat")
        np.testing.assert_allclose(shear, 0.0, atol=1e-6)


class TestForwardLean:
    """With torso lean, shear increases and compression decreases."""

    def test_shear_increases_with_lean(self, default_body: BodyModel, squat_dyn) -> None:
        qd = np.zeros(3)
        qdd = np.zeros(3)
        bar_mass = 60.0

        q_upright = np.array([0.0, 0.0, 0.0])
        q_leaned = np.array([0.0, 0.0, np.radians(30)])

        shear_upright = spinal_shear(q_upright, qd, qdd, default_body, bar_mass, "squat")
        shear_leaned = spinal_shear(q_leaned, qd, qdd, default_body, bar_mass, "squat")

        assert abs(shear_leaned) > abs(shear_upright)  # type: ignore

    def test_shear_proportional_to_sin(self, default_body: BodyModel, squat_dyn) -> None:
        qd = np.zeros(3)
        qdd = np.zeros(3)
        bar_mass = 60.0
        angle = np.radians(30)
        q = np.array([0.0, 0.0, angle])

        shear = spinal_shear(q, qd, qdd, default_body, bar_mass, "squat")

        m_above = default_body.m_squat[2]
        expected = (m_above + bar_mass) * default_body.g * np.sin(angle)
        np.testing.assert_allclose(shear, expected, rtol=1e-6)

    def test_compression_decreases_with_lean(self, default_body: BodyModel, squat_dyn) -> None:
        qd = np.zeros(3)
        qdd = np.zeros(3)
        bar_mass = 60.0

        q_upright = np.array([0.0, 0.0, 0.0])
        q_leaned = np.array([0.0, 0.0, np.radians(30)])

        comp_upright = spinal_compression(q_upright, qd, qdd, default_body, bar_mass, "squat")
        comp_leaned = spinal_compression(q_leaned, qd, qdd, default_body, bar_mass, "squat")

        assert comp_leaned < comp_upright

    def test_compression_cos_component(self, default_body: BodyModel, squat_dyn) -> None:
        qd = np.zeros(3)
        qdd = np.zeros(3)
        bar_mass = 60.0
        angle = np.radians(30)
        q = np.array([0.0, 0.0, angle])

        comp = spinal_compression(q, qd, qdd, default_body, bar_mass, "squat")

        m_above = default_body.m_squat[2]
        # Static part: cos component of gravity
        expected_static = (m_above + bar_mass) * default_body.g * np.cos(angle)
        # With zero accelerations, inertial term is zero
        np.testing.assert_allclose(comp, expected_static, rtol=1e-6)


class TestBatchSpineLoads:
    """Batch versions accept (N, 3) arrays and return (N,) arrays."""

    def test_batch_compression_shape(self, default_body: BodyModel, squat_dyn) -> None:
        n = 10
        rng = np.random.default_rng(42)
        q = rng.normal(0, 0.1, (n, 3))
        qd = rng.normal(0, 0.1, (n, 3))
        qdd = rng.normal(0, 0.1, (n, 3))

        comp = spinal_compression(q, qd, qdd, default_body, 60.0, "squat")
        assert comp.shape == (n,)  # type: ignore

    def test_batch_shear_shape(self, default_body: BodyModel, squat_dyn) -> None:
        n = 10
        rng = np.random.default_rng(42)
        q = rng.normal(0, 0.1, (n, 3))
        qd = rng.normal(0, 0.1, (n, 3))
        qdd = rng.normal(0, 0.1, (n, 3))

        shear = spinal_shear(q, qd, qdd, default_body, 60.0, "squat")
        assert shear.shape == (n,)  # type: ignore

    def test_batch_matches_loop(self, default_body: BodyModel, squat_dyn) -> None:
        n = 5
        rng = np.random.default_rng(42)
        q = rng.normal(0, 0.1, (n, 3))
        qd = np.zeros((n, 3))
        qdd = np.zeros((n, 3))

        batch_comp = spinal_compression(q, qd, qdd, default_body, 60.0, "squat")
        loop_comp = np.array(
            [spinal_compression(q[i], qd[i], qdd[i], default_body, 60.0, "squat") for i in range(n)]
        )
        np.testing.assert_allclose(batch_comp, loop_comp, rtol=1e-10)


class TestDeadliftExerciseType:
    """Deadlift exercise type uses m_deadlift mass."""

    def test_deadlift_standing_compression(self, default_body: BodyModel) -> None:
        q = np.zeros(3)
        qd = np.zeros(3)
        qdd = np.zeros(3)
        bar_mass = 60.0

        comp = spinal_compression(q, qd, qdd, default_body, bar_mass, "deadlift")

        # Deadlift: torso segment is trunk_head only (arms hang separately)
        m_above = default_body.m_deadlift[2]
        expected = (m_above + bar_mass) * default_body.g
        np.testing.assert_allclose(comp, expected, rtol=1e-6)


class TestShearInertialComponent:
    """Spinal shear should include inertial terms from angular acceleration and velocity."""

    def test_shear_with_acceleration(self, default_body: BodyModel) -> None:
        """Angular acceleration adds tangential shear: m * qdd * d_com * cos(angle)."""
        angle = np.radians(30)
        q = np.array([0.0, 0.0, angle])
        qd = np.zeros(3)
        qdd = np.array([0.0, 0.0, 5.0])  # torso accelerating
        bar_mass = 60.0

        shear = spinal_shear(q, qd, qdd, default_body, bar_mass, "squat")

        m_torso = default_body.m_squat[2]
        d_com = default_body.d[2]
        gravity_part = (m_torso + bar_mass) * default_body.g * np.sin(angle)
        tangential_part = m_torso * 5.0 * d_com * np.cos(angle)
        expected = gravity_part + tangential_part
        np.testing.assert_allclose(shear, expected, rtol=1e-6)

    def test_shear_with_velocity(self, default_body: BodyModel) -> None:
        """Angular velocity adds centripetal shear: m * qd^2 * d_com * sin(angle)."""
        angle = np.radians(30)
        q = np.array([0.0, 0.0, angle])
        qd = np.array([0.0, 0.0, 3.0])  # torso moving
        qdd = np.zeros(3)
        bar_mass = 60.0

        shear = spinal_shear(q, qd, qdd, default_body, bar_mass, "squat")

        m_torso = default_body.m_squat[2]
        d_com = default_body.d[2]
        gravity_part = (m_torso + bar_mass) * default_body.g * np.sin(angle)
        centripetal_part = m_torso * 3.0**2 * d_com * np.sin(angle)
        expected = gravity_part + centripetal_part
        np.testing.assert_allclose(shear, expected, rtol=1e-6)

    def test_shear_exceeds_static_during_motion(self, default_body: BodyModel) -> None:
        """Shear with inertial terms should exceed pure-static shear."""
        angle = np.radians(30)
        q = np.array([0.0, 0.0, angle])
        bar_mass = 60.0

        static_shear = spinal_shear(q, np.zeros(3), np.zeros(3), default_body, bar_mass, "squat")
        dynamic_shear = spinal_shear(
            q,
            np.array([0.0, 0.0, 3.0]),
            np.array([0.0, 0.0, 5.0]),
            default_body,
            bar_mass,
            "squat",
        )
        assert abs(dynamic_shear) > abs(static_shear)  # type: ignore


class TestNIOSHConstant:
    """The NIOSH compression limit constant exists."""

    def test_niosh_limit_value(self) -> None:
        assert NIOSH_COMPRESSION_LIMIT == 3400.0
