"""Edge-case and precondition tests for DoublePendulumDynamics."""

from __future__ import annotations

import math

import pytest
from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
    JointTorques,
    LowerSegmentProperties,
    SegmentProperties,
)


@pytest.fixture()
def state() -> DoublePendulumState:
    return DoublePendulumState(theta1=0.5, theta2=0.3, omega1=0.2, omega2=-0.1)


@pytest.fixture()
def dyn() -> DoublePendulumDynamics:
    return DoublePendulumDynamics()


class TestParametersProperties:
    def test_projected_gravity_disabled(self) -> None:
        params = DoublePendulumParameters.default()
        params.gravity_enabled = False
        assert params.projected_gravity == 0.0

    def test_projected_gravity_unconstrained(self) -> None:
        params = DoublePendulumParameters.default()
        params.constrained_to_plane = False
        assert params.projected_gravity == pytest.approx(params.gravity_m_s2)

    def test_projected_gravity_inclined(self) -> None:
        params = DoublePendulumParameters.default()
        expected = params.gravity_m_s2 * math.cos(
            math.radians(params.plane_inclination_deg)
        )
        assert params.projected_gravity == pytest.approx(expected)

    def test_plane_inclination_rad(self) -> None:
        params = DoublePendulumParameters.default()
        assert params.plane_inclination_rad == pytest.approx(
            math.radians(params.plane_inclination_deg)
        )


class TestSegmentProperties:
    def test_center_of_mass_distance(self) -> None:
        seg = SegmentProperties(
            length_m=2.0, mass_kg=1.0, center_of_mass_ratio=0.5, inertia_about_com=1.0
        )
        assert seg.center_of_mass_distance == pytest.approx(1.0)

    def test_inertia_about_proximal_joint(self) -> None:
        seg = SegmentProperties(
            length_m=2.0, mass_kg=3.0, center_of_mass_ratio=0.5, inertia_about_com=1.0
        )
        # 1.0 + 3.0 * 1.0**2 = 4.0
        assert seg.inertia_about_proximal_joint == pytest.approx(4.0)


class TestLowerSegmentProperties:
    def test_total_mass(self) -> None:
        seg = LowerSegmentProperties(
            length_m=1.0, shaft_mass_kg=0.2, clubhead_mass_kg=0.3, shaft_com_ratio=0.5
        )
        assert seg.total_mass == pytest.approx(0.5)

    def test_center_of_mass_distance(self) -> None:
        seg = LowerSegmentProperties(
            length_m=1.0, shaft_mass_kg=0.2, clubhead_mass_kg=0.3, shaft_com_ratio=0.5
        )
        # (0.5*0.2 + 1.0*0.3)/0.5 = 0.4/0.5 = 0.8
        assert seg.center_of_mass_distance == pytest.approx(0.8)

    def test_inertia_about_com_positive(self) -> None:
        seg = LowerSegmentProperties(
            length_m=1.0,
            shaft_mass_kg=0.15,
            clubhead_mass_kg=0.20,
            shaft_com_ratio=0.43,
        )
        assert seg.inertia_about_com > 0

    def test_inertia_about_proximal_joint_uses_parallel_axis(self) -> None:
        seg = LowerSegmentProperties(
            length_m=1.0,
            shaft_mass_kg=0.15,
            clubhead_mass_kg=0.20,
            shaft_com_ratio=0.43,
        )
        expected = (
            seg.inertia_about_com + seg.total_mass * seg.center_of_mass_distance**2
        )
        assert seg.inertia_about_proximal_joint == pytest.approx(expected)


class TestSingularMassMatrix:
    def test_zero_mass_lower_segment_singular(self) -> None:
        upper = SegmentProperties(
            length_m=1.0, mass_kg=1.0, center_of_mass_ratio=0.5, inertia_about_com=0.1
        )
        # Zero-mass, zero-inertia lower segment => det(M)=0
        lower = LowerSegmentProperties(
            length_m=1.0, shaft_mass_kg=0.0, clubhead_mass_kg=0.0, shaft_com_ratio=0.5
        )
        # total_mass=0 will divide-by-zero in center_of_mass_distance;
        # instead, force tiny but vanishing parameters via direct override.
        params = DoublePendulumParameters(
            upper_segment=upper,
            lower_segment=LowerSegmentProperties(
                length_m=1.0,
                shaft_mass_kg=1e-30,
                clubhead_mass_kg=1e-30,
                shaft_com_ratio=0.5,
            ),
        )
        dyn = DoublePendulumDynamics(params)
        with pytest.raises(ZeroDivisionError, match="singular|determinant"):
            dyn.control_affine(
                DoublePendulumState(theta1=0.0, theta2=0.0, omega1=0.0, omega2=0.0)
            )


class TestDynamicsComputations:
    def test_damping_vector(self, dyn: DoublePendulumDynamics) -> None:
        d1, d2 = dyn.damping_vector(1.0, -2.0)
        assert d1 == pytest.approx(dyn._d1 * 1.0)
        assert d2 == pytest.approx(dyn._d2 * -2.0)

    def test_gravity_vector_zero_at_zero_angles(
        self, dyn: DoublePendulumDynamics
    ) -> None:
        g1, g2 = dyn.gravity_vector(0.0, 0.0)
        assert g1 == pytest.approx(0.0)
        assert g2 == pytest.approx(0.0)

    def test_inverse_dynamics_identity(
        self, dyn: DoublePendulumDynamics, state: DoublePendulumState
    ) -> None:
        """Apply torques -> get accelerations -> invert -> recover torques."""
        # forward: compute accelerations from zero applied torque
        from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (  # noqa: E501
            DoublePendulumDynamics as DPD,
        )

        # Use forward via derivatives
        _, _, acc1, acc2 = dyn.derivatives(0.0, state)
        # Inverse dynamics returns torque needed
        tau1, tau2 = dyn.inverse_dynamics(state, (acc1, acc2))
        # Default forcing is zero -> recovered torques should be ~0
        assert tau1 == pytest.approx(0.0, abs=1e-9)
        assert tau2 == pytest.approx(0.0, abs=1e-9)
        # Re-construct the type to silence unused-import warning
        assert DPD is dyn.__class__

    def test_joint_torque_breakdown_returns_named_components(
        self, dyn: DoublePendulumDynamics, state: DoublePendulumState
    ) -> None:
        breakdown = dyn.joint_torque_breakdown(state, control=(0.7, -0.4))
        assert isinstance(breakdown, JointTorques)
        assert breakdown.applied == (0.7, -0.4)
        assert len(breakdown.gravitational) == 2
        assert len(breakdown.damping) == 2
        assert len(breakdown.coriolis_centripetal) == 2

    def test_applied_torques_with_default_zero_input(
        self, dyn: DoublePendulumDynamics, state: DoublePendulumState
    ) -> None:
        t1, t2 = dyn.applied_torques(0.0, state)
        assert t1 == 0.0
        assert t2 == 0.0

    def test_control_affine_returns_drift_and_input(
        self, dyn: DoublePendulumDynamics, state: DoublePendulumState
    ) -> None:
        f, g = dyn.control_affine(state)
        assert len(f) == 4
        assert f[0] == state.omega1
        assert f[1] == state.omega2
        # Control matrix: top two rows are zeros, bottom two are mass-inverse
        assert g[0] == (0.0, 0.0)
        assert g[1] == (0.0, 0.0)
        assert len(g[2]) == 2
        assert len(g[3]) == 2

    def test_step_preserves_phi(self, dyn: DoublePendulumDynamics) -> None:
        state = DoublePendulumState(
            theta1=0.1, theta2=0.2, omega1=0.0, omega2=0.0, phi=0.5, omega_phi=0.3
        )
        new_state = dyn.step(0.0, state, dt=0.001)
        assert new_state.phi == pytest.approx(0.5)
        assert new_state.omega_phi == pytest.approx(0.3)
