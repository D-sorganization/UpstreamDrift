"""Unit tests for the Double Pendulum dynamics module."""

import numpy as np
import pytest

from engines.pendulum_models.python.double_pendulum import (
    GRAVITY,
    I1,
    I2,
    C_times_qdot,
    M_matrix,
    c1,
    c2,
    double_pendulum_dynamics,
    g_vector,
    l1,
    m1,
    m2,
    tau_natural,
    u_pd,
)

# Constants for tolerance
RTOL = 1e-10
ATOL = 1e-10


class TestDoublePendulumDynamics:
    """Test suite for double pendulum dynamics."""

    def test_mass_matrix_symmetry(self) -> None:
        """Test that the mass matrix is symmetric and positive definite."""
        # Test at various configurations
        configs = [
            np.array([0.0, 0.0]),
            np.array([np.pi / 2, 0.0]),
            np.array([0.0, np.pi / 2]),
            np.array([np.pi, np.pi]),
            np.random.uniform(-np.pi, np.pi, 2),
        ]

        for q in configs:
            M = M_matrix(q)

            # Symmetry check
            assert np.allclose(
                M, M.T, rtol=RTOL, atol=ATOL
            ), f"Mass matrix not symmetric at q={q}"

            # Positive definite check (eigenvalues > 0)
            eigvals = np.linalg.eigvals(M)
            assert np.all(eigvals > 0), f"Mass matrix not positive definite at q={q}"

    def test_mass_matrix_values(self) -> None:
        """Test mass matrix values against analytical expectations for simple cases."""
        # Case 1: q2 = 0 (extended arm)
        # M11 term includes contributions from both links about joint 1.
        # Link 1 contribution: I1 + m1 * c1^2
        # Link 2 contribution: I2 + m2 * (l1^2 + c2^2 + 2*l1*c2*cos(q2))
        q = np.array([0.0, 0.0])
        M = M_matrix(q)
        # I1 + I2 + m1*c1^2 + m2*(l1^2 + c2^2 + 2*l1*c2)
        expected_M11 = I1 + I2 + m1 * c1**2 + m2 * (l1**2 + c2**2 + 2 * l1 * c2)
        assert M[0, 0] == pytest.approx(expected_M11, rel=RTOL)

        # Case 2: q2 = pi/2 (right angle)
        # cos(pi/2) = 0
        q = np.array([0.0, np.pi / 2])
        M = M_matrix(q)
        # I1 + I2 + m1*c1^2 + m2*(l1^2 + c2^2)
        expected_M11 = I1 + I2 + m1 * c1**2 + m2 * (l1**2 + c2**2)
        assert M[0, 0] == pytest.approx(expected_M11, rel=RTOL)

    def test_gravity_vector_static(self) -> None:
        """Test gravity vector at static equilibrium points."""
        # Downward position (0, 0) - Stable equilibrium if defined as 0
        # The code uses sin(q), so q=0 corresponds to gravity acting "down"
        # relative to the angle?
        # Let's check the implementation of g_vector:
        # g1 = (m1 * c1 + m2 * l1) * g * sin(q1) + ...
        # If q=[0,0], sin(0)=0 -> g=[0,0]. This implies 0 is a
        # stable/unstable equilibrium.

        q = np.array([0.0, 0.0])
        g_vec = g_vector(q)
        assert np.allclose(g_vec, np.zeros(2), rtol=RTOL, atol=ATOL)

        # Horizontal position (pi/2, 0)
        q = np.array([np.pi / 2, 0.0])
        g_vec = g_vector(q)
        # Expected torque is moment arm * weight
        # Link 1 CM dist: c1. Total Mass: m1. Torque: m1*g*c1
        # Link 2 CM dist from joint 1: l1 + c2. Mass: m2. Torque: m2*g*(l1+c2)
        expected_g1 = m1 * GRAVITY * c1 + m2 * GRAVITY * (l1 + c2)
        # Torque on joint 2 is just m2*g*c2
        expected_g2 = m2 * GRAVITY * c2

        assert g_vec[0] == pytest.approx(expected_g1, rel=RTOL)
        assert g_vec[1] == pytest.approx(expected_g2, rel=RTOL)

    def test_coriolis_term_zero_velocity(self) -> None:
        """Test that Coriolis/Centrifugal terms are zero when velocity is zero."""
        q = np.random.uniform(-np.pi, np.pi, 2)
        qdot = np.zeros(2)
        C = C_times_qdot(q, qdot)
        assert np.allclose(C, np.zeros(2), rtol=RTOL, atol=ATOL)

    def test_dynamics_consistency(self) -> None:
        """Test that forward dynamics is consistent with inverse dynamics
        (tau_natural)."""
        # Generate random state
        q = np.random.uniform(-np.pi, np.pi, 2)
        qdot = np.random.uniform(-10, 10, 2)
        # Random input torque
        u = np.random.uniform(-5, 5, 2)

        # Helper to provide u to dynamics
        def u_func(t, x):
            return u

        # Compute acceleration via forward dynamics
        x = np.concatenate([q, qdot])
        xdot = double_pendulum_dynamics(0.0, x, u_func)
        qddot_computed = xdot[2:4]

        # Now use inverse dynamics (tau_natural) to see if we get back the
        # applied torque
        # tau_nat = M*qddot + C*qdot + g
        # Our dynamics: qddot = M^-1(u - C*qdot - g) => M*qddot = u - C*qdot - g
        # => M*qddot + C*qdot + g = u
        # So tau_natural should equal u
        tau_computed = tau_natural(q, qdot, qddot_computed)

        assert np.allclose(tau_computed, u, rtol=RTOL, atol=ATOL)

    def test_pd_controller_output(self) -> None:
        """Test the PD controller function."""
        x = np.array([0.1, 0.2, 0.0, 0.0])  # Small displacement, zero velocity
        kp, kd = 100.0, 10.0

        # u = -kp*(q - q_des) - kd*(qdot - qdot_des)
        # q_des = 0, qdot_des = 0
        expected_u = -kp * x[0:2] - kd * x[2:4]

        u = u_pd(0.0, x, kp=kp, kd=kd)
        assert np.allclose(u, expected_u, rtol=RTOL, atol=ATOL)
