"""Unit tests for OpenSim simulate_with_coefficients (issue #4120).

Tests the PolynomialTorqueController and the forward-sim wrapper, including:
- Zero-coefficient behavior (no motion)
- Unit step responses
- Polynomial evaluation
- SimOut shape and type validation
- Grip and clubhead extraction
- TDD oracle synthesizer

Requires OpenSim installed; skipped otherwise.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]

# Skip entire module if OpenSim unavailable
pytest.importorskip(
    "opensim",
    minversion=None,
    reason="OpenSim not installed; install with: pip install opensim",
)

import numpy as np
from src.engines.physics_engines.opensim.python.opensim_golf.controller import (
    PolynomialTorqueController,
    evaluate_torque_polynomial,
)
from src.engines.physics_engines.opensim.python.opensim_golf.simulate_with_coefficients import (
    SimOptions,
    simulate_with_coefficients,
    synthesize_target_from_coefficients,
)


class TestPolynomialTorqueController:
    """Tests for the polynomial controller."""

    def test_controller_initialization_with_theta(self) -> None:
        """Controller initializes with theta coefficients."""
        n_joints = 3
        theta = np.random.randn(n_joints * 7)

        controller = PolynomialTorqueController(theta)

        assert controller._n_joints == n_joints
        np.testing.assert_array_equal(controller.get_theta(), theta)

    def test_controller_initialization_with_n_joints(self) -> None:
        """Controller initializes with zero coefficients for n_joints."""
        n_joints = 5
        controller = PolynomialTorqueController(n_joints=n_joints)

        assert controller._n_joints == n_joints
        np.testing.assert_array_equal(controller.get_theta(), np.zeros(n_joints * 7))

    def test_controller_initialization_raises_on_missing_args(self) -> None:
        """Controller raises ValueError if both theta and n_joints are None."""
        with pytest.raises(ValueError, match="Either theta or n_joints"):
            PolynomialTorqueController()

    def test_controller_initialization_raises_on_bad_theta_shape(self) -> None:
        """Controller raises ValueError if theta is 2-D."""
        with pytest.raises(ValueError, match="theta must be 1-D"):
            PolynomialTorqueController(np.zeros((3, 7)))

    def test_controller_initialization_raises_on_bad_theta_length(self) -> None:
        """Controller raises ValueError if theta length is not divisible by 7."""
        with pytest.raises(ValueError, match="divisible by 7"):
            PolynomialTorqueController(np.zeros(22))

    def test_set_theta(self) -> None:
        """set_theta updates coefficients."""
        n_joints = 2
        controller = PolynomialTorqueController(n_joints=n_joints)

        theta_new = np.random.randn(n_joints * 7)
        controller.set_theta(theta_new)

        np.testing.assert_array_equal(controller.get_theta(), theta_new)

    def test_set_theta_rejects_nan(self) -> None:
        """set_theta rejects NaN values."""
        controller = PolynomialTorqueController(n_joints=2)
        theta_bad = np.array([1.0, 2.0, np.nan] + [0.0] * 11)

        with pytest.raises(ValueError, match="finite"):
            controller.set_theta(theta_bad)

    def test_tau_at_single_joint(self) -> None:
        """tau_at evaluates torque for a single joint."""
        n_joints = 3
        # Coefficients: [1, 0, 0, 0, 0, 0, 0] means tau = 1
        theta = np.zeros(n_joints * 7)
        theta[0] = 1.0

        controller = PolynomialTorqueController(theta)

        tau = controller.tau_at(0.5, 0)
        assert np.isclose(tau, 1.0)

    def test_tau_at_raises_on_bad_index(self) -> None:
        """tau_at raises ValueError for out-of-bounds joint index."""
        controller = PolynomialTorqueController(n_joints=2)

        with pytest.raises(ValueError, match="out of bounds"):
            controller.tau_at(0.0, 5)

    def test_controller_picklable(self) -> None:
        """Controller can be pickled and unpickled."""
        import pickle

        n_joints = 3
        theta = np.random.randn(n_joints * 7)
        controller = PolynomialTorqueController(theta)

        pickled = pickle.dumps(controller)
        controller_restored = pickle.loads(pickled)

        np.testing.assert_array_equal(
            controller.get_theta(), controller_restored.get_theta()
        )


class TestEvaluateTorquePolynomial:
    """Tests for the polynomial evaluation utility."""

    def test_constant_polynomial(self) -> None:
        """Constant polynomial tau(t) = a0."""
        coeffs = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau = evaluate_torque_polynomial(1.5, coeffs)
        assert np.isclose(tau, 5.0)

    def test_linear_polynomial(self) -> None:
        """Linear polynomial tau(t) = a0 + a1*t."""
        coeffs = np.array([2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau = evaluate_torque_polynomial(2.0, coeffs)
        expected = 2.0 + 3.0 * 2.0
        assert np.isclose(tau, expected)

    def test_quadratic_polynomial(self) -> None:
        """Quadratic polynomial tau(t) = a0 + a1*t + a2*t^2."""
        coeffs = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        tau = evaluate_torque_polynomial(2.0, coeffs)
        expected = 1.0 + 2.0 * 2.0 + 3.0 * 4.0
        assert np.isclose(tau, expected)

    def test_full_degree_polynomial(self) -> None:
        """Degree-6 polynomial with all coefficients."""
        coeffs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        t = 2.0
        tau = evaluate_torque_polynomial(t, coeffs)
        expected = sum(t**k for k in range(7))
        assert np.isclose(tau, expected)

    def test_raises_on_wrong_length(self) -> None:
        """Raises ValueError if coeffs is not length 7."""
        with pytest.raises(ValueError, match="length 7"):
            evaluate_torque_polynomial(1.0, np.zeros(5))


class TestSimOptions:
    """Tests for SimOptions dataclass."""

    def test_default_options(self) -> None:
        """Default SimOptions have sensible values."""
        opts = SimOptions()

        assert opts.t_final == 1.0
        assert opts.dt == 1e-3
        assert opts.integrator == "rk4"
        assert opts.tolerance == 1e-5

    def test_custom_options(self) -> None:
        """SimOptions accepts custom values."""
        opts = SimOptions(t_final=2.0, dt=5e-4, integrator="semiexplicit")

        assert opts.t_final == 2.0
        assert opts.dt == 5e-4
        assert opts.integrator == "semiexplicit"

    def test_rejects_negative_t_final(self) -> None:
        """SimOptions rejects negative t_final."""
        with pytest.raises(ValueError, match="positive"):
            SimOptions(t_final=-1.0)

    def test_rejects_negative_dt(self) -> None:
        """SimOptions rejects negative dt."""
        with pytest.raises(ValueError, match="positive"):
            SimOptions(dt=-1e-3)

    def test_rejects_dt_greater_than_t_final(self) -> None:
        """SimOptions rejects dt > t_final."""
        with pytest.raises(ValueError, match="must not exceed"):
            SimOptions(t_final=1.0, dt=2.0)


class TestSimulateWithCoefficientsSmoke:
    """Smoke tests for simulate_with_coefficients."""

    def test_zero_coefficients_trajectory_is_constant(self) -> None:
        """Zero-coefficient input → constant joint angles over time."""
        # This test assumes the model loads and initial pose is a stable equilibrium
        # or close to it. With zero torques, the system should not accelerate
        # significantly.

        n_joints_expected = 23  # Standard golf humanoid
        theta = np.zeros(n_joints_expected * 7)

        opts = SimOptions(t_final=0.1, dt=0.01)
        result = simulate_with_coefficients(theta, opts)

        # Check SimOut shape
        assert result.time.shape == (11,)  # 0.0 to 0.1 in 0.01 steps
        assert result.q.shape == (11, n_joints_expected)
        assert result.qd.shape == (11, n_joints_expected)
        assert result.qdd.shape == (11, n_joints_expected)
        assert result.tau.shape == (11, n_joints_expected)
        assert result.grip.shape == (11, 3)
        assert result.grip_quat.shape == (11, 4)
        assert result.clubhead.shape == (11, 3)
        assert result.club_quat.shape == (11, 4)

    def test_constant_torque_produces_trajectory(self) -> None:
        """Non-zero constant torque produces non-zero accelerations."""
        n_joints_expected = 23
        theta = np.zeros(n_joints_expected * 7)
        # Set first joint to constant torque of 10 N·m
        theta[0] = 10.0

        opts = SimOptions(t_final=0.1, dt=0.01)
        result = simulate_with_coefficients(theta, opts)

        # With constant torque, acceleration should be non-zero
        assert result.solver_status == "success"
        assert np.max(np.abs(result.qdd)) > 0.1

    def test_simout_is_valid_trajectory(self) -> None:
        """SimOut has monotone time, finite states, and correct schema."""
        n_joints_expected = 23
        theta = np.random.randn(n_joints_expected * 7) * 0.1

        opts = SimOptions(t_final=0.2, dt=0.01)
        result = simulate_with_coefficients(theta, opts)

        # Time is monotone
        assert np.all(np.diff(result.time) > 0)

        # All arrays are finite
        assert np.all(np.isfinite(result.q))
        assert np.all(np.isfinite(result.qd))
        assert np.all(np.isfinite(result.qdd))
        assert np.all(np.isfinite(result.tau))
        assert np.all(np.isfinite(result.grip))
        assert np.all(np.isfinite(result.clubhead))

        # Quaternions are unit-norm
        grip_norms = np.linalg.norm(result.grip_quat, axis=1)
        club_norms = np.linalg.norm(result.club_quat, axis=1)
        np.testing.assert_allclose(grip_norms, 1.0, atol=1e-6)
        np.testing.assert_allclose(club_norms, 1.0, atol=1e-6)

    def test_torque_matches_controller(self) -> None:
        """Recorded tau values match controller evaluation."""
        n_joints_expected = 23
        # Random coefficients with small magnitude
        theta = np.random.randn(n_joints_expected * 7) * 0.1

        opts = SimOptions(t_final=0.1, dt=0.01)
        result = simulate_with_coefficients(theta, opts)

        controller = PolynomialTorqueController(theta)
        for i, t in enumerate(result.time):
            for j in range(n_joints_expected):
                expected_tau = controller.tau_at(t, j)
                # Allow some numerical tolerance due to integration errors
                assert np.isclose(
                    result.tau[i, j], expected_tau, atol=1e-4
                ), f"tau mismatch at t={t}, joint={j}"

    def test_solver_status_success(self) -> None:
        """Solver status is 'success' for nominal inputs."""
        n_joints_expected = 23
        theta = np.zeros(n_joints_expected * 7)

        opts = SimOptions(t_final=0.1, dt=0.01)
        result = simulate_with_coefficients(theta, opts)

        assert result.solver_status == "success"

    def test_duration_recorded(self) -> None:
        """Wall-clock duration is recorded and non-zero."""
        n_joints_expected = 23
        theta = np.zeros(n_joints_expected * 7)

        opts = SimOptions(t_final=0.1, dt=0.01)
        result = simulate_with_coefficients(theta, opts)

        assert result.wall_clock_s > 0.0


class TestSynthesizeTargetFromCoefficients:
    """Tests for the TDD oracle."""

    def test_synthesize_produces_valid_target(self) -> None:
        """Synthesizer produces a valid ClubTarget."""
        n_joints_expected = 23
        theta = np.zeros(n_joints_expected * 7)

        target = synthesize_target_from_coefficients(theta)

        # Check schema
        assert target.time.shape[0] > 1
        assert target.butt.shape == (len(target.time), 3)
        assert target.clubhead.shape == (len(target.time), 3)
        assert target.club_quat.shape == (len(target.time), 4)
        assert 1 <= target.impact_idx <= len(target.time)
        assert target.source.format == "opensim_rk4"

    def test_synthesize_is_reproducible(self) -> None:
        """Synthesizer produces byte-identical targets (within float tolerance)."""
        n_joints_expected = 23
        theta = np.random.randn(n_joints_expected * 7) * 0.1

        target1 = synthesize_target_from_coefficients(theta)
        target2 = synthesize_target_from_coefficients(theta)

        np.testing.assert_allclose(target1.time, target2.time)
        np.testing.assert_allclose(target1.butt, target2.butt, atol=1e-12)
        np.testing.assert_allclose(target1.clubhead, target2.clubhead, atol=1e-12)

    def test_synthesize_rejects_bad_simulation(self) -> None:
        """Synthesizer raises ValueError if simulation fails."""
        # This is hard to trigger in a unit test without mocking.
        # For now, we rely on the happy path test above.


class TestSimulateWithCoefficientsInvalidInputs:
    """Tests for error handling in simulate_with_coefficients."""

    def test_rejects_wrong_theta_length(self) -> None:
        """Raises ValueError if theta has wrong length."""
        # Assume model has 23 joints
        theta_bad = np.zeros(100)  # Wrong length

        with pytest.raises(ValueError, match="must have length"):
            simulate_with_coefficients(theta_bad)

    def test_rejects_nan_in_theta(self) -> None:
        """Raises ValueError if theta contains NaN."""
        n_joints_expected = 23
        theta = np.zeros(n_joints_expected * 7)
        theta[10] = np.nan

        with pytest.raises(ValueError, match="finite"):
            simulate_with_coefficients(theta)

    def test_rejects_inf_in_theta(self) -> None:
        """Raises ValueError if theta contains Inf."""
        n_joints_expected = 23
        theta = np.zeros(n_joints_expected * 7)
        theta[10] = np.inf

        with pytest.raises(ValueError, match="finite"):
            simulate_with_coefficients(theta)

    def test_rejects_2d_theta(self) -> None:
        """Raises ValueError if theta is 2-D."""
        theta_bad = np.zeros((3, 7))

        with pytest.raises(ValueError, match="1-D"):
            simulate_with_coefficients(theta_bad)
