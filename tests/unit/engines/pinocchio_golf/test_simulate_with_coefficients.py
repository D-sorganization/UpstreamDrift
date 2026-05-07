"""TDD tests for pinocchio_golf.simulate_with_coefficients (issue #4118).

Tests the forward-simulation wrapper with polynomial torque inputs using RK4 +
Articulated Body Algorithm (ABA).
"""

import unittest
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.pinocchio_golf.simulate_with_coefficients import (
    SimOptions,
    simulate_with_coefficients,
    synthesize_target_from_coefficients,
)

# Import ClubTarget directly to avoid loading c3d dependencies
try:
    from src.shared.python.motion_matching.club_target import ClubTarget
except ImportError:
    ClubTarget = None  # Will be skipped if pandas not available


class TestSimulateWithCoefficientsBasics(unittest.TestCase):
    """Test basic functionality of simulate_with_coefficients."""

    def setUp(self) -> None:
        """Load the Pinocchio golfer model once for the test suite."""
        # Model is loaded and cached at module import time
        pass

    def test_simulate_returns_complete_simout(self) -> None:
        """Verify SimOut has all documented fields."""
        # Create zero coefficients -> zero torque, gravity only
        n_joints = 43  # All revolute joints in the golfer.urdf model
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        result = simulate_with_coefficients(theta, opts)

        # Check all required fields exist
        self.assertTrue(hasattr(result, "time"))
        self.assertTrue(hasattr(result, "q"))
        self.assertTrue(hasattr(result, "qd"))
        self.assertTrue(hasattr(result, "qdd"))
        self.assertTrue(hasattr(result, "tau"))
        self.assertTrue(hasattr(result, "grip"))
        self.assertTrue(hasattr(result, "grip_quat"))
        self.assertTrue(hasattr(result, "clubhead"))
        self.assertTrue(hasattr(result, "club_quat"))
        self.assertTrue(hasattr(result, "solver_status"))

    def test_simulate_time_starts_at_zero_and_is_monotonic(self) -> None:
        """Verify time vector starts at 0 and is strictly increasing."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        result = simulate_with_coefficients(theta, opts)

        # Check time[0] == 0
        self.assertAlmostEqual(float(result.time[0]), 0.0, places=6)

        # Check strictly increasing
        diffs = np.diff(result.time)
        self.assertTrue(np.all(diffs > 0), "time must be strictly increasing")

    def test_simulate_field_lengths_consistent(self) -> None:
        """Verify all trajectory arrays have the same row count."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        result = simulate_with_coefficients(theta, opts)

        n = len(result.time)
        self.assertEqual(result.q.shape[0], n, "q rows must match time length")
        self.assertEqual(result.qd.shape[0], n, "qd rows must match time length")
        self.assertEqual(result.qdd.shape[0], n, "qdd rows must match time length")
        self.assertEqual(result.tau.shape[0], n, "tau rows must match time length")
        self.assertEqual(result.grip.shape[0], n, "grip rows must match time length")
        self.assertEqual(
            result.grip_quat.shape[0], n, "grip_quat rows must match time length"
        )
        self.assertEqual(
            result.clubhead.shape[0], n, "clubhead rows must match time length"
        )
        self.assertEqual(
            result.club_quat.shape[0], n, "club_quat rows must match time length"
        )

    def test_simulate_grip_quaternions_are_unit_norm(self) -> None:
        """Verify grip_quat rows are unit-norm quaternions."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        result = simulate_with_coefficients(theta, opts)

        qnorms = np.linalg.norm(result.grip_quat, axis=1)
        norms_close_to_one = np.allclose(qnorms, 1.0, atol=1e-6)
        self.assertTrue(norms_close_to_one, "grip_quat rows must be unit norm")

    def test_simulate_club_quaternions_are_unit_norm(self) -> None:
        """Verify club_quat rows are unit-norm quaternions."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        result = simulate_with_coefficients(theta, opts)

        qnorms = np.linalg.norm(result.club_quat, axis=1)
        norms_close_to_one = np.allclose(qnorms, 1.0, atol=1e-6)
        self.assertTrue(norms_close_to_one, "club_quat rows must be unit norm")

    def test_simulate_zero_torque_returns_success(self) -> None:
        """Verify zero-torque simulation returns 'success' status."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        result = simulate_with_coefficients(theta, opts)

        self.assertEqual(result.solver_status, "success")

    def test_simulate_position_outputs_are_finite(self) -> None:
        """Verify grip and clubhead positions are finite."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        result = simulate_with_coefficients(theta, opts)

        self.assertTrue(
            np.all(np.isfinite(result.grip)),
            "grip must contain only finite values",
        )
        self.assertTrue(
            np.all(np.isfinite(result.clubhead)),
            "clubhead must contain only finite values",
        )

    def test_simulate_rejects_non_finite_theta(self) -> None:
        """Verify non-finite theta is rejected with ValueError."""
        n_joints = 43
        theta_with_nan = np.zeros(n_joints * 7)
        theta_with_nan[0] = np.nan

        opts = SimOptions(t_final=0.1, dt=0.01)

        with self.assertRaises(ValueError) as cm:
            simulate_with_coefficients(theta_with_nan, opts)
        self.assertIn("finite", str(cm.exception).lower())

    def test_simulate_rejects_wrong_theta_length(self) -> None:
        """Verify theta length must be n_joints * 7."""
        theta_wrong_length = np.zeros(100)  # Wrong size
        opts = SimOptions(t_final=0.1, dt=0.01)

        with self.assertRaises(ValueError) as cm:
            simulate_with_coefficients(theta_wrong_length, opts)
        self.assertIn("theta", str(cm.exception).lower())

    def test_simulate_respects_dt_option(self) -> None:
        """Verify dt option controls integration step size."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)

        # Small dt -> more samples
        opts_fine = SimOptions(t_final=0.1, dt=0.001)
        result_fine = simulate_with_coefficients(theta, opts_fine)

        # Large dt -> fewer samples
        opts_coarse = SimOptions(t_final=0.1, dt=0.01)
        result_coarse = simulate_with_coefficients(theta, opts_coarse)

        n_fine = len(result_fine.time)
        n_coarse = len(result_coarse.time)

        self.assertGreater(
            n_fine, n_coarse, "finer dt should produce more time samples"
        )

    def test_simulate_respects_t_final_option(self) -> None:
        """Verify t_final option controls simulation end time."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        dt = 0.01

        opts_short = SimOptions(t_final=0.05, dt=dt)
        result_short = simulate_with_coefficients(theta, opts_short)

        opts_long = SimOptions(t_final=0.1, dt=dt)
        result_long = simulate_with_coefficients(theta, opts_long)

        self.assertLess(
            float(result_short.time[-1]),
            float(result_long.time[-1]),
            "shorter t_final should end earlier",
        )


class TestSynthesizeTargetFromCoefficients(unittest.TestCase):
    """Test the TDD oracle: synthesize_target_from_coefficients."""

    def test_synthesize_returns_club_target(self) -> None:
        """Verify output is a valid ClubTarget."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        target = synthesize_target_from_coefficients(theta, opts)

        self.assertIsInstance(target, ClubTarget)

    def test_synthesize_target_has_all_fields(self) -> None:
        """Verify synthesized ClubTarget has all required fields."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        target = synthesize_target_from_coefficients(theta, opts)

        self.assertTrue(hasattr(target, "time"))
        self.assertTrue(hasattr(target, "butt"))
        self.assertTrue(hasattr(target, "clubhead"))
        self.assertTrue(hasattr(target, "club_quat"))
        self.assertTrue(hasattr(target, "impact_idx"))
        self.assertTrue(hasattr(target, "source"))

    def test_synthesize_time_matches_sim_time(self) -> None:
        """Verify synthesized target time matches simulation time."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        target = synthesize_target_from_coefficients(theta, opts)

        # Time should match the simulation grid
        self.assertAlmostEqual(float(target.time[0]), 0.0, places=6)
        self.assertAlmostEqual(
            float(target.time[-1]), 0.1, places=2, msg="final time should ≈ t_final"
        )

    def test_synthesize_round_trip_time_consistency(self) -> None:
        """Verify theta -> target -> new_sim has consistent timegrid."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        # First round trip
        target1 = synthesize_target_from_coefficients(theta, opts)
        result1 = simulate_with_coefficients(theta, opts)

        # Verify time arrays match
        np.testing.assert_allclose(target1.time, result1.time, rtol=1e-10)


class TestSimulateWithCoefficientsOptions(unittest.TestCase):
    """Test SimOptions dataclass validation."""

    def test_simoptions_defaults(self) -> None:
        """Verify SimOptions has sensible defaults."""
        opts = SimOptions()
        self.assertEqual(opts.t_final, 1.0)
        self.assertEqual(opts.dt, 0.001)
        self.assertEqual(opts.integrator, "rk4")

    def test_simoptions_rejects_negative_t_final(self) -> None:
        """Verify t_final must be positive."""
        with self.assertRaises(ValueError):
            SimOptions(t_final=-0.1, dt=0.01)

    def test_simoptions_rejects_zero_t_final(self) -> None:
        """Verify t_final must be positive."""
        with self.assertRaises(ValueError):
            SimOptions(t_final=0.0, dt=0.01)

    def test_simoptions_rejects_negative_dt(self) -> None:
        """Verify dt must be positive."""
        with self.assertRaises(ValueError):
            SimOptions(t_final=0.1, dt=-0.01)

    def test_simoptions_rejects_zero_dt(self) -> None:
        """Verify dt must be positive."""
        with self.assertRaises(ValueError):
            SimOptions(t_final=0.1, dt=0.0)

    def test_simoptions_rejects_dt_greater_than_t_final(self) -> None:
        """Verify dt must not exceed t_final."""
        with self.assertRaises(ValueError):
            SimOptions(t_final=0.01, dt=0.1)


class TestPolynomialTorqueEvaluation(unittest.TestCase):
    """Test polynomial torque evaluation."""

    def test_simulate_zero_coefficients_produces_gravity_torques(self) -> None:
        """With zero torque coeffs, tau should reflect gravity/dynamics only."""
        n_joints = 43
        theta = np.zeros(n_joints * 7)
        opts = SimOptions(t_final=0.1, dt=0.01)

        result = simulate_with_coefficients(theta, opts)

        # With zero input, tau is determined by gravity + internal forces
        # We don't assert specific values, just that they're computed
        self.assertEqual(result.tau.shape, (len(result.time), n_joints))
        self.assertTrue(np.all(np.isfinite(result.tau)))

    def test_simulate_nonzero_coefficients_runs(self) -> None:
        """Verify simulation runs with non-zero polynomial coefficients."""
        n_joints = 43
        # Simple constant torque: theta = [1, 0, 0, 0, 0, 0, 0, ...] per joint
        theta = np.zeros(n_joints * 7)
        # Set constant term (a0) for a few joints to 0.1 N·m
        for j in range(3):
            theta[j * 7] = 0.1  # Constant term for joint j

        opts = SimOptions(t_final=0.1, dt=0.01)

        # Should complete without error
        result = simulate_with_coefficients(theta, opts)

        self.assertEqual(result.solver_status, "success")
        self.assertTrue(np.all(np.isfinite(result.q)))


if __name__ == "__main__":
    unittest.main()
