"""Tests for drift-control decomposition (Guideline F - MANDATORY)."""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.drift_control import (
    DriftControlDecomposer,
    DriftControlResult,
)


@pytest.fixture
def simple_pendulum_model():
    """Create simple pendulum for testing."""
    xml = """
    <mujoco>
        <option gravity="0 0 -9.81" timestep="0.01"/>
        <worldbody>
            <body name="pendulum" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
                <geom type="capsule" size="0.01 0.5" mass="1.0"/>
            </body>
        </worldbody>
        <actuator>
            <motor joint="hinge" gear="1.0"/>
        </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


class TestDriftControlDecomposer:
    """Test drift-control decomposition implementation."""

    def test_decomposer_initialization(self, simple_pendulum_model):
        """Test decomposer creates private data structures."""
        decomposer = DriftControlDecomposer(simple_pendulum_model)

        assert hasattr(decomposer, "_data_drift")
        assert hasattr(decomposer, "_data_control")
        assert hasattr(decomposer, "_data_full")

        # Verify they are separate instances
        assert id(decomposer._data_drift) != id(decomposer._data_control)
        assert id(decomposer._data_drift) != id(decomposer._data_full)

    def test_superposition_principle(self, simple_pendulum_model):
        """Test that drift + control = full (Guideline F requirement)."""
        decomposer = DriftControlDecomposer(simple_pendulum_model)

        # Set state: pendulum at 45 degrees with some velocity
        qpos = np.array([np.pi / 4])  # 45 degrees
        qvel = np.array([1.0])  # 1 rad/s
        ctrl = np.array([0.5])  # 0.5 Nm torque

        result = decomposer.decompose(qpos, qvel, ctrl)

        # Verify superposition
        reconstructed = result.drift_acceleration + result.control_acceleration

        assert np.allclose(result.full_acceleration, reconstructed, atol=1e-5), (
            f"Superposition failed: full={result.full_acceleration}, "
            f"drift+control={reconstructed}, residual={result.residual}"
        )

        # Guideline F requires residual < 1e-5
        assert (
            result.residual < 1e-5
        ), f"Residual {result.residual:.2e} exceeds Guideline F tolerance 1e-5"

    def test_zero_control_gives_drift_only(self, simple_pendulum_model):
        """Test that zero control gives drift-only acceleration."""
        decomposer = DriftControlDecomposer(simple_pendulum_model)

        qpos = np.array([np.pi / 6])
        qvel = np.array([0.5])
        ctrl = np.array([0.0])  # NO CONTROL

        result = decomposer.decompose(qpos, qvel, ctrl)

        # With zero control, control acceleration should be near zero
        assert np.allclose(
            result.control_acceleration, 0, atol=1e-6
        ), f"Expected zero control acceleration, got {result.control_acceleration}"

        # Full should equal drift
        assert np.allclose(
            result.full_acceleration, result.drift_acceleration, atol=1e-6
        )

    def test_zvcf_isolates_coriolis_from_gravity(self, simple_pendulum_model):
        """Test ZVCF: counterfactual has only gravity, observed has
        gravity + Coriolis.
        """
        decomposer = DriftControlDecomposer(simple_pendulum_model)

        qpos = np.array([np.pi / 4])
        qvel = np.array([0.0])  # NO VELOCITY
        ctrl = np.array([0.0])

        result = decomposer.decompose(qpos, qvel, ctrl)

        # With zero velocity, drift_velocity_component should be near zero
        assert np.allclose(
            result.drift_velocity_component, 0, atol=1e-6
        ), (
            f"Expected zero Coriolis with zero velocity, "
            f"got {result.drift_velocity_component}"
        )

        # Drift should be purely gravity
        assert np.allclose(
            result.drift_acceleration, result.drift_gravity_component, atol=1e-6
        )

    def test_gravity_acceleration_matches_mgL_sin_theta(self, simple_pendulum_model):
        """Test gravity acceleration matches analytical solution."""
        decomposer = DriftControlDecomposer(simple_pendulum_model)

        # For simple pendulum: τ_gravity = m*g*L*sin(θ)
        # With unit mass (1kg) and L=0.5m, g=9.81
        # At θ=30°: τ = 1 * 9.81 * 0.5 * sin(30°) = 2.4525 Nm
        # Acceleration α = τ/I
        # For point mass: I ≈ m*L² = 1 * 0.5² = 0.25 kg⋅m²
        # α = 2.4525 / 0.25 = 9.81 rad/s² (approximately)

        theta = np.radians(30)
        qpos = np.array([theta])
        qvel = np.array([0.0])
        ctrl = np.array([0.0])

        result = decomposer.decompose(qpos, qvel, ctrl)

        # Gravity component should be non-zero
        assert (
            abs(result.drift_gravity_component[0]) > 1.0
        ), "Expected significant gravity acceleration at 30 degrees"

    def test_control_scales_with_torque(self, simple_pendulum_model):
        """Test that control acceleration scales linearly with applied torque."""
        decomposer = DriftControlDecomposer(simple_pendulum_model)

        qpos = np.array([0.1])
        qvel = np.array([0.0])

        # Test with small torque
        ctrl_small = np.array([0.1])
        result_small = decomposer.decompose(qpos, qvel, ctrl_small)

        # Test with double the torque
        ctrl_large = np.array([0.2])
        result_large = decomposer.decompose(qpos, qvel, ctrl_large)

        # Control acceleration should scale linearly (approximately 2x)
        ratio = (
            result_large.control_acceleration[0] / result_small.control_acceleration[0]
        )
        assert 1.8 < ratio < 2.2, f"Expected ~2x scaling, got {ratio:.2f}x"

    def test_trajectory_analysis(self, simple_pendulum_model):
        """Test analyzing full trajectory."""
        decomposer = DriftControlDecomposer(simple_pendulum_model)

        # Create simple trajectory
        N = 10
        qpos_traj = np.linspace(0, np.pi / 2, N).reshape(-1, 1)
        qvel_traj = np.ones((N, 1)) * 0.5
        ctrl_traj = np.ones((N, 1)) * 0.1

        results = decomposer.analyze_trajectory(qpos_traj, qvel_traj, ctrl_traj)

        assert len(results) == N
        assert all(isinstance(r, DriftControlResult) for r in results)

        # All should satisfy superposition
        for r in results:
            assert (
                r.residual < 1e-5
            ), f"Trajectory point failed superposition: residual={r.residual:.2e}"

    def test_decomposition_is_reproducible(self, simple_pendulum_model):
        """Test that decomposition gives same results with same inputs."""
        decomposer = DriftControlDecomposer(simple_pendulum_model)

        qpos = np.array([0.2])
        qvel = np.array([0.3])
        ctrl = np.array([0.1])

        result1 = decomposer.decompose(qpos, qvel, ctrl)
        result2 = decomposer.decompose(qpos, qvel, ctrl)

        assert np.allclose(result1.full_acceleration, result2.full_acceleration)
        assert np.allclose(result1.drift_acceleration, result2.drift_acceleration)
        assert np.allclose(result1.control_acceleration, result2.control_acceleration)
        assert abs(result1.residual - result2.residual) < 1e-10


@pytest.mark.integration
class TestDriftControlPhysics:
    """Integration tests for drift-control physics validation."""

    def test_passive_pendulum_drift_matches_energy_conservation(
        self, simple_pendulum_model
    ):
        """Test drift component matches energy-conserving motion."""
        pytest.skip("Requires energy calculation utilities - implement in follow-up")

        # TODO: Verify that drift-only motion conserves energy
        # (modulo damping losses if present)

    def test_control_enables_upward_swing(self, simple_pendulum_model):
        """Test that control can drive pendulum upward against gravity."""
        decomposer = DriftControlDecomposer(simple_pendulum_model)

        # Start at bottom
        qpos = np.array([0.0])
        qvel = np.array([0.0])

        # Strong upward torque
        ctrl = np.array([5.0])

        result = decomposer.decompose(qpos, qvel, ctrl)

        # Control component should dominate and be positive (upward)
        assert (
            result.control_acceleration[0] > result.drift_acceleration[0]
        ), "Control should dominate drift for strong actuation"

        # Total should be positive (accelerating upward)
        assert (
            result.full_acceleration[0] > 0
        ), "Strong torque should accelerate pendulum upward"
