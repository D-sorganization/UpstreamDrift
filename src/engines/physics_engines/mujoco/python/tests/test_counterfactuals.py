"""Tests for counterfactual analysis (Guideline G - MANDATORY)."""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.counterfactuals import (
    CounterfactualAnalyzer,
)


@pytest.fixture
def simple_pendulum_model():
    """Create simple pendulum for testing."""
    xml = """
    <mujoco>
        <option gravity="0 0 -9.81" timestep="0.01"/>
        <worldbody>
            <body name="pendulum" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 1 0" damping="0.0"/>
                <geom type="capsule" size="0.01 0.5" mass="1.0"/>
            </body>
        </worldbody>
        <actuator>
            <motor joint="hinge" gear="1.0"/>
        </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


class TestZTCF:
    """Test Zero-Torque Counterfactual (Guideline G1)."""

    def test_ztcf_with_zero_control_gives_zero_delta(self, simple_pendulum_model):
        """Test ZTCF: zero control means observed = counterfactual."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        qpos = np.array([np.pi / 6])
        qvel = np.array([0.5])
        ctrl = np.array([0.0])  # NO CONTROL

        result = analyzer.ztcf(qpos, qvel, ctrl)

        # With zero control, observed should equal counterfactual
        assert np.allclose(
            result.observed_acceleration, result.counterfactual_acceleration, atol=1e-6
        )

        # Delta should be near zero
        assert np.allclose(result.delta_acceleration, 0, atol=1e-6)
        assert result.type == "ztcf"

    def test_ztcf_positive_torque_gives_positive_delta(self, simple_pendulum_model):
        """Test ZTCF: upward torque creates positive acceleration delta."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        # Pendulum hanging down
        qpos = np.array([0.0])
        qvel = np.array([0.0])
        ctrl = np.array([2.0])  # Upward torque

        result = analyzer.ztcf(qpos, qvel, ctrl)

        # Torque should increase acceleration (positive delta)
        assert result.delta_acceleration[0] > 0, (
            "Upward torque should create positive acceleration delta"
        )

        # Observed > counterfactual
        assert result.observed_acceleration[0] > result.counterfactual_acceleration[0]

    def test_ztcf_negative_torque_opposes_gravity(self, simple_pendulum_model):
        """Test ZTCF: downward torque opposes upward swing."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        # Pendulum swinging up
        qpos = np.array([np.pi / 4])
        qvel = np.array([2.0])  # Swinging upward
        ctrl = np.array([-1.0])  # Oppose upward swing

        result = analyzer.ztcf(qpos, qvel, ctrl)

        # Torque opposes motion (negative delta)
        assert result.delta_acceleration[0] < 0, (
            "Opposing torque should create negative acceleration delta"
        )

    def test_ztcf_delta_scales_with_torque(self, simple_pendulum_model):
        """Test ZTCF: delta scales linearly with applied torque."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        qpos = np.array([0.1])
        qvel = np.array([0.0])

        # Small torque
        ctrl_small = np.array([0.5])
        result_small = analyzer.ztcf(qpos, qvel, ctrl_small)

        # Double torque
        ctrl_large = np.array([1.0])
        result_large = analyzer.ztcf(qpos, qvel, ctrl_large)

        # Delta should scale approximately linearly
        ratio = result_large.delta_acceleration[0] / result_small.delta_acceleration[0]
        assert 1.8 < ratio < 2.2, f"Expected ~2x scaling, got {ratio:.2f}x"

    def test_ztcf_trajectory_integration(self, simple_pendulum_model):
        """Test ZTCF with trajectory prediction."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        qpos = np.array([0.0])
        qvel = np.array([0.0])
        ctrl = np.array([5.0])  # Strong torque

        result = analyzer.ztcf(qpos, qvel, ctrl, dt=0.01, compute_trajectories=True)

        # Should have position predictions
        assert result.observed_position is not None
        assert result.counterfactual_position is not None
        assert result.delta_position is not None

        # Observed position should differ from counterfactual
        assert abs(result.delta_position[0]) > 1e-6, (
            "Position delta should be non-zero with strong torque"
        )


class TestZVCF:
    """Test Zero-Velocity Counterfactual (Guideline G2)."""

    def test_zvcf_with_zero_velocity_gives_zero_delta(self, simple_pendulum_model):
        """Test ZVCF: zero velocity means observed = counterfactual."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        qpos = np.array([np.pi / 6])
        qvel = np.array([0.0])  # NO VELOCITY

        result = analyzer.zvcf(qpos, qvel)

        # With zero velocity, observed should equal counterfactual
        assert np.allclose(
            result.observed_acceleration, result.counterfactual_acceleration, atol=1e-6
        )

        # Delta should be near zero
        assert np.allclose(result.delta_acceleration, 0, atol=1e-6)
        assert result.type == "zvcf"

    def test_zvcf_nonzero_velocity_creates_delta(self, simple_pendulum_model):
        """Test ZVCF: non-zero velocity creates acceleration delta."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        # Pendulum at angle with velocity
        qpos = np.array([np.pi / 4])
        qvel = np.array([2.0])  # Significant velocity

        result = analyzer.zvcf(qpos, qvel)

        # Velocity creates Coriolis/centrifugal effects
        # Delta should be non-zero
        assert abs(result.delta_acceleration[0]) > 1e-3, (
            "Non-zero velocity should create acceleration delta"
        )

    def test_zvcf_delta_scales_with_velocity(self, simple_pendulum_model):
        """Test ZVCF: delta scales with velocity."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        qpos = np.array([np.pi / 6])

        # Low velocity
        qvel_low = np.array([0.5])
        result_low = analyzer.zvcf(qpos, qvel_low)

        # High velocity
        qvel_high = np.array([2.0])
        result_high = analyzer.zvcf(qpos, qvel_high)

        # Higher velocity should create larger delta
        # (Coriolis scales with velocity)
        assert abs(result_high.delta_acceleration[0]) > abs(
            result_low.delta_acceleration[0]
        ), "Higher velocity should create larger Coriolis effect"

    def test_zvcf_isolates_coriolis_from_gravity(self, simple_pendulum_model):
        """Test ZVCF: counterfactual has only gravity, observed has
        gravity + Coriolis.
        """
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        qpos = np.array([np.pi / 4])
        qvel = np.array([1.0])

        result = analyzer.zvcf(qpos, qvel)

        # Counterfactual should have only gravity acceleration
        # (no velocity terms)
        # For pendulum at 45°, gravity creates downward (negative) acceleration
        assert result.counterfactual_acceleration[0] < 0, (
            "Counterfactual should have gravity-driven negative acceleration"
        )

        # Delta is the Coriolis contribution
        # (can be positive or negative depending on direction)
        assert result.delta_acceleration is not None


class TestCounterfactualTrajectoryAnalysis:
    """Test trajectory-level counterfactual analysis."""

    def test_ztcf_trajectory_analysis(self, simple_pendulum_model):
        """Test ZTCF analysis on full trajectory."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        # Create simple trajectory
        N = 10
        qpos_traj = np.linspace(0, np.pi / 2, N).reshape(-1, 1)
        qvel_traj = np.ones((N, 1)) * 0.5
        ctrl_traj = np.ones((N, 1)) * 0.2

        results = analyzer.analyze_trajectory_ztcf(qpos_traj, qvel_traj, ctrl_traj)

        assert len(results) == N
        assert all(r.type == "ztcf" for r in results)

        # All should have non-zero delta (control is active)
        for r in results:
            assert abs(r.delta_acceleration[0]) > 1e-6

    def test_zvcf_trajectory_analysis(self, simple_pendulum_model):
        """Test ZVCF analysis on full trajectory."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        N = 10
        qpos_traj = np.linspace(0, np.pi / 3, N).reshape(-1, 1)
        qvel_traj = np.linspace(0, 1.5, N).reshape(-1, 1)  # Increasing velocity

        results = analyzer.analyze_trajectory_zvcf(qpos_traj, qvel_traj)

        assert len(results) == N
        assert all(r.type == "zvcf" for r in results)

        # Delta should increase with velocity
        deltas = [abs(r.delta_acceleration[0]) for r in results]
        # Later deltas should generally be larger (higher velocity)
        assert deltas[-1] > deltas[0], (
            "Delta should increase with velocity in trajectory"
        )


@pytest.mark.integration
class TestCounterfactualPhysics:
    """Integration tests for counterfactual physics validation."""

    def test_ztcf_plus_counterfactual_equals_observed(self, simple_pendulum_model):
        """Test physics: counterfactual + delta = observed."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        qpos = np.array([0.2])
        qvel = np.array([0.3])
        ctrl = np.array([0.5])

        result = analyzer.ztcf(qpos, qvel, ctrl)

        # Reconstruction test
        reconstructed = result.counterfactual_acceleration + result.delta_acceleration

        assert np.allclose(result.observed_acceleration, reconstructed, atol=1e-6), (
            "Physics violation: observed != counterfactual + delta"
        )

    def test_zvcf_plus_counterfactual_equals_observed(self, simple_pendulum_model):
        """Test physics: counterfactual + delta = observed."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        qpos = np.array([0.3])
        qvel = np.array([1.0])

        result = analyzer.zvcf(qpos, qvel)

        # Reconstruction test
        reconstructed = result.counterfactual_acceleration + result.delta_acceleration

        assert np.allclose(result.observed_acceleration, reconstructed, atol=1e-6), (
            "Physics violation: observed != counterfactual + delta"
        )

    def test_ztcf_reveals_control_authority(self, simple_pendulum_model):
        """Test that ZTCF correctly identifies control authority."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        # Scenario: pendulum falling vs being driven upward
        qpos = np.array([0.0])
        qvel = np.array([0.0])

        # Strong upward torque
        ctrl = np.array([10.0])

        result = analyzer.ztcf(qpos, qvel, ctrl)

        # Counterfactual: pendulum stays stationary (at bottom, no velocity)
        # Observed: strong upward acceleration from torque
        # Delta should be large and positive
        assert result.delta_acceleration[0] > 5.0, (
            "Strong torque should create large positive delta"
        )

        # This delta represents the control authority
        assert result.torque_attributed_effect is not None
        assert result.torque_attributed_effect[0] > 0
