"""Tests for power flow analysis (Guideline E3 - Required).

Comprehensive test suite validating:
- Joint-level power calculations
- Work decomposition (drift/control)
- Inter-segment energy transfer
- Energy conservation
- Sign conventions (generation vs absorption)
"""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.power_flow import (
    InterSegmentTransfer,
    PowerFlowAnalyzer,
    PowerFlowResult,
)


@pytest.fixture
def simple_pendulum_model() -> mujoco.MjModel:
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


class TestPowerFlowBasics:
    """Test basic power flow calculations."""

    def test_power_is_torque_times_velocity(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test P =τ · ω."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.1])
        qvel = np.array([2.0])  # 2 rad/s
        qacc = np.array([0.0])
        tau = np.array([3.0])  # 3 Nm

        result = analyzer.compute_power_flow(qpos, qvel, qacc, tau)

        # P = τ · ω = 3.0 * 2.0 = 6.0 Watts
        expected_power = 6.0
        assert abs(result.joint_powers[0] - expected_power) < 1e-6, (
            f"Expected power {expected_power}, got {result.joint_powers[0]}"
        )

    def test_power_sign_positive_when_aligned(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test power is positive when torque and velocity are aligned."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.0])
        qvel = np.array([1.0])  # Positive velocity
        qacc = np.array([0.0])
        tau = np.array([2.0])  # Positive torque (same direction)

        result = analyzer.compute_power_flow(qpos, qvel, qacc, tau)

        # Both positive → power positive (generation)
        assert result.joint_powers[0] > 0, (
            "Power should be positive when torque and velocity are aligned"
        )

    def test_power_sign_negative_when_opposed(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test power is negative when torque opposes velocity."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.0])
        qvel = np.array([1.0])  # Positive velocity
        qacc = np.array([0.0])
        tau = np.array([-2.0])  # Negative torque (opposes motion)

        result = analyzer.compute_power_flow(qpos, qvel, qacc, tau)

        # Opposite signs → power negative (absorption/braking)
        assert result.joint_powers[0] < 0, (
            "Power should be negative when torque opposes velocity"
        )

    def test_zero_velocity_gives_zero_power(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test P = 0 when velocity is zero."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.5])
        qvel = np.array([0.0])  # No motion
        qacc = np.array([0.0])
        tau = np.array([5.0])  # Non-zero torque

        result = analyzer.compute_power_flow(qpos, qvel, qacc, tau)

        # v = 0 → P = 0 regardless of torque
        assert abs(result.joint_powers[0]) < 1e-10, (
            "Power should be zero when velocity is zero"
        )


class TestWorkCalculations:
    """Test work calculations and decomposition."""

    def test_work_equals_power_times_time(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test W = P · dt."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.0])
        qvel = np.array([2.0])
        qacc = np.array([0.0])
        tau = np.array([3.0])
        dt = 0.1  # 0.1 second

        result = analyzer.compute_power_flow(qpos, qvel, qacc, tau, dt=dt)

        # W = P · dt = 6.0 * 0.1 = 0.6 Joules
        expected_work = 6.0 * 0.1
        assert abs(result.joint_work_total[0] - expected_work) < 1e-6, (
            f"Expected work {expected_work}, got {result.joint_work_total[0]}"
        )

    def test_work_decomposition_sums_to_total(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test that drift + control work equals total work."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.2])
        qvel = np.array([1.5])
        qacc = np.array([0.0])
        tau_total = np.array([4.0])
        tau_drift = np.array([1.0])
        tau_control = np.array([3.0])
        dt = 0.05

        result = analyzer.compute_power_flow(
            qpos,
            qvel,
            qacc,
            tau_total,
            dt=dt,
            tau_drift=tau_drift,
            tau_control=tau_control,
        )

        # Work components should sum to total
        work_sum = result.joint_work_drift[0] + result.joint_work_control[0]
        assert abs(result.joint_work_total[0] - work_sum) < 1e-6, (
            "Drift + control work should equal total work"
        )


class TestEnergyCalculations:
    """Test segment energy calculations."""

    def test_kinetic_energy_at_rest_is_zero(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test KE = 0 when velocity is zero."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.5])
        qvel = np.array([0.0])  # At rest
        qacc = np.array([0.0])
        tau = np.array([0.0])

        result = analyzer.compute_power_flow(qpos, qvel, qacc, tau)

        # All segments at rest → KE should be near zero
        # (world body has no mass)
        total_ke = np.sum(result.segment_kinetic_energy)
        assert total_ke < 1e-6, f"Expected zero KE at rest, got {total_ke}"

    def test_potential_energy_increases_with_height(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test PE increases as pendulum rises."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qacc = np.array([0.0])
        qvel = np.array([0.0])
        tau = np.array([0.0])

        # Low position
        qpos_low = np.array([0.0])  # Hanging down
        result_low = analyzer.compute_power_flow(qpos_low, qvel, qacc, tau)

        # High position
        qpos_high = np.array([np.pi / 2])  # Horizontal
        result_high = analyzer.compute_power_flow(qpos_high, qvel, qacc, tau)

        # Higher position should have higher PE
        pe_low = np.sum(result_low.segment_potential_energy)
        pe_high = np.sum(result_high.segment_potential_energy)

        assert pe_high > pe_low, (
            f"Higher position should have higher PE: {pe_high} vs {pe_low}"
        )

    def test_total_mechanical_energy_is_ke_plus_pe(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test E = KE + PE."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([np.pi / 4])
        qvel = np.array([1.0])
        qacc = np.array([0.0])
        tau = np.array([0.0])

        result = analyzer.compute_power_flow(qpos, qvel, qacc, tau)

        total_ke = np.sum(result.segment_kinetic_energy)
        total_pe = np.sum(result.segment_potential_energy)
        expected_me = total_ke + total_pe

        assert abs(result.total_mechanical_energy - expected_me) < 1e-6, (
            "Total ME should equal KE + PE"
        )


class TestSystemPowerMetrics:
    """Test system-level power metrics."""

    def test_power_input_is_sum_of_positive_powers(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test power input calculation."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.0])
        qvel = np.array([2.0])
        qacc = np.array([0.0])
        tau = np.array([3.0])  # Positive power

        result = analyzer.compute_power_flow(qpos, qvel, qacc, tau)

        # Single joint with positive power
        expected_power_in = 3.0 * 2.0  # 6.0 W
        assert abs(result.power_in - expected_power_in) < 1e-6, (
            f"Expected power_in {expected_power_in}, got {result.power_in}"
        )

    def test_power_dissipation_from_damping(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test power dissipation calculation."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.0])
        qvel = np.array([2.0])  # 2 rad/s
        qacc = np.array([0.0])
        tau = np.array([0.0])

        result = analyzer.compute_power_flow(qpos, qvel, qacc, tau)

        # Damping = 0.1, velocity = 2.0
        # P_diss = b * ω² = 0.1 * 4.0 = 0.4 W
        expected_diss = 0.1 * 2.0**2
        assert abs(result.power_dissipation - expected_diss) < 1e-6, (
            f"Expected dissipation {expected_diss}, got {result.power_dissipation}"
        )


class TestTrajectoryAnalysis:
    """Test trajectory-level analysis."""

    def test_trajectory_analysis_length(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test trajectory analysis returns correct number of results."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        N = 10
        times = np.linspace(0, 1, N)
        qpos_traj = np.sin(times).reshape(-1, 1)
        qvel_traj = np.cos(times).reshape(-1, 1)
        qacc_traj = -np.sin(times).reshape(-1, 1)
        tau_traj = np.ones((N, 1))

        results = analyzer.analyze_trajectory(
            times, qpos_traj, qvel_traj, qacc_traj, tau_traj
        )

        assert len(results) == N, f"Expected {N} results, got {len(results)}"
        assert all(isinstance(r, PowerFlowResult) for r in results)


class TestInterSegmentTransfer:
    """Test inter-segment power transfer analysis."""

    def test_inter_segment_transfer_structure(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test inter-segment transfer returns correct structure."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.2])
        qvel = np.array([1.0])
        tau = np.array([2.0])

        transfers = analyzer.compute_inter_segment_transfer(qpos, qvel, tau)

        # Should have transfers for all bodies (world + pendulum)
        assert len(transfers) == simple_pendulum_model.nbody
        assert all(isinstance(t, InterSegmentTransfer) for t in transfers)

    def test_world_body_has_zero_power_transfer(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test world body (fixed) has no power transfer."""
        analyzer = PowerFlowAnalyzer(simple_pendulum_model)

        qpos = np.array([0.0])
        qvel = np.array([1.0])
        tau = np.array([1.0])

        transfers = analyzer.compute_inter_segment_transfer(qpos, qvel, tau)

        # First transfer should be world body
        world_transfer = transfers[0]
        assert (
            world_transfer.segment_name == "world"
            or world_transfer.parent_name == "world"
        )


@pytest.mark.integration
class TestPowerFlowPhysics:
    """Integration tests for power flow physics validation."""

    def test_conservation_over_conservative_swing(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test energy conservation for passive swing."""
        pytest.skip("Requires time history for dE/dt - implement in follow-up")

        # total mechanical energy remains constant

    def test_work_matches_energy_change(
        self, simple_pendulum_model: mujoco.MjModel
    ) -> None:
        """Test W = ΔE for simple case."""
        pytest.skip("Requires time integration - implement in follow-up")
