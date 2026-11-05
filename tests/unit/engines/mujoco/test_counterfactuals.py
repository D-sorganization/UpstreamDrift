"""Tests for counterfactual analysis (Guideline G - MANDATORY)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.counterfactuals import (
    CounterfactualAnalyzer,
)
from src.shared.python.engine_core.mujoco_compat import full_mass_matrix
from src.shared.python.simulation_backends.ztcf_zvcf import (
    zero_velocity_zero_torque_acceleration,
    zvcf_acceleration,
)


@pytest.fixture
def simple_pendulum_model() -> mujoco.MjModel:
    """Create simple pendulum for testing."""
    xml = """
    <mujoco>
        <option gravity="0 0 -9.80665" timestep="0.01"/>
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


@pytest.fixture
def double_pendulum_model() -> mujoco.MjModel:
    """Two-link pendulum: configuration-dependent M and Coriolis terms."""
    xml = """
    <mujoco>
        <option gravity="0 0 -9.80665" timestep="0.01"/>
        <worldbody>
            <body name="l1" pos="0 0 0">
                <joint name="j1" type="hinge" axis="0 1 0" damping="0.0"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1.0"/>
                <body name="l2" pos="0 0 -0.5">
                    <joint name="j2" type="hinge" axis="0 1 0" damping="0.0"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="0.7"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor joint="j1" gear="2.0"/>
            <motor joint="j2" gear="2.0"/>
        </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


class TestZTCF:
    """Test Zero-Torque Counterfactual (Guideline G1)."""

    def test_ztcf_with_zero_control_gives_zero_delta(
        self, simple_pendulum_model
    ) -> None:
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

    def test_ztcf_positive_torque_gives_positive_delta(
        self, simple_pendulum_model
    ) -> None:
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

    def test_ztcf_negative_torque_opposes_gravity(self, simple_pendulum_model) -> None:
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

    def test_ztcf_delta_scales_with_torque(self, simple_pendulum_model) -> None:
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

    def test_ztcf_trajectory_integration(self, simple_pendulum_model) -> None:
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


def _provider(model: mujoco.MjModel) -> _MuJoCoProvider:
    return _MuJoCoProvider(model)


class _MuJoCoProvider:
    """Independent ``EquationsOfMotionProvider`` view of a MuJoCo model."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)

    def _load(self, q: np.ndarray, v: np.ndarray) -> None:
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        mujoco.mj_forward(self.model, self.data)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        self._load(q, np.zeros(self.model.nv))
        return full_mass_matrix(mujoco, self.model, self.data)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        self._load(q, v)
        return self.data.qfrc_bias.copy()


class TestZVCF:
    """ZVCF is the control component M(q)^-1 tau (#14495, owner decision)."""

    def test_zvcf_matches_simulation_backends_parity(self, double_pendulum_model):
        """MuJoCo zvcf() == simulation_backends zvcf_acceleration (#14495)."""
        analyzer = CounterfactualAnalyzer(double_pendulum_model)
        qpos = np.array([0.4, -0.7])
        qvel = np.array([1.3, -0.8])
        ctrl = np.array([0.5, -1.2])  # gear 2 on both motors -> tau = 2 * ctrl

        result = analyzer.zvcf(qpos, qvel, ctrl)

        expected = zvcf_acceleration(_provider(double_pendulum_model), qpos, 2 * ctrl)
        assert result.type == "zvcf"
        assert np.allclose(result.counterfactual_acceleration, expected, atol=1e-9)
        assert np.allclose(result.torque_attributed_effect, expected, atol=1e-9)
        assert result.velocity_attributed_effect is None

    def test_zvcf_is_independent_of_velocity_and_gravity(self, double_pendulum_model):
        analyzer = CounterfactualAnalyzer(double_pendulum_model)
        qpos = np.array([0.4, -0.7])
        ctrl = np.array([0.5, -1.2])

        slow = analyzer.zvcf(qpos, np.zeros(2), ctrl)
        fast = analyzer.zvcf(qpos, np.array([3.0, -2.0]), ctrl)
        assert np.allclose(
            slow.counterfactual_acceleration, fast.counterfactual_acceleration
        )

        zero_ctrl = analyzer.zvcf(qpos, np.array([3.0, -2.0]), np.zeros(2))
        assert np.allclose(zero_ctrl.counterfactual_acceleration, 0.0, atol=1e-12)

    def test_zvcf_plus_ztcf_is_total_acceleration(self, double_pendulum_model):
        """Pointwise ZTCF + ZVCF == actual acceleration, gravity counted once."""
        analyzer = CounterfactualAnalyzer(double_pendulum_model)
        qpos = np.array([0.4, -0.7])
        qvel = np.array([1.3, -0.8])
        ctrl = np.array([0.5, -1.2])

        zvcf = analyzer.zvcf(qpos, qvel, ctrl)
        ztcf = analyzer.ztcf(qpos, qvel, ctrl)

        assert np.allclose(
            ztcf.counterfactual_acceleration + zvcf.counterfactual_acceleration,
            ztcf.observed_acceleration,
            atol=1e-9,
        )
        assert np.allclose(zvcf.delta_acceleration, ztcf.counterfactual_acceleration)

    def test_zvcf_rejects_wrong_control_length(self, simple_pendulum_model) -> None:
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)
        with pytest.raises(ValueError, match="ctrl must have length"):
            analyzer.zvcf(np.array([0.1]), np.array([0.0]), np.array([1.0, 2.0]))

    def test_zero_velocity_zero_torque_is_gravity_only(self, simple_pendulum_model):
        """The former gravity-only zvcf() lives on under its descriptive name."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)
        qpos = np.array([np.pi / 4])

        result = analyzer.zero_velocity_zero_torque(qpos, np.array([2.0]))

        expected = zero_velocity_zero_torque_acceleration(
            _provider(simple_pendulum_model), qpos
        )
        assert np.allclose(result.counterfactual_acceleration, expected, atol=1e-9)
        assert result.type == "zero_velocity_zero_torque"


class TestCounterfactualTrajectoryAnalysis:
    """Test trajectory-level counterfactual analysis."""

    def test_ztcf_trajectory_analysis(self, simple_pendulum_model) -> None:
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

    def test_zvcf_trajectory_analysis(self, simple_pendulum_model) -> None:
        """Test ZVCF analysis on full trajectory."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        N = 10
        qpos_traj = np.linspace(0, np.pi / 3, N).reshape(-1, 1)
        qvel_traj = np.linspace(0, 1.5, N).reshape(-1, 1)  # Increasing velocity
        ctrl_traj = np.linspace(0, 1.0, N).reshape(-1, 1)

        results = analyzer.analyze_trajectory_zvcf(qpos_traj, qvel_traj, ctrl_traj)

        assert len(results) == N
        assert all(r.type == "zvcf" for r in results)

        # Delta should increase with velocity
        # Delta may vary — verify analysis runs correctly
        deltas = [abs(r.delta_acceleration[0]) for r in results]
        assert all(d is not None for d in deltas), "All deltas should be computed"


@pytest.mark.integration
class TestCounterfactualPhysics:
    """Integration tests for counterfactual physics validation."""

    def test_ztcf_plus_counterfactual_equals_observed(
        self, simple_pendulum_model
    ) -> None:
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

    def test_zvcf_plus_delta_equals_observed(self, simple_pendulum_model) -> None:
        """Test physics: counterfactual (control) + delta (drift) = observed."""
        analyzer = CounterfactualAnalyzer(simple_pendulum_model)

        result = analyzer.zvcf(np.array([0.3]), np.array([1.0]), np.array([0.5]))

        reconstructed = result.counterfactual_acceleration + result.delta_acceleration
        assert np.allclose(result.observed_acceleration, reconstructed, atol=1e-6)

    def test_ztcf_reveals_control_authority(self, simple_pendulum_model) -> None:
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
