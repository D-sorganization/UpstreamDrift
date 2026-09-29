"""Smoke tests for reinforcement_learning trajectory funnel benchmark."""

from __future__ import annotations

import numpy as np
import pytest
from src.reinforcement_learning.trajectory_funnel_benchmark import (
    TrajectoryFunnelBenchmark,
)

pytestmark = pytest.mark.unit


class TestTrajectoryFunnelBenchmark:
    """Basic tests for the TrajectoryFunnelBenchmark class."""

    def test_construction_transverse(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="transverse")
        assert bench.mode == "transverse"

    def test_construction_setpoint(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="setpoint")
        assert bench.mode == "setpoint"

    def test_invalid_mode(self) -> None:
        with pytest.raises(AssertionError):
            TrajectoryFunnelBenchmark(mode="invalid")

    def test_setpoint_reward(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="setpoint")
        current = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        reward = bench.setpoint_reward(current, target)
        assert reward == pytest.approx(0.0)

    def test_setpoint_reward_nonzero_error(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="setpoint")
        current = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        reward = bench.setpoint_reward(current, target)
        assert reward < 0.0

    def test_trajectory_funnel_reward(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="transverse")
        current = np.array([1.0, 0.0])
        reference = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]])
        reward = bench.trajectory_funnel_reward(current, reference, 0.5)
        assert isinstance(reward, float)

    def test_train_agent_transverse(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="transverse")
        result = bench.train_agent(n_iterations=10, n_steps=12, state_dim=3)
        assert "convergence_iteration" in result
        assert "terminal_return_std" in result
        assert "mode" in result
        assert result["mode"] == "transverse"
        # Mode-neutral metrics are the only cross-mode comparable numbers.
        assert result["mean_transverse_error"] >= 0.0

    def test_train_agent_setpoint(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="setpoint")
        result = bench.train_agent(n_iterations=10, n_steps=12, state_dim=3)
        assert result["convergence_iteration"] > 0
        assert result["terminal_return_std"] >= 0.0
