import numpy as np
import pytest

from src.reinforcement_learning.trajectory_funnel_benchmark import (
    TrajectoryFunnelBenchmark,
)


def test_initialization():
    bench = TrajectoryFunnelBenchmark("transverse")
    assert bench.mode == "transverse"

    bench2 = TrajectoryFunnelBenchmark("setpoint")
    assert bench2.mode == "setpoint"

    with pytest.raises(AssertionError):
        TrajectoryFunnelBenchmark("invalid")


def test_setpoint_reward():
    bench = TrajectoryFunnelBenchmark("setpoint")

    current = np.array([1.0, 2.0])
    target = np.array([1.0, 3.0])

    res = bench.setpoint_reward(current, target)
    assert res == -1.0

    with pytest.raises(AssertionError):
        bench.setpoint_reward(None, target)


def test_trajectory_funnel_reward():
    bench = TrajectoryFunnelBenchmark("transverse")

    current = np.array([0.0, 0.5])
    reference = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

    # closest point is [0.0, 0.0] -> dist=0.5 or [0.0, 1.0] -> dist=0.5
    # project_phase_idx will be 0
    res = bench.trajectory_funnel_reward(current, reference, 0)

    # transverse cost = -10.0 * 0.5^2 = -2.5
    # phase reward = 0.5 * (0 / 3) = 0.0
    # total = -2.5
    assert np.isclose(res, -2.5)


def test_simulate_agent_training_mock():
    bench_setpoint = TrajectoryFunnelBenchmark("setpoint")
    res1 = bench_setpoint.simulate_agent_training_mock()
    assert res1["convergence_epochs"] == 15000
    assert res1["terminal_variance"] == 4.5

    bench_transverse = TrajectoryFunnelBenchmark("transverse")
    res2 = bench_transverse.simulate_agent_training_mock()
    assert res2["convergence_epochs"] == 2400
    assert res2["terminal_variance"] == 0.03
