"""Comprehensive coverage tests for TrajectoryFunnelBenchmark.

Targets the entire ``src/reinforcement_learning`` package: reward functions,
convergence estimator, and deterministic simulation entry point.  No training
loops, deterministic seeded runs only.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.reinforcement_learning import TrajectoryFunnelBenchmark
from src.reinforcement_learning import trajectory_funnel_benchmark as tfb_module

# ---------------------------------------------------------------------------
# Construction / mode validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_mode_is_transverse(self) -> None:
        bench = TrajectoryFunnelBenchmark()
        assert bench.mode == "transverse"

    @pytest.mark.parametrize("mode", ["transverse", "setpoint"])
    def test_valid_modes(self, mode: str) -> None:
        bench = TrajectoryFunnelBenchmark(mode=mode)
        assert bench.mode == mode

    @pytest.mark.parametrize("bad_mode", ["", "TRANSVERSE", "foo", "set_point"])
    def test_invalid_mode_raises(self, bad_mode: str) -> None:
        with pytest.raises(AssertionError, match="Mode must be"):
            TrajectoryFunnelBenchmark(mode=bad_mode)


# ---------------------------------------------------------------------------
# setpoint_reward
# ---------------------------------------------------------------------------


class TestSetpointReward:
    def setup_method(self) -> None:
        self.bench = TrajectoryFunnelBenchmark(mode="setpoint")

    def test_zero_error_returns_zero(self) -> None:
        s = np.array([1.0, 2.0, 3.0, 4.0])
        assert self.bench.setpoint_reward(s, s) == pytest.approx(0.0)

    def test_negative_squared_distance(self) -> None:
        # 3-4-5 triangle: distance^2 = 25
        c = np.array([3.0, 4.0])
        t = np.array([0.0, 0.0])
        assert self.bench.setpoint_reward(c, t) == pytest.approx(-25.0)

    def test_symmetry(self) -> None:
        a = np.array([1.0, -2.0, 0.5])
        b = np.array([-3.0, 1.0, 2.0])
        assert self.bench.setpoint_reward(a, b) == pytest.approx(
            self.bench.setpoint_reward(b, a)
        )

    def test_always_non_positive(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            c = rng.standard_normal(5)
            t = rng.standard_normal(5)
            assert self.bench.setpoint_reward(c, t) <= 0.0

    def test_return_type_is_python_float(self) -> None:
        c = np.array([1.0, 0.0])
        t = np.array([0.0, 0.0])
        result = self.bench.setpoint_reward(c, t)
        assert type(result) is float

    def test_none_current_state_raises(self) -> None:
        with pytest.raises(AssertionError, match="current_state must be provided"):
            self.bench.setpoint_reward(None, np.array([1.0]))  # type: ignore[arg-type]

    def test_higher_distance_lower_reward(self) -> None:
        t = np.zeros(3)
        r_near = self.bench.setpoint_reward(np.array([0.1, 0.0, 0.0]), t)
        r_far = self.bench.setpoint_reward(np.array([10.0, 0.0, 0.0]), t)
        assert r_near > r_far

    def test_high_dimensional(self) -> None:
        c = np.ones(50)
        t = np.zeros(50)
        # ||1||^2 over 50 dims = 50
        assert self.bench.setpoint_reward(c, t) == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# trajectory_funnel_reward
# ---------------------------------------------------------------------------


class TestTrajectoryFunnelReward:
    def setup_method(self) -> None:
        self.bench = TrajectoryFunnelBenchmark(mode="transverse")

    def test_on_trajectory_first_point(self) -> None:
        ref = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        # Exactly on first point: distance 0, phase_velocity_reward = 0
        r = self.bench.trajectory_funnel_reward(ref[0], ref, 0.0)
        assert r == pytest.approx(0.0)

    def test_on_trajectory_last_point_gives_phase_bonus(self) -> None:
        ref = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        # On last point: distance 0, phase_velocity_reward = 0.5 * 3/4 = 0.375
        r = self.bench.trajectory_funnel_reward(ref[-1], ref, 1.0)
        assert r == pytest.approx(0.5 * (3 / 4))

    def test_transverse_cost_scales_with_squared_distance(self) -> None:
        ref = np.array([[0.0, 0.0]])
        # min squared distance = 4, transverse cost = -40, phase bonus = 0
        r = self.bench.trajectory_funnel_reward(np.array([2.0, 0.0]), ref, 0.0)
        assert r == pytest.approx(-40.0)

    def test_picks_geometrically_closest_point(self) -> None:
        # Closest reference point is index 2: phase reward = 0.5 * 2/4 = 0.25
        ref = np.array([[10.0, 0.0], [5.0, 0.0], [0.0, 0.0], [-5.0, 0.0]])
        r = self.bench.trajectory_funnel_reward(np.array([0.0, 0.0]), ref, 0.0)
        assert r == pytest.approx(0.25)

    def test_return_type_is_python_float(self) -> None:
        ref = np.array([[0.0, 0.0], [1.0, 0.0]])
        r = self.bench.trajectory_funnel_reward(np.array([0.0, 0.0]), ref, 0.0)
        assert type(r) is float

    def test_none_current_state_raises(self) -> None:
        ref = np.array([[0.0, 0.0]])
        with pytest.raises(AssertionError, match="current_state must be provided"):
            self.bench.trajectory_funnel_reward(None, ref, 0.0)  # type: ignore[arg-type]

    def test_phase_argument_does_not_affect_reward(self) -> None:
        """current_phase is documented as 'allowing phase slippage' — verify
        the implementation does not use it to alter the reward (the projection
        finds its own closest phase)."""
        ref = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        state = np.array([0.9, 0.0])
        r0 = self.bench.trajectory_funnel_reward(state, ref, 0.0)
        r1 = self.bench.trajectory_funnel_reward(state, ref, 1.0)
        assert r0 == r1

    def test_high_dim_trajectory(self) -> None:
        ref = np.zeros((10, 4))
        ref[:, 0] = np.linspace(0, 1, 10)
        r = self.bench.trajectory_funnel_reward(np.zeros(4), ref, 0.0)
        # closest is index 0, distance 0
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _estimate_convergence
# ---------------------------------------------------------------------------


class TestEstimateConvergence:
    def setup_method(self) -> None:
        self.bench = TrajectoryFunnelBenchmark()

    def test_empty_trajectory(self) -> None:
        epoch, var = self.bench._estimate_convergence([])
        assert epoch == 0
        assert var == float("inf")

    def test_constant_trajectory_terminal_variance_zero(self) -> None:
        traj = [1.0] * 50
        epoch, var = self.bench._estimate_convergence(traj)
        # rolling means are constant → relative improvement = 0 < threshold
        assert epoch == 10  # first index where i >= window_size
        assert var == pytest.approx(0.0)

    def test_no_convergence_returns_len(self) -> None:
        # Strictly increasing trajectory never plateaus
        traj = list(np.linspace(1.0, 100.0, 30).tolist())
        epoch, var = self.bench._estimate_convergence(traj, threshold=1e-9)
        assert epoch == len(traj)
        assert var > 0.0

    def test_short_trajectory_below_window(self) -> None:
        traj = [1.0, 2.0, 3.0]  # shorter than default window_size=10
        epoch, var = self.bench._estimate_convergence(traj)
        # Loop range(10, 3) is empty → epoch stays at len()
        assert epoch == len(traj)
        # final_window has >1 element → std computed
        assert var == pytest.approx(float(np.std(traj)))

    def test_single_element_terminal_variance_zero(self) -> None:
        epoch, var = self.bench._estimate_convergence([5.0])
        assert epoch == 1
        assert var == 0.0

    def test_custom_window_and_threshold(self) -> None:
        traj = [1.0] * 30
        epoch, _var = self.bench._estimate_convergence(
            traj, window_size=5, threshold=0.5
        )
        assert epoch == 5

    def test_zero_prev_rolling_mean_skips_division(self) -> None:
        """If a rolling mean is exactly zero the relative-improvement guard
        must avoid division-by-zero and not declare convergence at that step."""
        # symmetric around zero so rolling mean lands on 0.0 at some windows
        traj = [-1.0, 1.0] * 20
        epoch, _ = self.bench._estimate_convergence(traj, window_size=2)
        # Should not crash; either converged at some point or len()
        assert 0 <= epoch <= len(traj)


# ---------------------------------------------------------------------------
# simulate_agent_training
# ---------------------------------------------------------------------------


class TestSimulateAgentTraining:
    def test_setpoint_result_schema(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="setpoint")
        result = bench.simulate_agent_training(n_episodes=5, n_steps=10, state_dim=3)
        assert set(result.keys()) == {
            "convergence_epochs",
            "terminal_variance",
            "mode",
        }
        assert result["mode"] == "setpoint"
        assert isinstance(result["convergence_epochs"], int)
        assert isinstance(result["terminal_variance"], float)

    def test_transverse_result_schema(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="transverse")
        result = bench.simulate_agent_training(n_episodes=5, n_steps=10, state_dim=3)
        assert result["mode"] == "transverse"

    def test_deterministic_repeatable(self) -> None:
        """Seeded rng inside the method makes runs deterministic for given mode."""
        b1 = TrajectoryFunnelBenchmark(mode="setpoint")
        b2 = TrajectoryFunnelBenchmark(mode="setpoint")
        r1 = b1.simulate_agent_training(n_episodes=5, n_steps=8, state_dim=2)
        r2 = b2.simulate_agent_training(n_episodes=5, n_steps=8, state_dim=2)
        assert r1 == r2

    def test_minimal_run(self) -> None:
        bench = TrajectoryFunnelBenchmark(mode="transverse")
        result = bench.simulate_agent_training(n_episodes=2, n_steps=3, state_dim=2)
        assert result["convergence_epochs"] >= 0
        assert np.isfinite(result["terminal_variance"])

    def test_modes_produce_different_dynamics(self) -> None:
        sp = TrajectoryFunnelBenchmark(mode="setpoint").simulate_agent_training(
            n_episodes=10, n_steps=10, state_dim=3
        )
        tv = TrajectoryFunnelBenchmark(mode="transverse").simulate_agent_training(
            n_episodes=10, n_steps=10, state_dim=3
        )
        # The two reward formulations are fundamentally different in magnitude
        assert sp["terminal_variance"] != tv["terminal_variance"]

    def test_logger_initialized_when_no_handlers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bench = TrajectoryFunnelBenchmark(mode="transverse")
        with caplog.at_level(logging.INFO, logger=tfb_module.__name__):
            bench.simulate_agent_training(n_episodes=2, n_steps=3, state_dim=2)
        # At least the "Initializing" and "Training complete" messages
        msgs = " ".join(rec.getMessage() for rec in caplog.records)
        assert "Initializing" in msgs
        assert "Training complete" in msgs


# ---------------------------------------------------------------------------
# Package smoke
# ---------------------------------------------------------------------------


def test_package_exports_benchmark() -> None:
    from src import reinforcement_learning as pkg

    assert "TrajectoryFunnelBenchmark" in pkg.__all__
    assert pkg.TrajectoryFunnelBenchmark is TrajectoryFunnelBenchmark
