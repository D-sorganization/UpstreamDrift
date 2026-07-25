"""Regression tests for issue #7983.

The old ``simulate_agent_training`` reported an "empirical" RL result with no
agent: ``state = reference[step] + rng.normal(0, noise_scale)``. The state was
never produced by a policy, never influenced by the reward, and was
bit-identical between the two modes being compared. The unconditional closing
log line claimed a result the run had not measured.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.reinforcement_learning import trajectory_funnel_benchmark as tfb
from src.reinforcement_learning.trajectory_funnel_benchmark import (
    TrajectoryFunnelBenchmark,
)

pytestmark = pytest.mark.unit


class TestThereIsActuallyAnAgent:
    def test_actions_move_the_state(self) -> None:
        """A non-zero policy must change the trajectory it produces."""
        bench = TrajectoryFunnelBenchmark("transverse")
        reference = bench.build_reference(20, 3)
        rng = np.random.default_rng(0)
        _, zero_states = bench.rollout(np.zeros((3, 5)), reference, rng)
        theta = np.zeros((3, 5))
        theta[:, -1] = 1.0  # constant bias action
        _, biased_states = bench.rollout(theta, reference, rng)
        assert not np.allclose(zero_states, biased_states)

    def test_state_is_not_the_reference_plus_noise(self) -> None:
        """A zero policy holds the initial state - it does not track the ref."""
        bench = TrajectoryFunnelBenchmark("transverse")
        reference = bench.build_reference(20, 3)
        _, states = bench.rollout(np.zeros((3, 5)), reference, np.random.default_rng(0))
        np.testing.assert_allclose(states, np.tile(reference[0], (20, 1)))

    def test_training_improves_the_mode_return(self) -> None:
        """Return must rise across iterations - real optimisation, not a schedule."""
        for mode in ("setpoint", "transverse"):
            result = TrajectoryFunnelBenchmark(mode).train_agent()
            assert float(result["final_return"]) > float(result["initial_return"]), (
                f"{mode}: no improvement "
                f"{result['initial_return']} -> {result['final_return']}"
            )

    def test_the_two_modes_do_not_visit_identical_states(self) -> None:
        """The core defect: both 'agents' used to see bit-identical states."""
        setpoint = TrajectoryFunnelBenchmark("setpoint")
        transverse = TrajectoryFunnelBenchmark("transverse")
        setpoint.train_agent(n_iterations=20, n_steps=15, state_dim=3)
        transverse.train_agent(n_iterations=20, n_steps=15, state_dim=3)
        assert setpoint.learning_curve != transverse.learning_curve

    def test_reward_influences_the_learned_policy(self) -> None:
        """Different rewards must yield different neutral-metric outcomes."""
        sp = TrajectoryFunnelBenchmark("setpoint").train_agent()
        fn = TrajectoryFunnelBenchmark("transverse").train_agent()
        assert sp["mean_transverse_error"] != fn["mean_transverse_error"]


class TestMetricUnits:
    def test_convergence_index_is_an_iteration_not_a_step(self) -> None:
        """5000 samples were once reported as 'convergence_epochs' for 100 eps."""
        bench = TrajectoryFunnelBenchmark("transverse")
        result = bench.train_agent(n_iterations=12, n_steps=40, state_dim=2)
        assert len(bench.learning_curve) == 12
        assert int(result["convergence_iteration"]) <= 12

    def test_neutral_metrics_are_reported(self) -> None:
        """Cross-mode comparison needs metrics independent of reward scale."""
        result = TrajectoryFunnelBenchmark("transverse").train_agent(n_iterations=3)
        assert result["mean_transverse_error"] >= 0.0
        assert result["terminal_setpoint_error"] >= 0.0


class TestConclusionIsDerived:
    def test_main_conclusion_matches_the_numbers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The closing line must be derived from the run, not hardcoded."""
        with caplog.at_level(logging.INFO, logger=tfb.__name__):
            tfb._main()
        messages = [rec.getMessage() for rec in caplog.records]
        conclusion = [m for m in messages if m.startswith("Result:")]
        assert len(conclusion) == 1
        assert "geometrically accelerates convergence" not in conclusion[0]
        assert any(word in conclusion[0] for word in ("lower", "higher", "equal"))
