"""Behavioral gates for optional bounded receding-horizon feedback (F05)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import src.shared.python.motion_matching.bounded_nmpc as bounded_nmpc_module

from src.shared.python.motion_matching.bounded_nmpc import (
    BoundedNMPC,
    MPCConfig,
    MPCProblem,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _step(state: np.ndarray, effort: np.ndarray) -> np.ndarray:
    dt = 0.1
    return np.array(
        [state[0] + dt * state[1] + 0.5 * dt**2 * effort[0], state[1] + dt * effort[0]],
        dtype=float,
    )


def _problem() -> MPCProblem:
    return MPCProblem(
        step=_step,
        time_step_s=0.1,
        target_states=np.tile(np.array([1.0, 0.0]), (9, 1)),
        state_weights=np.array([10.0, 0.1]),
        terminal_weights=np.array([30.0, 0.1]),
        input_weights=np.array([0.01]),
        input_lower=np.array([-2.0]),
        input_upper=np.array([2.0]),
        state_lower=np.array([-2.0, -3.0]),
        state_upper=np.array([2.0, 3.0]),
        state_component_ids=("hinge_q", "hinge_v"),
        state_units=("rad", "rad/s"),
        input_channel_ids=("hip",),
        input_units=("N*m",),
    )


def test_bounded_horizon_improves_over_verified_zero_feedback() -> None:
    controller = BoundedNMPC(
        _problem(),
        MPCConfig(horizon_steps=6, max_evaluations=300, max_wall_s=1.0),
        fallback=lambda state, time_s: np.array([0.0]),
    )
    receipt = controller.command_for_step(
        0, np.array([0.0, 0.0]), observation_time_s=0.0, current_time_s=0.0
    )
    assert receipt.status == "optimized"
    assert receipt.applied.shape == (1,)
    assert 0.0 < receipt.applied[0] <= 2.0
    assert receipt.objective < receipt.fallback_objective
    assert receipt.input_boundary == "post_limit_actuator_command"
    assert receipt.information_pattern == "exact_simulated_state"
    assert receipt.evaluations <= 300
    with pytest.raises(ValueError):
        receipt.applied[0] = 0.0


def test_stale_cancelled_and_expired_solve_use_logged_safe_fallback() -> None:
    ticks = iter((0.0, 0.001, 0.02, 0.021, 0.03, 0.05, 0.051))
    controller = BoundedNMPC(
        _problem(),
        MPCConfig(
            horizon_steps=3,
            max_evaluations=100,
            max_wall_s=0.01,
            max_observation_age_s=0.005,
        ),
        fallback=lambda state, time_s: np.array([0.0]),
        clock=lambda: next(ticks),
    )
    stale = controller.command_for_step(
        1, np.zeros(2), observation_time_s=0.08, current_time_s=0.1
    )
    assert stale.status == "fallback_stale_observation"
    assert stale.applied.tolist() == [0.0]
    cancelled = controller.command_for_step(
        2,
        np.zeros(2),
        observation_time_s=0.2,
        current_time_s=0.2,
        cancelled=True,
    )
    assert cancelled.status == "fallback_cancelled"
    assert cancelled.applied.tolist() == [0.0]
    expired = controller.command_for_step(
        3, np.zeros(2), observation_time_s=0.3, current_time_s=0.3
    )
    assert expired.status == "fallback_timeout"
    assert expired.applied.tolist() == [0.0]
    assert stale.elapsed_s == pytest.approx(0.001)
    assert cancelled.elapsed_s == pytest.approx(0.001)
    assert expired.elapsed_s == pytest.approx(0.021)
    assert tuple(item.status for item in controller.applied_history) == (
        "fallback_stale_observation",
        "fallback_cancelled",
        "fallback_timeout",
    )


def test_unsafe_fallback_cannot_be_driven_or_logged_as_applied() -> None:
    controller = BoundedNMPC(
        _problem(),
        MPCConfig(horizon_steps=3, max_evaluations=100, max_wall_s=1.0),
        fallback=lambda state, time_s: np.array([9.0]),
    )
    with pytest.raises(ValueError, match="fallback.*bounds"):
        controller.command_for_step(
            0,
            np.zeros(2),
            observation_time_s=0.0,
            current_time_s=0.0,
            cancelled=True,
        )
    assert controller.applied_history == ()


def test_candidate_must_satisfy_all_declared_plant_scenarios() -> None:
    def high_gain(state: np.ndarray, effort: np.ndarray) -> np.ndarray:
        return _step(state, 1.5 * effort)

    base = _problem()
    problem = MPCProblem(
        step=base.step,
        scenario_steps=(high_gain,),
        time_step_s=base.time_step_s,
        target_states=base.target_states,
        state_weights=base.state_weights,
        terminal_weights=base.terminal_weights,
        input_weights=base.input_weights,
        input_lower=base.input_lower,
        input_upper=base.input_upper,
        state_lower=np.array([-2.0, -0.2]),
        state_upper=np.array([2.0, 0.2]),
        state_component_ids=base.state_component_ids,
        state_units=base.state_units,
        input_channel_ids=base.input_channel_ids,
        input_units=base.input_units,
    )
    controller = BoundedNMPC(
        problem,
        MPCConfig(horizon_steps=3, max_evaluations=400, max_wall_s=1.0),
        fallback=lambda state, time_s: np.array([0.0]),
    )
    receipt = controller.command_for_step(
        0, np.zeros(2), observation_time_s=0.0, current_time_s=0.0
    )
    assert receipt.status in {"optimized", "fallback_infeasible"}
    assert abs(high_gain(np.zeros(2), receipt.applied)[1]) <= 0.2 + 1e-9


def test_unsupported_state_manifold_and_unnamed_units_are_rejected() -> None:
    base = _problem()
    with pytest.raises(ValueError, match="Euclidean"):
        replace(base, state_identity="quaternion_configuration")
    with pytest.raises(ValueError, match="state.*units"):
        replace(base, state_units=("rad", ""))
    with pytest.raises(ValueError, match="input.*units"):
        replace(base, input_units=("N",))


def test_unavoidable_horizon_constraint_reports_infeasible_fallback() -> None:
    def drift(state: np.ndarray, effort: np.ndarray) -> np.ndarray:
        return state + np.array([0.1, 0.0])

    problem = replace(
        _problem(),
        step=drift,
        state_upper=np.array([0.15, 3.0]),
    )
    controller = BoundedNMPC(
        problem,
        MPCConfig(horizon_steps=3, max_evaluations=300, max_wall_s=1.0),
        fallback=lambda state, time_s: np.array([0.0]),
    )
    receipt = controller.command_for_step(
        0, np.zeros(2), observation_time_s=0.0, current_time_s=0.0
    )
    assert receipt.status == "fallback_infeasible"
    assert receipt.applied.tolist() == [0.0]


def test_nonfinite_solver_candidate_keeps_verified_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bounded_nmpc_module,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(success=True, x=np.full(3, np.nan)),
    )
    controller = BoundedNMPC(
        _problem(),
        MPCConfig(horizon_steps=3, max_evaluations=100, max_wall_s=1.0),
        fallback=lambda state, time_s: np.array([0.0]),
    )
    receipt = controller.command_for_step(
        0, np.zeros(2), observation_time_s=0.0, current_time_s=0.0
    )
    assert receipt.status == "fallback_solver_failure"
    assert receipt.applied.tolist() == [0.0]
    assert len(controller.applied_history) == 1
