"""Native command optimization must end in guarded replay, including fallback."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from src.engines.myosuite_project_forecast import ProjectTaskForecaster
from src.engines.myosuite_project_tracking import (
    NativeTrackingReference,
    NativeTrackingScales,
    ProjectTaskTrackingObjective,
)
from src.shared.python.motion_matching.bounded_candidate_search import (
    CandidateSearchPolicy,
)
from .test_project_task_forecast import _observation, _task
from .test_project_task_native_search import _search
from .test_project_task_producer import _integration_state, _native_runtime

pytestmark = pytest.mark.unit
TARGET = np.array([[0.15, 0.3], [0.2, 0.4], [0.25, 0.45]])


@contextmanager
def _fixture(tmp_path: Path) -> Iterator[tuple[Any, Any, Any, Any, Any]]:
    mj = _native_runtime()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    try:
        with ProjectTaskForecaster(task) as forecast:
            from src.engines import myosuite_project_native_search as native

            with _search(native, forecast, observation) as search:
                target = search.predict(observation, TARGET)
                reference = NativeTrackingReference(
                    observation.native_time_seconds + target.time_seconds,
                    target.qpos,
                    target.qvel,
                    target.activation,
                    hashlib.sha256(TARGET.tobytes()).hexdigest(),
                )
                scales = NativeTrackingScales(
                    np.array([0.001]),
                    np.array([0.1]),
                    np.array([0.02]),
                    np.array([10.0, 10.0]),
                )
                objective = ProjectTaskTrackingObjective(
                    forecast, observation, reference, scales
                )
                yield mj, task, observation, search, objective
    finally:
        task.close()


def _arguments(observation: Any) -> dict[str, Any]:
    return {
        "initial_plan": np.tile(observation.control, (3, 1)),
        "fallback_plan": np.tile(observation.control, (3, 1)),
        "lower": np.array([-1.0, 0.0]),
        "upper": np.array([1.0, 1.0]),
        "max_command_increment": np.array([0.2, 0.3]),
        "state_margins": lambda states, commands: np.array([1.0]),
        "guarded_admission": lambda history: True,
        "criteria_provenance_sha256": hashlib.sha256(
            b"native-motor-filter-fixture-criteria-v1"
        ).hexdigest(),
        "policy": CandidateSearchPolicy(2000, 60.0, 100, 1e-8),
    }


def _solve(*args: Any, **kwargs: Any) -> Any:
    from src.engines.myosuite_project_command_solve import (
        NativeCommandSolveProblem,
        solve_native_command_plan,
    )

    policy = kwargs.pop("policy")
    options = {
        name: kwargs.pop(name)
        for name in ("solver", "cancel_requested")
        if name in kwargs
    }
    return solve_native_command_plan(
        *args, NativeCommandSolveProblem(**kwargs), policy, **options
    )


def _propose(plan: np.ndarray) -> Any:
    def solver(*args: Any, **kwargs: Any) -> OptimizeResult:
        return OptimizeResult(x=plan.ravel(), success=True)

    return solver


def test_actual_native_optimized_plan_improves_and_replays_without_live_step(
    tmp_path: Path,
    record_property: Any,
) -> None:
    with _fixture(tmp_path) as (mj, task, observation, search, objective):
        before = _integration_state(mj, task.model, task.data)
        result = _solve(search, observation, objective, **_arguments(observation))
        assert result.status == "optimized"
        assert result.selected_plan.objective < result.fallback_plan.objective
        assert (
            result.selected_plan.objective
            == objective.history_cost(result.selected_plan.history).total
        )
        assert result.selected_plan.replay_artifact is not None
        assert result.elapsed_s >= result.selected_plan.validation_seconds
        for key, value in {
            "optimized_objective": result.selected_plan.objective,
            "fallback_objective": result.fallback_plan.objective,
            "solve_elapsed_s": result.elapsed_s,
            "numerical_evaluations": result.numerical.evaluations,
            "solve_parameters_sha256": result.solve_parameters_sha256,
            "criteria_provenance_sha256": result.criteria_provenance_sha256,
        }.items():
            record_property(key, value)
        np.testing.assert_array_equal(
            _integration_state(mj, task.model, task.data), before
        )


@pytest.mark.parametrize("location", [0, 2])
def test_post_mapping_slew_rejects_first_or_final_knot(
    tmp_path: Path, location: int
) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        proposal = TARGET.copy()
        proposal[location, 0] = 0.9
        result = _solve(
            search,
            observation,
            objective,
            solver=_propose(proposal),
            **_arguments(observation),
        )
        assert result.status == "fallback_infeasible"
        np.testing.assert_array_equal(
            result.selected_plan.history.applied_actuator_commands,
            np.tile(observation.control, (3, 1)),
        )


def test_guarded_only_rejection_keeps_independently_admitted_fallback(
    tmp_path: Path,
) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        arguments = _arguments(observation)
        fallback = arguments["fallback_plan"]
        arguments["guarded_admission"] = lambda h: bool(
            np.array_equal(h.applied_actuator_commands, fallback)
        )
        result = _solve(
            search, observation, objective, solver=_propose(TARGET), **arguments
        )
        assert result.status == "fallback_guarded_rejection"
        assert result.selected_plan is result.fallback_plan


def test_invalid_complete_fallback_stops_before_search(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        arguments = _arguments(observation)
        arguments["fallback_plan"][2, 0] = 0.9
        with pytest.raises(ValueError, match="hard criteria"):
            _solve(search, observation, objective, **arguments)


def test_cancelled_search_still_has_guarded_replayable_fallback(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        result = _solve(
            search,
            observation,
            objective,
            cancel_requested=lambda: True,
            **_arguments(observation),
        )
        assert result.status == "fallback_cancelled"
        assert result.selected_plan is result.fallback_plan
        assert result.selected_plan.replay_artifact is not None


def test_missing_complete_state_margins_is_rejected(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        arguments = _arguments(observation)
        arguments["state_margins"] = None
        with pytest.raises(TypeError, match="margins"):
            _solve(search, observation, objective, **arguments)


def test_criteria_provenance_requires_explicit_sha256(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        arguments = _arguments(observation)
        arguments["criteria_provenance_sha256"] = "undeclared"
        with pytest.raises(ValueError, match="SHA-256"):
            _solve(search, observation, objective, **arguments)


def test_full_horizon_activation_margin_rejects_late_state(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (mj, task, observation, search, objective):
        arguments = _arguments(observation)
        scratch = mj.MjData(task.model)

        def margins(states: np.ndarray, commands: np.ndarray) -> np.ndarray:
            values = []
            for state in states:
                mj.mj_setState(
                    task.model, scratch, state, mj.mjtState.mjSTATE_INTEGRATION
                )
                values.append(0.3001 - float(scratch.act[0]))
            return np.array(values)

        arguments["state_margins"] = margins
        result = _solve(
            search, observation, objective, solver=_propose(TARGET), **arguments
        )
        assert result.status == "fallback_infeasible"
        assert result.selected_plan is result.fallback_plan


def test_expired_budget_includes_fallback_validation_cost(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        arguments = _arguments(observation)
        arguments["policy"] = CandidateSearchPolicy(100, 1e-9, 10, 0.0)
        result = _solve(search, observation, objective, **arguments)
        assert result.status == "fallback_timeout"
        assert result.elapsed_s > arguments["policy"].max_wall_s
        assert result.selected_plan.replay_artifact is not None


def test_changed_live_anchor_cannot_return_earlier_fallback(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (_, task, observation, search, objective):
        arguments = _arguments(observation)

        def mutate(*args: Any, **kwargs: Any) -> OptimizeResult:
            task.data.qvel[:] += 0.2
            return OptimizeResult(x=TARGET.ravel(), success=True)

        with pytest.raises(ValueError):
            _solve(search, observation, objective, solver=mutate, **arguments)


def test_changed_adapter_source_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        from src.engines import myosuite_project_command_solve as module

        calls = iter(("a" * 64, "b" * 64))
        monkeypatch.setattr(module, "_source_sha256", lambda: next(calls))
        with pytest.raises(ValueError, match="solve source changed"):
            _solve(
                search,
                observation,
                objective,
                cancel_requested=lambda: True,
                **_arguments(observation),
            )


def test_guarded_objective_disagreement_cannot_promote_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        from src.engines.myosuite_project_tracking import NativeTrackingCost

        original = objective.history_cost
        fallback_commands = np.tile(observation.control, (3, 1))

        def changed(history: Any) -> Any:
            original(history)  # Retain actual current-state/source admission.
            value = (
                1.0
                if np.array_equal(history.applied_actuator_commands, fallback_commands)
                else 2.0
            )
            return NativeTrackingCost(value, 0.0, 0.0, 0.0)

        monkeypatch.setattr(objective, "history_cost", changed)
        result = _solve(
            search,
            observation,
            objective,
            solver=_propose(TARGET),
            **_arguments(observation),
        )
        assert result.status == "fallback_guarded_no_benefit"
        assert result.selected_plan is result.fallback_plan


def test_changed_hard_criteria_declaration_cannot_be_recorded_as_original(
    tmp_path: Path,
) -> None:
    with _fixture(tmp_path) as (_, _, observation, search, objective):
        from src.engines.myosuite_project_command_solve import (
            NativeCommandSolveProblem,
            solve_native_command_plan,
        )

        arguments = _arguments(observation)
        policy = arguments.pop("policy")
        problem = NativeCommandSolveProblem(**arguments)

        def mutate(*args: Any, **kwargs: Any) -> OptimizeResult:
            object.__setattr__(problem, "criteria_provenance_sha256", "b" * 64)
            return OptimizeResult(x=TARGET.ravel(), success=True)

        with pytest.raises(ValueError, match="parameters changed"):
            solve_native_command_plan(
                search, observation, objective, problem, policy, solver=mutate
            )
