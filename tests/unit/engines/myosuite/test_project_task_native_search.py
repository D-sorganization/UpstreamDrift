"""Provisional native search never supplies guarded execution qualification."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.myosuite_project_forecast import ProjectTaskForecaster
from .test_project_task_forecast import _observation, _task
from .test_project_task_producer import (
    _AUDITED_BASE_SHA256,
    _PRODUCTION_SHA256,
    _integration_state,
    _native_runtime,
)

pytestmark = pytest.mark.unit


def _module() -> Any:
    from src.engines import myosuite_project_native_search

    return myosuite_project_native_search


def _search(module: Any, forecast: Any, observation: Any) -> Any:
    seed = forecast.predict(observation, np.full((2, 2), 0.2)).history
    return module.ProjectTaskNativeSearch(
        forecast,
        seed,
        model_id="fixture",
        variant_id="mixed",
        experiment_id="owned-native-search",
        max_steps=20,
    )


def test_search_matches_guarded_complete_history_and_repeated_candidates(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    task.data.qacc_warmstart[:] = 0.25
    observation = _observation(task)
    initial = _integration_state(mj, task.model, task.data)
    commands = np.array([[-0.1, 0.2], [0.1, 0.6], [-0.2, 0.3]])
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                first = search.predict(observation, commands)
                search.predict(observation, np.full((2, 2), 0.4))
                repeated = search.predict(observation, commands)
                guarded = forecast.predict(observation, commands).history
                np.testing.assert_array_equal(
                    first.integration_states, guarded.integration_states
                )
                np.testing.assert_array_equal(
                    first.integration_states, repeated.integration_states
                )
                np.testing.assert_array_equal(first.applied_actuator_commands, commands)
                np.testing.assert_array_equal(
                    first.time_seconds, guarded.time_seconds - 1.25
                )
                assert first.classification == "provisional-native-search"
                assert first.planned_bundle.model == search._seed.bundle.model
                assert (
                    first.planned_bundle.initial_state
                    == search._seed.bundle.initial_state
                )
                assert (
                    first.planned_bundle.applied_input_sha256
                    == first.planned_input_sha256
                )
                assert search.last_attempt_seconds > 0
                with pytest.raises(ValueError):
                    first.integration_states.setflags(write=True)
            # Search closure never owns the caller's SDK forecaster.
            forecast.predict(observation, commands)
        np.testing.assert_array_equal(
            _integration_state(mj, task.model, task.data), initial
        )
    finally:
        task.close()


@pytest.mark.parametrize(
    "commands", [np.array([[0.0, 1.2]]), np.array([[np.nan, 0.2]])]
)
def test_search_invalid_candidate_is_rejected_and_next_candidate_is_independent(
    tmp_path: Path,
    commands: np.ndarray,
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    valid = np.array([[0.1, 0.2], [-0.1, 0.3]])
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                first = search.predict(observation, valid)
                with pytest.raises(ValueError):
                    search.predict(observation, commands)
                repeated = search.predict(observation, valid)
                np.testing.assert_array_equal(
                    first.integration_states, repeated.integration_states
                )
                assert task.data.time == 1.25
    finally:
        task.close()


def test_search_rejects_stale_observation_before_owned_execution(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                task.data.qvel[:] += 0.1
                with pytest.raises(ValueError, match="observation"):
                    search.predict(observation, np.zeros((2, 2)))
                assert task.data.time == 1.25
    finally:
        task.close()


def test_search_promotion_recomputes_guarded_cost_and_independent_replay(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    commands = np.array([[-0.1, 0.2], [0.1, 0.6]])
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                provisional = search.predict(observation, commands)
                poisoned = replace(
                    provisional,
                    integration_states=np.zeros_like(provisional.integration_states),
                )
                admitted = []

                def admit(history: Any) -> bool:
                    admitted.append(history.integration_states.copy())
                    return True

                plan = search.promote(
                    observation,
                    poisoned.applied_actuator_commands,
                    objective=lambda history: float(
                        np.sum(history.integration_states**2)
                    ),
                    admit=admit,
                )
                assert plan.objective > 0
                np.testing.assert_array_equal(
                    plan.history.integration_states, provisional.integration_states
                )
                np.testing.assert_array_equal(
                    admitted[0], provisional.integration_states
                )
                assert plan.replay_artifact.producer_history is plan.history
                assert plan.validation_seconds > 0
                assert task.data.time == 1.25
    finally:
        task.close()


def test_search_promotion_refuses_failed_hard_criteria(tmp_path: Path) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                with pytest.raises(ValueError, match="hard"):
                    search.promote(
                        observation,
                        np.full((2, 2), 0.2),
                        objective=lambda history: 0.0,
                        admit=lambda history: False,
                    )
                assert task.data.time == 1.25
    finally:
        task.close()


def test_search_detects_deferred_conversion_live_mutation(tmp_path: Path) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)

    class Commands:
        def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
            task.data.act[:] += 0.1
            return np.full((2, 2), 0.2)

    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                with pytest.raises(ValueError, match="live plant"):
                    search.predict(observation, Commands())
                assert task.data.time == 1.25
    finally:
        task.close()


@pytest.mark.parametrize("callback", ["admit", "objective"])
def test_promotion_callbacks_cannot_reenable_history_writes(
    tmp_path: Path, callback: str
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)

    def mutate(history: Any) -> Any:
        history.applied_actuator_commands.setflags(write=True)
        return True if callback == "admit" else 0.0

    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                with pytest.raises(ValueError, match="WRITEABLE"):
                    search.promote(
                        observation,
                        np.full((2, 2), 0.2),
                        objective=mutate if callback == "objective" else lambda h: 0.0,
                        admit=mutate if callback == "admit" else lambda h: True,
                    )
                search.predict(observation, np.full((2, 2), 0.2))
                assert task.data.time == 1.25
    finally:
        task.close()


@pytest.mark.parametrize("rows", [0, 21])
def test_search_rejects_out_of_budget_horizon_without_owned_step(
    tmp_path: Path, rows: int
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                before = _integration_state(mj, search._model, search._data)
                with pytest.raises(ValueError, match="horizon"):
                    search.predict(observation, np.full((rows, 2), 0.2))
                np.testing.assert_array_equal(
                    _integration_state(mj, search._model, search._data), before
                )
    finally:
        task.close()


def test_search_closure_is_idempotent_and_rejects_further_use(tmp_path: Path) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    try:
        with ProjectTaskForecaster(task) as forecast:
            search = _search(module, forecast, observation)
            search.close()
            search.close()
            with pytest.raises(RuntimeError, match="closed"):
                search.predict(observation, np.full((2, 2), 0.2))
            forecast.predict(observation, np.full((2, 2), 0.2))
    finally:
        task.close()


@pytest.mark.parametrize("field", ["qfrc_applied", "xfrc_applied"])
def test_search_refuses_unsupported_live_load_before_owned_restore(
    tmp_path: Path, field: str
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                getattr(task.data, field).flat[0] = 0.1
                before = _integration_state(mj, search._model, search._data)
                with pytest.raises(ValueError):
                    search.predict(_observation(task), np.full((2, 2), 0.2))
                np.testing.assert_array_equal(
                    _integration_state(mj, search._model, search._data), before
                )
                assert getattr(task.data, field).flat[0] == 0.1
    finally:
        task.close()


def test_search_owned_model_mutation_is_rejected(tmp_path: Path) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                search._model.body_mass[1] *= 2
                with pytest.raises(ValueError, match="model changed"):
                    search.predict(observation, np.full((2, 2), 0.2))
                assert task.data.time == 1.25
    finally:
        task.close()


def test_search_reentrant_promotion_is_rejected(tmp_path: Path) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    commands = np.full((2, 2), 0.2)
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:

                def admit(history: Any) -> bool:
                    search.predict(observation, commands)
                    return True

                with pytest.raises(RuntimeError, match="busy"):
                    search.promote(
                        observation, commands, objective=lambda h: 0.0, admit=admit
                    )
                search.predict(observation, commands)
    finally:
        task.close()


@pytest.mark.parametrize("cost", [float("nan"), float("inf")])
def test_promotion_rejects_nonfinite_objective(tmp_path: Path, cost: float) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                with pytest.raises(ValueError, match="finite"):
                    search.promote(
                        observation,
                        np.full((2, 2), 0.2),
                        objective=lambda h: cost,
                        admit=lambda h: True,
                    )
    finally:
        task.close()


def test_search_recovers_after_partial_owned_native_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    commands = np.array([[-0.1, 0.2], [0.1, 0.6], [-0.2, 0.3]])
    live_before = _integration_state(mj, task.model, task.data)
    try:
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                expected = search.predict(observation, commands)
                original = module.direct._require_no_native_warnings
                stepped_times = []

                def reject_after_native_step(data: Any) -> None:
                    stepped_times.append(float(data.time))
                    raise RuntimeError("injected failure after real native step")

                monkeypatch.setattr(
                    module.direct,
                    "_require_no_native_warnings",
                    reject_after_native_step,
                )
                with pytest.raises(RuntimeError, match="after real native step"):
                    search.predict(observation, commands)
                assert stepped_times and stepped_times[0] > 1.25
                monkeypatch.setattr(
                    module.direct, "_require_no_native_warnings", original
                )
                recovered = search.predict(observation, commands)
                guarded = forecast.predict(observation, commands).history
                np.testing.assert_array_equal(
                    recovered.integration_states, expected.integration_states
                )
                np.testing.assert_array_equal(
                    recovered.integration_states, guarded.integration_states
                )
                np.testing.assert_array_equal(
                    _integration_state(mj, task.model, task.data), live_before
                )
    finally:
        task.close()


@pytest.mark.parametrize("variant", ["driver", "iron"])
def test_original_production_search_matches_guarded_full_twenty_step_plan(
    variant: str,
) -> None:
    mj = _native_runtime()
    module = _module()
    root = os.environ.get("FEEDBACK_MYOSUITE_PRODUCTION_ROOT")
    if not root:
        pytest.skip("production search requires retained complete resource root")
    from src.engines.myosuite_project_task_producer import (
        MyoSuiteSdkBinding,
        create_project_golf_task,
    )

    path = Path(root) / "golf" / "body" / f"golfer_myobody_{variant}.xml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _PRODUCTION_SHA256[variant]
    task = create_project_golf_task(
        path,
        MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256),
        resource_root=Path(root),
    )
    try:
        task.data.time = 1.25
        task.data.act[:] = np.linspace(0.02, 0.08, task.model.na)
        task.data.qvel[:] = np.linspace(-1e-4, 1e-4, task.model.nv)
        mj.mj_forward(task.model, task.data)
        task.data.qacc_warmstart[:] = np.linspace(-0.01, 0.01, task.model.nv)
        observation = _observation(task)
        initial = _integration_state(mj, task.model, task.data)
        commands = np.full((20, task.model.nu), 0.05)
        commands[:, :14] = 0.001
        commands[10:, 14:] = 0.06
        with ProjectTaskForecaster(task) as forecast:
            seed = forecast.predict(observation, commands[:2]).history
            with module.ProjectTaskNativeSearch(
                forecast,
                seed,
                model_id="myosuite-golfer",
                variant_id=variant,
                experiment_id=f"opaque:native-search-{variant}",
                max_steps=20,
            ) as search:
                first = search.predict(observation, commands)
                changed = commands.copy()
                changed[:, 0] += 0.001
                search.predict(observation, changed)
                repeated = search.predict(observation, commands)
                plan = search.promote(
                    observation,
                    commands,
                    objective=lambda h: float(np.sum(h.applied_actuator_commands**2)),
                    admit=lambda h: bool(np.isfinite(h.integration_states).all()),
                )
                np.testing.assert_array_equal(
                    first.integration_states, repeated.integration_states
                )
                np.testing.assert_array_equal(
                    first.integration_states, plan.history.integration_states
                )
                np.testing.assert_array_equal(first.applied_actuator_commands, commands)
                np.testing.assert_array_equal(
                    first.time_seconds, plan.history.time_seconds - 1.25
                )
        np.testing.assert_array_equal(
            _integration_state(mj, task.model, task.data), initial
        )
    finally:
        task.close()
