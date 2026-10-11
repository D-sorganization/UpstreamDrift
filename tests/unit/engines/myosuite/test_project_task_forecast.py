"""Owned native forecasts must not step, reset or substitute the live plant."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from .test_project_task_feedback import _policy
from .test_project_task_producer import (
    _AUDITED_BASE_SHA256,
    _MIXED_XML,
    _PRODUCTION_SHA256,
    _integration_state,
    _native_runtime,
)

pytestmark = pytest.mark.unit


def _module() -> Any:
    from src.engines import myosuite_project_forecast

    return myosuite_project_forecast


def _task(tmp_path: Path, mj: Any) -> Any:
    from src.engines.myosuite_project_task_producer import (
        MyoSuiteSdkBinding,
        create_project_golf_task,
    )

    source = tmp_path / "mixed.xml"
    source.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        source, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    task.data.time = 1.25
    task.data.qpos[:] = 0.2
    task.data.qvel[:] = 0.5
    task.data.act[:] = 0.3
    task.data.ctrl[:] = [0.1, 0.2]
    mj.mj_forward(task.model, task.data)
    return task


def _observation(task: Any) -> Any:
    from src.engines.myosuite_project_feedback import snapshot_project_task_state

    return snapshot_project_task_state(task.model, task.data, sample_index=0)


def test_native_forecasts_repeat_complete_state_without_changing_live_plant(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    initial = _integration_state(mj, task.model, task.data)
    observation = _observation(task)
    commands = np.array([[-0.1, 0.2], [0.1, 1.5], [-0.2, 0.0]])
    try:
        with module.ProjectTaskForecaster(task) as forecast:
            first = forecast.predict(observation, commands)
            # A different preceding candidate must not contaminate a repeated one.
            forecast.predict(observation, np.full((2, 2), 0.4))
            repeated = forecast.predict(observation, commands)
            np.testing.assert_array_equal(first.history.integration_states[0], initial)
            np.testing.assert_array_equal(
                first.history.integration_states, repeated.history.integration_states
            )
            np.testing.assert_array_equal(
                first.history.applied_actuator_commands,
                [[-0.1, 0.2], [0.1, 1.0], [-0.2, 0.0]],
            )
            assert first.history.time_seconds[0] == 1.25
            assert first.history.time_seconds[-1] > 1.25
            assert (
                first.forecast_adapter_source_sha256
                == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            )
            np.testing.assert_array_equal(
                _integration_state(mj, task.model, task.data), initial
            )
        with pytest.raises(RuntimeError, match="closed"):
            forecast.predict(observation, commands)
        # Closing a forecaster never closes or steps the live SDK task.
        task.step(np.array([0.0, 0.2]))
        assert task.data.time > 1.25
    finally:
        task.close()


def test_native_lookahead_feedback_exports_controller_off_replay(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    from src.engines import myosuite_project_feedback as feedback
    from src.engines.project_task_replay_artifacts import freeze_project_task_history

    task = _task(tmp_path, mj)
    live = True
    choices = []
    try:
        with module.ProjectTaskForecaster(task) as forecast:

            def controller(observation: Any) -> np.ndarray:
                assert live, "independent replay called the controller"
                candidates = (
                    np.tile([-0.1, 0.2], (2, 1)),
                    np.tile([0.1, 0.2], (2, 1)),
                )
                results = [forecast.predict(observation, value) for value in candidates]
                # The candidate with lower motor command predicts a lower hinge position.
                assert (
                    results[0].history.integration_states[-1, 1]
                    < (results[1].history.integration_states[-1, 1])
                )
                choices.append(results[0].forecast_adapter_source_sha256)
                return candidates[0][0]

            source = Path(__file__).resolve()
            policy = replace(
                _policy(feedback),
                source_path=source,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            recording = feedback.record_project_task_feedback(
                task, controller, steps=3, policy=policy
            )
        assert len(choices) == 3
        np.testing.assert_array_equal(
            recording.history.applied_actuator_commands,
            np.tile([-0.1, 0.2], (3, 1)),
        )
        assert recording.history.integration_states.shape[0] == 4
        live = False
        artifact = freeze_project_task_history(
            task,
            recording.history,
            resource_root=tmp_path,
            resources=task.project_model_source.resources,
            model_id="fixture",
            variant_id="lookahead",
            experiment_id="owned-forecast-feedback",
        )
        assert artifact.producer_history is recording.history
    finally:
        task.close()


@pytest.mark.parametrize(
    "field", ["qpos", "activation", "native_time_seconds", "order"]
)
def test_native_forecast_rejects_inconsistent_observation(
    tmp_path: Path, field: str
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    if field in ("qpos", "activation"):
        observation = replace(observation, **{field: getattr(observation, field) + 1})
    elif field == "native_time_seconds":
        observation = replace(observation, native_time_seconds=3.0)
    else:
        observation = replace(observation, ordered_actuator_ids=("activation", "motor"))
    try:
        with module.ProjectTaskForecaster(task) as forecast:
            with pytest.raises(ValueError, match="observation"):
                forecast.predict(observation, np.zeros((2, 2)))
            assert task.data.time == 1.25
    finally:
        task.close()


def test_native_forecast_rejects_stale_complete_state(tmp_path: Path) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    task.data.qvel[:] += 0.1
    try:
        with module.ProjectTaskForecaster(task) as forecast:
            with pytest.raises(ValueError, match="observation"):
                forecast.predict(observation, np.zeros((2, 2)))
            assert task.data.time == 1.25
    finally:
        task.close()


@pytest.mark.parametrize("owner", ["live", "owned"])
def test_native_forecast_rejects_changed_model(tmp_path: Path, owner: str) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    try:
        with module.ProjectTaskForecaster(task) as forecast:
            changed = task if owner == "live" else forecast._candidate
            changed.model.body_mass[1] *= 2
            with pytest.raises(ValueError, match="model"):
                forecast.predict(observation, np.zeros((2, 2)))
            assert task.data.time == 1.25
    finally:
        task.close()


def test_native_forecast_failure_detects_conversion_side_effect(tmp_path: Path) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)

    class Commands:
        def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
            task.data.qvel[:] += 0.1
            return np.zeros((2, 2), dtype=np.float64)

    try:
        with module.ProjectTaskForecaster(task) as forecast:
            with pytest.raises(ValueError, match="live plant"):
                forecast.predict(observation, Commands())
            assert task.data.time == 1.25
    finally:
        task.close()


def test_native_forecast_restores_nonzero_warmstart_across_candidates(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    task.data.qacc_warmstart[:] = np.linspace(0.15, 0.35, task.model.nv)
    original = _integration_state(mj, task.model, task.data)
    observation = _observation(task)
    first_commands = np.array([[-0.1, 0.2], [0.1, 0.3], [-0.2, 0.4]])
    other_commands = np.array([[0.5, 0.6], [0.4, 0.7]])
    try:
        with module.ProjectTaskForecaster(task) as forecast:
            first = forecast.predict(observation, first_commands)
            forecast.predict(observation, other_commands)
            repeated = forecast.predict(observation, first_commands)
        with module.ProjectTaskForecaster(task) as fresh:
            independent = fresh.predict(observation, first_commands)
        for history in (first.history, repeated.history, independent.history):
            np.testing.assert_array_equal(history.integration_states[0], original)
        np.testing.assert_array_equal(
            first.history.integration_states, repeated.history.integration_states
        )
        np.testing.assert_array_equal(
            first.history.integration_states, independent.history.integration_states
        )
        np.testing.assert_array_equal(
            _integration_state(mj, task.model, task.data), original
        )
        np.testing.assert_array_equal(
            task.data.qacc_warmstart, np.linspace(0.15, 0.35, task.model.nv)
        )
    finally:
        task.close()


@pytest.mark.parametrize(
    "invalid",
    [np.array([[0.1, np.nan]]), np.array([[0.1, 0.2, 0.3]])],
    ids=["nonfinite", "wrong-channel-count"],
)
def test_failed_forecast_candidate_cannot_contaminate_next_prediction(
    tmp_path: Path, invalid: np.ndarray
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    original = _integration_state(mj, task.model, task.data)
    observation = _observation(task)
    commands = np.array([[0.1, 0.2], [-0.1, 0.3]])
    try:
        with module.ProjectTaskForecaster(task) as forecast:
            with pytest.raises(ValueError):
                forecast.predict(observation, invalid)
            after_failure = forecast.predict(observation, commands)
        with module.ProjectTaskForecaster(task) as fresh:
            independent = fresh.predict(observation, commands)
        np.testing.assert_array_equal(
            after_failure.history.integration_states,
            independent.history.integration_states,
        )
        np.testing.assert_array_equal(
            _integration_state(mj, task.model, task.data), original
        )
    finally:
        task.close()


def test_forecaster_close_is_idempotent_and_does_not_close_live_task(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    original = _integration_state(mj, task.model, task.data)
    observation = _observation(task)
    try:
        forecast = module.ProjectTaskForecaster(task)
        forecast.close()
        forecast.close()
        with pytest.raises(RuntimeError, match="closed"):
            forecast.predict(observation, np.zeros((1, task.model.nu)))
        np.testing.assert_array_equal(
            _integration_state(mj, task.model, task.data), original
        )
        task.step(np.array([0.0, 0.2]))
        assert task.data.time > 1.25
    finally:
        task.close()


def test_reentrant_forecast_is_rejected_without_touching_live_plant(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    original = _integration_state(mj, task.model, task.data)
    observation = _observation(task)
    commands = np.array([[0.1, 0.2], [-0.1, 0.3]])
    try:
        with module.ProjectTaskForecaster(task) as forecast:

            class ReentrantCommands:
                def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
                    with pytest.raises(RuntimeError, match="reentrant|concurrent|busy"):
                        forecast.predict(observation, commands)
                    return commands.copy()

            forecast.predict(observation, ReentrantCommands())
            np.testing.assert_array_equal(
                _integration_state(mj, task.model, task.data), original
            )
    finally:
        task.close()


def test_forecaster_constructor_closes_owned_sdk_task_after_identity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mj = _native_runtime()
    module = _module()
    live = _task(tmp_path, mj)
    original = _integration_state(mj, live.model, live.data)
    real_factory = module.create_project_golf_task
    owned_closed: list[bool] = []

    def changed_owned_factory(*args: Any, **kwargs: Any) -> Any:
        owned = real_factory(*args, **kwargs)
        assert owned is not live
        real_close = owned.close

        def record_close() -> None:
            owned_closed.append(True)
            real_close()

        owned.close = record_close
        owned.model.body_mass[1] *= 2
        return owned

    monkeypatch.setattr(module, "create_project_golf_task", changed_owned_factory)
    try:
        with pytest.raises(ValueError, match="owned source-identical model"):
            module.ProjectTaskForecaster(live)
        assert owned_closed == [True]
        np.testing.assert_array_equal(
            _integration_state(mj, live.model, live.data), original
        )
        live.step(np.array([0.0, 0.2]))
        assert live.data.time > 1.25
    finally:
        live.close()


@pytest.mark.parametrize("load_kind", ["qfrc_applied", "xfrc_applied"])
def test_forecast_rejects_live_external_load_before_owned_step(
    tmp_path: Path, load_kind: str
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    try:
        with module.ProjectTaskForecaster(task) as forecast:
            owned_before = _integration_state(
                mj, forecast._candidate.model, forecast._candidate.data
            )
            if load_kind == "qfrc_applied":
                task.data.qfrc_applied[0] = 0.5
            else:
                task.data.xfrc_applied[1, 0] = 0.5
            live_before = _integration_state(mj, task.model, task.data)
            observation = _observation(task)
            with pytest.raises(ValueError, match="external|applied"):
                forecast.predict(observation, np.array([[0.1, 0.2]]))
            np.testing.assert_array_equal(
                _integration_state(
                    mj, forecast._candidate.model, forecast._candidate.data
                ),
                owned_before,
            )
            np.testing.assert_array_equal(
                _integration_state(mj, task.model, task.data), live_before
            )
            assert np.any(getattr(task.data, load_kind))
    finally:
        task.close()


@pytest.mark.parametrize("variant", ["driver", "iron"])
def test_actual_production_suffix_forecast_replays_full_native_state(
    variant: str,
) -> None:
    mj = _native_runtime()
    root = os.environ.get("FEEDBACK_MYOSUITE_PRODUCTION_ROOT")
    if not root:
        pytest.skip("production forecast requires retained complete resource root")
    model_path = Path(root) / "golf" / "body" / f"golfer_myobody_{variant}.xml"
    assert (
        hashlib.sha256(model_path.read_bytes()).hexdigest()
        == _PRODUCTION_SHA256[variant]
    )
    from src.engines import myosuite_project_forecast as module
    from src.engines.myosuite_project_task_producer import (
        MyoSuiteSdkBinding,
        create_project_golf_task,
    )
    from src.engines.native_direct_model_provider import create_native_direct_model
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        replay_direct_model_actuator_commands,
    )
    from src.engines.project_task_replay_artifacts import freeze_project_task_history

    task = create_project_golf_task(
        model_path,
        MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256),
        resource_root=Path(root),
    )
    try:
        task.data.time = 1.25
        task.data.act[:] = np.linspace(0.02, 0.08, task.model.na)
        task.data.qvel[:] = np.linspace(-1e-4, 1e-4, task.model.nv)
        mj.mj_forward(task.model, task.data)
        seed_command = np.full(task.model.nu, 0.05)
        seed_command[:14] = 0.001
        task.step(seed_command)
        task.data.qacc_warmstart[:] = np.linspace(-0.01, 0.01, task.model.nv)
        suffix_initial = _integration_state(mj, task.model, task.data)
        observation = _observation(task)
        actions = np.full((2, task.model.nu), 0.05)
        actions[:, :14] = [[-0.001], [0.002]]
        actions[1, 14:] = 0.06
        with module.ProjectTaskForecaster(task) as forecast:
            history = forecast.predict(observation, actions).history
        np.testing.assert_array_equal(history.integration_states[0], suffix_initial)
        np.testing.assert_array_equal(
            _integration_state(mj, task.model, task.data), suffix_initial
        )
        assert history.time_seconds[0] > 1.25
        assert np.any(task.data.act)
        assert np.any(task.data.qacc_warmstart)
        artifact = freeze_project_task_history(
            task,
            history,
            resource_root=Path(root),
            resources=task.project_model_source.resources,
            model_id="myosuite-golfer",
            variant_id=variant,
            experiment_id=f"opaque:forecast-suffix-{variant}",
        )
        replay = replay_direct_model_actuator_commands(
            artifact.bundle,
            artifact.registration,
            create_native_direct_model,
            compiled_profile_bytes=artifact.compiled_profile_bytes,
        )
        np.testing.assert_array_equal(
            replay.integration_states, history.integration_states
        )
        np.testing.assert_array_equal(
            replay.applied_actuator_commands, history.applied_actuator_commands
        )
    finally:
        task.close()
