"""Frozen native manifold costs must agree across search and guarded histories."""

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
from .test_project_task_native_search import _search
from .test_project_task_producer import (
    _AUDITED_BASE_SHA256,
    _MIXED_XML,
    _PRODUCTION_SHA256,
    _integration_state,
    _native_runtime,
)

pytestmark = pytest.mark.unit


def _module() -> Any:
    from src.engines import myosuite_project_tracking

    return myosuite_project_tracking


def _inputs(module: Any, observation: Any, steps: int = 2) -> tuple[Any, Any]:
    reference = module.NativeTrackingReference(
        observation.native_time_seconds + np.arange(steps + 1) * 0.001,
        np.tile(observation.qpos, (steps + 1, 1)),
        np.tile(observation.qvel, (steps + 1, 1)),
        np.tile(observation.activation, (steps + 1, 1)),
        hashlib.sha256(observation.integration_state.tobytes()).hexdigest(),
    )
    scales = module.NativeTrackingScales(
        np.full(len(observation.qvel), 0.5),
        np.full(len(observation.qvel), 2.0),
        np.full(len(observation.activation), 0.25),
        np.array([0.1, 0.2]),
    )
    return reference, scales


def test_scaled_objective_matches_hand_computed_terms_and_guarded_promotion(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    from src.engines import myosuite_project_native_search as native_search

    task = _task(tmp_path, mj)
    observation = _observation(task)
    initial = _integration_state(mj, task.model, task.data)
    commands = np.array([[0.1, 0.2], [0.2, 0.3]])
    reference, scales = _inputs(module, observation)
    try:
        with ProjectTaskForecaster(task) as forecast:
            objective = module.ProjectTaskTrackingObjective(
                forecast, observation, reference, scales
            )
            with _search(native_search, forecast, observation) as search:
                prediction = search.predict(observation, commands)
                value = objective.prediction_cost(prediction)
                plan = search.promote(
                    observation,
                    commands,
                    objective=lambda h: objective.history_cost(h).total,
                    admit=lambda h: bool(np.isfinite(h.integration_states).all()),
                )
                assert plan.objective == value.total
                position = float(np.sum(((prediction.qpos - 0.2) / 0.5) ** 2))
                velocity = float(np.sum(((prediction.qvel - 0.5) / 2.0) ** 2))
                activation = float(np.sum(((prediction.activation - 0.3) / 0.25) ** 2))
                # First control is unchanged; second motor/muscle increments .1.
                slew = 1.0 + 0.25
                assert value.position == pytest.approx(position)
                assert value.velocity == pytest.approx(velocity)
                assert value.activation == pytest.approx(activation)
                assert value.command_slew == pytest.approx(slew)
                assert value.total == pytest.approx(
                    position + velocity + activation + slew
                )
                assert objective.position_units == ("rad",)
                assert (
                    objective.source_sha256
                    == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                )
        np.testing.assert_array_equal(
            _integration_state(mj, task.model, task.data), initial
        )
    finally:
        task.close()


@pytest.mark.parametrize("changed", ["source", "producer", "anchor", "parameters"])
def test_tracking_rejects_changed_lineage_or_solve_anchor(
    tmp_path: Path, changed: str
) -> None:
    mj = _native_runtime()
    module = _module()
    from src.engines import myosuite_project_native_search as native_search

    task = _task(tmp_path, mj)
    observation = _observation(task)
    reference, scales = _inputs(module, observation)
    try:
        with ProjectTaskForecaster(task) as forecast:
            objective = module.ProjectTaskTrackingObjective(
                forecast, observation, reference, scales
            )
            with _search(native_search, forecast, observation) as search:
                prediction = search.predict(observation, np.full((2, 2), 0.2))
                history = forecast.predict(observation, np.full((2, 2), 0.2)).history
                if changed == "source":
                    from src.engines.native_replay_contracts import (
                        native_replay_contract_types,
                    )

                    contracts = native_replay_contract_types()
                    bundle = prediction.planned_bundle
                    history_inputs = bundle.input_history
                    prediction = replace(
                        prediction,
                        planned_bundle=contracts.build_experiment_replay_bundle(
                            experiment_id=bundle.experiment_id,
                            model=replace(bundle.model, source_model_sha256="a" * 64),
                            capabilities=bundle.capabilities,
                            initial_state_values=tuple(
                                (v.component_id, v.values) for v in bundle.initial_state
                            ),
                            channels=history_inputs.channels,
                            input_kind=history_inputs.input_kind,
                            interpolation=history_inputs.interpolation,
                            time_seconds=history_inputs.time_seconds,
                            input_values=history_inputs.values,
                            policy=bundle.policy,
                        ),
                    )
                elif changed == "producer":
                    history = replace(history, project_task_source_sha256="a" * 64)
                elif changed == "anchor":
                    task.data.qacc_warmstart[0] += 0.01
                else:
                    objective._scales = replace(scales, position=scales.position * 2)
                with pytest.raises(ValueError):
                    if changed == "producer":
                        objective.history_cost(history)
                    else:
                        objective.prediction_cost(prediction)
    finally:
        task.close()


def test_reference_and_scales_are_immutable_and_bind_objective_parameters(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    reference, scales = _inputs(module, observation)
    try:
        with pytest.raises(ValueError):
            reference.qpos.setflags(write=True)
        with pytest.raises(ValueError):
            scales.position.setflags(write=True)
        with ProjectTaskForecaster(task) as forecast:
            first = module.ProjectTaskTrackingObjective(
                forecast, observation, reference, scales
            )
            second = module.ProjectTaskTrackingObjective(
                forecast,
                observation,
                reference,
                replace(scales, position=scales.position * 2),
            )
            assert first.parameters_sha256 != second.parameters_sha256
            assert len(first.parameters_sha256) == 64
    finally:
        task.close()


@pytest.mark.parametrize("invalid", [0.0, float("nan"), float("inf")])
def test_tracking_refuses_invalid_physical_scale(
    tmp_path: Path, invalid: float
) -> None:
    _native_runtime()
    module = _module()
    with pytest.raises(ValueError):
        module.NativeTrackingScales(
            np.array([invalid]), np.ones(1), np.ones(1), np.ones(2)
        )


def test_tracking_refuses_reference_clock_mismatch(tmp_path: Path) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    observation = _observation(task)
    reference, scales = _inputs(module, observation)
    try:
        with ProjectTaskForecaster(task) as forecast:
            with pytest.raises(ValueError, match="clock"):
                module.ProjectTaskTrackingObjective(
                    forecast,
                    observation,
                    replace(reference, time_seconds=reference.time_seconds + 0.01),
                    scales,
                )
    finally:
        task.close()


@pytest.mark.parametrize("changed", ["order", "initial", "horizon"])
def test_tracking_refuses_changed_prediction_binding(
    tmp_path: Path, changed: str
) -> None:
    mj = _native_runtime()
    module = _module()
    from src.engines import myosuite_project_native_search as native_search

    task = _task(tmp_path, mj)
    observation = _observation(task)
    reference, scales = _inputs(module, observation)
    try:
        with ProjectTaskForecaster(task) as forecast:
            objective = module.ProjectTaskTrackingObjective(
                forecast, observation, reference, scales
            )
            with _search(native_search, forecast, observation) as search:
                prediction = search.predict(observation, np.full((2, 2), 0.2))
                if changed == "order":
                    prediction = replace(
                        prediction,
                        ordered_actuator_ids=prediction.ordered_actuator_ids[::-1],
                    )
                elif changed == "initial":
                    states = prediction.integration_states.copy()
                    states[0, -1] += 0.1
                    prediction = replace(prediction, integration_states=states)
                else:
                    prediction = replace(
                        prediction,
                        integration_states=prediction.integration_states[:-1],
                    )
                with pytest.raises(ValueError):
                    objective.prediction_cost(prediction)
    finally:
        task.close()


def test_native_quaternion_antipodes_have_equal_tracking_cost(tmp_path: Path) -> None:
    mj = _native_runtime()
    module = _module()
    from src.engines.myosuite_project_task_producer import (
        MyoSuiteSdkBinding,
        create_project_golf_task,
    )

    path = tmp_path / "free-mixed.xml"
    xml = _MIXED_XML.replace(
        '<worldbody><body><joint name="hinge"/>',
        '<worldbody><body><freejoint/><geom type="sphere" size=".1" mass="2"/>'
        '<body pos="0 0 .2"><joint name="hinge"/>',
    ).replace("</body></worldbody>", "</body></body></worldbody>")
    path.write_text(xml, encoding="utf-8")
    task = create_project_golf_task(
        path, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    try:
        task.data.time = 1.25
        mj.mj_forward(task.model, task.data)
        observation = _observation(task)
        reference, scales = _inputs(module, observation)
        flipped = reference.qpos.copy()
        flipped[:, 3:7] *= -1
        with ProjectTaskForecaster(task) as forecast:
            history = forecast.predict(observation, np.full((2, 2), 0.2)).history
            first = module.ProjectTaskTrackingObjective(
                forecast, observation, reference, scales
            )
            second = module.ProjectTaskTrackingObjective(
                forecast, observation, replace(reference, qpos=flipped), scales
            )
            assert first.history_cost(history).total == pytest.approx(
                second.history_cost(history).total, abs=1e-14
            )
            assert first.position_units[:6] == ("m", "m", "m", "rad", "rad", "rad")
    finally:
        task.close()


@pytest.mark.parametrize("variant", ["driver", "iron"])
def test_original_golfer_twenty_step_tracking_agrees_with_guarded_replay(
    variant: str,
) -> None:
    mj = _native_runtime()
    root = os.environ.get("FEEDBACK_MYOSUITE_PRODUCTION_ROOT")
    if not root:
        pytest.skip("tracking requires retained original production resources")
    module = _module()
    from src.engines import myosuite_project_native_search as native_search
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
        mj.mj_forward(task.model, task.data)
        task.data.qacc_warmstart[:] = np.linspace(-0.01, 0.01, task.model.nv)
        observation = _observation(task)
        initial = _integration_state(mj, task.model, task.data)
        steps = 20
        reference = module.NativeTrackingReference(
            observation.native_time_seconds + np.arange(steps + 1) * 0.001,
            np.tile(observation.qpos, (steps + 1, 1)),
            np.tile(observation.qvel, (steps + 1, 1)),
            np.tile(observation.activation, (steps + 1, 1)),
            hashlib.sha256(initial.tobytes()).hexdigest(),
        )
        scales = module.NativeTrackingScales(
            np.full(task.model.nv, 0.5),
            np.full(task.model.nv, 2.0),
            np.full(task.model.na, 0.25),
            np.full(task.model.nu, 0.1),
        )
        commands = np.full((steps, task.model.nu), 0.05)
        commands[:, :14] = 0.001
        with ProjectTaskForecaster(task) as forecast:
            objective = module.ProjectTaskTrackingObjective(
                forecast, observation, reference, scales
            )
            seed = forecast.predict(observation, commands[:2]).history
            with native_search.ProjectTaskNativeSearch(
                forecast,
                seed,
                model_id="myosuite-golfer",
                variant_id=variant,
                experiment_id=f"opaque:tracking-{variant}",
                max_steps=steps,
            ) as search:
                prediction = search.predict(observation, commands)
                cost = objective.prediction_cost(prediction)
                plan = search.promote(
                    observation,
                    commands,
                    objective=lambda h: objective.history_cost(h).total,
                    admit=lambda h: bool(np.isfinite(h.integration_states).all()),
                )
                assert plan.objective == cost.total
                assert np.isfinite(cost.total)
                np.testing.assert_array_equal(
                    plan.history.integration_states, prediction.integration_states
                )
        np.testing.assert_array_equal(
            _integration_state(mj, task.model, task.data), initial
        )
    finally:
        task.close()
