"""State-feedback admission; native snapshots alone do not qualify an SDK run."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from .test_project_task_producer import (
    _AUDITED_BASE_SHA256,
    _MIXED_XML,
    _fingerprint_native_runtime,
    _integration_state,
    _native_runtime,
)

pytestmark = pytest.mark.unit


def _feedback_module() -> Any:
    from src.engines import myosuite_project_feedback

    return myosuite_project_feedback


def _policy(module: Any) -> Any:
    source = Path(__file__).resolve()
    return module.DeclaredFeedbackPolicy(
        name="fixture-state-feedback",
        version="1.0.0",
        source_path=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        parameters_sha256=hashlib.sha256(b"gain=0.1;activation=0.2").hexdigest(),
    )


def test_native_observation_owns_immutable_complete_state() -> None:
    mj = _fingerprint_native_runtime()
    module = _feedback_module()
    model = mj.MjModel.from_xml_string(_MIXED_XML)
    data = mj.MjData(model)
    data.qpos[:] = 0.25
    data.qvel[:] = -0.5
    data.act[:] = 0.2
    data.ctrl[:] = [0.3, 0.4]
    data.time = 1.25
    mj.mj_forward(model, data)
    initial = _integration_state(mj, model, data)

    observation = module.snapshot_project_task_state(model, data, sample_index=3)

    assert observation.sample_index == 3
    assert observation.native_time_seconds == 1.25
    assert observation.ordered_actuator_ids == ("motor", "activation")
    np.testing.assert_array_equal(observation.integration_state, initial)
    for field in ("qpos", "qvel", "activation", "control", "integration_state"):
        values = getattr(observation, field)
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)
    data.qpos[:] = 0.75
    assert observation.qpos[0] == 0.25
    assert not hasattr(observation, "model")
    assert not hasattr(observation, "data")


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_feedback_requires_positive_integer_budget_before_sdk_access(
    steps: Any,
) -> None:
    module = _feedback_module()
    with pytest.raises(ValueError, match="positive integer"):
        module.record_project_task_feedback(
            None, lambda observation: np.zeros(2), steps=steps, policy=_policy(module)
        )


def test_real_sdk_feedback_changes_commands_and_independently_replays(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _feedback_module()
    from src.engines.myosuite_project_task_producer import (
        MyoSuiteSdkBinding,
        create_project_golf_task,
    )
    from src.engines.project_task_replay_artifacts import freeze_project_task_history

    source = tmp_path / "mixed.xml"
    source.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        source, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    task.data.qvel[:] = 0.5
    mj.mj_forward(task.model, task.data)
    observations = []
    policy_live = True

    def controller(observation: Any) -> np.ndarray:
        assert policy_live, "independent replay reused feedback"
        observations.append(observation)
        return np.array([-0.1 * observation.qvel[0], 0.2])

    try:
        recording = module.record_project_task_feedback(
            task, controller, steps=4, policy=_policy(module)
        )
        history = recording.history
        assert len(observations) == 4
        assert history.integration_states.shape[0] == 5
        assert len(np.unique(history.applied_actuator_commands[:, 0])) == 4
        for index, observation in enumerate(observations):
            np.testing.assert_array_equal(
                observation.integration_state, history.integration_states[index]
            )
            assert history.applied_actuator_commands[index, 0] == (
                -0.1 * observation.qvel[0]
            )
        assert recording.policy == _policy(module)
        assert (
            recording.applied_command_array_sha256
            == hashlib.sha256(history.applied_actuator_commands.tobytes()).hexdigest()
        )
        artifact = freeze_project_task_history(
            # Policy sidecar array hashes are distinct from canonical T01 hashes.
            task,
            history,
            resource_root=source.parent,
            resources=task.project_model_source.resources,
            model_id="feedback-fixture",
            variant_id="mixed",
            experiment_id="closed-loop-to-open-loop",
        )
        assert artifact.producer_history is history
        assert (
            recording.initial_state_array_sha256
            == hashlib.sha256(history.integration_states[0].tobytes()).hexdigest()
        )
        assert (
            recording.feedback_adapter_source_sha256
            == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        )
        policy_live = False
        from src.engines.native_direct_model_provider import create_native_direct_model
        from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
            replay_direct_model_actuator_commands,
        )

        task.step = lambda _: (_ for _ in ()).throw(AssertionError("SDK reused"))
        replay = replay_direct_model_actuator_commands(
            artifact.bundle,
            artifact.registration,
            create_native_direct_model,
            compiled_profile_bytes=artifact.compiled_profile_bytes,
        )
        np.testing.assert_array_equal(
            replay.integration_states, history.integration_states
        )
    finally:
        task.close()


def test_real_sdk_controller_exception_preserves_only_completed_step(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    module = _feedback_module()
    from src.engines.myosuite_project_task_producer import (
        MyoSuiteSdkBinding,
        create_project_golf_task,
    )

    source = tmp_path / "mixed.xml"
    source.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        source, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    calls = []

    def controller(observation: Any) -> np.ndarray:
        calls.append(observation.sample_index)
        if len(calls) == 2:
            raise RuntimeError("controller stopped")
        return np.array([0.1, 0.2])

    try:
        with pytest.raises(RuntimeError, match="controller stopped"):
            module.record_project_task_feedback(
                task, controller, steps=4, policy=_policy(module)
            )
        assert calls == [0, 1]
        assert task.data.time == float(task.model.opt.timestep)
        np.testing.assert_array_equal(task.data.ctrl, [0.1, 0.2])
    finally:
        task.close()


@pytest.mark.parametrize(
    "mutation", ["state", "model", "callback", "nonfinite", "shape", "deferred_state"]
)
def test_real_sdk_rejects_controller_plant_mutation_or_bad_action_before_step(
    tmp_path: Path,
    mutation: str,
) -> None:
    mj = _native_runtime()
    module = _feedback_module()
    from src.engines.myosuite_project_task_producer import (
        MyoSuiteSdkBinding,
        create_project_golf_task,
    )

    source = tmp_path / "mixed.xml"
    source.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        source, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    before_time = float(task.data.time)

    def controller(observation: Any) -> Any:
        if mutation == "state":
            task.data.qpos[:] += 0.1
        elif mutation == "model":
            task.model.body_mass[1] += 1.0
        elif mutation == "callback":
            mj.set_mjcb_control(lambda model, data: None)
        elif mutation == "nonfinite":
            return np.array([np.nan, 0.2])
        elif mutation == "shape":
            return np.array([0.2])
        elif mutation == "deferred_state":

            class DeferredAction:
                def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
                    task.data.qpos[:] += 0.1
                    return np.asarray([0.1, 0.2], dtype=dtype)

            return DeferredAction()
        return np.array([0.1, 0.2])

    try:
        with pytest.raises(ValueError):
            module.record_project_task_feedback(
                task, controller, steps=2, policy=_policy(module)
            )
        assert float(task.data.time) == before_time
    finally:
        mj.set_mjcb_control(None)
        task.close()
