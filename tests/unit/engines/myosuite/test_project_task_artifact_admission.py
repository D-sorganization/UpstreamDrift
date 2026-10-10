"""Adversarial admission checks for actual SDK-produced native replay artifacts."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import pytest

from src.engines.myosuite_project_task_producer import (
    MyoSuiteSdkBinding,
    ProjectTaskCommandHistory,
    create_project_golf_task,
    record_project_task_commands,
)
from src.engines.project_task_replay_artifacts import freeze_project_task_history

if TYPE_CHECKING:
    from src.engines.project_task_replay_artifacts import FrozenProjectTaskReplay

_RecordedTask = tuple[Any, ProjectTaskCommandHistory, Path, tuple[Any, ...], Any]

pytestmark = pytest.mark.unit
_AUDITED_BASE_SHA256 = (
    "53ad8bc71fc07a2e7fcead164ca027a33418b9ac1aa953ea6c4042fa48eed154"
)
_MIXED_XML = """<mujoco model="artifact-admission">
  <option timestep="0.001" integrator="RK4"/>
  <worldbody><body name="load"><joint name="hinge"/>
    <geom type="sphere" size="0.05" mass="1"/></body></worldbody>
  <actuator><motor name="direct_motor" joint="hinge"/>
    <general name="limited_activation" joint="hinge" dyntype="filter"
      dynprm="0.02" ctrllimited="true" ctrlrange="0 1"/></actuator>
</mujoco>"""


@pytest.fixture
def recorded_task(tmp_path: Path) -> Iterator[_RecordedTask]:
    pytest.importorskip("myosuite")
    mj = pytest.importorskip("mujoco")
    if mj.__version__ != "3.6.0":
        pytest.skip("reviewed SDK lane requires actual MuJoCo 3.6.0")
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        DeclaredModelResource,
    )

    path = tmp_path / "mixed.xml"
    path.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        path, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    task.data.act[:] = 0.2
    mj.mj_forward(task.model, task.data)
    history = record_project_task_commands(
        task, np.array([[0.001, 0.2], [-0.002, 0.1]])
    )
    resources = (
        DeclaredModelResource(
            "mixed.xml", hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    try:
        yield task, history, path, resources, mj
    finally:
        task.close()


def _freeze(
    recorded_task: _RecordedTask,
    *,
    history: ProjectTaskCommandHistory | None = None,
    resources: tuple[Any, ...] | None = None,
) -> FrozenProjectTaskReplay:
    task, original, path, declared, _ = recorded_task
    return freeze_project_task_history(
        task,
        original if history is None else history,
        resource_root=path.parent,
        resources=declared if resources is None else resources,
        model_id="synthetic-mixed",
        variant_id="admission-fixture",
        experiment_id="opaque:adversarial-admission",
    )


def test_admitted_history_replays_without_sdk_lifecycle(
    recorded_task: _RecordedTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    from myosuite.envs.gymnasium_env import MyoGymnasiumEnv
    from src.engines.native_direct_model_provider import create_native_direct_model
    from src.engines.native_replay_contracts import native_replay_contract_types
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        replay_direct_model_actuator_commands,
    )

    artifact = _freeze(recorded_task)
    task, history, _, _, _ = recorded_task
    contracts = native_replay_contract_types()
    reloaded = contracts.load_experiment_replay_bundle(
        contracts.dumps_experiment_replay_bundle(artifact.bundle)
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("independent replay invoked an SDK lifecycle hook")

    with monkeypatch.context() as patch:
        for name in ("reset", "step", "_step_physics", "get_reward_dict"):
            patch.setattr(MyoGymnasiumEnv, name, forbidden)
        replay = replay_direct_model_actuator_commands(
            reloaded,
            artifact.registration,
            create_native_direct_model,
            compiled_profile_bytes=artifact.compiled_profile_bytes,
        )
    np.testing.assert_array_equal(replay.integration_states, history.integration_states)
    assert task.data.time == history.time_seconds[-1]


def test_independent_replay_rejects_permuted_compiled_channel_order(
    recorded_task: _RecordedTask,
) -> None:
    from src.engines.native_direct_model_provider import create_native_direct_model
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        replay_direct_model_actuator_commands,
    )

    artifact = _freeze(recorded_task)
    assert artifact.registration.ordered_channel_ids == (
        "command:direct_motor",
        "command:limited_activation",
    )
    swapped = replace(
        artifact.registration,
        ordered_channel_ids=tuple(reversed(artifact.registration.ordered_channel_ids)),
    )
    with pytest.raises(ValueError, match="channel|order|identity"):
        replay_direct_model_actuator_commands(
            artifact.bundle,
            swapped,
            create_native_direct_model,
            compiled_profile_bytes=artifact.compiled_profile_bytes,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "clock_shape",
        "state_shape",
        "command_shape",
        "nan_state",
        "nan_command",
        "source",
        "loaded",
    ],
)
def test_freezer_rejects_malformed_or_foreign_producer_history(
    recorded_task: _RecordedTask, damage: str
) -> None:
    _, history, _, _, _ = recorded_task
    changes: dict[str, object]
    if damage == "clock_shape":
        changes = {"time_seconds": history.time_seconds[:-1].copy()}
    elif damage == "state_shape":
        changes = {"integration_states": history.integration_states[:, :-1].copy()}
    elif damage == "command_shape":
        changes = {
            "applied_actuator_commands": history.applied_actuator_commands[
                :, :-1
            ].copy()
        }
    elif damage == "nan_state":
        states = history.integration_states.copy()
        states[1, 1] = np.nan
        changes = {"integration_states": states}
    elif damage == "nan_command":
        commands = history.applied_actuator_commands.copy()
        commands[0, 1] = np.nan
        changes = {"applied_actuator_commands": commands}
    elif damage == "source":
        changes = {"project_task_source_sha256": "0" * 64}
    else:
        changes = {"loaded_native_model_sha256": "0" * 64}
    corrupt = replace(history, **changes)
    with pytest.raises((ValueError, TypeError)):
        _freeze(recorded_task, history=corrupt)


def test_freezer_binds_native_state_clock_to_declared_history_clock(
    recorded_task: _RecordedTask,
) -> None:
    task, history, _, _, mj = recorded_task
    states = history.integration_states.copy()
    state = mj.MjData(task.model)
    mj.mj_setState(task.model, state, states[1], mj.mjtState.mjSTATE_INTEGRATION)
    state.time += 0.25
    mj.mj_getState(task.model, state, states[1], mj.mjtState.mjSTATE_INTEGRATION)
    corrupt = replace(history, integration_states=states)
    with pytest.raises(ValueError, match="clock|time|state"):
        _freeze(recorded_task, history=corrupt)


def test_freezer_rejects_commands_outside_compiled_muscle_control_range(
    recorded_task: _RecordedTask,
) -> None:
    _, history, _, _, _ = recorded_task
    commands = history.applied_actuator_commands.copy()
    commands[0, 1] = 1.2
    with pytest.raises(ValueError, match="control|command|history"):
        _freeze(
            recorded_task,
            history=replace(history, applied_actuator_commands=commands),
        )


def test_freezer_rejects_unexecuted_finite_state_transition(
    recorded_task: _RecordedTask,
) -> None:
    _, history, _, _, _ = recorded_task
    states = history.integration_states.copy()
    states[1, 1] += 0.01
    with pytest.raises(ValueError, match="transition|state|history"):
        _freeze(recorded_task, history=replace(history, integration_states=states))


def test_freezer_rejects_changed_source_or_resource_digest(
    recorded_task: _RecordedTask,
) -> None:
    _, history, path, resources, _ = recorded_task
    assert isinstance(history, ProjectTaskCommandHistory)
    with pytest.raises(ValueError, match="resource|source|closure|SHA"):
        _freeze(recorded_task, resources=(replace(resources[0], sha256="0" * 64),))
    path.write_text(
        _MIXED_XML.replace('timestep="0.001"', 'timestep="0.002"'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="resource|source|closure|SHA|compiled"):
        _freeze(recorded_task)


@pytest.mark.parametrize("annotation", ["<!-- changed origin -->", "  "])
def test_freezer_rejects_rehashed_source_bytes_after_sdk_execution(
    recorded_task: _RecordedTask, annotation: str
) -> None:
    from src.engines.myosuite_project_task_producer import _model_sha256

    _, history, path, resources, mj = recorded_task
    changed = _MIXED_XML.replace("</mujoco>", f"{annotation}</mujoco>")
    path.write_text(changed, encoding="utf-8")
    recompiled = mj.MjModel.from_xml_path(str(path))
    assert _model_sha256(mj, recompiled) == history.loaded_native_model_sha256
    refreshed = replace(
        resources[0], sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="source|origin|history|provenance"):
        _freeze(recorded_task, resources=(refreshed,))


def test_freezer_rejects_compiled_actuator_law_mutation(
    recorded_task: _RecordedTask,
) -> None:
    task, _, _, _, _ = recorded_task
    task.model.actuator_gainprm[0, 0] += 0.01
    with pytest.raises(ValueError, match="model|native|compiled|history"):
        _freeze(recorded_task)
