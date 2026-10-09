"""Project task contracts; portable bounds checks do not qualify native SDK use."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import hashlib
import os
import sys
from types import SimpleNamespace, FrameType
from typing import Any

import numpy as np
import pytest

from src.engines.myosuite_project_task_producer import (
    MyoSuiteSdkBinding,
    create_project_golf_task,
    native_action_bounds,
    record_project_task_commands,
)

pytestmark = pytest.mark.unit


def test_mixed_action_bounds_preserve_unlimited_motors_and_limited_muscles() -> None:
    model = SimpleNamespace(
        nu=3,
        actuator_ctrllimited=np.array([False, True, True]),
        actuator_ctrlrange=np.array([[0.0, 0.0], [0.0, 1.0], [-2.0, 3.0]]),
    )
    low, high = native_action_bounds(model)
    np.testing.assert_array_equal(low, [-np.inf, 0.0, -2.0])
    np.testing.assert_array_equal(high, [np.inf, 1.0, 3.0])
    np.testing.assert_array_equal(model.actuator_ctrlrange[0], [0.0, 0.0])


@pytest.mark.parametrize("limits", [[[1.0, 0.0]], [[np.nan, 1.0]], [[0.0, np.inf]]])
def test_limited_action_channels_require_finite_ordered_ranges(
    limits: list[list[float]],
) -> None:
    model = SimpleNamespace(
        nu=1, actuator_ctrllimited=np.array([True]), actuator_ctrlrange=np.array(limits)
    )
    with pytest.raises(ValueError, match="limited actuator"):
        native_action_bounds(model)


def test_action_bounds_reject_incomplete_compiled_channel_layout() -> None:
    model = SimpleNamespace(
        nu=2,
        actuator_ctrllimited=np.array([True]),
        actuator_ctrlrange=np.array([[0, 1]]),
    )
    with pytest.raises(ValueError, match="layout"):
        native_action_bounds(model)


@pytest.mark.parametrize("digest", ["", "f" * 63, "G" * 64, "f" * 65])
def test_sdk_binding_requires_exact_base_source_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        MyoSuiteSdkBinding("3.0.0", "3.6.0", digest)


@pytest.mark.parametrize("sdk,runtime", [("", "3.6.0"), ("3.0.0", ""), ("  ", "3.6.0")])
def test_sdk_binding_requires_separate_sdk_and_native_runtime_versions(
    sdk: str, runtime: str
) -> None:
    with pytest.raises(ValueError, match="version"):
        MyoSuiteSdkBinding(sdk, runtime, "f" * 64)


_AUDITED_BASE_SHA256 = (
    "53ad8bc71fc07a2e7fcead164ca027a33418b9ac1aa953ea6c4042fa48eed154"
)
_MIXED_XML = """<mujoco><option timestep="0.001" integrator="RK4"/>
<worldbody><body><joint name="hinge"/><geom type="sphere" size="0.05" mass="1"/>
</body></worldbody><actuator><motor name="motor" joint="hinge"/>
<general name="activation" joint="hinge" dyntype="filter" dynprm="0.02"
 ctrllimited="true" ctrlrange="0 1"/></actuator></mujoco>"""


def _native_runtime() -> Any:
    pytest.importorskip("myosuite")
    mj = pytest.importorskip("mujoco")
    if mj.__version__ != "3.6.0":
        pytest.skip("reviewed MyoSuite task lane requires actual MuJoCo 3.6.0")
    return mj


def _integration_state(mj: Any, model: Any, data: Any) -> np.ndarray:
    kind = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(model, kind))
    mj.mj_getState(model, data, state, kind)
    return state


def test_real_sdk_reset_and_step_preserve_mixed_commands_and_native_state(
    tmp_path: Path,
) -> None:
    mj = _native_runtime()
    from myosuite.envs.gymnasium_env import MyoGymnasiumEnv

    model_path = tmp_path / "mixed.xml"
    model_path.write_text(_MIXED_XML, encoding="utf-8")
    binding = MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    sdk_calls: list[str] = []
    sdk_codes = {MyoGymnasiumEnv.reset.__code__, MyoGymnasiumEnv.step.__code__}

    def trace(frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code in sdk_codes:
            sdk_calls.append(frame.f_code.co_name)

    previous_profiler = sys.getprofile()
    try:
        sys.setprofile(trace)
        task = create_project_golf_task(model_path, binding)
        task.data.act[:] = 0.2
        mj.mj_forward(task.model, task.data)
        initial = _integration_state(mj, task.model, task.data)
        result = task.step(np.array([-3.0, 2.0]))
    finally:
        sys.setprofile(previous_profiler)

    assert sdk_calls == ["reset", "step"]
    assert task.project_sdk_binding == binding
    assert task.project_task_kind == "project-defined-golf-state-bookkeeping"
    assert task.model_path == str(model_path.resolve())
    assert task.frame_skip == 1
    assert result[1:4] == (0.0, False, False)
    np.testing.assert_array_equal(task.data.ctrl, [-3.0, 1.0])
    reference_model = mj.MjModel.from_xml_path(str(model_path))
    reference = mj.MjData(reference_model)
    mj.mj_setState(reference_model, reference, initial, mj.mjtState.mjSTATE_INTEGRATION)
    reference.ctrl[:] = [-3.0, 1.0]
    mj.mj_step(reference_model, reference)
    mj.mj_forward(reference_model, reference)
    np.testing.assert_array_equal(
        _integration_state(mj, task.model, task.data),
        _integration_state(mj, reference_model, reference),
    )
    assert task.data.time == reference.time == 0.001
    task.close()


@pytest.mark.parametrize("wrong", ["sdk", "runtime", "source"])
def test_real_sdk_constructor_rejects_mismatched_provenance(
    tmp_path: Path, wrong: str
) -> None:
    _native_runtime()
    model_path = tmp_path / "mixed.xml"
    model_path.write_text(_MIXED_XML, encoding="utf-8")
    binding = MyoSuiteSdkBinding(
        "wrong" if wrong == "sdk" else "3.0.0",
        "wrong" if wrong == "runtime" else "3.6.0",
        "0" * 64 if wrong == "source" else _AUDITED_BASE_SHA256,
    )
    with pytest.raises(ValueError, match="differs"):
        create_project_golf_task(model_path, binding)


@pytest.mark.parametrize(
    "field", ["helper_source_sha256", "distribution_metadata_sha256"]
)
def test_real_sdk_metadata_binding_rejects_changed_execution_source(
    tmp_path: Path, field: str
) -> None:
    _native_runtime()
    path = tmp_path / "mixed.xml"
    path.write_text(_MIXED_XML, encoding="utf-8")
    binding = replace(
        MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256),
        **{field: "0" * 64},
    )
    with pytest.raises(ValueError, match="helper|metadata|source"):
        create_project_golf_task(path, binding)


@pytest.mark.parametrize("epoch", [0.0, 1.0])
def test_real_sdk_command_recording_matches_uninterrupted_native_reference(
    tmp_path: Path,
    epoch: float,
) -> None:
    mj = _native_runtime()
    model_path = tmp_path / "mixed.xml"
    model_path.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        model_path, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    task.data.act[:] = 0.2
    task.data.time = epoch
    mj.mj_forward(task.model, task.data)
    initial = _integration_state(mj, task.model, task.data)
    actions = np.array([[-3.0, 1.2], [-2.0, 0.1], [0.3, 0.6]])

    history = record_project_task_commands(task, actions)

    np.testing.assert_array_equal(history.integration_states[0], initial)
    np.testing.assert_array_equal(
        history.applied_actuator_commands, [[-3.0, 1.0], [-2.0, 0.1], [0.3, 0.6]]
    )
    assert history.integration_states.shape[0] == 4
    assert not history.integration_states.flags.writeable
    assert not history.applied_actuator_commands.flags.writeable
    reference_model = mj.MjModel.from_xml_path(str(model_path))
    reference = mj.MjData(reference_model)
    mj.mj_setState(reference_model, reference, initial, mj.mjtState.mjSTATE_INTEGRATION)
    for index, command in enumerate(history.applied_actuator_commands):
        reference.ctrl[:] = command
        mj.mj_step(reference_model, reference)
        mj.mj_forward(reference_model, reference)
        np.testing.assert_array_equal(
            history.integration_states[index + 1],
            _integration_state(mj, reference_model, reference),
        )
        assert history.time_seconds[index + 1] == reference.time
    task.close()


@pytest.mark.parametrize("changed_method", ["step", "get_reward_dict", "_get_obs_dict"])
def test_command_producer_rejects_task_hook_replacement_before_execution(
    tmp_path: Path, changed_method: str
) -> None:
    mj = _native_runtime()
    model_path = tmp_path / "mixed.xml"
    model_path.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        model_path, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    initial = _integration_state(mj, task.model, task.data)
    setattr(task, changed_method, lambda *_args: None)
    with pytest.raises(ValueError, match="method"):
        record_project_task_commands(task, np.array([[0.1, 0.2]]))
    np.testing.assert_array_equal(
        _integration_state(mj, task.model, task.data), initial
    )
    task.close()


_PRODUCTION_SHA256 = {
    "driver": "e4a55d603af42db841d2a6e7b5fc5de71fb43a49a843c5d78b2062927202a8c5",
    "iron": "b5c05b4e305398ac4c5e68bc6b84c89df273446dda3eab9532116a82117f660b",
}


@pytest.mark.parametrize("variant", ["driver", "iron"])
def test_actual_production_sdk_task_matches_its_native_command_reference(
    variant: str,
) -> None:
    mj = _native_runtime()
    root = os.environ.get("FEEDBACK_MYOSUITE_PRODUCTION_ROOT")
    if not root:
        pytest.skip("production proof requires the retained complete resource root")
    model_path = Path(root) / "golf" / "body" / f"golfer_myobody_{variant}.xml"
    assert (
        hashlib.sha256(model_path.read_bytes()).hexdigest()
        == _PRODUCTION_SHA256[variant]
    )
    task = create_project_golf_task(
        model_path,
        MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256),
        resource_root=Path(root),
    )
    assert (task.model.nq, task.model.nv, task.model.nu, task.model.na) == (
        89,
        83,
        100,
        86,
    )
    assert np.count_nonzero(task.model.actuator_ctrllimited) == 86
    task.data.act[:] = np.linspace(0.02, 0.08, 86)
    task.data.qvel[:] = np.linspace(-1e-4, 1e-4, 83)
    mj.mj_forward(task.model, task.data)
    initial = _integration_state(mj, task.model, task.data)
    actions = np.tile(np.linspace(0.03, 0.07, 100), (3, 1))
    actions[:, :14] = np.array([-0.001, 0.002, -0.003])[:, None]
    actions[1:, 14:] += 0.01

    history = record_project_task_commands(task, actions)

    np.testing.assert_array_equal(history.applied_actuator_commands, actions)
    np.testing.assert_array_equal(history.integration_states[0], initial)
    reference_model = mj.MjModel.from_xml_path(str(model_path))
    reference = mj.MjData(reference_model)
    mj.mj_setState(reference_model, reference, initial, mj.mjtState.mjSTATE_INTEGRATION)
    for index, command in enumerate(actions):
        reference.ctrl[:] = command
        mj.mj_step(reference_model, reference)
        mj.mj_forward(reference_model, reference)
        np.testing.assert_array_equal(
            history.integration_states[index + 1],
            _integration_state(mj, reference_model, reference),
        )
        assert history.time_seconds[index + 1] == reference.time
    assert np.isfinite(history.integration_states).all()
    assert history.sdk_binding.native_runtime_version == "3.6.0"
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        DeclaredModelResource,
        _discover_native_resource_files,
        resource_closure_sha256,
    )

    resource_root = Path(root).resolve(strict=True)
    discovered, _ = _discover_native_resource_files(model_path, resource_root)
    resources = tuple(
        DeclaredModelResource(
            path.relative_to(resource_root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(discovered)
    )
    closure_digest = resource_closure_sha256(
        resource_root,
        model_path,
        resources,
        expected_loaded_native_model_sha256=history.loaded_native_model_sha256,
    )
    assert len(resources) > 2
    assert len(closure_digest) == 64
    from src.engines.project_task_replay_artifacts import freeze_project_task_history
    from src.engines.native_direct_model_provider import create_native_direct_model
    from src.engines.native_replay_contracts import native_replay_contract_types
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        replay_direct_model_actuator_commands,
    )

    artifact = freeze_project_task_history(
        task,
        history,
        resource_root=resource_root,
        resources=resources,
        model_id="myosuite-golfer",
        variant_id=variant,
        experiment_id=f"opaque:production-task-{variant}",
    )
    contracts = native_replay_contract_types()
    reloaded = contracts.load_experiment_replay_bundle(
        contracts.dumps_experiment_replay_bundle(artifact.bundle)
    )
    replay = replay_direct_model_actuator_commands(
        reloaded,
        artifact.registration,
        create_native_direct_model,
        compiled_profile_bytes=bytes(artifact.compiled_profile_bytes),
    )
    np.testing.assert_array_equal(replay.integration_states, history.integration_states)
    np.testing.assert_array_equal(replay.applied_actuator_commands, actions)
    suffix = replace(
        history,
        time_seconds=history.time_seconds[1:],
        integration_states=history.integration_states[1:],
        applied_actuator_commands=history.applied_actuator_commands[1:],
    )
    suffix_artifact = freeze_project_task_history(
        task,
        suffix,
        resource_root=resource_root,
        resources=resources,
        model_id="myosuite-golfer",
        variant_id=variant,
        experiment_id=f"opaque:production-suffix-{variant}",
    )
    suffix_replay = replay_direct_model_actuator_commands(
        suffix_artifact.bundle,
        suffix_artifact.registration,
        create_native_direct_model,
        compiled_profile_bytes=suffix_artifact.compiled_profile_bytes,
    )
    np.testing.assert_array_equal(
        suffix_replay.integration_states, history.integration_states[1:]
    )
    np.testing.assert_array_equal(
        suffix_replay.applied_actuator_commands[0], actions[1]
    )
    mj.mj_setState(
        task.model,
        task.data,
        suffix.integration_states[0],
        mj.mjtState.mjSTATE_INTEGRATION,
    )
    mj.mj_forward(task.model, task.data)
    changed_actions = actions[1:].copy()
    changed_actions[:, 14:] += 0.005
    changed = record_project_task_commands(task, changed_actions)
    np.testing.assert_array_equal(
        changed.integration_states[0], suffix.integration_states[0]
    )
    changed_artifact = freeze_project_task_history(
        task,
        changed,
        resource_root=resource_root,
        resources=resources,
        model_id="myosuite-golfer",
        variant_id=variant,
        experiment_id=f"opaque:production-changed-future-{variant}",
    )
    changed_replay = replay_direct_model_actuator_commands(
        changed_artifact.bundle,
        changed_artifact.registration,
        create_native_direct_model,
        compiled_profile_bytes=changed_artifact.compiled_profile_bytes,
    )
    np.testing.assert_array_equal(
        changed_replay.integration_states, changed.integration_states
    )
    assert not np.array_equal(
        changed_replay.integration_states[-1], suffix_replay.integration_states[-1]
    )
    task.close()


def test_actual_sdk_runtime_admits_its_own_compiled_resource_closure(
    tmp_path: Path,
) -> None:
    _native_runtime()
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        DeclaredModelResource,
        resource_closure_sha256,
    )

    model_path = tmp_path / "mixed.xml"
    model_path.write_text(_MIXED_XML, encoding="utf-8")
    resource = DeclaredModelResource(
        "mixed.xml", hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    digest = resource_closure_sha256(tmp_path, model_path, (resource,))
    assert len(digest) == 64


@pytest.mark.parametrize("epoch", [1e20, 2.0**42])
def test_actual_sdk_rejects_stalled_or_distorted_native_clock(
    tmp_path: Path, epoch: float
) -> None:
    mj = _native_runtime()
    model_path = tmp_path / "mixed.xml"
    model_path.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        model_path, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    task.data.time = epoch
    mj.mj_forward(task.model, task.data)
    with pytest.raises(ValueError, match="clock"):
        record_project_task_commands(task, np.array([[0.001, 0.02]]))
    task.close()


def test_actual_sdk_runtime_law_manifest_binds_available_native_fields(
    tmp_path: Path,
) -> None:
    _native_runtime()
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        actuator_law_manifest_sha256,
    )

    model_path = tmp_path / "mixed.xml"
    model_path.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        model_path, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    channels = tuple(
        SimpleNamespace(
            channel_id=f"command:{name}",
            target_id=f"actuator:{name}",
            unit="1",
            coordinate_id=None,
            frame_id=None,
        )
        for name in ("motor", "activation")
    )
    before = actuator_law_manifest_sha256(task.model, channels)
    task.model.actuator_gainprm[0, 0] += 0.01
    assert actuator_law_manifest_sha256(task.model, channels) != before
    task.close()


def test_native_direct_factory_loads_fresh_model_without_sdk_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mj = _native_runtime()
    from myosuite.envs.gymnasium_env import MyoGymnasiumEnv
    from src.engines.native_direct_model_provider import create_native_direct_model

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("independent model loading called an SDK lifecycle hook")

    monkeypatch.setattr(MyoGymnasiumEnv, "reset", forbidden)
    monkeypatch.setattr(MyoGymnasiumEnv, "step", forbidden)
    model_path = tmp_path / "mixed.xml"
    model_path.write_text(_MIXED_XML, encoding="utf-8")
    first = create_native_direct_model(str(model_path))
    second = create_native_direct_model(str(model_path))
    assert isinstance(first.model, mj.MjModel)
    assert isinstance(first.data, mj.MjData)
    assert first.model is not second.model
    assert first.data is not second.data
    assert first.model_path == str(model_path.resolve())
    assert not hasattr(first, "step")
    assert not hasattr(first, "reset")
    assert first.data.time == 0.0
    first.close()
    second.close()


def test_project_history_exports_existing_t01_and_independently_replays(
    tmp_path: Path,
) -> None:
    _native_runtime()
    from src.engines.project_task_replay_artifacts import freeze_project_task_history
    from src.engines.native_direct_model_provider import create_native_direct_model
    from src.engines.native_replay_contracts import native_replay_contract_types
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        DeclaredModelResource,
        replay_direct_model_actuator_commands,
    )

    path = tmp_path / "mixed.xml"
    path.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        path, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    history = record_project_task_commands(
        task, np.array([[0.001, 0.2], [-0.002, 0.1]])
    )
    artifact = freeze_project_task_history(
        task,
        history,
        resource_root=tmp_path,
        resources=(
            DeclaredModelResource(
                "mixed.xml", hashlib.sha256(path.read_bytes()).hexdigest()
            ),
        ),
        model_id="synthetic-mixed",
        variant_id="test-only",
        experiment_id="opaque:task-test",
    )
    contracts = native_replay_contract_types()
    reloaded = contracts.load_experiment_replay_bundle(
        contracts.dumps_experiment_replay_bundle(artifact.bundle)
    )
    assert (
        reloaded.input_history.input_kind
        is contracts.ActuationInputKind.ACTUATOR_COMMAND
    )
    assert len(reloaded.input_history.values) == 3
    assert reloaded.input_history.values[-1] == reloaded.input_history.values[-2]
    replay = replay_direct_model_actuator_commands(
        reloaded,
        artifact.registration,
        create_native_direct_model,
        compiled_profile_bytes=artifact.compiled_profile_bytes,
    )
    np.testing.assert_array_equal(replay.integration_states, history.integration_states)
    np.testing.assert_array_equal(
        replay.applied_actuator_commands, history.applied_actuator_commands
    )
    task.close()


@pytest.mark.parametrize("split", [0, 2])
def test_native_replay_preserves_nonzero_epoch_and_exact_suffix_state(
    tmp_path: Path, split: int
) -> None:
    mj = _native_runtime()
    from src.engines.project_task_replay_artifacts import freeze_project_task_history
    from src.engines.native_direct_model_provider import create_native_direct_model
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        DeclaredModelResource,
        replay_direct_model_actuator_commands,
    )

    path = tmp_path / "mixed.xml"
    path.write_text(_MIXED_XML, encoding="utf-8")
    task = create_project_golf_task(
        path, MyoSuiteSdkBinding("3.0.0", "3.6.0", _AUDITED_BASE_SHA256)
    )
    task.data.time = 1.25
    task.data.act[:] = 0.2
    mj.mj_forward(task.model, task.data)
    actions = np.array([[0.003, 0.1], [-0.002, 0.2], [0.001, 0.05], [-0.004, 0.3]])
    history = record_project_task_commands(task, actions)
    suffix = replace(
        history,
        time_seconds=history.time_seconds[split:],
        integration_states=history.integration_states[split:],
        applied_actuator_commands=history.applied_actuator_commands[split:],
    )
    artifact = freeze_project_task_history(
        task,
        suffix,
        resource_root=tmp_path,
        resources=(
            DeclaredModelResource(
                "mixed.xml", hashlib.sha256(path.read_bytes()).hexdigest()
            ),
        ),
        model_id="synthetic-mixed",
        variant_id="test-only",
        experiment_id="opaque:suffix-test",
    )
    replay = replay_direct_model_actuator_commands(
        artifact.bundle,
        artifact.registration,
        create_native_direct_model,
        compiled_profile_bytes=artifact.compiled_profile_bytes,
    )
    np.testing.assert_array_equal(replay.integration_states, suffix.integration_states)
    np.testing.assert_array_equal(replay.applied_actuator_commands, actions[split:])
    np.testing.assert_array_equal(
        replay.time_seconds, suffix.time_seconds - suffix.time_seconds[0]
    )
    assert replay.time_seconds[0] == 0.0
    task.close()
