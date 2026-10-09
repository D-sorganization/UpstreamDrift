"""Project-defined MyoSuite task creation; native replay keeps its MuJoCo identity.

This adapter uses the real SDK lifecycle. It is neither an official supplied
golf benchmark nor another physics engine. Independent replay remains owned by
``native_direct_model_replay`` and consumes post-mapping actuator commands.
The SDK adapter imports independently of the legacy GUI/engine wrapper package.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MyoSuiteSdkBinding:
    """Separate declared SDK/base-source and native-runtime identities."""

    sdk_version: str
    native_runtime_version: str
    base_source_sha256: str
    helper_source_sha256: str = (
        "eb2d5af2b2c4e842626aa3b29eb1a9c77c93a62a4650a6a50bda492832d3b107"
    )
    distribution_metadata_sha256: str = (
        "39e5fbbea88bb4aa29c42fc1d13f5846877b179094402d7c80e839f25ceb9698"
    )

    def __post_init__(self) -> None:
        for version in (self.sdk_version, self.native_runtime_version):
            if not isinstance(version, str) or not version.strip():
                raise ValueError("SDK and native runtime version must be nonempty")
        for digest in (
            self.base_source_sha256,
            self.helper_source_sha256,
            self.distribution_metadata_sha256,
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    "SDK execution source must have a lowercase SHA-256 digest"
                )


def native_action_bounds(model: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Preserve actual compiled limits, including genuinely unlimited motors.

    Returned arrays are independent of the model. Unused ctrlrange entries on
    unlimited channels do not imply zero limits or normalized muscle actions.
    """
    limited = np.asarray(model.actuator_ctrllimited)
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    count = int(model.nu)
    if (
        count <= 0
        or limited.shape != (count,)
        or ranges.shape != (count, 2)
        or not np.isin(limited, [0, 1]).all()
    ):
        raise ValueError("compiled actuator control layout is incomplete")
    limited = limited.astype(bool)
    active_ranges = ranges[limited]
    if not np.isfinite(active_ranges).all() or np.any(
        active_ranges[:, 0] > active_ranges[:, 1]
    ):
        raise ValueError("limited actuator ranges must be finite and ordered")
    low = np.where(limited, ranges[:, 0], -np.inf)
    high = np.where(limited, ranges[:, 1], np.inf)
    return low, high


@dataclass(frozen=True)
class ProjectModelSourceBinding:
    """Source bytes frozen before constructing the SDK's executed native model."""

    resource_root: Path
    resources: tuple[Any, ...]
    source_model_sha256: str
    resource_closure_sha256: str


def _prepare_model_source(model_path: Path, root: Path) -> ProjectModelSourceBinding:
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        DeclaredModelResource,
        _discover_native_resource_files,
        resource_closure_sha256,
    )

    root = root.resolve(strict=True)
    files, _ = _discover_native_resource_files(model_path, root)
    resources = tuple(
        DeclaredModelResource(
            path.relative_to(root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(files)
    )
    digest = resource_closure_sha256(root, model_path, resources)
    source = next(
        item.sha256
        for item in resources
        if item.relative_path == model_path.relative_to(root).as_posix()
    )
    return ProjectModelSourceBinding(root, resources, source, digest)


def create_project_golf_task(
    model_path: Path, binding: MyoSuiteSdkBinding, *, resource_root: Path | None = None
) -> Any:
    """Construct and SDK-reset a project task for the exact source model path.

    Postcondition: SDK reset/observation hooks have run, the task uses one native
    step per action, and its action bounds preserve compiled native semantics.
    Resource-closure/profile admission is still required before qualifying replay.
    """
    if not isinstance(binding, MyoSuiteSdkBinding):
        raise TypeError("an explicit MyoSuite SDK binding is required")
    model_path = model_path.resolve(strict=True)
    if not model_path.is_file() or model_path.suffix.lower() != ".xml":
        raise ValueError("project task requires an existing MJCF XML model path")

    import mujoco as mj

    from src.engines.native_replay_contracts import require_no_global_mujoco_callbacks

    require_no_global_mujoco_callbacks(mj)
    from myosuite.envs.gymnasium_env import MyoGymnasiumEnv
    import gymnasium as gym

    require_no_global_mujoco_callbacks(mj)
    _validate_sdk_binding(binding, MyoGymnasiumEnv, mj.__version__)
    source_binding = _prepare_model_source(
        model_path, resource_root or model_path.parent
    )

    class ProjectGolfTask(MyoGymnasiumEnv):
        """Physical-state bookkeeping task using the unmodified SDK step/reset."""

        def __init__(self) -> None:
            super().__init__(frame_skip=1, render_mode=None)
            self.model_path = str(model_path)
            model = mj.MjModel.from_xml_path(self.model_path)
            self.model = model
            self.data = mj.MjData(model)
            self._ctrl_dt = float(model.opt.timestep)
            low, high = native_action_bounds(model)
            self.action_space = gym.spaces.Box(low, high, dtype=np.float64)
            observation_count = self.model.nq + self.model.nv + self.model.na
            self.observation_space = self._unbounded_obs_space(observation_count)

        def _get_obs_dict(self, accessor: Any) -> dict[str, NDArray[np.float64]]:
            return {
                "qpos": accessor.joint_pos(),
                "qvel": accessor.joint_vel(),
                "activation": accessor.muscle_act(),
            }

        def get_reward_dict(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
            # Required SDK bookkeeping only; zero is not a motion-matching score.
            return {"dense": 0.0, "done": False}

    task = ProjectGolfTask()
    from src.engines.physics_engines.myosuite.python.native_direct_model_replay import (
        resource_closure_sha256,
    )

    resource_closure_sha256(
        source_binding.resource_root,
        model_path,
        source_binding.resources,
        expected_loaded_native_model_sha256=_model_sha256(mj, task.model),
    )
    require_no_global_mujoco_callbacks(mj)
    task.reset(seed=0)
    require_no_global_mujoco_callbacks(mj)
    task.project_sdk_binding = binding
    task.project_model_source = source_binding
    task.project_task_source_sha256 = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    task.project_task_kind = "project-defined-golf-state-bookkeeping"
    task._project_methods = {
        name: (getattr(task, name).__func__, getattr(task, name).__func__.__code__)
        for name in (
            "reset",
            "step",
            "_step_physics",
            "_get_obs_dict",
            "get_reward_dict",
            "reset_task",
            "_finalize_step",
            "_obs_dict_to_vec",
            "_ensure_obs_gymnasium_compliant",
        )
    }
    return task


@dataclass(frozen=True)
class ProjectTaskCommandHistory:
    """Native producer samples for existing T01 export, not a replay format."""

    time_seconds: NDArray[np.float64]
    integration_states: NDArray[np.float64]
    applied_actuator_commands: NDArray[np.float64]
    loaded_native_model_sha256: str
    project_task_source_sha256: str
    sdk_binding: MyoSuiteSdkBinding
    source_model_sha256: str
    resource_closure_sha256: str


def record_project_task_commands(
    task: Any, actions: NDArray[np.float64]
) -> ProjectTaskCommandHistory:
    """Exercise the SDK's actual step path and freeze post-mapping commands.

    No reset or observation feedback is performed here. All actions are validated
    before execution. Outputs contain actual clocks/full native states and are
    read-only; independent replay and scientific scoring remain separate.
    """
    import mujoco as mj

    from src.engines.native_replay_contracts import require_no_global_mujoco_callbacks

    _admit_task(task, mj)
    _verify_source_files(task.project_model_source)
    model, data = task.model, task.data
    actions = np.asarray(actions, dtype=np.float64).copy()
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != model.nu:
        raise ValueError("actions must contain nonempty ordered native actuator rows")
    if not np.isfinite(actions).all():
        raise ValueError("all project task actions must be finite")
    model_digest = _model_sha256(mj, model)
    initial = _read_native_state(mj, model, data)
    states = [initial]
    times = [float(data.time)]
    applied = []
    low, high = native_action_bounds(model)
    for action in actions:
        _admit_task(task, mj)
        require_no_global_mujoco_callbacks(mj)
        if _model_sha256(mj, model) != model_digest:
            raise ValueError("project task changed the admitted native model")
        expected = np.clip(action, low, high)
        task.step(action)
        _admit_task(task, mj)
        require_no_global_mujoco_callbacks(mj)
        if not np.array_equal(data.ctrl, expected):
            raise ValueError("SDK applied controls differ from declared action mapping")
        _require_native_step_clock(
            times[-1], float(data.time), float(model.opt.timestep)
        )
        if _model_sha256(mj, model) != model_digest:
            raise ValueError("SDK task mutated the admitted native model")
        applied.append(data.ctrl.copy())
        states.append(_read_native_state(mj, model, data))
        times.append(float(data.time))
    arrays = tuple(
        np.asarray(value, dtype=np.float64) for value in (times, states, applied)
    )
    for array in arrays:
        array.setflags(write=False)
    _verify_source_files(task.project_model_source)
    return ProjectTaskCommandHistory(
        arrays[0],
        arrays[1],
        arrays[2],
        model_digest,
        task.project_task_source_sha256,
        task.project_sdk_binding,
        task.project_model_source.source_model_sha256,
        task.project_model_source.resource_closure_sha256,
    )


def _admit_task(task: Any, mj: Any) -> None:
    from myosuite.envs.gymnasium_env import MyoGymnasiumEnv

    if type(task).__bases__ != (MyoGymnasiumEnv,) or type(task).__module__ != __name__:
        raise ValueError("producer requires the exact project-defined SDK task")
    _validate_sdk_binding(task.project_sdk_binding, MyoGymnasiumEnv, mj.__version__)
    if (
        hashlib.sha256(Path(task.model_path).read_bytes()).hexdigest()
        != task.project_model_source.source_model_sha256
    ):
        raise ValueError("project model source changed after native construction")
    if (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        != task.project_task_source_sha256
    ):
        raise ValueError("project task source changed after construction")
    model = task.model
    if task.frame_skip != 1 or task._ctrl_dt != float(model.opt.timestep):
        raise ValueError("project task requires its declared single-native-step policy")
    for name, (expected, expected_code) in task._project_methods.items():
        actual = getattr(getattr(task, name, None), "__func__", None)
        if (
            actual is not expected
            or getattr(actual, "__code__", None) is not expected_code
        ):
            raise ValueError("project/SDK method changed after construction")
    low, high = native_action_bounds(task.model)
    if not np.array_equal(task.action_space.low, low) or not np.array_equal(
        task.action_space.high, high
    ):
        raise ValueError("project action bounds differ from compiled native limits")


def _model_sha256(mj: Any, model: Any) -> str:
    payload = np.zeros(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, buffer=payload)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _verify_source_files(binding: ProjectModelSourceBinding) -> None:
    for resource in binding.resources:
        if (
            hashlib.sha256(
                (binding.resource_root / resource.relative_path).read_bytes()
            ).hexdigest()
            != resource.sha256
        ):
            raise ValueError(
                "project model source resource changed after native construction"
            )


def _require_native_step_clock(before: float, after: float, step: float) -> None:
    """Apply T04's interval precision policy independently of epoch/run length."""
    from src.engines.native_replay_contracts import require_native_step_clock

    require_native_step_clock(before, after, step)


def _read_native_state(mj: Any, model: Any, data: Any) -> NDArray[np.float64]:
    if any(int(warning.number) for warning in data.warning):
        raise ValueError("project SDK task produced a native numerical warning")
    kind = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(model, kind), dtype=np.float64)
    mj.mj_getState(model, data, state, kind)
    if not np.isfinite(state).all():
        raise ValueError("project SDK task produced nonfinite native state")
    normalized = data.qpos.copy()
    mj.mj_normalizeQuat(model, normalized)
    if not np.allclose(data.qpos, normalized, atol=8 * np.finfo(float).eps, rtol=0):
        raise ValueError("project SDK task has a nonunit native quaternion")
    return state


def _validate_sdk_binding(
    binding: MyoSuiteSdkBinding, base_class: Any, native_version: str
) -> None:
    actual_sdk = importlib.metadata.version("myosuite")
    if (
        actual_sdk != binding.sdk_version
        or native_version != binding.native_runtime_version
    ):
        raise ValueError("installed SDK/native runtime differs from declared versions")
    if (actual_sdk, native_version) != ("3.0.0", "3.6.0"):
        raise ValueError("SDK/native runtime combination has not been reviewed")
    source = inspect.getsourcefile(base_class)
    if source is None:
        raise ValueError("actual SDK base source is unavailable")
    source_digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if source_digest != binding.base_source_sha256:
        raise ValueError("actual SDK base source differs from declared SHA-256")
    from myosuite.envs import muscle_stages

    helper_source = inspect.getsourcefile(muscle_stages)
    if (
        helper_source is None
        or hashlib.sha256(Path(helper_source).read_bytes()).hexdigest()
        != binding.helper_source_sha256
    ):
        raise ValueError(
            "actual SDK execution helper source differs from declared SHA-256"
        )
    metadata = importlib.metadata.distribution("myosuite").read_text("METADATA")
    if (
        metadata is None
        or hashlib.sha256(metadata.encode("utf-8")).hexdigest()
        != binding.distribution_metadata_sha256
    ):
        raise ValueError(
            "actual SDK distribution metadata differs from declared SHA-256"
        )
