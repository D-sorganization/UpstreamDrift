"""Copied native feedback observations and the existing admitted SDK recorder.

Policy provenance is a declaration with execution-source guards, not a signature
or proof of hidden dependencies. Actual post-mapping inputs own replay identity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.myosuite_project_task_producer import (
    ProjectTaskCommandHistory,
    _read_native_state,
    _record_project_task_actions,
)


@dataclass(frozen=True)
class DeclaredFeedbackPolicy:
    """Controller source and caller-declared frozen parameter provenance."""

    name: str
    version: str
    source_path: Path
    source_sha256: str
    parameters_sha256: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (self.name, self.version)
        ):
            raise ValueError("feedback policy name and version must be nonempty")
        for digest in (self.source_sha256, self.parameters_sha256):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("feedback provenance requires lowercase SHA-256")
        object.__setattr__(
            self, "source_path", Path(self.source_path).resolve(strict=True)
        )


@dataclass(frozen=True)
class ProjectTaskObservation:
    """Native-order coordinates; qpos and tangent velocity may have different sizes."""

    sample_index: int
    native_time_seconds: float
    ordered_actuator_ids: tuple[str, ...]
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    activation: NDArray[np.float64]
    control: NDArray[np.float64]
    integration_state: NDArray[np.float64]


@dataclass(frozen=True)
class ProjectFeedbackRecording:
    """Producer provenance alongside existing history, without a new replay format."""

    history: ProjectTaskCommandHistory
    policy: DeclaredFeedbackPolicy
    feedback_adapter_source_sha256: str
    initial_state_array_sha256: str
    applied_command_array_sha256: str


def _immutable(values: NDArray[np.float64]) -> NDArray[np.float64]:
    copied = np.asarray(values, dtype=np.float64)
    return np.frombuffer(copied.tobytes(), dtype=np.float64).reshape(copied.shape)


def snapshot_project_task_state(
    model: Any, data: Any, *, sample_index: int
) -> ProjectTaskObservation:
    """Copy a complete native state without exposing mutable simulator handles."""
    import mujoco as mj

    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample index must be a nonnegative integer")
    full_state = _read_native_state(mj, model, data)
    names = tuple(
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    if any(not name for name in names):
        raise ValueError("feedback requires named ordered native actuators")
    return ProjectTaskObservation(
        sample_index,
        float(data.time),
        names,
        *(
            _immutable(values)
            for values in (
                data.qpos,
                data.qvel,
                data.act,
                data.ctrl,
                full_state,
            )
        ),
    )


def _policy_function(controller: Callable[..., Any]) -> Any:
    return controller if inspect.isfunction(controller) else type(controller).__call__


def _verify_policy(policy: DeclaredFeedbackPolicy, function: Any) -> None:
    source = inspect.getsourcefile(function)
    if source is None or Path(source).resolve() != policy.source_path:
        raise ValueError("feedback callable differs from declared source file")
    if (
        hashlib.sha256(policy.source_path.read_bytes()).hexdigest()
        != policy.source_sha256
    ):
        raise ValueError("feedback policy source changed")


def _adapter_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _feedback_actions(
    task: Any,
    controller: Callable[[ProjectTaskObservation], NDArray[np.float64]],
    steps: int,
    policy: DeclaredFeedbackPolicy,
    adapter_source_sha256: str,
) -> Iterator[NDArray[np.float64]]:
    import mujoco as mj

    from src.engines.native_replay_contracts import require_no_global_mujoco_callbacks

    function = _policy_function(controller)
    code = getattr(function, "__code__", None)
    for index in range(steps):
        require_no_global_mujoco_callbacks(mj)
        _verify_policy(policy, function)
        if _adapter_source_sha256() != adapter_source_sha256:
            raise ValueError("feedback adapter source changed")
        observation = snapshot_project_task_state(
            task.model, task.data, sample_index=index
        )
        # Conversion may run user __array__/__float__; keep it inside the guard.
        action = np.asarray(controller(observation), dtype=np.float64).copy()
        require_no_global_mujoco_callbacks(mj)
        if (
            _policy_function(controller) is not function
            or getattr(function, "__code__", None) is not code
        ):
            raise ValueError("feedback callable implementation changed")
        _verify_policy(policy, function)
        if _adapter_source_sha256() != adapter_source_sha256:
            raise ValueError("feedback adapter source changed")
        if not np.array_equal(
            observation.integration_state, _read_native_state(mj, task.model, task.data)
        ):
            raise ValueError("feedback callback changed native plant state")
        yield action


def record_project_task_feedback(
    task: Any,
    controller: Callable[[ProjectTaskObservation], NDArray[np.float64]],
    *,
    steps: int,
    policy: DeclaredFeedbackPolicy,
) -> ProjectFeedbackRecording:
    """Record one feedback action per SDK/native step; replay never calls a policy.

    Controller failures propagate. No partial history is returned. A caller must
    retain this provenance alongside exported T01 inputs; the native executor
    needs only the independently frozen initial state and applied commands.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("feedback step budget must be a positive integer")
    if not callable(controller) or not isinstance(policy, DeclaredFeedbackPolicy):
        raise TypeError("feedback requires a callable and declared policy provenance")
    adapter_source_sha256 = _adapter_source_sha256()
    history = _record_project_task_actions(
        task, _feedback_actions(task, controller, steps, policy, adapter_source_sha256)
    )
    return ProjectFeedbackRecording(
        history,
        policy,
        adapter_source_sha256,
        hashlib.sha256(history.integration_states[0].tobytes()).hexdigest(),
        hashlib.sha256(history.applied_actuator_commands.tobytes()).hexdigest(),
    )
