"""Stateless generalized-effort request policy contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

from src.shared.python.motion_matching.distributed_feedback import (
    ActuatorMap,
    AllocationReceipt,
    ControlTask,
    DistributedFeedbackController,
    DistributedFeedbackPolicy,
    NominalTrajectory,
    TaskGroup,
    TaskKinematics,
)

pytestmark = pytest.mark.unit


class _QuaternionLikeModel:
    nq = 4
    nv = 3

    def difference(self, q: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return q[1:] - reference[1:]

    def mass_inverse(self, q: np.ndarray) -> np.ndarray:
        return np.eye(self.nv)


class _CountingTask:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self, q: np.ndarray, v: np.ndarray) -> TaskKinematics:
        self.calls += 1
        jacobian = np.array([[1.0, 0.0, 0.0]])
        return TaskKinematics(np.array([q[1]]), jacobian @ v, jacobian)

    def difference(self, target: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return target - actual


class _RecordingAllocator:
    def __init__(
        self,
        actuators: ActuatorMap,
        *,
        fail_once: bool = False,
        invalid_once: str | None = None,
    ) -> None:
        self.actuators = actuators
        self.fail_once = fail_once
        self.invalid_once = invalid_once
        self.calls: list[tuple[np.ndarray, np.ndarray | None, float]] = []

    def allocate(
        self,
        requested: np.ndarray,
        previous: np.ndarray | None,
        dt: float,
        q: np.ndarray,
        v: np.ndarray,
    ) -> AllocationReceipt:
        self.calls.append(
            (requested.copy(), None if previous is None else previous.copy(), dt)
        )
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("synthetic allocator failure")
        if self.invalid_once == "shape":
            self.invalid_once = None
            return AllocationReceipt(np.array([1.8]), "test", 0.0)
        if self.invalid_once == "finite":
            self.invalid_once = None
            return AllocationReceipt(np.array([np.nan, 2.7]), "test", 0.0)
        return AllocationReceipt(np.array([1.8, 2.7]), "test", 0.0)


def _inputs(
    *,
    feedforward: np.ndarray | None = None,
    gains: np.ndarray | None = None,
    task_provider: _CountingTask | None = None,
) -> tuple[
    _QuaternionLikeModel, ActuatorMap, NominalTrajectory, tuple[ControlTask, ...]
]:
    model = _QuaternionLikeModel()
    actuators = ActuatorMap(("distal", "proximal"), (2, 0), (1,))
    schedule = NominalTrajectory(
        times=np.array([0.0, 0.1]),
        q=np.array([[1.0, 0.0, 0.0, 0.0]]),
        v=np.zeros((1, 3)),
        feedforward=np.array([[2.0, 3.0]]) if feedforward is None else feedforward,
        gains=(
            np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
            if gains is None
            else gains
        ),
        phase_names=("swing",),
        channel_ids=actuators.channel_ids,
    )
    tasks: tuple[ControlTask, ...] = ()
    if task_provider is not None:
        tasks = (
            ControlTask(
                name="club",
                group=TaskGroup.CLUB,
                priority=0,
                active_phases=("swing",),
                provider=task_provider,
                target_positions=np.array([[2.0]]),
                target_velocities=np.zeros((1, 1)),
                kp=2.0,
                kd=0.5,
                frame_id="clubhead",
                position_unit="m",
            ),
        )
    return model, actuators, schedule, tasks


def _policy(
    model: _QuaternionLikeModel,
    actuators: ActuatorMap,
    schedule: NominalTrajectory,
    tasks: tuple[ControlTask, ...] = (),
) -> DistributedFeedbackPolicy:
    return DistributedFeedbackPolicy(model, actuators, schedule, tasks=tasks)


def test_policy_returns_immutable_full_nv_effort_for_permuted_nq_nv_map() -> None:
    model, actuators, schedule, tasks = _inputs()
    policy = _policy(model, actuators, schedule, tasks)
    q = np.array([1.0, 0.2, -0.1, 0.3])
    v = np.array([0.4, -0.2, 0.1])
    request = policy.request_for_step(0.0, q, v, 0.05, expected_time_s=0.0)

    assert request.time_s == 0.0
    assert request.dt_s == 0.05
    assert request.schedule_index == 0
    assert request.channel_ids == ("distal", "proximal")
    assert request.generalized_indices == (2, 0)
    assert request.selected_coordinate_mask == (True, False, True)
    assert not hasattr(policy, "_previous")
    assert not hasattr(policy, "_next_time")
    assert not hasattr(policy, "allocator")
    assert request.nominal_feedforward == (2.0, 3.0)
    assert request.feedback_correction == pytest.approx((-0.2, -0.3))
    assert request.total_requested == pytest.approx((1.8, 2.7))
    assert request.generalized_effort == pytest.approx((2.7, 0.0, 1.8))
    with pytest.raises(FrozenInstanceError):
        field_name = "time_s"
        setattr(request, field_name, 1.0)
    with pytest.raises(TypeError):
        cast(Any, request.generalized_effort)[0] = 100.0
    np.testing.assert_array_equal(q, [1.0, 0.2, -0.1, 0.3])
    np.testing.assert_array_equal(v, [0.4, -0.2, 0.1])


def test_policy_is_repeatable_and_samples_each_active_task_once_per_call() -> None:
    provider = _CountingTask()
    model, actuators, schedule, tasks = _inputs(task_provider=provider)
    policy = _policy(model, actuators, schedule, tasks)
    q = np.array([1.0, 0.2, -0.1, 0.3])
    v = np.array([0.4, -0.2, 0.1])

    first = policy.request_for_step(0.0, q, v, 0.05, expected_time_s=0.0)
    second = policy.request_for_step(0.0, q, v, 0.05, expected_time_s=0.0)

    assert provider.calls == 2
    assert first == second
    assert first.active_tasks == ("club",)
    # 2*(2 - 0.2) + 0.5*(0 - 0.4) = 3.4 on generalized coordinate 0.
    assert first.generalized_effort == pytest.approx((6.4, 0.0, 1.8))


def test_controller_delegates_full_nv_request_and_commits_only_after_allocate() -> None:
    model, actuators, schedule, tasks = _inputs()
    allocator = _RecordingAllocator(actuators, fail_once=True)
    controller = DistributedFeedbackController(
        model, actuators, schedule, allocator, tasks=tasks
    )
    q = np.array([1.0, 0.2, -0.1, 0.3])
    v = np.array([0.4, -0.2, 0.1])

    with pytest.raises(RuntimeError, match="synthetic allocator failure"):
        controller.command_for_step(0.0, q, v, 0.05)
    step = controller.command_for_step(0.0, q, v, 0.05)

    assert len(allocator.calls) == 2
    np.testing.assert_allclose(allocator.calls[0][0], [2.7, 0.0, 1.8])
    np.testing.assert_allclose(allocator.calls[1][0], [2.7, 0.0, 1.8])
    assert allocator.calls[0][1] is None
    assert allocator.calls[1][1] is None
    assert step.phase == "swing"
    np.testing.assert_allclose(step.total_requested, [1.8, 2.7])
    np.testing.assert_allclose(step.applied, [1.8, 2.7])
    with pytest.raises(ValueError, match="contiguous"):
        controller.command_for_step(0.0, q, v, 0.05)


def test_nonfinite_effort_overflow_is_rejected_before_allocator() -> None:
    overflow_gains = np.zeros((1, 2, 6))
    overflow_gains[0, 0, 0] = 1.0e308
    model, actuators, schedule, _ = _inputs(
        feedforward=np.zeros((1, 2)), gains=overflow_gains
    )
    allocator = _RecordingAllocator(actuators)
    controller = DistributedFeedbackController(model, actuators, schedule, allocator)

    with pytest.raises(ValueError, match="finite"):
        controller.command_for_step(
            0.0, np.array([1.0, 2.0, 0.0, 0.0]), np.zeros(3), 0.05
        )
    assert allocator.calls == []


def test_disabled_policy_returns_nominal_only_request() -> None:
    model, actuators, schedule, _ = _inputs()
    policy = DistributedFeedbackPolicy(model, actuators, schedule, enabled=False)

    request = policy.request_for_step(
        0.0,
        np.array([1.0, 0.2, -0.1, 0.3]),
        np.zeros(3),
        0.05,
        expected_time_s=0.0,
    )

    assert request.feedback_correction == (0.0, 0.0)
    assert request.total_requested == (2.0, 3.0)
    assert request.generalized_effort == (3.0, 0.0, 2.0)


@pytest.mark.parametrize("invalid_kind", ["shape", "finite"])
def test_invalid_allocator_output_does_not_advance_controller_history(
    invalid_kind: str,
) -> None:
    model, actuators, schedule, _ = _inputs()
    allocator = _RecordingAllocator(actuators, invalid_once=invalid_kind)
    controller = DistributedFeedbackController(model, actuators, schedule, allocator)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.zeros(3)

    with pytest.raises(ValueError, match="invalid applied input"):
        controller.command_for_step(0.0, q, v, 0.05)
    step = controller.command_for_step(0.0, q, v, 0.05)

    assert allocator.calls[0][1] is None
    assert allocator.calls[1][1] is None
    assert step.time_s == 0.0
    assert controller._next_time == 0.05
