"""Opt-in, step-transactional numerical RK4 stage evidence.

Stages are numerical evaluation states, not a continuous-time certificate.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
StageObserver = Callable[[int, float, Array, Array], None]


@dataclass(frozen=True)
class Stage:
    step: int
    hold: int
    segment: int
    substep: int
    stage: int
    time_s: float
    q: Array
    v: Array
    tau: Array
    applied_drive: Array | None = None
    applied_external: Array | None = None
    effort_semantics: str | None = None


@dataclass(frozen=True)
class RunPlan:
    requested_steps: int
    dt_s: float
    substeps: int
    effort_holds: int
    project: bool
    coordinate_order: tuple[str, ...]


@dataclass(frozen=True)
class SegmentPlan:
    step: int
    hold: int
    segment: int
    start_s: float
    duration_s: float
    substeps: int
    force_active: bool
    step_sizes_s: tuple[float, ...] = ()


def fixed_substep_sizes(
    duration_s: float, nominal_step_s: float, start_offset_s: float = 0.0
) -> tuple[float, ...]:
    """Partition an interval at the nominal lattice anchored to the hold start."""
    if not np.isfinite([duration_s, nominal_step_s, start_offset_s]).all():
        raise ValueError("step durations must be finite")
    if duration_s <= 0.0 or nominal_step_s <= 0.0 or start_offset_s < 0.0:
        raise ValueError("step durations must be positive")
    phase = start_offset_s / nominal_step_s
    nearest = round(phase)
    if abs(phase - nearest) <= 8 * np.finfo(float).eps * max(1.0, abs(phase)):
        to_boundary = nominal_step_s
    else:
        to_boundary = (np.floor(phase) + 1.0 - phase) * nominal_step_s
    first = min(duration_s, to_boundary)
    remaining = duration_s - first
    count = int(np.floor(remaining / nominal_step_s))
    remainder = remaining - count * nominal_step_s
    return (
        (first,) + (nominal_step_s,) * count + ((remainder,) if remainder > 0 else ())
    )


@dataclass(frozen=True)
class RunOutcome:
    requested_steps: int
    completed_steps: int
    status: str
    failure: str | None = None


@dataclass(frozen=True)
class TracedStep:
    stages: tuple[Stage, ...]
    segments: tuple[SegmentPlan, ...]
    pre_projection_q: Array
    pre_projection_v: Array
    post_projection_q: Array
    post_projection_v: Array


class StageRecorder:
    """Commit one complete integration step at a time; failed buffers vanish."""

    def __init__(
        self, *, capture_ideal_drive: bool = False, capture_motor_drive: bool = False
    ) -> None:
        if capture_ideal_drive and capture_motor_drive:
            raise ValueError("choose one drive evidence mode")
        self.capture_ideal_drive = capture_ideal_drive
        self.capture_drive = capture_ideal_drive or capture_motor_drive
        self.drive_semantics = (
            "mujoco_post_saturation_generalized_motor"
            if capture_motor_drive
            else "ideal_direct_generalized_no_saturation"
        )
        self.steps: list[TracedStep] = []
        self._pending: list[Stage] = []
        self._segments: list[SegmentPlan] = []
        self.plan: RunPlan | None = None
        self.outcome: RunOutcome | None = None

    def start_run(self, plan: RunPlan) -> None:
        if self.plan is not None:
            raise ValueError("a stage recorder cannot be reused across runs")
        self.plan = plan
        self.outcome = RunOutcome(plan.requested_steps, 0, "running")

    def begin(self) -> None:
        self._pending = []
        self._segments = []

    def segment(self, plan: SegmentPlan) -> None:
        self._segments.append(plan)

    def finish(self, failure: str | None = None) -> None:
        if self.plan is None:
            raise ValueError("run plan is required")
        if failure is None:
            failure = self._integrity_failure()
        status = (
            "complete"
            if failure is None and len(self.steps) == self.plan.requested_steps
            else "partial"
        )
        self.outcome = RunOutcome(
            self.plan.requested_steps, len(self.steps), status, failure
        )

    def _integrity_failure(self) -> str | None:
        """Check trace against run and accepted-segment plans, not physics."""
        plan = self.plan
        if plan is None:
            return "missing run plan"
        if len(self.steps) != plan.requested_steps or self._pending or self._segments:
            return "incomplete steps or open transaction"
        nv = len(plan.coordinate_order)
        if (
            not nv
            or any(
                not isinstance(name, str) or not name.strip()
                for name in plan.coordinate_order
            )
            or len(set(plan.coordinate_order)) != nv
        ):
            return "coordinate inventory must be nonempty and unique"
        if (
            not nv
            or not np.isfinite(plan.dt_s)
            or plan.dt_s <= 0
            or plan.effort_holds < 1
            or plan.substeps < 1
            or plan.substeps % plan.effort_holds
        ):
            return "invalid run plan"
        hold_dt = plan.dt_s / plan.effort_holds
        for k, step in enumerate(self.steps):
            held_efforts: dict[int, Array] = {}
            for value in (
                step.pre_projection_q,
                step.pre_projection_v,
                step.post_projection_q,
                step.post_projection_v,
            ):
                if value.shape != (nv,) or not np.isfinite(value).all():
                    return "invalid projection boundary"
            segments = iter(step.segments)
            current = next(segments, None)
            for hold in range(plan.effort_holds):
                start = (k * plan.effort_holds + hold) * hold_dt
                hold_start = start
                end = start + hold_dt
                segment_id = 0
                while current is not None and current.hold == hold:
                    if (
                        current.step != k
                        or current.segment != segment_id
                        or not np.isfinite(current.start_s)
                        or not np.isfinite(current.duration_s)
                        or current.duration_s <= 0
                        or current.substeps < 1
                        or abs(current.start_s - start) > 1e-12
                    ):
                        return "invalid segment schedule"
                    expected_sizes = fixed_substep_sizes(
                        current.duration_s,
                        hold_dt / (plan.substeps // plan.effort_holds),
                        max(0.0, current.start_s - hold_start),
                    )
                    if current.substeps != len(expected_sizes) or (
                        len(current.step_sizes_s) != len(expected_sizes)
                        or not np.allclose(
                            current.step_sizes_s, expected_sizes, rtol=0.0, atol=1e-15
                        )
                    ):
                        return "invalid segment substep schedule"
                    start = current.start_s + current.duration_s
                    if start > end + 1e-12:
                        return "segment exceeds hold"
                    segment_id += 1
                    current = next(segments, None)
                if segment_id == 0 or abs(start - end) > 1e-12:
                    return "missing or incomplete hold"
            if current is not None:
                return "extra or reordered segment"
            for stage in step.stages:
                for value in (stage.q, stage.v, stage.tau):
                    if value.shape != (nv,) or not np.isfinite(value).all():
                        return "invalid stage values"
                if stage.hold in held_efforts:
                    if not np.array_equal(stage.tau, held_efforts[stage.hold]):
                        return "held effort changes within a hold"
                else:
                    held_efforts[stage.hold] = stage.tau
                if self.capture_drive and (
                    stage.applied_drive is None
                    or stage.applied_external is None
                    or stage.applied_drive.shape != (nv,)
                    or stage.applied_external.shape != (nv,)
                    or not np.isfinite(stage.applied_drive).all()
                    or not np.isfinite(stage.applied_external).all()
                    or stage.effort_semantics != self.drive_semantics
                ):
                    return "missing or invalid ideal drive evidence"
        expected = self.expected_stages()
        actual = [
            (s.step, s.hold, s.segment, s.substep, s.stage, s.time_s)
            for step in self.steps
            for s in step.stages
        ]
        if len(expected) != len(actual):
            return "missing or extra stage"
        for e, a in zip(expected, actual, strict=True):
            if e[:5] != a[:5] or not np.isfinite(a[5]) or abs(e[5] - a[5]) > 1e-12:
                return "stage identity or time mismatch"
        return None

    def expected_stages(self) -> list[tuple[int, int, int, int, int, float]]:
        """Expand committed segment plans; these are not a physical certificate."""
        expected = []
        for step in self.steps:
            for segment in step.segments:
                start = segment.start_s
                for substep, h in enumerate(segment.step_sizes_s):
                    for stage, offset in enumerate((0.0, h / 2, h / 2, h), 1):
                        expected.append(
                            (
                                segment.step,
                                segment.hold,
                                segment.segment,
                                substep,
                                stage,
                                start + offset,
                            )
                        )
                    start += h
        return expected

    def record(
        self,
        step: int,
        hold: int,
        segment: int,
        substep: int,
        stage: int,
        time_s: float,
        q: Array,
        v: Array,
        tau: Array,
    ) -> None:
        self._pending.append(
            Stage(
                step,
                hold,
                segment,
                substep,
                stage,
                time_s,
                q.copy(),
                v.copy(),
                tau.copy(),
            )
        )

    def observer(
        self,
        step: int,
        hold: int,
        segment: int,
        substep: int,
        time_s: float,
        tau: Array,
    ) -> StageObserver:
        def record_stage(stage: int, offset: float, q: Array, v: Array) -> None:
            self.record(step, hold, segment, substep, stage, time_s + offset, q, v, tau)

        return record_stage

    def record_drive(self, drive: Array, external: Array) -> None:
        if not self._pending:
            raise ValueError("drive evidence requires a pending stage")
        self._pending[-1] = replace(
            self._pending[-1],
            applied_drive=np.asarray(drive, dtype=float).copy(),
            applied_external=np.asarray(external, dtype=float).copy(),
            effort_semantics=self.drive_semantics,
        )

    def commit(
        self,
        pre_projection_q: Array,
        pre_projection_v: Array,
        post_projection_q: Array,
        post_projection_v: Array,
    ) -> None:
        self.steps.append(
            TracedStep(
                tuple(self._pending),
                tuple(self._segments),
                pre_projection_q.copy(),
                pre_projection_v.copy(),
                post_projection_q.copy(),
                post_projection_v.copy(),
            )
        )
        self._pending = []
        self._segments = []

    def abort(self) -> None:
        self._pending = []
        self._segments = []
