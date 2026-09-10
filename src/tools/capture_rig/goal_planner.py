"""Pure capture outcome planning; architecture edges never execute commands."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SafeId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,79}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
State = Literal["done", "ready", "blocked", "skipped"]


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class CaptureStep(_Record):
    id: SafeId
    action: SafeId
    node_id: SafeId | None = None
    requires: tuple[SafeId, ...] = Field(default=(), max_length=64)
    workflow_key: SafeId | None = None
    minimum_views: int = Field(default=0, ge=0, le=12, strict=True)
    maximum_views: int | None = Field(default=None, ge=1, le=12, strict=True)
    needs_calibration: bool = Field(default=False, strict=True)
    optional: bool = Field(default=False, strict=True)


class CaptureGoal(_Record):
    id: SafeId
    title: str = Field(min_length=1, max_length=200)
    steps: tuple[SafeId, ...] = Field(min_length=1, max_length=64)


class CaptureGoalCatalog(_Record):
    schema_version: Literal["capture-goals/1"] = "capture-goals/1"
    steps: tuple[CaptureStep, ...] = Field(min_length=1, max_length=128)
    goals: tuple[CaptureGoal, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        by_id = {step.id: step for step in self.steps}
        if len(by_id) != len(self.steps) or len({g.id for g in self.goals}) != len(
            self.goals
        ):
            raise ValueError("Duplicate capture step or goal ID")
        for step in self.steps:
            if len(set(step.requires)) != len(step.requires):
                raise ValueError("Duplicate prerequisite")
            if (
                step.maximum_views is not None
                and step.minimum_views > step.maximum_views
            ):
                raise ValueError("Step has incompatible camera requirements")
        references = [key for step in self.steps for key in step.requires]
        references.extend(key for goal in self.goals for key in goal.steps)
        if any(key not in by_id for key in references):
            raise ValueError("Unknown capture prerequisite or goal step")
        try:
            tuple(
                TopologicalSorter(
                    {step.id: step.requires for step in self.steps}
                ).static_order()
            )
        except CycleError as exc:
            raise ValueError("Capture prerequisites contain a cycle") from exc
        return self

    @property
    def revision(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CaptureRoute:
    goals: tuple[str, ...]
    steps: tuple[CaptureStep, ...]
    catalog_revision: str
    minimum_views: int
    maximum_views: int | None

    @property
    def step_ids(self) -> tuple[str, ...]:
        return tuple(step.id for step in self.steps)


def resolve(catalog: CaptureGoalCatalog, selected: Iterable[str]) -> CaptureRoute:
    """Return a stable prerequisite closure with each shared step exactly once."""
    goals = tuple(sorted(set(selected)))
    if not goals:
        raise ValueError("Select at least one capture outcome")
    by_goal = {goal.id: goal for goal in catalog.goals}
    if any(goal not in by_goal for goal in goals):
        raise ValueError(
            "Unknown capture outcome; choose from the current capability map"
        )
    by_step = {step.id: step for step in catalog.steps}
    pending = [key for goal in goals for key in by_goal[goal].steps]
    required: set[str] = set()
    while pending:
        key = pending.pop()
        if key not in required:
            required.add(key)
            pending.extend(by_step[key].requires)
    dependencies = {key: sorted(by_step[key].requires) for key in sorted(required)}
    order = tuple(TopologicalSorter(dependencies).static_order())
    steps = tuple(by_step[key] for key in order)
    minimum = max(step.minimum_views for step in steps)
    maximums = [step.maximum_views for step in steps if step.maximum_views is not None]
    maximum = min(maximums) if maximums else None
    if maximum is not None and minimum > maximum:
        raise ValueError(
            "Selected outcomes have incompatible camera routes; run them separately"
        )
    return CaptureRoute(goals, steps, catalog.revision, minimum, maximum)


@dataclass(frozen=True)
class Readiness:
    status: State
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"done", "ready", "blocked", "skipped"}:
            raise ValueError("Unknown capture readiness status")
        if not isinstance(self.reason, str):
            raise TypeError("Capture readiness reason must be text")


def evaluate(
    route: CaptureRoute,
    evidence: Mapping[str, Readiness],
    *,
    view_count: int,
    calibration_compatible: bool,
) -> tuple[Readiness, ...]:
    """Use existing workflow evidence; stale prerequisites invalidate descendants."""
    if (
        isinstance(view_count, bool)
        or not isinstance(view_count, int)
        or view_count < 0
    ):
        raise ValueError("Camera view count must be a nonnegative integer")
    if not isinstance(calibration_compatible, bool):
        raise TypeError("Calibration compatibility must be explicitly true or false")
    result: dict[str, Readiness] = {}
    for step in route.steps:
        state = evidence.get(
            step.id, Readiness("blocked", "Open this step to supply its inputs")
        )
        if state.status == "skipped" and not step.optional:
            state = Readiness(
                "blocked", "This required step is unavailable for the current session"
            )
        if view_count < step.minimum_views:
            state = Readiness(
                "blocked",
                f"Add recordings from at least {step.minimum_views} camera views",
            )
        elif step.maximum_views is not None and view_count > step.maximum_views:
            state = Readiness(
                "blocked",
                f"Use a session with at most {step.maximum_views} camera views",
            )
        elif step.needs_calibration and not calibration_compatible:
            state = Readiness(
                "blocked", "Review compatible camera calibration and optical settings"
            )
        missing = [
            key
            for key in step.requires
            if result[key].status not in {"done", "skipped"}
        ]
        if missing:
            state = Readiness(
                "blocked", f"Complete or refresh prerequisite: {', '.join(missing)}"
            )
        result[step.id] = state
    return tuple(result[step.id] for step in route.steps)


class CaptureProgress(_Record):
    schema_version: Literal["capture-progress/1"] = "capture-progress/1"
    capture_id: str = Field(min_length=1, max_length=200)
    goals: tuple[SafeId, ...] = Field(min_length=1, max_length=64)
    catalog_revision: Revision
    input_revision: Revision
    current_step: SafeId
    skipped: tuple[SafeId, ...] = Field(default=(), max_length=64)


class CaptureGoalRequest(_Record):
    """Portable map selection; contains no executable command or filesystem path."""

    schema_version: Literal["capture-goal-request/1"] = "capture-goal-request/1"
    goals: tuple[SafeId, ...] = Field(min_length=1, max_length=64)
    catalog_revision: Revision

    def resolve(self, catalog: CaptureGoalCatalog) -> CaptureRoute:
        if self.catalog_revision != catalog.revision:
            raise ValueError(
                "Capability map changed; select outcomes from the current map"
            )
        return resolve(catalog, self.goals)


def restore(
    catalog: CaptureGoalCatalog,
    progress: CaptureProgress,
    *,
    capture_id: str,
    input_revision: str,
) -> CaptureRoute:
    """Require explicit review after changes; never silently bind another swing."""
    if progress.capture_id != capture_id:
        raise ValueError("Saved workflow belongs to another capture")
    if progress.catalog_revision != catalog.revision:
        raise ValueError(
            "Capability graph changed; review the selected outcomes before resuming"
        )
    if progress.input_revision != input_revision:
        raise ValueError("Capture inputs changed; review readiness before resuming")
    route = resolve(catalog, progress.goals)
    if progress.current_step not in route.step_ids:
        raise ValueError("Saved step is not part of the selected outcome route")
    optional = {step.id for step in route.steps if step.optional}
    if not set(progress.skipped).issubset(optional):
        raise ValueError("Saved progress skips a required or unknown step")
    return route
