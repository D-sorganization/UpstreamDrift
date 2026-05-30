"""Read-model dataclasses for the training-controller dashboard.

These types are the *only* surface the future PyQt6 widget layer needs
to render. They are intentionally PyQt-free so the controller and its
unit tests can build them headlessly.

Each dataclass is frozen + slots per the repo style (see CLAUDE.md
§"Coding Standards") and validates its preconditions in
``__post_init__``. The dataclasses are deliberately *flat* and
JSON-shaped; callers should not reach through the object graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.shared.python.training import JobId, TrainingJob
from src.shared.python.training.metrics import MetricKind


__all__ = [
    "DashboardModel",
    "GpuSnapshot",
    "JobRow",
    "MetricSeries",
    "ResourceSnapshot",
    "job_row_from_training_job",
]


@dataclass(frozen=True, slots=True)
class JobRow:
    """One row of the dashboard's job list.

    A view-model projection of :class:`TrainingJob` that exposes only
    the columns the dashboard renders. ``elapsed_s`` is derived at
    build time (``now - started_at`` for active jobs,
    ``completed_at - started_at`` for terminal jobs, ``0.0`` when the
    job has not yet started).

    Invariants:
        - ``job_id`` is a non-empty string.
        - ``framework`` is a non-empty string.
        - ``status`` is a non-empty string (the :class:`TrainingStatus`
          ``.value``).
        - ``elapsed_s`` is a non-negative float.
        - ``dataset_id`` is ``None`` or a non-empty string.
        - ``error_message`` is ``None`` or a non-empty string.
    """

    job_id: str
    framework: str
    status: str
    dataset_id: str | None
    elapsed_s: float
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        if not isinstance(self.framework, str) or not self.framework:
            raise ValueError("framework must be a non-empty string")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a non-empty string")
        if not isinstance(self.elapsed_s, (int, float)) or self.elapsed_s < 0:
            raise ValueError(
                f"elapsed_s must be a non-negative number (got {self.elapsed_s!r})"
            )
        if self.dataset_id is not None and (
            not isinstance(self.dataset_id, str) or not self.dataset_id
        ):
            raise ValueError("dataset_id must be None or a non-empty string")
        if self.error_message is not None and (
            not isinstance(self.error_message, str) or not self.error_message
        ):
            raise ValueError("error_message must be None or a non-empty string")
        object.__setattr__(self, "elapsed_s", float(self.elapsed_s))


def job_row_from_training_job(job: TrainingJob, *, now: float) -> JobRow:
    """Project a :class:`TrainingJob` into a :class:`JobRow`.

    Args:
        job: The job to project.
        now: Wall-clock time, used to compute elapsed seconds for jobs
            still running.

    Returns:
        A frozen :class:`JobRow` summarising the job's dashboard
        columns. Elapsed is ``0.0`` when the job has not yet started
        (``PENDING`` / ``QUEUED``).
    """

    if not isinstance(job, TrainingJob):
        raise TypeError(f"expected TrainingJob (got {type(job).__name__})")
    if not isinstance(now, (int, float)) or now < 0:
        raise ValueError(f"now must be a non-negative number (got {now!r})")
    if job.started_at is None:
        elapsed = 0.0
    elif job.completed_at is not None:
        elapsed = max(0.0, float(job.completed_at - job.started_at))
    else:
        elapsed = max(0.0, float(now) - float(job.started_at))
    return JobRow(
        job_id=job.job_id.value,
        framework=job.config.framework.value,
        status=job.status.value,
        dataset_id=job.config.dataset_id,
        elapsed_s=elapsed,
        error_message=job.error_message,
    )


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """Per-metric time series, with optional pre-smoothed values.

    The GUI plot binds ``steps`` (x-axis) against ``values`` (y-axis).
    When ``smoothed`` is set it has the same length as ``values`` and
    represents a rolling-mean overlay (used for noisy RL rewards).

    Invariants:
        - ``name`` is a non-empty string.
        - ``kind`` is a :class:`MetricKind`.
        - ``steps`` and ``values`` are tuples of equal length.
        - ``smoothed`` (when set) has the same length as ``values``.
        - All ``steps`` are non-negative ints.
    """

    name: str
    kind: MetricKind
    steps: tuple[int, ...]
    values: tuple[float, ...]
    smoothed: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.kind, MetricKind):
            raise TypeError(
                f"kind must be a MetricKind (got {type(self.kind).__name__})"
            )
        if not isinstance(self.steps, tuple) or not isinstance(self.values, tuple):
            raise TypeError("steps and values must be tuples")
        if len(self.steps) != len(self.values):
            raise ValueError(
                "steps and values must have equal length "
                f"(got {len(self.steps)} vs {len(self.values)})"
            )
        for step in self.steps:
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError(f"steps must be non-negative ints (got {step!r})")
        for value in self.values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"values must be real numbers (got {value!r})")
        if self.smoothed is not None:
            if not isinstance(self.smoothed, tuple):
                raise TypeError("smoothed must be a tuple or None")
            if len(self.smoothed) != len(self.values):
                raise ValueError(
                    "smoothed must have the same length as values "
                    f"(got {len(self.smoothed)} vs {len(self.values)})"
                )
            for value in self.smoothed:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(
                        f"smoothed values must be real numbers (got {value!r})"
                    )


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    """Per-GPU snapshot for the resource strip.

    Mirrors :class:`training.resource_monitor.GpuSample` but flattened
    so the GUI does not need to import the monitor types.

    Invariants:
        - ``index`` is a non-negative int.
        - ``name`` is a non-empty string.
        - ``utilization_percent`` is ``None`` or in ``[0.0, 100.0]``.
        - Memory values are non-negative; used <= total.
    """

    index: int
    name: str
    utilization_percent: float | None
    memory_used_mb: int
    memory_total_mb: int

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"index must be non-negative (got {self.index!r})")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.utilization_percent is not None:
            if not isinstance(self.utilization_percent, (int, float)):
                raise TypeError("utilization_percent must be a number or None")
            if not 0.0 <= float(self.utilization_percent) <= 100.0:
                raise ValueError(
                    f"utilization_percent must be in [0, 100] "
                    f"(got {self.utilization_percent!r})"
                )
        if self.memory_used_mb < 0 or self.memory_total_mb < 0:
            raise ValueError("memory values must be non-negative")
        if self.memory_used_mb > self.memory_total_mb:
            raise ValueError(
                "memory_used_mb cannot exceed memory_total_mb "
                f"({self.memory_used_mb} > {self.memory_total_mb})"
            )


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Flat view of the host's resource situation.

    Either populated from a :class:`training.resource_monitor.ResourceSample`
    or constructed in an "unavailable" form (every numeric field
    ``None``) for the case where ``psutil`` is missing.

    Invariants:
        - ``cpu_percent`` is ``None`` or in ``[0.0, 100.0]``.
        - ``memory_percent`` is ``None`` or in ``[0.0, 100.0]``.
        - ``gpus`` is a tuple of :class:`GpuSnapshot`.
        - When ``available`` is ``False`` all numeric fields are
          ``None`` and ``gpus`` is empty.
    """

    cpu_percent: float | None
    memory_percent: float | None
    gpus: tuple[GpuSnapshot, ...] = field(default_factory=tuple)
    available: bool = True

    def __post_init__(self) -> None:
        for field_name in ("cpu_percent", "memory_percent"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{field_name} must be a number or None (got {value!r})"
                )
            if not 0.0 <= float(value) <= 100.0:
                raise ValueError(f"{field_name} must be in [0, 100] (got {value!r})")
        if not isinstance(self.gpus, tuple):
            raise TypeError("gpus must be a tuple")
        for gpu in self.gpus:
            if not isinstance(gpu, GpuSnapshot):
                raise TypeError(
                    f"gpus entries must be GpuSnapshot (got {type(gpu).__name__})"
                )
        if not self.available and (
            self.cpu_percent is not None or self.memory_percent is not None or self.gpus
        ):
            raise ValueError(
                "unavailable ResourceSnapshot must have all-None fields "
                "and an empty gpus tuple"
            )

    @classmethod
    def unavailable(cls) -> ResourceSnapshot:
        """Construct the ``available=False`` sentinel value."""

        return cls(
            cpu_percent=None,
            memory_percent=None,
            gpus=(),
            available=False,
        )


@dataclass(frozen=True, slots=True)
class DashboardModel:
    """Top-level read-model the controller hands the GUI.

    Attributes:
        jobs: Snapshot of every known job, in registry order.
        selected_job_id: The currently-selected row, or ``None``.
        metric_series_for_selected: Per-name series for the selected
            job. Empty when no job is selected or no metrics observed.
        resources: Latest host-resource sample (or the "unavailable"
            sentinel).

    Invariants:
        - ``jobs`` is a tuple of :class:`JobRow`.
        - When ``selected_job_id`` is set, at least one row in ``jobs``
          must have a matching ``job_id``. The check is enforced so the
          GUI cannot end up "highlighting nothing".
        - ``metric_series_for_selected`` is a tuple of
          :class:`MetricSeries`; empty when no job is selected.
    """

    jobs: tuple[JobRow, ...]
    selected_job_id: JobId | None
    metric_series_for_selected: tuple[MetricSeries, ...]
    resources: ResourceSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.jobs, tuple):
            raise TypeError("jobs must be a tuple")
        for row in self.jobs:
            if not isinstance(row, JobRow):
                raise TypeError(
                    f"jobs entries must be JobRow (got {type(row).__name__})"
                )
        if self.selected_job_id is not None:
            if not isinstance(self.selected_job_id, JobId):
                raise TypeError(
                    "selected_job_id must be a JobId or None "
                    f"(got {type(self.selected_job_id).__name__})"
                )
            known = {row.job_id for row in self.jobs}
            if self.selected_job_id.value not in known:
                raise ValueError(
                    f"selected_job_id {self.selected_job_id.value!r} is not in jobs"
                )
        if not isinstance(self.metric_series_for_selected, tuple):
            raise TypeError("metric_series_for_selected must be a tuple")
        for series in self.metric_series_for_selected:
            if not isinstance(series, MetricSeries):
                raise TypeError(
                    "metric_series_for_selected entries must be MetricSeries "
                    f"(got {type(series).__name__})"
                )
        if self.selected_job_id is None and self.metric_series_for_selected:
            raise ValueError(
                "metric_series_for_selected must be empty when no job is selected"
            )
        if not isinstance(self.resources, ResourceSnapshot):
            raise TypeError(
                "resources must be a ResourceSnapshot "
                f"(got {type(self.resources).__name__})"
            )

    @property
    def selected_row(self) -> JobRow | None:
        """The :class:`JobRow` matching ``selected_job_id``, if any."""

        if self.selected_job_id is None:
            return None
        for row in self.jobs:
            if row.job_id == self.selected_job_id.value:
                return row
        # Unreachable: __post_init__ enforces membership.
        return None  # pragma: no cover - defensive

    def find_row(self, job_id: JobId) -> JobRow | None:
        """Return the row matching ``job_id`` or ``None``.

        Does not raise — the GUI calls this opportunistically.
        """

        if not isinstance(job_id, JobId):
            raise TypeError(f"job_id must be a JobId (got {type(job_id).__name__})")
        for row in self.jobs:
            if row.job_id == job_id.value:
                return row
        return None
