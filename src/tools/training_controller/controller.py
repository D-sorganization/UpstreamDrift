"""Headless MVC controller for the training-controller dashboard tab.

This module is GUI-free on purpose. It owns:

- A reference to the backend :class:`training.Scheduler`.
- A reference to the backend :class:`training.DatasetRegistry`.
- A reference to the backend :class:`training.CompatibilityChecker`.
- A list of observer callbacks fired whenever the read-model changes.

The PyQt6 widget layer (deferred to a follow-up of #6012) instantiates
one controller per dashboard tab, calls :meth:`current_model` to render,
and re-renders whenever its registered ``on_model_change`` callback
fires. Submit / cancel / pause / resume all go through the controller
so the compatibility-check gate is enforced regardless of which widget
triggered the action.

Thread safety: every public mutator takes an internal :class:`RLock`,
and observer fan-out is performed off the lock so user callbacks cannot
deadlock against further controller calls.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.training import (
    CompatibilityChecker,
    CompatibilityReport,
    DatasetRegistry,
    JobId,
    Scheduler,
    StatusChangeEvent,
    TrainingConfig,
    TrainingJob,
    summarize_by_kind,
)
from src.shared.python.training.metric_summary import RollingMean
from src.shared.python.training.metrics import MetricKind, TrainingMetric
from src.shared.python.training.resource_monitor import ResourceSample

from .view_model import (
    DashboardModel,
    GpuSnapshot,
    JobRow,
    MetricSeries,
    ResourceSnapshot,
    job_row_from_training_job,
)

__all__ = [
    "DEFAULT_ROLLING_WINDOW",
    "ModelChangeCallback",
    "ResourceProvider",
    "TrainingDashboardController",
]


logger = get_logger(__name__)


ModelChangeCallback = Callable[[], None]
"""Callback type registered via :meth:`TrainingDashboardController.on_model_change`."""

ResourceProvider = Callable[[], ResourceSample | None]
"""Pluggable source of the latest :class:`ResourceSample`.

The dashboard wires this to ``ResourceMonitor.latest`` when psutil is
available, or to a lambda returning ``None`` otherwise. The controller
itself does not know which is which — it just calls the provider and
maps the result onto :class:`ResourceSnapshot`.
"""

DEFAULT_ROLLING_WINDOW = 32
"""Rolling-mean window used to smooth noisy :attr:`MetricKind.REWARD` series."""


class TrainingDashboardController:
    """Headless MVC controller that binds the scheduler to the read-model.

    Args:
        scheduler: The backend scheduler whose registry and lifecycle
            calls this controller drives.
        dataset_registry: Dataset library exposed to the GUI. Kept on
            the controller (not just looked up via ``scheduler.registry``)
            so the GUI can render the library dock without a separate
            handle.
        compatibility_checker: Compatibility gate the controller runs
            before every :meth:`submit_job` so the GUI cannot dispatch
            an incompatible (config, engine) pair regardless of what
            the scheduler-level gate is configured with.
        resource_provider: Optional callable returning the latest
            :class:`ResourceSample`. Defaults to a provider that always
            returns ``None`` (so the GUI renders the "monitoring
            unavailable" sentinel until a real monitor is wired).
        rolling_window: Window length passed to :class:`RollingMean`
            when computing the smoothed overlay for reward series. Must
            be a positive int.
        clock: Wall-clock source. Defaulted to :func:`time.time` so
            tests can inject a deterministic clock.

    Raises:
        TypeError: When any of the backend references are the wrong type.
        ValueError: When ``rolling_window`` is not a positive int.
    """

    __slots__ = (
        "_clock",
        "_compat",
        "_datasets",
        "_lock",
        "_metrics",
        "_observers",
        "_resource_provider",
        "_rolling_window",
        "_scheduler",
        "_selected_job_id",
        "_status_unsubscribe",
    )

    def __init__(
        self,
        scheduler: Scheduler,
        dataset_registry: DatasetRegistry,
        compatibility_checker: CompatibilityChecker,
        *,
        resource_provider: ResourceProvider | None = None,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(scheduler, Scheduler):
            raise TypeError("scheduler must be a Scheduler")
        if not isinstance(dataset_registry, DatasetRegistry):
            raise TypeError("dataset_registry must be a DatasetRegistry")
        if not isinstance(compatibility_checker, CompatibilityChecker):
            raise TypeError("compatibility_checker must be a CompatibilityChecker")
        if resource_provider is not None and not callable(resource_provider):
            raise TypeError("resource_provider must be callable or None")
        if not isinstance(rolling_window, int) or rolling_window < 1:
            raise ValueError(
                f"rolling_window must be a positive int (got {rolling_window!r})"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._scheduler = scheduler
        self._datasets = dataset_registry
        self._compat = compatibility_checker
        self._resource_provider: ResourceProvider = (
            resource_provider if resource_provider is not None else _no_resources
        )
        self._rolling_window = rolling_window
        self._clock = clock
        self._lock = threading.RLock()
        self._observers: list[ModelChangeCallback] = []
        self._selected_job_id: JobId | None = None
        self._metrics: dict[JobId, list[TrainingMetric]] = {}
        # Hook the scheduler so any backend state change feeds the GUI.
        self._status_unsubscribe = scheduler.on_status_change(
            self._handle_status_change
        )

    # ------------------------------------------------------------------ read

    @property
    def scheduler(self) -> Scheduler:
        """Backend scheduler this controller is bound to."""

        return self._scheduler

    @property
    def dataset_registry(self) -> DatasetRegistry:
        """Dataset library the GUI's dataset dock renders."""

        return self._datasets

    @property
    def compatibility_checker(self) -> CompatibilityChecker:
        """Compatibility gate the controller runs before submit."""

        return self._compat

    def current_model(self) -> DashboardModel:
        """Build and return the current :class:`DashboardModel`."""

        with self._lock:
            return self._build_model_locked()

    def select_job(self, job_id: JobId | None) -> None:
        """Update the selected job and notify observers.

        ``job_id`` may be ``None`` to clear the selection. Unknown ids
        raise :class:`KeyError` rather than silently dropping the
        selection — the GUI must not pass an id that's no longer in the
        registry.
        """

        with self._lock:
            if job_id is None:
                self._selected_job_id = None
            else:
                if not isinstance(job_id, JobId):
                    raise TypeError(
                        f"job_id must be a JobId or None (got {type(job_id).__name__})"
                    )
                if not self._scheduler.registry.has(job_id):
                    raise KeyError(f"cannot select unknown job_id {job_id.value!r}")
                self._selected_job_id = job_id
        self._notify()

    @property
    def selected_job_id(self) -> JobId | None:
        """The currently-selected job id, if any."""

        with self._lock:
            return self._selected_job_id

    # ------------------------------------------------------------------ writes

    def submit_job(
        self,
        config: TrainingConfig,
        target_engine: str | None = None,
    ) -> TrainingJob:
        """Run the compatibility check, then queue the job.

        The controller's compatibility check runs **before** the
        scheduler is asked to admit the job, even if the scheduler is
        already configured with its own checker. The check is the
        single idiot-proof gate the GUI's Submit button cannot bypass.

        Args:
            config: The validated job configuration.
            target_engine: Optional engine name. When set, the
                controller's compatibility checker runs first; on
                failure :class:`training.errors.CompatibilityError` is
                raised before the scheduler is touched.

        Returns:
            The newly-queued :class:`TrainingJob`.

        Raises:
            CompatibilityError: When the (config, engine) pairing
                fails the controller's compatibility check.
            TypeError: When ``config`` is not a :class:`TrainingConfig`.
        """

        if not isinstance(config, TrainingConfig):
            raise TypeError(
                f"config must be a TrainingConfig (got {type(config).__name__})"
            )
        if target_engine is not None and not isinstance(target_engine, str):
            raise TypeError("target_engine must be a string or None")
        if target_engine is not None:
            report = self._compat.check(config, target_engine)
            if not report.is_compatible:
                raise self._compat_error(report, target_engine)
        job = self._scheduler.submit(config, target_engine=target_engine)
        # No explicit notify here: scheduler.submit fires StatusChangeEvent
        # which threads through _handle_status_change → _notify.
        return job

    def cancel_job(self, job_id: JobId) -> TrainingJob:
        """Cancel a job via the scheduler. See :meth:`Scheduler.cancel`."""

        if not isinstance(job_id, JobId):
            raise TypeError(f"job_id must be a JobId (got {type(job_id).__name__})")
        return self._scheduler.cancel(job_id)

    def pause_job(self, job_id: JobId) -> TrainingJob:
        """Pause a ``RUNNING`` job. See :meth:`Scheduler.pause`."""

        if not isinstance(job_id, JobId):
            raise TypeError(f"job_id must be a JobId (got {type(job_id).__name__})")
        return self._scheduler.pause(job_id)

    def resume_job(self, job_id: JobId) -> TrainingJob:
        """Resume a ``PAUSED`` job. See :meth:`Scheduler.resume`."""

        if not isinstance(job_id, JobId):
            raise TypeError(f"job_id must be a JobId (got {type(job_id).__name__})")
        return self._scheduler.resume(job_id)

    # ------------------------------------------------------------------ metrics

    def ingest_metric(self, job_id: JobId, metric: TrainingMetric) -> None:
        """Buffer a metric observation for ``job_id``.

        The GUI binds this to a :class:`TrainingJobLiveSubscriber`
        so each per-job realtime payload feeds the controller's
        in-memory series. When ``job_id`` matches the current
        selection, observers are notified.
        """

        if not isinstance(job_id, JobId):
            raise TypeError(f"job_id must be a JobId (got {type(job_id).__name__})")
        if not isinstance(metric, TrainingMetric):
            raise TypeError(
                f"metric must be a TrainingMetric (got {type(metric).__name__})"
            )
        with self._lock:
            self._metrics.setdefault(job_id, []).append(metric)
            should_notify = self._selected_job_id == job_id
        if should_notify:
            self._notify()

    def metrics_for(self, job_id: JobId) -> tuple[TrainingMetric, ...]:
        """Snapshot of every buffered metric for ``job_id``.

        Returns an empty tuple when nothing has been ingested for the
        id. Never raises on an unknown id — the GUI calls this
        opportunistically while jobs are spinning up.
        """

        if not isinstance(job_id, JobId):
            raise TypeError(f"job_id must be a JobId (got {type(job_id).__name__})")
        with self._lock:
            buffered = self._metrics.get(job_id)
            return tuple(buffered) if buffered else ()

    def clear_metrics(self, job_id: JobId | None = None) -> None:
        """Drop buffered metrics for ``job_id`` (or all of them)."""

        with self._lock:
            if job_id is None:
                self._metrics.clear()
            else:
                if not isinstance(job_id, JobId):
                    raise TypeError(
                        f"job_id must be a JobId or None (got {type(job_id).__name__})"
                    )
                self._metrics.pop(job_id, None)

    # ------------------------------------------------------------------ observers

    def on_model_change(self, callback: ModelChangeCallback) -> Callable[[], None]:
        """Register a no-arg callback fired when the model changes.

        Args:
            callback: Zero-arg callable. Invoked off the controller's
                internal lock so it may call back into the controller
                (e.g. ``current_model``) without deadlocking.

        Returns:
            An unsubscribe function. Calling it twice is a no-op.
        """

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._observers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._observers.remove(callback)
                except ValueError:
                    return

        return _unsubscribe

    def close(self) -> None:
        """Tear down scheduler hooks. Idempotent.

        The dashboard's tab-close handler calls this so the controller
        stops receiving scheduler events. Observers are *not* fired by
        ``close`` — closing is a lifecycle event, not a model change.
        """

        unsubscribe = self._status_unsubscribe
        self._status_unsubscribe = _noop
        try:
            unsubscribe()
        except (RuntimeError, ValueError, TypeError):
            logger.exception(
                "TrainingDashboardController.close: status unsubscribe failed"
            )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _compat_error(report: CompatibilityReport, engine: str):
        # Local import keeps the public import block tidy; the error
        # type lives next to the rest of the training errors.
        from src.shared.python.training.errors import (
            CompatibilityError,
        )  # noqa: PLC0415

        msgs = "; ".join(i.message for i in report.errors)
        return CompatibilityError(
            f"engine {engine!r} is incompatible with the submitted config: {msgs}"
        )

    def _handle_status_change(self, event: StatusChangeEvent) -> None:
        # Any status change can affect the rendered model — let observers
        # decide whether to re-render the visible row.
        del event  # the model is rebuilt from the registry, not the event
        # If the selected job has been removed (unusual but possible if
        # the GUI clears a stale selection), drop the selection rather
        # than letting the model builder raise.
        with self._lock:
            if self._selected_job_id is not None and not (
                self._scheduler.registry.has(self._selected_job_id)
            ):
                self._selected_job_id = None
        self._notify()

    def _notify(self) -> None:
        with self._lock:
            observers = tuple(self._observers)
        for observer in observers:
            try:
                observer()
            except (RuntimeError, ValueError, TypeError, OSError, LookupError):
                logger.exception(
                    "training controller observer raised; continuing fan-out"
                )

    def _build_model_locked(self) -> DashboardModel:
        # Must be called with self._lock held.
        now = float(self._clock())
        jobs = tuple(
            job_row_from_training_job(job, now=now)
            for job in self._scheduler.registry.list()
        )
        selected = self._selected_job_id
        if selected is not None and not _row_index(jobs, selected.value):
            # Selection became stale between the registry snapshot and
            # the build; drop it so DashboardModel doesn't raise.
            selected = None
            self._selected_job_id = None
        if selected is None:
            series: tuple[MetricSeries, ...] = ()
        else:
            buffered = self._metrics.get(selected, ())
            series = _build_series(buffered, window=self._rolling_window)
        resources = _resource_snapshot(self._resource_provider())
        return DashboardModel(
            jobs=jobs,
            selected_job_id=selected,
            metric_series_for_selected=series,
            resources=resources,
        )


# --------------------------------------------------------------------- helpers


def _no_resources() -> ResourceSample | None:
    return None


def _noop() -> None:
    return None


def _row_index(rows: tuple[JobRow, ...], value: str) -> bool:
    return any(row.job_id == value for row in rows)


def _build_series(
    metrics: Iterable[TrainingMetric], *, window: int
) -> tuple[MetricSeries, ...]:
    by_kind = summarize_by_kind(metrics)
    out: list[MetricSeries] = []
    for kind, observed in by_kind.items():
        by_name: dict[str, list[TrainingMetric]] = {}
        for m in observed:
            by_name.setdefault(m.name, []).append(m)
        for name in sorted(by_name):
            ordered = sorted(by_name[name], key=lambda m: (m.step, m.timestamp))
            steps = tuple(m.step for m in ordered)
            values = tuple(m.value for m in ordered)
            smoothed = (
                _rolling_smoothed(values, window=window)
                if kind is MetricKind.REWARD
                else None
            )
            out.append(
                MetricSeries(
                    name=name,
                    kind=kind,
                    steps=steps,
                    values=values,
                    smoothed=smoothed,
                )
            )
    return tuple(out)


def _rolling_smoothed(values: tuple[float, ...], *, window: int) -> tuple[float, ...]:
    rolling = RollingMean(window)
    out: list[float] = []
    for value in values:
        rolling.push(value)
        current = rolling.value
        # ``RollingMean.value`` returns None only when no values have
        # been pushed; we just pushed one, so this is always a float.
        out.append(float(current) if current is not None else 0.0)
    return tuple(out)


def _resource_snapshot(sample: ResourceSample | None) -> ResourceSnapshot:
    if sample is None:
        return ResourceSnapshot.unavailable()
    gpus = tuple(
        GpuSnapshot(
            index=g.index,
            name=g.name,
            utilization_percent=(
                None if g.utilization_percent is None else float(g.utilization_percent)
            ),
            memory_used_mb=int(g.memory_used_mb),
            memory_total_mb=int(g.memory_total_mb),
        )
        for g in sample.gpus
    )
    return ResourceSnapshot(
        cpu_percent=float(sample.cpu_percent),
        memory_percent=float(sample.memory_percent),
        gpus=gpus,
        available=True,
    )
