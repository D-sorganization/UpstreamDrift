"""Concrete :class:`ProgressSink` implementations.

A sink is the bridge between a running training loop and the rest of
the system: tests inspect an in-memory sink, the dashboard subscribes
to a realtime-channel sink (PR3), and durable storage uses the
JSONL-file sink. The :class:`CompositeProgressSink` fans an emission
out to several sinks so callers do not have to wrap one sink in a
threading-aware multiplexer.

All sinks are safe to call from the runner's thread; the JSONL sink
is also safe across threads — appends are flushed under a lock.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from pathlib import Path

from ..errors import TrainingError
from ..metrics import TrainingMetric
from ..persistence import training_metric_to_dict
from ..status import TrainingStatus

__all__ = [
    "CompositeProgressSink",
    "InMemoryProgressSink",
    "JsonlFileProgressSink",
    "NullProgressSink",
    "ProgressSinkError",
]


class ProgressSinkError(TrainingError, OSError):
    """Raised when a sink cannot persist an emission (e.g. disk full)."""


class NullProgressSink:
    """A sink that drops every emission silently.

    Useful as a default in test helpers that don't care about
    progress, and as a placeholder when a runner is invoked without a
    real sink.
    """

    __slots__ = ()

    def emit_metric(self, metric: TrainingMetric) -> None:
        return None

    def emit_status(
        self, status: TrainingStatus, *, message: str | None = None
    ) -> None:
        return None


class InMemoryProgressSink:
    """Records every emission in lists for in-process inspection.

    Thread-safe; the snapshot accessors return tuples so callers can
    iterate without coordinating with concurrent producers.
    """

    __slots__ = ("_lock", "_metrics", "_statuses")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: list[TrainingMetric] = []
        self._statuses: list[tuple[TrainingStatus, str | None]] = []

    def emit_metric(self, metric: TrainingMetric) -> None:
        if not isinstance(metric, TrainingMetric):
            raise TypeError("emit_metric expects a TrainingMetric")
        with self._lock:
            self._metrics.append(metric)

    def emit_status(
        self, status: TrainingStatus, *, message: str | None = None
    ) -> None:
        if not isinstance(status, TrainingStatus):
            raise TypeError("emit_status expects a TrainingStatus")
        with self._lock:
            self._statuses.append((status, message))

    @property
    def metrics(self) -> tuple[TrainingMetric, ...]:
        """Snapshot of every metric emission, in order."""

        with self._lock:
            return tuple(self._metrics)

    @property
    def statuses(self) -> tuple[tuple[TrainingStatus, str | None], ...]:
        """Snapshot of every status emission, in order."""

        with self._lock:
            return tuple(self._statuses)

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()
            self._statuses.clear()


class JsonlFileProgressSink:
    """Append-only JSONL sink for durable metric streams.

    Each metric becomes one line; status changes are also written as
    ``{"event": "status", ...}`` records so a downstream consumer can
    reconstruct the lifecycle from the file alone.
    """

    __slots__ = ("_lock", "_metrics_path", "_status_path")

    def __init__(
        self,
        metrics_path: Path,
        *,
        status_path: Path | None = None,
    ) -> None:
        if not isinstance(metrics_path, Path):
            raise TypeError(
                f"metrics_path must be a pathlib.Path (got {type(metrics_path).__name__})"
            )
        if status_path is not None and not isinstance(status_path, Path):
            raise TypeError(
                f"status_path must be a pathlib.Path or None "
                f"(got {type(status_path).__name__})"
            )
        self._metrics_path = metrics_path
        self._status_path = status_path or metrics_path.with_suffix(".status.jsonl")
        self._lock = threading.Lock()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._status_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def metrics_path(self) -> Path:
        return self._metrics_path

    @property
    def status_path(self) -> Path:
        return self._status_path

    def emit_metric(self, metric: TrainingMetric) -> None:
        if not isinstance(metric, TrainingMetric):
            raise TypeError("emit_metric expects a TrainingMetric")
        line = json.dumps(training_metric_to_dict(metric), separators=(",", ":"))
        with self._lock:
            try:
                with self._metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                raise ProgressSinkError(
                    f"failed to append metric to {self._metrics_path}: {exc}"
                ) from exc

    def emit_status(
        self, status: TrainingStatus, *, message: str | None = None
    ) -> None:
        if not isinstance(status, TrainingStatus):
            raise TypeError("emit_status expects a TrainingStatus")
        payload = {"event": "status", "status": status.value, "message": message}
        line = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            try:
                with self._status_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                raise ProgressSinkError(
                    f"failed to append status to {self._status_path}: {exc}"
                ) from exc


class CompositeProgressSink:
    """Fans an emission out to multiple sinks.

    Errors from individual sinks are isolated — a failure in one sink
    does not prevent the rest from receiving the emission. The first
    error (if any) is re-raised after the fan-out completes so callers
    notice without losing the partial broadcast.
    """

    __slots__ = ("_sinks",)

    def __init__(self, sinks: Iterable[object]) -> None:
        self._sinks: tuple[object, ...] = tuple(sinks)
        if not self._sinks:
            raise ValueError("CompositeProgressSink requires at least one sink")

    def emit_metric(self, metric: TrainingMetric) -> None:
        self._broadcast("emit_metric", (metric,), {})

    def emit_status(
        self, status: TrainingStatus, *, message: str | None = None
    ) -> None:
        self._broadcast("emit_status", (status,), {"message": message})

    def _broadcast(
        self,
        method_name: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        first_error: BaseException | None = None
        for sink in self._sinks:
            method = getattr(sink, method_name, None)
            if method is None:
                continue
            try:
                method(*args, **kwargs)
            except (ProgressSinkError, OSError, ValueError, TypeError) as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
