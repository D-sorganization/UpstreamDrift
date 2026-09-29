"""Fixture training entry points used by subprocess-driver tests.

This module is NOT a test module — it is imported by worker
subprocesses via :class:`SubprocessDriver` and exercised through the
public ``entry_point`` resolution path. The functions here mimic the
real :class:`TrainingJobRunner` shape but stay framework-free so the
worker has no heavy imports to load.

All entry points have the signature::

    fn(config, *, progress, cancel) -> RunResult

matching the worker's call convention.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from training import (
    MetricKind,
    RunResult,
    TrainingMetric,
    TrainingStatus,
    new_run_id,
)


def emit_metrics(
    config: Any,
    *,
    progress: Any,
    cancel: Any,
) -> RunResult:
    """Emit N metrics then return COMPLETED.

    ``config.hyperparameters`` may contain:

    - ``num_metrics`` (int, default ``3``): how many metric events to
      emit before completing.
    - ``per_metric_sleep`` (float, default ``0.0``): seconds to sleep
      between emissions; allows the parent to inject a cancel midway.
    - ``status_message`` (str|None, default ``None``): an initial status
      announcement.
    """

    hyper = dict(config.hyperparameters)
    num_metrics = int(hyper.get("num_metrics", 3))
    per_metric_sleep = float(hyper.get("per_metric_sleep", 0.0))
    status_message = hyper.get("status_message")

    start = time.monotonic()
    progress.emit_status(TrainingStatus.RUNNING, message=status_message)
    for i in range(num_metrics):
        if cancel.is_cancelled:
            return RunResult(
                run_id=new_run_id(),
                status=TrainingStatus.CANCELLED,
                duration_s=time.monotonic() - start,
            )
        progress.emit_metric(
            TrainingMetric(
                name="loss",
                value=1.0 / (i + 1),
                step=i,
                timestamp=time.time(),
                kind=MetricKind.LOSS,
            )
        )
        if per_metric_sleep > 0:
            time.sleep(per_metric_sleep)
    return RunResult(
        run_id=new_run_id(),
        status=TrainingStatus.COMPLETED,
        duration_s=time.monotonic() - start,
    )


def slow_until_cancel(
    config: Any,
    *,
    progress: Any,
    cancel: Any,
) -> RunResult:
    """Loop forever (with cooperative cancel) — used for cancel tests.

    ``config.hyperparameters`` may contain ``poll_interval`` (float,
    default ``0.02``).
    """

    hyper = dict(config.hyperparameters)
    poll = float(hyper.get("poll_interval", 0.02))
    start = time.monotonic()
    progress.emit_status(TrainingStatus.RUNNING)
    while not cancel.is_cancelled:
        time.sleep(poll)
        # Defensive ceiling so a runaway test never blocks CI forever.
        if time.monotonic() - start > 30.0:
            return RunResult(
                run_id=new_run_id(),
                status=TrainingStatus.FAILED,
                duration_s=time.monotonic() - start,
                error="slow_until_cancel: 30s ceiling reached without cancel",
            )
    return RunResult(
        run_id=new_run_id(),
        status=TrainingStatus.CANCELLED,
        duration_s=time.monotonic() - start,
    )


def raise_immediately(
    config: Any,
    *,
    progress: Any,
    cancel: Any,
) -> RunResult:
    """Raise a :class:`RuntimeError` immediately — used for crash tests."""

    del config, progress, cancel
    raise RuntimeError("fixture: simulated training failure")


def emit_metrics_with_stderr(
    config: Any,
    *,
    progress: Any,
    cancel: Any,
) -> RunResult:
    """Write to stderr while emitting metrics — verifies stream isolation."""

    sys.stderr.write("noise on stderr — not part of the wire protocol\n")
    sys.stderr.flush()
    return emit_metrics(config, progress=progress, cancel=cancel)
