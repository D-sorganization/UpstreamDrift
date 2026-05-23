"""Training runtime — runner registry, drivers, and progress sinks.

This sub-package is the execution layer that sits between the
declarative :class:`Scheduler` and the per-framework
:class:`TrainingJobRunner` adapters. It is intentionally separate from
the contracts package so that headless consumers of the contracts can
import them without dragging in threading / subprocess machinery.

Public types (see individual modules for full docs):

- :class:`RunnerRegistry` — framework → adapter lookup.
- :class:`InMemoryProgressSink`, :class:`JsonlFileProgressSink`,
  :class:`CompositeProgressSink` — progress drains.
- :class:`Driver` — Protocol for execution backends.
- :class:`InProcessDriver` — thread-per-job execution in the current
  process. The default backend; subprocess + ray backends layer on
  later.
"""

from __future__ import annotations

from .adapters import PyTorchCVAERunner
from .driver import (
    Driver,
    DriverError,
    JobHandle,
    JobHandleStatus,
)
from .in_process_driver import InProcessDriver
from .progress_sinks import (
    CompositeProgressSink,
    InMemoryProgressSink,
    JsonlFileProgressSink,
    NullProgressSink,
    RealtimeChannelProgressSink,
    training_channel_for,
)
from .runner_registry import RunnerRegistry

__all__ = [
    "CompositeProgressSink",
    "Driver",
    "DriverError",
    "InMemoryProgressSink",
    "InProcessDriver",
    "JobHandle",
    "JobHandleStatus",
    "JsonlFileProgressSink",
    "NullProgressSink",
    "PyTorchCVAERunner",
    "RealtimeChannelProgressSink",
    "RunnerRegistry",
    "training_channel_for",
]
