"""Training-Controller dashboard tab — headless portion (issue #6012).

This package houses the in-launcher training dashboard. The PyQt6
widget surface is deferred to a follow-up PR (the remote environment
authoring this package has no display). What ships now is the
GUI-free triad:

- :class:`controller.TrainingDashboardController` — MVC controller
  binding the backend :class:`training.Scheduler` to the read-model.
- :class:`live_subscriber.TrainingJobLiveSubscriber` — realtime
  subscription wrapper that decodes ``training/<job_id>/progress``
  payloads into typed events.
- :mod:`view_model` — frozen dataclasses the GUI layer renders.

Layout follows the ``starting_pose_matcher`` precedent (AGENTS.md §C):
PyQt-free modules live alongside the eventual ``gui.py`` /
``_embed_adapter.py`` files so the follow-up PR is a pure-PyQt diff
with no headless logic mixed in.
"""

from __future__ import annotations

import contextlib

from .controller import (
    DEFAULT_ROLLING_WINDOW,
    ModelChangeCallback,
    ResourceProvider,
    TrainingDashboardController,
)
from .live_subscriber import (
    MetricCallback,
    StatusCallback,
    TrainingJobLiveSubscriber,
)
from .view_model import (
    DashboardModel,
    GpuSnapshot,
    JobRow,
    MetricSeries,
    ResourceSnapshot,
    job_row_from_training_job,
)


def _register_embed_adapter() -> None:
    """Register the launcher embed adapter without importing PyQt."""

    with contextlib.suppress(ImportError, ValueError):
        from src.shared.python.launcher_embed import (
            get_embeddable_tool,
            register_embeddable_tool,
        )

        from ._embed_adapter import _TrainingControllerEmbedAdapter

        adapter = _TrainingControllerEmbedAdapter()
        if get_embeddable_tool(adapter.tool_id) is None:
            register_embeddable_tool(adapter)


_register_embed_adapter()

__all__ = [
    "DEFAULT_ROLLING_WINDOW",
    "DashboardModel",
    "GpuSnapshot",
    "JobRow",
    "MetricCallback",
    "MetricSeries",
    "ModelChangeCallback",
    "ResourceProvider",
    "ResourceSnapshot",
    "StatusCallback",
    "TrainingDashboardController",
    "TrainingJobLiveSubscriber",
    "_register_embed_adapter",
    "job_row_from_training_job",
]
