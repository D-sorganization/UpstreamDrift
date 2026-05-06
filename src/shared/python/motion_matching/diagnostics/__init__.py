"""Motion-matching diagnostic comparators.

Public entry points:

- :func:`compare_clubhead_traces` — numeric report comparing two ``ClubTarget``s
- :class:`TraceCompareOptions` / :class:`TraceCompareReport`
- ``plot_3d_overlay``, ``plot_per_axis_timeseries``, ``plot_speed_comparison``,
  ``plot_setup_pose_skeletons`` — matplotlib renderers for the report.
"""

from __future__ import annotations

from .clubhead_trace import (
    TraceCompareOptions,
    TraceCompareReport,
    compare_clubhead_traces,
    plot_3d_overlay,
    plot_per_axis_timeseries,
    plot_setup_pose_skeletons,
    plot_speed_comparison,
)

__all__ = [
    "TraceCompareOptions",
    "TraceCompareReport",
    "compare_clubhead_traces",
    "plot_3d_overlay",
    "plot_per_axis_timeseries",
    "plot_setup_pose_skeletons",
    "plot_speed_comparison",
]
