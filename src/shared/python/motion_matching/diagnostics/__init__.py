"""Motion-matching diagnostics.

Public entry points:

- :func:`compare_clubhead_traces` — numeric report comparing two ``ClubTarget``s,
  with renderers ``plot_3d_overlay``, ``plot_per_axis_timeseries``,
  ``plot_speed_comparison``, ``plot_setup_pose_skeletons``.
- :mod:`initial_state_diff` — compares the pose specified in a Simscape input
  MAT file against the pose the constraint solver settles to at ``t=0``
  (loop-closure projection): ``InitialStateDiffReport``, ``load_diff_report``,
  ``plot_skeleton_overlay``, ``plot_per_joint_delta_bars``,
  ``plot_cartesian_delta_summary``, ``summarize_for_pr_comment``.
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
from .initial_state_diff import (
    InitialStateDiffReport,
    load_diff_report,
    plot_cartesian_delta_summary,
    plot_per_joint_delta_bars,
    plot_skeleton_overlay,
    summarize_for_pr_comment,
)

__all__ = [
    "InitialStateDiffReport",
    "TraceCompareOptions",
    "TraceCompareReport",
    "compare_clubhead_traces",
    "load_diff_report",
    "plot_3d_overlay",
    "plot_cartesian_delta_summary",
    "plot_per_axis_timeseries",
    "plot_per_joint_delta_bars",
    "plot_setup_pose_skeletons",
    "plot_skeleton_overlay",
    "plot_speed_comparison",
    "summarize_for_pr_comment",
]
