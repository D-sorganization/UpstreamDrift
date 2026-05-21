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
- :func:`forward_kinematics` — minimal Python FK over the model's joint
  hierarchy. Not a Simscape replacement; for fast input-MAT inspection.
- :func:`reference_golfer_setup` / :func:`compare_to_reference` — codified
  "credible golfer setup" pose for sanity-checking input MATs.

See ``docs/golf-model/INPUT_POSE_INVESTIGATION.md`` for design context.
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
from .body_target_video import (
    BodyTargetVideoResult,
    save_body_target_video,
    save_c3d_body_video,
)
from .forward_kinematics import (
    SegmentLengths,
    SkeletonPose,
    forward_kinematics,
)
from .initial_state_diff import (
    InitialStateDiffReport,
    load_diff_report,
    plot_cartesian_delta_summary,
    plot_per_joint_delta_bars,
    plot_skeleton_overlay,
    summarize_for_pr_comment,
)
from .reference_pose import (
    REFERENCE_GOLFER_FIELDS,
    compare_to_reference,
    reference_golfer_setup,
)

__all__ = [
    "InitialStateDiffReport",
    "BodyTargetVideoResult",
    "REFERENCE_GOLFER_FIELDS",
    "SegmentLengths",
    "SkeletonPose",
    "TraceCompareOptions",
    "TraceCompareReport",
    "compare_clubhead_traces",
    "compare_to_reference",
    "forward_kinematics",
    "load_diff_report",
    "plot_3d_overlay",
    "plot_cartesian_delta_summary",
    "plot_per_axis_timeseries",
    "plot_per_joint_delta_bars",
    "plot_setup_pose_skeletons",
    "plot_skeleton_overlay",
    "plot_speed_comparison",
    "reference_golfer_setup",
    "save_body_target_video",
    "save_c3d_body_video",
    "summarize_for_pr_comment",
]
