"""Engine-independent swing comparison metrics package (Issue #11164).

Provides unified data models and pure functions for comparing two golf swings
(e.g., student vs pro, simulation vs mocap, owner vs tour-average baseline).
"""

from __future__ import annotations

from src.shared.python.swing_comparison.events import (
    SwingEvents,
    detect_events,
)
from src.shared.python.swing_comparison.metrics import (
    ClubMetrics,
    ComparisonReport,
    HandPathMetrics,
    KinematicSequenceMetrics,
    LeadArmMetrics,
    SegmentKinematicPeak,
    SegmentRotationMetrics,
    SwingMetrics,
    TempoMetrics,
    WristMetrics,
    compare,
    compute_all_metrics,
    compute_club_metrics,
    compute_hand_path_metrics,
    compute_kinematic_sequence,
    compute_lead_arm_metrics,
    compute_segment_rotations,
    compute_tempo,
    compute_wrist_metrics,
)
from src.shared.python.swing_comparison.motion import (
    CAPTURE_A_CLUBHEAD_LABELS,
    CAPTURE_A_GRIP_LABELS,
    CAPTURE_A_MARKER_LABELS,
    CAPTURE_A_PELVIS_LEFT_LABELS,
    CAPTURE_A_PELVIS_RIGHT_LABELS,
    CAPTURE_A_SHOULDER_LEFT_LABELS,
    CAPTURE_A_SHOULDER_RIGHT_LABELS,
    SwingMotion,
    swing_motion_from_markers,
)
from src.shared.python.swing_comparison.report import (
    comparison_to_dict,
    comparison_to_markdown,
)

__all__ = [
    "CAPTURE_A_CLUBHEAD_LABELS",
    "CAPTURE_A_GRIP_LABELS",
    "CAPTURE_A_MARKER_LABELS",
    "CAPTURE_A_PELVIS_LEFT_LABELS",
    "CAPTURE_A_PELVIS_RIGHT_LABELS",
    "CAPTURE_A_SHOULDER_LEFT_LABELS",
    "CAPTURE_A_SHOULDER_RIGHT_LABELS",
    "ClubMetrics",
    "ComparisonReport",
    "HandPathMetrics",
    "KinematicSequenceMetrics",
    "LeadArmMetrics",
    "SegmentKinematicPeak",
    "SegmentRotationMetrics",
    "SwingEvents",
    "SwingMetrics",
    "SwingMotion",
    "TempoMetrics",
    "WristMetrics",
    "compare",
    "comparison_to_dict",
    "comparison_to_markdown",
    "compute_all_metrics",
    "compute_club_metrics",
    "compute_hand_path_metrics",
    "compute_kinematic_sequence",
    "compute_lead_arm_metrics",
    "compute_segment_rotations",
    "compute_tempo",
    "compute_wrist_metrics",
    "detect_events",
    "swing_motion_from_markers",
]
