"""Pure-Python segment length and scale estimation from tour capture (OS-3b).

Computes pairwise marker distances on frame 0 for:
- femur (knee-hip proxy via waist)
- tibia (knee-ankle)
- foot (ankle-toe)
- humerus (shoulder-elbow)
- forearm (elbow-wrist)
- torso (waist-shoulder)

Evaluates the rigid assumption over the first 20 frames (address stance),
outputs scale factors per body relative to nominal model dimensions, and
records provenance for every segment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from src.shared.python.motion_matching.tour_capture_contract import TourCapture

# Nominal model segment lengths in metres (Rajagopal humanoid model with waist proxy).
DEFAULT_NOMINAL_LENGTHS_M: MappingProxyType[str, float] = MappingProxyType(
    {
        "femur_r": 0.5450,
        "femur_l": 0.5350,
        "tibia_r": 0.4385,
        "tibia_l": 0.4344,
        "calcn_r": 0.1621,
        "calcn_l": 0.1602,
        "humerus_r": 0.2863,
        "humerus_l": 0.2863,
        "radius_r": 0.2358,
        "radius_l": 0.2358,
        "torso": 0.4700,
    }
)


@dataclass(frozen=True)
class SegmentScaleResult:
    """Estimated segment scales, measured lengths, nominal lengths, and residuals."""

    scale_factors: dict[str, float]
    measured_lengths_m: dict[str, float]
    nominal_lengths_m: dict[str, float]
    rigid_residuals_m: dict[str, float]
    provenance: dict[str, str]


def _centroid_points(
    capture: TourCapture, labels: tuple[str, ...], frame: int
) -> np.ndarray:
    """Calculate the centroid of valid specified markers at a given frame."""
    valid_points = []
    for label in labels:
        if label in capture.labels:
            idx = capture.index(label)
            if (
                capture.valid[frame, idx]
                and np.isfinite(capture.points_m[frame, idx]).all()
            ):
                valid_points.append(capture.points_m[frame, idx])
    if not valid_points:
        raise ValueError(
            f"No valid markers available at frame {frame} among candidate labels: {labels}"
        )
    return np.mean(valid_points, axis=0)


def _segment_distance(
    capture: TourCapture,
    proximal_labels: tuple[str, ...],
    distal_labels: tuple[str, ...],
    frame: int,
) -> float:
    """Compute Euclidean distance between proximal and distal marker centroids at a frame."""
    p_prox = _centroid_points(capture, proximal_labels, frame)
    p_dist = _centroid_points(capture, distal_labels, frame)
    return float(np.linalg.norm(p_prox - p_dist))


def estimate_segment_scales(
    capture: TourCapture,
    *,
    nominal_lengths_m: Mapping[str, float] | None = None,
    evaluation_frames: int = 20,
) -> SegmentScaleResult:
    """Estimate segment lengths, scaling factors, and rigid residuals from capture markers.

    Preconditions:
    - capture must have at least evaluation_frames (default 20)
    - nominal_lengths_m, if provided, must map positive finite values
    - markers required for segment pairs must be present and valid at frame 0

    Returns:
    - SegmentScaleResult with scale_factors, measured_lengths_m, nominal_lengths_m,
      rigid_residuals_m, and provenance.
    """
    if capture.frames < evaluation_frames:
        raise ValueError(
            f"Capture must have at least {evaluation_frames} frames to evaluate the rigid assumption; "
            f"got {capture.frames} frames"
        )

    nominal = dict(
        DEFAULT_NOMINAL_LENGTHS_M if nominal_lengths_m is None else nominal_lengths_m
    )
    for seg, length in nominal.items():
        if not np.isfinite(length) or length <= 0:
            raise ValueError(
                f"Nominal length for {seg} must be a positive finite float; got {length}"
            )

    # Segment definitions: (proximal_labels, distal_labels, description)
    segment_defs: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
        "femur_r": (
            ("WaistRight", "WaistRBack"),
            ("RKneeOut",),
            "waist_r centroid to RKneeOut (knee-hip proxy via waist)",
        ),
        "femur_l": (
            ("WaistLeft", "WaistLBack"),
            ("LKneeOut",),
            "waist_l centroid to LKneeOut (knee-hip proxy via waist)",
        ),
        "tibia_r": (
            ("RKneeOut",),
            ("RAnkleOut",),
            "RKneeOut to RAnkleOut (knee-ankle)",
        ),
        "tibia_l": (
            ("LKneeOut",),
            ("LAnkleOut",),
            "LKneeOut to LAnkleOut (knee-ankle)",
        ),
        "calcn_r": (
            ("RAnkleOut",),
            ("RToeIn", "RToeOut"),
            "RAnkleOut to toe_r centroid (ankle-toe)",
        ),
        "calcn_l": (
            ("LAnkleOut",),
            ("LToeIn", "LToeOut"),
            "LAnkleOut to toe_l centroid (ankle-toe)",
        ),
        "humerus_r": (
            ("RShoulderTop", "RShoulderBack"),
            ("RElbowOut",),
            "shoulder_r centroid (RShoulderBack fallback) to RElbowOut (shoulder-elbow)",
        ),
        "humerus_l": (
            ("LShoulderTop", "LShoulderBack"),
            ("LElbowOut",),
            "shoulder_l centroid to LElbowOut (shoulder-elbow)",
        ),
        "radius_r": (
            ("RElbowOut",),
            ("RWristTop",),
            "RElbowOut to RWristTop (elbow-wrist)",
        ),
        "radius_l": (
            ("LElbowOut",),
            ("LWristTop",),
            "LElbowOut to LWristTop (elbow-wrist)",
        ),
        "torso": (
            ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack"),
            ("LShoulderTop", "LShoulderBack", "RShoulderTop", "RShoulderBack"),
            "waist centroid to shoulder centroid (waist-shoulder)",
        ),
    }

    measured: dict[str, float] = {}
    scales: dict[str, float] = {}
    residuals: dict[str, float] = {}
    provenance: dict[str, str] = {}

    for seg, (prox_labels, dist_labels, desc) in segment_defs.items():
        if seg not in nominal:
            continue
        d0 = _segment_distance(capture, prox_labels, dist_labels, frame=0)
        measured[seg] = d0
        scales[seg] = d0 / nominal[seg]
        provenance[seg] = desc

        # Evaluate rigid assumption across first evaluation_frames
        series = [
            _segment_distance(capture, prox_labels, dist_labels, frame=f)
            for f in range(evaluation_frames)
        ]
        residuals[seg] = float(np.std(series))

    return SegmentScaleResult(
        scale_factors=scales,
        measured_lengths_m=measured,
        nominal_lengths_m=nominal,
        rigid_residuals_m=residuals,
        provenance=provenance,
    )
