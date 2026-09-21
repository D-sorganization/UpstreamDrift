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
import math

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
    diff = p_prox - p_dist
    return float(math.sqrt(np.vdot(diff, diff)))  # Bolt optimization


# Segment definitions: (proximal_labels, distal_labels, description)
SEGMENT_DEFINITIONS: MappingProxyType[
    str, tuple[tuple[str, ...], tuple[str, ...], str]
] = MappingProxyType(
    {
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
)


def _validate_scale_inputs(
    capture: TourCapture, nominal: Mapping[str, float], evaluation_frames: int
) -> None:
    if capture.frames < evaluation_frames:
        raise ValueError(
            f"Capture must have at least {evaluation_frames} frames to evaluate the rigid assumption; "
            f"got {capture.frames} frames"
        )
    for seg, length in nominal.items():
        if not np.isfinite(length) or length <= 0:
            raise ValueError(
                f"Nominal length for {seg} must be a positive finite float; got {length}"
            )


def _marker_is_valid(capture: TourCapture, label: str, frame: int) -> bool:
    """Check if a marker is present and valid at a given frame."""
    if label not in capture.labels:
        return False
    idx = capture.index(label)
    return bool(
        capture.valid[frame, idx] and np.isfinite(capture.points_m[frame, idx]).all()
    )


def _humerus_distance(
    capture: TourCapture,
    side: str,
    frame: int,
) -> tuple[float, str]:
    """Compute humerus segment distance with bilateral acromion proxy correction.

    When both Top and Back shoulder markers are valid, uses their centroid.
    When one side is missing the Top marker (e.g. RShoulderTop occluded at address),
    reconstructs the acromion proxy distance using the contralateral side's
    (centroid / back) ratio to prevent artificial length inflation.
    """
    top_label = f"{side}ShoulderTop"
    back_label = f"{side}ShoulderBack"
    elbow_label = f"{side}ElbowOut"

    has_top = _marker_is_valid(capture, top_label, frame)
    has_back = _marker_is_valid(capture, back_label, frame)

    if has_top and has_back:
        d = _segment_distance(capture, (top_label, back_label), (elbow_label,), frame)
        return d, f"shoulder_{side.lower()} centroid to {elbow_label}"

    if has_back:
        # Check if contralateral side has both markers to supply a geometric ratio
        contra = "L" if side == "R" else "R"
        c_top = f"{contra}ShoulderTop"
        c_back = f"{contra}ShoulderBack"
        c_elbow = f"{contra}ElbowOut"
        if _marker_is_valid(capture, c_top, frame) and _marker_is_valid(
            capture, c_back, frame
        ):
            d_c_cent = _segment_distance(capture, (c_top, c_back), (c_elbow,), frame)
            d_c_back = _segment_distance(capture, (c_back,), (c_elbow,), frame)
            ratio = d_c_cent / d_c_back if d_c_back > 0 else 1.0
            d_back = _segment_distance(capture, (back_label,), (elbow_label,), frame)
            d = d_back * ratio
            return (
                d,
                f"{back_label} to {elbow_label} reconstructed via contralateral centroid/back ratio ({ratio:.4f})",
            )
        # Fallback to back marker alone
        d = _segment_distance(capture, (back_label,), (elbow_label,), frame)
        return d, f"{back_label} fallback to {elbow_label}"

    if has_top:
        d = _segment_distance(capture, (top_label,), (elbow_label,), frame)
        return d, f"{top_label} fallback to {elbow_label}"

    raise ValueError(f"Neither {top_label} nor {back_label} is valid at frame {frame}")


def estimate_segment_scales(
    capture: TourCapture,
    *,
    nominal_lengths_m: Mapping[str, float] | None = None,
    evaluation_frames: int = 20,
) -> SegmentScaleResult:
    """Estimate segment lengths, scaling factors, and rigid residuals from capture markers."""
    nominal = dict(
        DEFAULT_NOMINAL_LENGTHS_M if nominal_lengths_m is None else nominal_lengths_m
    )
    _validate_scale_inputs(capture, nominal, evaluation_frames)

    measured: dict[str, float] = {}
    scales: dict[str, float] = {}
    residuals: dict[str, float] = {}
    provenance: dict[str, str] = {}

    for seg, (prox_labels, dist_labels, desc) in SEGMENT_DEFINITIONS.items():
        if seg not in nominal:
            continue

        if seg in ("humerus_r", "humerus_l"):
            side = "R" if seg == "humerus_r" else "L"
            d0, prov_desc = _humerus_distance(capture, side, frame=0)
            series = [
                _humerus_distance(capture, side, frame=f)[0]
                for f in range(evaluation_frames)
            ]
        else:
            d0 = _segment_distance(capture, prox_labels, dist_labels, frame=0)
            prov_desc = desc
            series = [
                _segment_distance(capture, prox_labels, dist_labels, frame=f)
                for f in range(evaluation_frames)
            ]

        measured[seg] = d0
        scales[seg] = d0 / nominal[seg]
        provenance[seg] = prov_desc
        residuals[seg] = float(np.std(series))

    return SegmentScaleResult(
        scale_factors=scales,
        measured_lengths_m=measured,
        nominal_lengths_m=nominal,
        rigid_residuals_m=residuals,
        provenance=provenance,
    )
