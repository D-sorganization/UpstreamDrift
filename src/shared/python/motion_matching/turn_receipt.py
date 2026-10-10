"""Shared ``turn`` block for matched-swing receipts (issue #12042, slice 2).

Every matched-swing receipt writer calls :func:`attach_turn_block` (or
:func:`build_receipt_turn_block`) instead of re-deriving shoulder, trunk and
pelvis turn itself.  The block reports the shoulder-girdle, upper-trunk and
pelvis lines and both X-factors at address, top and impact plus the maximum
backswing, for the capture markers and the model side by side.  It is
descriptive only: it never fails a receipt and carries no thresholds (slice 8
adds the gate).  When turn cannot be computed the block still exists and says
why (``unavailable_reason``); unavailable is never reported as zero.

Frames: capture points arrive Y-up in metres and are mapped ``(x, -z, y)`` into
the Z-up native world (golfer faces -X, target -Y); model points must already
be in that world.  See ``swing_comparison.turn`` for the line definitions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.motion_matching.ground_support import capture_to_native_world
from src.shared.python.swing_comparison.events import SwingEvents, detect_events
from src.shared.python.swing_comparison.motion import swing_motion_from_markers
from src.shared.python.swing_comparison.turn import (
    TurnMetrics,
    build_turn_block,
    marker_turn_lines,
    model_turn_lines,
    unavailable_turn_block,
)

logger = logging.getLogger(__name__)

#: Receipt key holding the block.
TURN_KEY = "turn"

#: Writers that call this helper (checked by ``tests/unit/motion_matching``).
WIRED_WRITERS: tuple[str, ...] = (
    "src/shared/python/motion_matching/pipeline/cli.py",
    "src/engines/physics_engines/pinocchio/python/full_body_fit.py",
)
#: Matched-swing writers that do not call it yet, with the reason (follow-ups).
FOLLOW_UP_WRITERS: dict[str, str] = {
    "src/engines/physics_engines/drake/python/full_body_fit.py": (
        "Drake IK is being reworked by another agent; wire after it lands"
    ),
    "src/engines/physics_engines/opensim/python/tour_matching/document_ik.py": (
        "TRC marker frame (Y-up) and OpenSim forward markers need a frame check"
    ),
    "src/engines/physics_engines/opensim/python/tour_matching/full_swing_tracking.py": (
        "Moco receipt has no marker arrays in scope"
    ),
    "src/engines/physics_engines/myosuite/python/replay.py": (
        "Candidate carries only the club markers; needs the full capture"
    ),
    "src/engines/physics_engines/mujoco/python/replay_evidence.py": (
        "Strict pydantic receipt model; needs a schema field first"
    ),
    "src/shared/python/motion_matching/simscape_replay_harness.py": (
        "Simscape receipt is produced from MATLAB-side arrays"
    ),
    "src/shared/python/motion_matching/gs3dx_variants.py": (
        "Simscape variant receipt is produced from MATLAB-side arrays"
    ),
}


@dataclass(frozen=True)
class CaptureTurnInputs:
    """Capture markers in the native Z-up world plus the swing events."""

    t: np.ndarray
    markers: dict[str, np.ndarray]
    events: SwingEvents


def capture_turn_inputs(capture: Any) -> CaptureTurnInputs:
    """Build turn inputs from a ``TourCapture`` (Y-up metres, validity mask).

    Invalid samples become NaN.  The capture must carry the club markers that
    ``detect_events`` needs, so pass the full capture, not a body-only subset.

    Raises:
        ValueError: when no swing events can be detected from the capture.
    """
    native = capture_to_native_world(np.asarray(capture.points_m, dtype=float))
    valid = np.asarray(capture.valid, dtype=bool)
    native = np.where(valid[..., None], native, np.nan)
    t = np.asarray(capture.time_s, dtype=float)
    markers = {str(label): native[:, i, :] for i, label in enumerate(capture.labels)}
    try:
        events = detect_events(swing_motion_from_markers(t, markers))
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"swing events unavailable from capture: {exc}") from exc
    return CaptureTurnInputs(t=t, markers=markers, events=events)


def model_points_from_markers(
    labels: Sequence[str], markers_m: np.ndarray
) -> dict[str, np.ndarray]:
    """Name a model marker-site array (frames, labels, 3) for ``model_turn_lines``."""
    arr = np.asarray(markers_m, dtype=float)
    if arr.ndim != 3 or arr.shape[1] != len(labels) or arr.shape[2] != 3:
        raise ValueError(
            f"markers_m must be (frames, {len(labels)}, 3), got {arr.shape}"
        )
    return {str(label): arr[:, i, :] for i, label in enumerate(labels)}


def build_receipt_turn_block(
    capture: Any,
    *,
    model_time_s: np.ndarray,
    model_points: Mapping[str, np.ndarray],
    model_source: str,
) -> dict[str, Any]:
    """Build the receipt ``turn`` block; never raises for missing data.

    Args:
        capture: Full ``TourCapture`` (Y-up metres) carrying the club markers.
        model_time_s: Model time stamps (N,), seconds, on the capture clock.
        model_points: Named model points in the native Z-up world (metres); see
            ``swing_comparison.turn.MODEL_POINT_PAIRS`` for the accepted names.
        model_source: Provenance label for the model side (engine and method).

    Returns:
        A block validated by ``validate_turn_block``; when the inputs cannot be
        evaluated, a block with ``unavailable_reason`` instead of numbers.
    """
    try:
        cap = capture_turn_inputs(capture)
        markers: TurnMetrics = marker_turn_lines(cap.markers, cap.t, cap.events)
        model: TurnMetrics = model_turn_lines(
            model_points, np.asarray(model_time_s, dtype=float), cap.events
        )
        return build_turn_block(
            cap.events, markers=markers, model=model, model_source=model_source
        )
    except (ValueError, TypeError) as exc:
        logger.warning("turn block unavailable: %s", exc)
        return unavailable_turn_block(f"{type(exc).__name__}: {exc}")


def attach_turn_block(
    receipt: dict[str, Any],
    capture: Any,
    *,
    model_time_s: np.ndarray,
    model_points: Mapping[str, np.ndarray],
    model_source: str,
) -> dict[str, Any]:
    """Set ``receipt["turn"]`` (see ``build_receipt_turn_block``) and return it."""
    if not isinstance(receipt, dict):
        raise TypeError("receipt must be a dict")
    receipt[TURN_KEY] = build_receipt_turn_block(
        capture,
        model_time_s=model_time_s,
        model_points=model_points,
        model_source=model_source,
    )
    return receipt


def attach_turn_block_from_markers(
    receipt: dict[str, Any],
    capture: Any,
    *,
    model_time_s: np.ndarray,
    labels: Sequence[str],
    model_markers_m: np.ndarray,
    model_source: str,
) -> dict[str, Any]:
    """``attach_turn_block`` for writers that hold model marker sites (frames, labels, 3).

    The model side then uses like-for-like sites (WaistLeft/WaistRight,
    BackLeft/BackRight, ShoulderBack) rather than joint centres.  A shape
    mismatch is recorded as an unavailable block, not raised.
    """
    try:
        points = model_points_from_markers(labels, model_markers_m)
    except ValueError as exc:
        logger.warning("turn block unavailable: %s", exc)
        if not isinstance(receipt, dict):
            raise TypeError("receipt must be a dict") from exc
        receipt[TURN_KEY] = unavailable_turn_block(f"ValueError: {exc}")
        return receipt
    return attach_turn_block(
        receipt,
        capture,
        model_time_s=model_time_s,
        model_points=points,
        model_source=model_source,
    )
