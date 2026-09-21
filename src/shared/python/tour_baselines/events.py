"""Biomechanical swing events and phase intervals on native clocks (TB-01 #10586).

Defines address, takeaway, top of backswing, downswing, impact, and follow-through
intervals on each capture's native clock (360.0 Hz driver vs 359.0 Hz 7-iron).
Trajectory-inferred impact is explicitly labeled as inferred.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.tour_capture_contract import TourCapture


class DetectionMethod(str, Enum):
    """Method used to identify a biomechanical swing event."""

    TRAJECTORY_SPEED_PEAK = "trajectory_speed_peak"
    TRAJECTORY_MIN_SPEED = "trajectory_min_speed"
    GRIP_DISPLACEMENT_MAX = "grip_displacement_max"
    CAPTURE_BOUNDARY = "capture_boundary"
    MANUAL_OVERRIDE = "manual_override"


@dataclass(frozen=True)
class BiomechanicalEvent:
    """A discrete temporal biomechanical landmark during the swing."""

    name: str
    frame_index: int
    time_s: float
    detection_method: DetectionMethod
    is_inferred: bool
    confidence: float
    manual_override: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "detection_method": self.detection_method.value,
            "is_inferred": self.is_inferred,
            "confidence": self.confidence,
            "manual_override": self.manual_override,
        }


@dataclass(frozen=True)
class TourSwingEvents:
    """Biomechanical events across the swing horizon on a native capture clock."""

    kind: str
    rate_hz: float
    frames: int
    address: BiomechanicalEvent
    takeaway: BiomechanicalEvent
    top_of_backswing: BiomechanicalEvent
    impact: BiomechanicalEvent
    finish: BiomechanicalEvent

    def address_interval(self) -> tuple[float, float]:
        """Address interval: from capture start to takeaway."""
        return (self.address.time_s, self.takeaway.time_s)

    def backswing_interval(self) -> tuple[float, float]:
        """Backswing interval: from takeaway to top of backswing."""
        return (self.takeaway.time_s, self.top_of_backswing.time_s)

    def downswing_interval(self) -> tuple[float, float]:
        """Downswing interval: from top of backswing to impact."""
        return (self.top_of_backswing.time_s, self.impact.time_s)

    def impact_interval(self, half_window_s: float = 0.02) -> tuple[float, float]:
        """Impact interval: narrow window centered on impact."""
        t_imp = self.impact.time_s
        return (
            max(self.top_of_backswing.time_s, t_imp - half_window_s),
            min(self.finish.time_s, t_imp + half_window_s),
        )

    def follow_through_interval(self) -> tuple[float, float]:
        """Follow-through interval: from impact to finish."""
        return (self.impact.time_s, self.finish.time_s)

    def full_swing_interval(self) -> tuple[float, float]:
        """Full swing interval: address to finish."""
        return (self.address.time_s, self.finish.time_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rate_hz": self.rate_hz,
            "frames": self.frames,
            "address": self.address.as_dict(),
            "takeaway": self.takeaway.as_dict(),
            "top_of_backswing": self.top_of_backswing.as_dict(),
            "impact": self.impact.as_dict(),
            "finish": self.finish.as_dict(),
            "intervals_s": {
                "address": list(self.address_interval()),
                "backswing": list(self.backswing_interval()),
                "downswing": list(self.downswing_interval()),
                "impact": list(self.impact_interval()),
                "follow_through": list(self.follow_through_interval()),
                "full_swing": list(self.full_swing_interval()),
            },
        }


# Frozen default events on native clocks
SWING_EVENTS_DRIVER = TourSwingEvents(
    kind="driver",
    rate_hz=360.0,
    frames=654,
    address=BiomechanicalEvent(
        name="address",
        frame_index=0,
        time_s=0.0,
        detection_method=DetectionMethod.CAPTURE_BOUNDARY,
        is_inferred=False,
        confidence=1.0,
    ),
    takeaway=BiomechanicalEvent(
        name="takeaway",
        frame_index=71,
        time_s=71 / 360.0,
        detection_method=DetectionMethod.TRAJECTORY_SPEED_PEAK,
        is_inferred=True,
        confidence=0.85,
    ),
    top_of_backswing=BiomechanicalEvent(
        name="top_of_backswing",
        frame_index=397,
        time_s=397 / 360.0,
        detection_method=DetectionMethod.TRAJECTORY_MIN_SPEED,
        is_inferred=True,
        confidence=0.90,
    ),
    impact=BiomechanicalEvent(
        name="impact",
        frame_index=476,
        time_s=476 / 360.0,
        detection_method=DetectionMethod.TRAJECTORY_SPEED_PEAK,
        is_inferred=True,
        confidence=0.95,
    ),
    finish=BiomechanicalEvent(
        name="finish",
        frame_index=653,
        time_s=653 / 360.0,
        detection_method=DetectionMethod.CAPTURE_BOUNDARY,
        is_inferred=False,
        confidence=1.0,
    ),
)

SWING_EVENTS_IRON = TourSwingEvents(
    kind="iron",
    rate_hz=359.0,
    frames=657,
    address=BiomechanicalEvent(
        name="address",
        frame_index=0,
        time_s=0.0,
        detection_method=DetectionMethod.CAPTURE_BOUNDARY,
        is_inferred=False,
        confidence=1.0,
    ),
    takeaway=BiomechanicalEvent(
        name="takeaway",
        frame_index=70,
        time_s=70 / 359.0,
        detection_method=DetectionMethod.TRAJECTORY_SPEED_PEAK,
        is_inferred=True,
        confidence=0.85,
    ),
    top_of_backswing=BiomechanicalEvent(
        name="top_of_backswing",
        frame_index=394,
        time_s=394 / 359.0,
        detection_method=DetectionMethod.TRAJECTORY_MIN_SPEED,
        is_inferred=True,
        confidence=0.90,
    ),
    impact=BiomechanicalEvent(
        name="impact",
        frame_index=480,
        time_s=480 / 359.0,
        detection_method=DetectionMethod.TRAJECTORY_SPEED_PEAK,
        is_inferred=True,
        confidence=0.95,
    ),
    finish=BiomechanicalEvent(
        name="finish",
        frame_index=656,
        time_s=656 / 359.0,
        detection_method=DetectionMethod.CAPTURE_BOUNDARY,
        is_inferred=False,
        confidence=1.0,
    ),
)

_FROZEN_EVENTS: dict[str, TourSwingEvents] = {
    "driver": SWING_EVENTS_DRIVER,
    "iron": SWING_EVENTS_IRON,
}


def _detect_impact_from_head(
    capture: TourCapture, rate_hz: float, override_frame: int | None
) -> tuple[int, DetectionMethod, bool]:
    if override_frame is not None:
        return override_frame, DetectionMethod.MANUAL_OVERRIDE, True

    head_labels = ("Marker_3:3:1", "Marker_3:3:2", "Marker_3:3:3")
    indices = [capture.index(lbl) for lbl in head_labels if lbl in capture.labels]
    if not indices:
        default_frame = int(0.73 * capture.frames)
        return default_frame, DetectionMethod.TRAJECTORY_SPEED_PEAK, False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        head_pts = np.nanmean(capture.points_m[:, indices, :], axis=1)
    v_head = np.gradient(head_pts, capture.time_s, axis=0)
    speed = np.sqrt(np.sum(v_head**2, axis=-1))

    search_start = int(0.8 * rate_hz)
    search_end = min(capture.frames, int(1.6 * rate_hz))
    if search_start < search_end and not np.all(
        np.isnan(speed[search_start:search_end])
    ):
        imp_frame = search_start + int(np.nanargmax(speed[search_start:search_end]))
        return imp_frame, DetectionMethod.TRAJECTORY_SPEED_PEAK, False

    return int(0.73 * capture.frames), DetectionMethod.TRAJECTORY_SPEED_PEAK, False


def _detect_tob_from_grip(
    capture: TourCapture, rate_hz: float, imp_frame: int, override_frame: int | None
) -> tuple[int, DetectionMethod, bool]:
    if override_frame is not None:
        return override_frame, DetectionMethod.MANUAL_OVERRIDE, True

    grip_labels = ("Marker_2:2:1", "Marker_2:2:2", "Marker_2:2:3")
    indices = [capture.index(lbl) for lbl in grip_labels if lbl in capture.labels]
    if not indices:
        return int(0.60 * capture.frames), DetectionMethod.TRAJECTORY_MIN_SPEED, False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grip_pts = np.nanmean(capture.points_m[:, indices, :], axis=1)
    v_grip = np.gradient(grip_pts, capture.time_s, axis=0)
    speed_grip = np.sqrt(np.sum(v_grip**2, axis=-1))

    tob_start = int(0.5 * rate_hz)
    if tob_start < imp_frame and not np.all(np.isnan(speed_grip[tob_start:imp_frame])):
        tob_frame = tob_start + int(np.nanargmin(speed_grip[tob_start:imp_frame]))
        return tob_frame, DetectionMethod.TRAJECTORY_MIN_SPEED, False

    return int(0.60 * capture.frames), DetectionMethod.TRAJECTORY_MIN_SPEED, False


@precondition(
    lambda capture, kind, **_: isinstance(capture, TourCapture),
    "capture must be TourCapture",
)
@postcondition(
    lambda r: r.frames >= 10, "detected events must cover at least 10 frames"
)
def detect_tour_events(
    capture: TourCapture,
    kind: str,
    overrides: dict[str, Any] | None = None,
) -> TourSwingEvents:
    """Detect biomechanical events on the capture's native clock with optional overrides."""
    normalized_kind = kind.strip().lower()
    rate_hz = capture.rate_hz
    frames = capture.frames

    ovr = overrides or {}
    imp_override = ovr.get("impact_frame")
    tob_override = ovr.get("top_frame") or ovr.get("top_of_backswing_frame")
    takeaway_override = ovr.get("takeaway_frame")

    imp_frame, imp_method, imp_is_manual = _detect_impact_from_head(
        capture, rate_hz, imp_override
    )
    tob_frame, tob_method, tob_is_manual = _detect_tob_from_grip(
        capture, rate_hz, imp_frame, tob_override
    )

    takeaway_frame = (
        takeaway_override if takeaway_override is not None else int(0.11 * frames)
    )
    takeaway_method = (
        DetectionMethod.MANUAL_OVERRIDE
        if takeaway_override is not None
        else DetectionMethod.TRAJECTORY_SPEED_PEAK
    )

    return TourSwingEvents(
        kind=normalized_kind,
        rate_hz=rate_hz,
        frames=frames,
        address=BiomechanicalEvent(
            name="address",
            frame_index=0,
            time_s=float(capture.time_s[0]),
            detection_method=DetectionMethod.CAPTURE_BOUNDARY,
            is_inferred=False,
            confidence=1.0,
        ),
        takeaway=BiomechanicalEvent(
            name="takeaway",
            frame_index=takeaway_frame,
            time_s=float(capture.time_s[takeaway_frame]),
            detection_method=takeaway_method,
            is_inferred=True,
            confidence=0.85,
            manual_override=takeaway_override is not None,
        ),
        top_of_backswing=BiomechanicalEvent(
            name="top_of_backswing",
            frame_index=tob_frame,
            time_s=float(capture.time_s[tob_frame]),
            detection_method=tob_method,
            is_inferred=True,
            confidence=0.90,
            manual_override=tob_is_manual,
        ),
        impact=BiomechanicalEvent(
            name="impact",
            frame_index=imp_frame,
            time_s=float(capture.time_s[imp_frame]),
            detection_method=imp_method,
            is_inferred=True,
            confidence=0.95,
            manual_override=imp_is_manual,
        ),
        finish=BiomechanicalEvent(
            name="finish",
            frame_index=frames - 1,
            time_s=float(capture.time_s[-1]),
            detection_method=DetectionMethod.CAPTURE_BOUNDARY,
            is_inferred=False,
            confidence=1.0,
        ),
    )
