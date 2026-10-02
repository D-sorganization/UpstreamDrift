"""Bind authored contact phases to immutable original capture evidence."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from src.shared.python.motion_matching.historical_fit import ImageFitConfig
from src.shared.python.motion_matching.historical_fit.contact_schedule import (
    ContactPinPhase,
    ScheduledConstraintOptions,
)
from .necromatcher_review import CaptureReview


def _source_pts(frame: dict[str, Any]) -> Fraction:
    return Fraction(
        frame["pts_ticks"] * frame["timebase_numerator"],
        frame["timebase_denominator"],
    )


def _phase_binding(
    phase: ContactPinPhase, frames: list[dict[str, Any]], times: list[Fraction]
) -> dict[str, Any]:
    start, end = Fraction(*phase.start_pts), Fraction(*phase.end_pts)
    boundaries = []
    for boundary in (start, end):
        matches = [index for index, time in enumerate(times) if time == boundary]
        if len(matches) != 1:
            raise ValueError("Contact boundary must identify one exact source frame")
        boundaries.append(matches[0])
    reviewed = []
    for digest in phase.review_frame_sha256:
        matches = [
            index
            for index, (frame, time) in enumerate(zip(frames, times, strict=True))
            if frame["frame_sha256"].removeprefix("sha256:")
            == digest.removeprefix("sha256:")
            and start <= time <= end
        ]
        if len(matches) != 1:
            raise ValueError("Contact review hash must identify one frame in its phase")
        reviewed.append(matches[0])
    return {
        "start_pts": list(phase.start_pts),
        "end_pts": list(phase.end_pts),
        "pinned_spheres": list(phase.pinned_spheres),
        "boundary_frame_indices": boundaries,
        "review_frame_indices": reviewed,
        "review_frames": [frames[index] for index in reviewed],
    }


def contact_schedule_binding(
    config: ImageFitConfig, source: dict[str, Any], review: CaptureReview
) -> dict[str, Any] | None:
    """Validate identities before any optimizer; returned evidence is not contact truth.

    The caller must first load the parent through the hash-verifying library.
    Phase boundaries and review hashes must resolve to original frames in the
    same source asset, shot and camera. Duplicate image hashes are ambiguous
    within a phase and require a different review frame.
    """
    options = config.constraint_options
    if not isinstance(options, ScheduledConstraintOptions):
        return None
    schedule = options.schedule
    if (
        schedule.capture_id != source["capture_id"]
        or schedule.capture_sha256 != source["capture_hash"]
        or review.capture_id != source["capture_id"]
    ):
        raise ValueError("Contact schedule differs from verified capture identity")
    frames = [review.frame(index)["frame"] for index in range(review.frame_count)]
    times = [_source_pts(frame) for frame in frames]
    anchor = source["frames"][0]
    identities = ("asset_id", "shot_id", "swing_id", "camera_id")
    if any(
        tuple(frame[key] for key in identities)
        != tuple(anchor[key] for key in identities)
        for frame in frames
    ):
        raise ValueError("Contact review crosses source asset, shot, swing or camera")
    return {
        "status": schedule.status,
        "capture_id": schedule.capture_id,
        "capture_hash": schedule.capture_sha256,
        "phases": [_phase_binding(phase, frames, times) for phase in schedule.phases],
        "continuous_certified": False,
        "normal_height_hypothesis_only": True,
        "tangential_no_slip_enforced": False,
    }
