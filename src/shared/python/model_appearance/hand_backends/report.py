"""Pose records shared by the solver, the metrics and the tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FlatPlacement:
    """Dieter's flat hand, measured before the palm arch eats the cross.

    ``index_axis_distance_m`` is the proximal tangent point, not the dorsal
    knuckle. ``index_azimuth_rad`` and ``pinky_azimuth_rad`` are the knuckle
    azimuths before shaft roll.
    """

    index_gap_m: float
    pinky_gap_m: float
    index_axis_distance_m: float
    index_to_shaft_deg: float
    index_azimuth_rad: float
    pinky_azimuth_rad: float


@dataclass(frozen=True)
class SegmentPose:
    """One posed bone capsule in the appearance grip frame, after shaft roll."""

    name: str
    finger: str
    segment: str
    p0: tuple[float, float, float]
    p1: tuple[float, float, float]
    angle_rad: float
    gap_m: float
    grip_gap_m: float
    penetration_m: float
    neighbour_gap_m: float | None
    wrap_rad: float
    radius_m: float
    seat_radius_m: float
    tangent_azimuth_rad: float
    check_grip: bool
    check_neighbour: bool


@dataclass(frozen=True)
class ParallelDonor:
    """One bone the overlap pinky copies, already in the unrolled grip frame."""

    p0: tuple[float, float, float]
    p1: tuple[float, float, float]
    radius_m: float
    angle_rad: float
    wrap_rad: float


@dataclass(frozen=True)
class FingerOverride:
    """Per-finger replacement used for overlap and interlock.

    ``azimuth_rad`` is the tangent solver's base azimuth (before the
    ``atan2(before, d)`` offset). ``seat_radius_m`` skips the curl-down when
    the finger must sit on a neighbour instead of the grip. ``parallel_donors``
    lays the finger on those bones instead of on its own tangent circle.
    """

    station_z_m: float | None = None
    azimuth_rad: float | None = None
    clearance_extra_m: float | None = None
    seat_radius_m: float | None = None
    curl_sign: int | None = None
    grip_required_segments: tuple[str, ...] = ("proximal", "middle", "distal")
    neighbour_required_segments: tuple[str, ...] = ()
    parallel_donors: tuple[ParallelDonor, ...] | None = None


@dataclass(frozen=True)
class HandPose:
    """One hand: joint angles, capsules, the flat snapshot and the palm patch."""

    side: str
    curl_sign: int
    shaft_roll_deg: float
    angles_rad: Mapping[str, float]
    segments: tuple[SegmentPose, ...]
    flat: FlatPlacement | None
    palm_contact_area_m2: float

    def angle(self, name: str) -> float:
        """Joint angle in radians. Raises ``KeyError`` when ``name`` is absent."""
        return float(self.angles_rad[name])

    def segment(self, finger: str, segment: str) -> SegmentPose:
        """The capsule named by ``finger`` and ``segment``."""
        for item in self.segments:
            if item.finger == finger and item.segment == segment:
                return item
        raise KeyError(f"no segment {finger}/{segment} on the {self.side} hand")


@dataclass(frozen=True)
class GripPoseSolution:
    """Lead and trail poses plus the contact-quality record and the GB2 note."""

    lead: HandPose
    trail: HandPose
    metrics: Mapping[str, Any]
    gb2: Mapping[str, Any]
    trigger_finger_applied: bool
    coupling_overrides_applied: bool


__all__ = [
    "FingerOverride",
    "FlatPlacement",
    "GripPoseSolution",
    "HandPose",
    "ParallelDonor",
    "SegmentPose",
]
