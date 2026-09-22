"""Simscape kinematic topology classification for MS-61 (#10348).

The committed 27-coordinate GolfSwing3D Kinetic model attaches head markers to
the Hub body and has no independent neck DOFs. That reduced topology is a
validation oracle, not a full-body G3 flagship (MS-104 / MS-109). Classification
is fail-closed and does not invent licensed MATLAB success.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.shared.python.contracts import postcondition, precondition

REDUCED_27_COORDINATE_COUNT = 27
HEAD_MARKER_NAMES = ("HeadTop", "HeadFront", "HeadSide")
_NECK_TOKEN = "neck"


class SimscapeTopologyProfile(str, Enum):
    """Kinematic capability profile for Simscape matched-swing models."""

    REDUCED_27_NO_NECK = "reduced_27_no_neck"
    FULL_BODY_WITH_NECK = "full_body_with_neck"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SimscapeTopologyReport:
    """Structured topology diagnosis for receipts and native gates."""

    profile: SimscapeTopologyProfile
    coordinate_count: int
    has_independent_neck: bool
    head_markers: tuple[str, ...]
    head_marker_bodies: tuple[str, ...]
    neck_coordinate_names: tuple[str, ...]
    limitation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "coordinate_count": self.coordinate_count,
            "has_independent_neck": self.has_independent_neck,
            "head_markers": list(self.head_markers),
            "head_marker_bodies": list(self.head_marker_bodies),
            "neck_coordinate_names": list(self.neck_coordinate_names),
            "limitation": self.limitation,
        }


def _neck_coordinates(coordinate_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in coordinate_names if _NECK_TOKEN in name.lower())


@precondition(
    lambda coordinate_names, marker_labels, marker_bodies: (
        isinstance(coordinate_names, Sequence)
        and isinstance(marker_labels, Sequence)
        and isinstance(marker_bodies, Sequence)
    ),
    "topology inputs must be sequences",
)
def validate_simscape_topology(
    *,
    coordinate_names: Sequence[str],
    marker_labels: Sequence[str],
    marker_bodies: Sequence[str],
) -> None:
    """Fail closed if topology identity inputs are incomplete or inconsistent."""
    if not coordinate_names:
        raise ValueError("coordinate_names must be non-empty")
    if any(not str(name).strip() for name in coordinate_names):
        raise ValueError("coordinate_names must be non-empty strings")
    if len(marker_labels) != len(marker_bodies):
        raise ValueError(
            "marker_bodies length must match marker_labels "
            f"({len(marker_bodies)} != {len(marker_labels)})"
        )
    if not marker_labels:
        raise ValueError("marker_labels must be non-empty")
    if any(not str(label).strip() for label in marker_labels):
        raise ValueError("marker_labels must be non-empty strings")
    if any(not str(body).strip() for body in marker_bodies):
        raise ValueError("marker_bodies must be non-empty strings")
    missing_heads = [name for name in HEAD_MARKER_NAMES if name not in marker_labels]
    if missing_heads:
        raise ValueError(f"missing required head markers: {missing_heads}")


@postcondition(
    lambda report: isinstance(report, SimscapeTopologyReport),
    "topology report must be a SimscapeTopologyReport",
)
def classify_simscape_topology(
    *,
    coordinate_names: Sequence[str],
    marker_labels: Sequence[str],
    marker_bodies: Sequence[str],
) -> SimscapeTopologyReport:
    """Classify Simscape model topology; never claims native qualification."""
    validate_simscape_topology(
        coordinate_names=coordinate_names,
        marker_labels=marker_labels,
        marker_bodies=marker_bodies,
    )
    necks = _neck_coordinates(coordinate_names)
    head_bodies = tuple(
        str(marker_bodies[marker_labels.index(name)]) for name in HEAD_MARKER_NAMES
    )
    has_neck = len(necks) > 0
    if has_neck:
        profile = SimscapeTopologyProfile.FULL_BODY_WITH_NECK
        limitation = (
            "Independent neck DOFs present; full-body topology capability noted. "
            "Native G1/G3 acceptance still requires MS-100 receipts and MS-104 coverage."
        )
    elif len(coordinate_names) == REDUCED_27_COORDINATE_COUNT:
        profile = SimscapeTopologyProfile.REDUCED_27_NO_NECK
        limitation = (
            "Reduced 27-coordinate model has no independent neck; head markers attach "
            "to Hub. Terminal full-marker error is thorax-head coupled. Retained as a "
            "reduced-model oracle; MS-104 (#10378) owns full-body Simscape flagship work."
        )
    else:
        profile = SimscapeTopologyProfile.UNKNOWN
        limitation = (
            f"Unrecognized coordinate count {len(coordinate_names)} without neck DOFs; "
            "fail closed for full-body claims until topology is inventoried (MS-102)."
        )
    return SimscapeTopologyReport(
        profile=profile,
        coordinate_count=len(coordinate_names),
        has_independent_neck=has_neck,
        head_markers=HEAD_MARKER_NAMES,
        head_marker_bodies=head_bodies,
        neck_coordinate_names=necks,
        limitation=limitation,
    )
