"""Versioned measurement maps distinguishing marker semantics (TB-01 #10586).

Classifies each capture label into:
1. Observed retroreflective surface markers
2. Inferred anatomical joint centers (with explicit offset assumptions)
3. Observed rigid-body cluster centroids (grip and head)
4. Calibrated clubhead / clubface points (explicitly UNAVAILABLE in raw C3D)
5. Unassigned sentinels or unlabeled markers
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from src.shared.python.contracts import postcondition, precondition

MEASUREMENT_MAP_VERSION = "tour-measurement-map/1.0.0"


class MeasurementClass(str, Enum):
    """Categorical source classification of a motion measurement."""

    OBSERVED_SURFACE = "observed_surface"
    INFERRED_JOINT_CENTER = "inferred_joint_center"
    OBSERVED_CLUSTER_CENTROID = "observed_cluster_centroid"
    CALIBRATED_CLUB_POINT = "calibrated_club_point"
    UNASSIGNED_OR_SENTINEL = "unassigned_or_sentinel"


@dataclass(frozen=True)
class MarkerMeasurementSemantics:
    """Semantic measurement specification for a single capture channel."""

    label: str
    measurement_class: MeasurementClass
    segment: str
    is_observed: bool
    offset_assumptions: str = ""
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "measurement_class": self.measurement_class.value,
            "segment": self.segment,
            "is_observed": self.is_observed,
            "offset_assumptions": self.offset_assumptions,
            "description": self.description,
        }


# Base classifications shared between driver and 7-iron captures
_SURFACE_DEFINITIONS: dict[str, tuple[str, str]] = {
    "HeadTop": ("head", "Superior skull surface marker"),
    "HeadFront": ("head", "Anterior forehead surface marker"),
    "HeadSide": ("head", "Lateral temporal surface marker"),
    "BackTop": ("trunk", "C7 spinous process surface proxy for neck base"),
    "BackLeft": ("trunk", "Left posterior thoracic / scapular marker"),
    "BackRight": ("trunk", "Right posterior thoracic / scapular marker"),
    "WaistLeft": ("pelvis", "Left anterior superior iliac spine (ASIS) proxy"),
    "WaistRight": ("pelvis", "Right anterior superior iliac spine (ASIS) proxy"),
    "WaistLBack": ("pelvis", "Left posterior superior iliac spine (PSIS) proxy"),
    "WaistRBack": ("pelvis", "Right posterior superior iliac spine (PSIS) proxy"),
    "LShoulderTop": ("left_arm", "Left acromioclavicular joint surface marker"),
    "LShoulderBack": ("left_arm", "Left posterior shoulder / scapular marker"),
    "LElbowOut": ("left_arm", "Left lateral humeral epicondyle surface marker"),
    "LUArmHigh": ("left_arm", "Left upper arm lateral surface tracking marker"),
    "LWristTop": ("left_arm", "Left dorsal wrist joint surface marker"),
    "RShoulderTop": ("right_arm", "Right acromioclavicular joint surface marker"),
    "RShoulderBack": ("right_arm", "Right posterior shoulder / scapular marker"),
    "RElbowOut": ("right_arm", "Right lateral humeral epicondyle surface marker"),
    "RUArmHigh": ("right_arm", "Right upper arm lateral surface tracking marker"),
    "RWristTop": ("right_arm", "Right dorsal wrist joint surface marker"),
    "LKneeOut": ("left_leg", "Left lateral femoral epicondyle surface marker"),
    "LAnkleOut": ("left_leg", "Left lateral malleolus surface marker"),
    "LToeIn": ("left_leg", "Left first metatarsal head surface marker"),
    "LToeOut": ("left_leg", "Left fifth metatarsal head surface marker"),
    "RKneeOut": ("right_leg", "Right lateral femoral epicondyle surface marker"),
    "RAnkleOut": ("right_leg", "Right lateral malleolus surface marker"),
    "RToeIn": ("right_leg", "Right first metatarsal head surface marker"),
    "RToeOut": ("right_leg", "Right fifth metatarsal head surface marker"),
}

_CLUSTER_DEFINITIONS: dict[str, tuple[str, str]] = {
    "Marker_2:2:1": ("club_grip", "Proximal shaft / grip triad marker 1"),
    "Marker_2:2:2": ("club_grip", "Proximal shaft / grip triad marker 2"),
    "Marker_2:2:3": ("club_grip", "Proximal shaft / grip triad marker 3"),
    "Marker_3:3:1": ("club_head", "Distal clubhead cluster triad marker 1"),
    "Marker_3:3:2": ("club_head", "Distal clubhead cluster triad marker 2"),
    "Marker_3:3:3": ("club_head", "Distal clubhead cluster triad marker 3"),
}


def _build_measurement_map(
    unassigned_labels: tuple[str, ...],
) -> dict[str, MarkerMeasurementSemantics]:
    """Construct a full measurement map for a specific set of unassigned labels."""
    mapping: dict[str, MarkerMeasurementSemantics] = {}

    for lbl, (seg, desc) in _SURFACE_DEFINITIONS.items():
        mapping[lbl] = MarkerMeasurementSemantics(
            label=lbl,
            measurement_class=MeasurementClass.OBSERVED_SURFACE,
            segment=seg,
            is_observed=True,
            offset_assumptions="Surface skin/suit marker; subject to soft-tissue artifact.",
            description=desc,
        )

    for lbl, (seg, desc) in _CLUSTER_DEFINITIONS.items():
        mapping[lbl] = MarkerMeasurementSemantics(
            label=lbl,
            measurement_class=MeasurementClass.OBSERVED_CLUSTER_CENTROID,
            segment=seg,
            is_observed=True,
            offset_assumptions="Rigid triad on shaft/head; not anatomical or clubface center.",
            description=desc,
        )

    for lbl in unassigned_labels:
        mapping[lbl] = MarkerMeasurementSemantics(
            label=lbl,
            measurement_class=MeasurementClass.UNASSIGNED_OR_SENTINEL,
            segment="unassigned",
            is_observed=False,
            offset_assumptions="Stuck coordinate, system sentinel, or unlabeled marker.",
            description=f"Unassigned capture marker channel ({lbl})",
        )

    return mapping


_DRIVER_MAP_DICT = _build_measurement_map(
    ("Marker_0:0:0", "Uname*36", "Uname*37", "Uname*38")
)
_IRON_MAP_DICT = _build_measurement_map(
    ("Marker_0:0:0", "Uname*36", "Uname*37", "pelvis")
)

MEASUREMENT_MAP_DRIVER: MappingProxyType[str, MarkerMeasurementSemantics] = (
    MappingProxyType(_DRIVER_MAP_DICT)
)
MEASUREMENT_MAP_IRON: MappingProxyType[str, MarkerMeasurementSemantics] = (
    MappingProxyType(_IRON_MAP_DICT)
)

_MEASUREMENT_MAPS: dict[str, MappingProxyType[str, MarkerMeasurementSemantics]] = {
    "driver": MEASUREMENT_MAP_DRIVER,
    "iron": MEASUREMENT_MAP_IRON,
}


@precondition(lambda kind: isinstance(kind, str), "kind must be str")
@postcondition(
    lambda r: len(r) == 38, "measurement map must contain exactly 38 channels"
)
def get_measurement_map(kind: str) -> MappingProxyType[str, MarkerMeasurementSemantics]:
    """Return versioned measurement map for the requested capture kind ('driver' or 'iron')."""
    normalized = kind.strip().lower()
    if normalized not in _MEASUREMENT_MAPS:
        raise ValueError(
            f"Unknown capture kind {kind!r}; expected one of {sorted(_MEASUREMENT_MAPS)}"
        )
    return _MEASUREMENT_MAPS[normalized]
