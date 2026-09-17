"""Engine-agnostic contract for the tour-average driver capture (C3D_TA_Driver).

One frozen specification of the capture every engine lane matches against:
its content hash, clock, units, vertical axis and the 38 marker labels grouped
by body segment. ``load_tour_capture`` reads the C3D through ezc3d, verifies
the frozen identity and returns validated arrays; nothing here interpolates,
retimes, filters or repairs marker data. Engine-specific marker-to-body maps
live with each engine and consume :func:`tracked_labels`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class TourCaptureSpec:
    """Frozen identity of the canonical capture file."""

    sha256: str
    rate_hz: float
    frames: int
    units: str
    vertical_axis: str
    labels: tuple[str, ...]

    @property
    def duration_s(self) -> float:
        return (self.frames - 1) / self.rate_hz


TOUR_CAPTURE = TourCaptureSpec(  # the driver swing, the original canonical file
    sha256="545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba",
    rate_hz=360.0,
    frames=654,
    units="m",
    vertical_axis="y",
    labels=(
        "Marker_0:0:0",
        "WaistLeft",
        "WaistRight",
        "WaistLBack",
        "WaistRBack",
        "BackTop",
        "BackLeft",
        "BackRight",
        "HeadTop",
        "HeadFront",
        "HeadSide",
        "LShoulderTop",
        "LShoulderBack",
        "LElbowOut",
        "LUArmHigh",
        "LWristTop",
        "RShoulderTop",
        "RShoulderBack",
        "RElbowOut",
        "RUArmHigh",
        "RWristTop",
        "LKneeOut",
        "LToeIn",
        "LToeOut",
        "LAnkleOut",
        "RKneeOut",
        "RToeIn",
        "RToeOut",
        "RAnkleOut",
        "Marker_2:2:1",
        "Marker_2:2:2",
        "Marker_2:2:3",
        "Marker_3:3:1",
        "Marker_3:3:2",
        "Marker_3:3:3",
        "Uname*36",
        "Uname*37",
        "Uname*38",
    ),
)

MARKER_SEGMENTS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "head": ("HeadTop", "HeadFront", "HeadSide"),
        "trunk": ("BackTop", "BackLeft", "BackRight"),
        "pelvis": ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack"),
        "left_arm": (
            "LShoulderTop",
            "LShoulderBack",
            "LUArmHigh",
            "LElbowOut",
            "LWristTop",
        ),
        "right_arm": (
            "RShoulderTop",
            "RShoulderBack",
            "RUArmHigh",
            "RElbowOut",
            "RWristTop",
        ),
        "left_leg": ("LKneeOut", "LAnkleOut", "LToeIn", "LToeOut"),
        "right_leg": ("RKneeOut", "RAnkleOut", "RToeIn", "RToeOut"),
        "club": (
            "Marker_2:2:1",
            "Marker_2:2:2",
            "Marker_2:2:3",
            "Marker_3:3:1",
            "Marker_3:3:2",
            "Marker_3:3:3",
        ),
        "unassigned": ("Marker_0:0:0", "Uname*36", "Uname*37", "Uname*38"),
    }
)


def tracked_labels() -> tuple[str, ...]:
    """Return capture labels with an anatomical or club role, in capture order."""
    excluded = set(MARKER_SEGMENTS["unassigned"])
    return tuple(label for label in TOUR_CAPTURE.labels if label not in excluded)


@dataclass(frozen=True)
class MarkerPolicyEntry:
    """Validity characteristics and tracking policy for a capture marker label."""

    valid_samples: int
    missing_samples: int
    nominal_weight: float = 1.0
    excluded: bool = False

    def weight(self, is_valid: bool = True) -> float:
        """Return effective tracking weight: 0.0 if excluded or invalid, nominal_weight otherwise."""
        if self.excluded or not is_valid:
            return 0.0
        return self.nominal_weight

    def __getitem__(self, item: str) -> Any:
        if item == "valid_samples":
            return self.valid_samples
        if item == "missing_samples":
            return self.missing_samples
        if item == "nominal_weight":
            return self.nominal_weight
        if item == "excluded":
            return self.excluded
        raise KeyError(item)


class MarkerValidityPolicy(dict[str, MarkerPolicyEntry]):
    """Frozen mapping of marker validity policies with helper methods."""

    @precondition(
        lambda self, label, is_valid=True: isinstance(label, str), "label must be str"
    )
    @postcondition(lambda r: r >= 0.0, "weight must be non-negative")
    def weight_for(self, label: str, is_valid: bool = True) -> float:
        """Return tracking weight for the specified label and validity status."""
        if label not in self:
            raise ValueError(f"Unknown capture label: {label}")
        return self[label].weight(is_valid=is_valid)


_RAW_VALIDITY_DATA: dict[str, tuple[int, int]] = {
    "Marker_0:0:0": (649, 5),
    "WaistLeft": (654, 0),
    "WaistRight": (649, 5),
    "WaistLBack": (654, 0),
    "WaistRBack": (654, 0),
    "BackTop": (654, 0),
    "BackLeft": (654, 0),
    "BackRight": (654, 0),
    "HeadTop": (654, 0),
    "HeadFront": (654, 0),
    "HeadSide": (654, 0),
    "LShoulderTop": (654, 0),
    "LShoulderBack": (654, 0),
    "LElbowOut": (654, 0),
    "LUArmHigh": (654, 0),
    "LWristTop": (654, 0),
    "RShoulderTop": (128, 526),
    "RShoulderBack": (654, 0),
    "RElbowOut": (654, 0),
    "RUArmHigh": (654, 0),
    "RWristTop": (654, 0),
    "LKneeOut": (654, 0),
    "LToeIn": (654, 0),
    "LToeOut": (654, 0),
    "LAnkleOut": (654, 0),
    "RKneeOut": (654, 0),
    "RToeIn": (654, 0),
    "RToeOut": (654, 0),
    "RAnkleOut": (654, 0),
    "Marker_2:2:1": (618, 36),
    "Marker_2:2:2": (618, 36),
    "Marker_2:2:3": (618, 36),
    "Marker_3:3:1": (633, 21),
    "Marker_3:3:2": (633, 21),
    "Marker_3:3:3": (633, 21),
    "Uname*36": (649, 5),
    "Uname*37": (654, 0),
    "Uname*38": (649, 5),
}

MARKER_VALIDITY_POLICY: MarkerValidityPolicy = MarkerValidityPolicy(
    {
        label: MarkerPolicyEntry(
            valid_samples=_RAW_VALIDITY_DATA[label][0],
            missing_samples=_RAW_VALIDITY_DATA[label][1],
            nominal_weight=0.0 if label in MARKER_SEGMENTS["unassigned"] else 1.0,
            excluded=label in MARKER_SEGMENTS["unassigned"],
        )
        for label in TOUR_CAPTURE.labels
    }
)


@dataclass(frozen=True)
class TourCapture:
    """Validated marker samples: time, labels, world points in metres, validity.

    Invariants (checked at construction): strictly increasing time starting at
    zero, unique labels, points shaped (frames, markers, 3), validity shaped
    (frames, markers), and every valid point finite. Invalid points may be NaN.
    """

    time_s: Array
    labels: tuple[str, ...]
    points_m: Array
    valid: BoolArray
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float)
        points = np.asarray(self.points_m, dtype=float)
        valid = np.asarray(self.valid, dtype=bool)
        labels = tuple(self.labels)
        if (
            time.ndim != 1
            or time.size < 1
            or not np.isfinite(time).all()
            or time[0] != 0.0
            or np.any(np.diff(time) <= 0)
        ):
            raise ValueError("Capture time must start at zero and increase strictly")
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Capture labels must be unique and nonempty")
        if points.shape != (time.size, len(labels), 3):
            raise ValueError("Capture points must be shaped (frames, markers, 3)")
        if valid.shape != (time.size, len(labels)):
            raise ValueError("Capture validity must be shaped (frames, markers)")
        if not np.isfinite(points[valid]).all():
            raise ValueError("Every valid capture point must be finite")
        for name, value in (("time_s", time), ("points_m", points), ("valid", valid)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "labels", labels)

    @property
    def frames(self) -> int:
        return int(self.time_s.size)

    def index(self, label: str) -> int:
        """Return the column of a label; unknown labels raise ValueError."""
        if label not in self.labels:
            raise ValueError(f"Unknown capture label: {label}")
        return self.labels.index(label)

    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid))

    def subset(self, labels: Sequence[str]) -> TourCapture:
        """Return the same clock restricted to the given labels, in that order."""
        columns = [self.index(label) for label in labels]
        return TourCapture(
            self.time_s,
            tuple(labels),
            self.points_m[:, columns],
            self.valid[:, columns],
            self.source_sha256,
        )


TOUR_CAPTURE_IRON = TourCaptureSpec(  # the 7-iron swing of the same golfer
    sha256="395deb1f91006586819020fc85180409e716f07e1c680f9fb2ca114759f80845",
    rate_hz=359.0,
    frames=657,
    units="m",
    vertical_axis="y",
    labels=TOUR_CAPTURE.labels[:35] + ("Uname*36", "Uname*37", "pelvis"),
)
TOUR_CAPTURES: dict[str, TourCaptureSpec] = {
    "driver": TOUR_CAPTURE,
    "iron": TOUR_CAPTURE_IRON,
}


def capture_kind(digest: str) -> str:
    """Name of the canonical capture with this SHA-256, else ValueError."""
    for name, spec in TOUR_CAPTURES.items():
        if spec.sha256 == digest:
            return name
    raise ValueError("File is not a canonical tour-average capture")


def load_tour_capture(path: Path) -> TourCapture:
    """Read a canonical C3D (driver or 7-iron) and verify it against its spec.

    Points with negative residuals or nonfinite coordinates are marked invalid.
    Rejects any file whose hash, rate, units, frame count or labels differ.
    """
    import ezc3d  # local import: optional dependency, heavy to load

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    spec = TOUR_CAPTURES[capture_kind(digest)]
    c3d = ezc3d.c3d(str(path))
    parameters = c3d["parameters"]["POINT"]
    labels = tuple(parameters["LABELS"]["value"])
    rate = float(parameters["RATE"]["value"][0])
    units = str(parameters["UNITS"]["value"][0])
    points = np.asarray(c3d["data"]["points"], dtype=float)  # (4, markers, frames)
    residuals = np.asarray(c3d["data"]["meta_points"]["residuals"][0], dtype=float)
    if (
        labels != spec.labels
        or rate != spec.rate_hz
        or units != spec.units
        or points.shape != (4, len(labels), spec.frames)
    ):
        raise ValueError("Capture parameters differ from the frozen specification")
    xyz = np.transpose(points[:3], (2, 1, 0)).copy()
    valid = np.isfinite(xyz).all(axis=2) & (residuals.T >= 0)
    xyz[~valid] = np.nan
    time = np.arange(spec.frames) / spec.rate_hz
    return TourCapture(time, labels, xyz, valid, digest)
