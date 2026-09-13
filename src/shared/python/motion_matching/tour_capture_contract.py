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

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


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


TOUR_CAPTURE = TourCaptureSpec(
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


def load_tour_capture(path: Path) -> TourCapture:
    """Read the canonical C3D and verify it against :data:`TOUR_CAPTURE`.

    Points with negative residuals or nonfinite coordinates are marked invalid.
    Rejects any file whose hash, rate, units, frame count or labels differ.
    """
    import ezc3d  # local import: optional dependency, heavy to load

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != TOUR_CAPTURE.sha256:
        raise ValueError("File is not the canonical tour-average capture")
    c3d = ezc3d.c3d(str(path))
    parameters = c3d["parameters"]["POINT"]
    labels = tuple(parameters["LABELS"]["value"])
    rate = float(parameters["RATE"]["value"][0])
    units = str(parameters["UNITS"]["value"][0])
    points = np.asarray(c3d["data"]["points"], dtype=float)  # (4, markers, frames)
    residuals = np.asarray(c3d["data"]["meta_points"]["residuals"][0], dtype=float)
    if (
        labels != TOUR_CAPTURE.labels
        or rate != TOUR_CAPTURE.rate_hz
        or units != TOUR_CAPTURE.units
        or points.shape != (4, len(labels), TOUR_CAPTURE.frames)
    ):
        raise ValueError("Capture parameters differ from the frozen specification")
    xyz = np.transpose(points[:3], (2, 1, 0)).copy()
    valid = np.isfinite(xyz).all(axis=2) & (residuals.T >= 0)
    xyz[~valid] = np.nan
    time = np.arange(TOUR_CAPTURE.frames) / TOUR_CAPTURE.rate_hz
    return TourCapture(time, labels, xyz, valid, digest)
