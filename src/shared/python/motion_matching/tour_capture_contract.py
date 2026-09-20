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
from collections.abc import Mapping, Sequence
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
class CaptureValidationReport:
    """Structured result of validating a capture against a CaptureContract."""

    is_valid: bool
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = MappingProxyType({})

    def raise_for_status(self) -> None:
        """Raise ValueError if the capture does not conform to the contract."""
        if not self.is_valid:
            msg = "; ".join(self.reasons) if self.reasons else "Validation failed"
            raise ValueError(f"Capture does not conform to contract: {msg}")


@dataclass(frozen=True)
class CaptureContract:
    """Generic contract for motion captures: expected units, rate, markers, and segments."""

    units: str = "m"
    vertical_axis: str = "y"
    rate_hz: float | None = None
    min_frames: int = 2
    max_gap_fraction: float = 0.5
    required_labels: tuple[str, ...] | None = None
    required_segments: tuple[str, ...] | None = None
    label_map: Mapping[str, str] | None = None
    static_calibration_required: bool = False
    frozen_sha256: str | None = None


@dataclass(frozen=True)
class TourCaptureSpec(CaptureContract):
    """Frozen identity of the canonical capture file."""

    sha256: str = ""
    rate_hz: float = 0.0
    frames: int = 0
    units: str = "m"
    vertical_axis: str = "y"
    labels: tuple[str, ...] = ()
    max_gap_fraction: float = 0.9

    def __post_init__(self) -> None:
        if self.frozen_sha256 is None and self.sha256:
            object.__setattr__(self, "frozen_sha256", self.sha256)

    @property
    def duration_s(self) -> float:
        return (self.frames - 1) / self.rate_hz if self.rate_hz > 0 else 0.0


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


def _extract_c3d_metadata(path: Path) -> dict[str, Any]:
    import ezc3d

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    c3d = ezc3d.c3d(str(path))
    params = c3d["parameters"]["POINT"]
    labels = tuple(str(x) for x in params["LABELS"]["value"])
    rate = float(params["RATE"]["value"][0])
    units = str(params["UNITS"]["value"][0])
    points = np.asarray(c3d["data"]["points"], dtype=float)
    residuals = np.asarray(c3d["data"]["meta_points"]["residuals"][0], dtype=float)
    return {
        "digest": digest,
        "labels": labels,
        "rate_hz": rate,
        "units": units,
        "frames": points.shape[2],
        "points": points,
        "residuals": residuals,
    }


def _check_contract_reasons(
    contract: CaptureContract,
    meta: Mapping[str, Any],
    effective_label_map: Mapping[str, str],
) -> list[str]:
    reasons: list[str] = []
    if contract.units and meta["units"].lower() != contract.units.lower():
        reasons.append(
            f"invalid_units: expected '{contract.units}', found '{meta['units']}'"
        )
    if contract.rate_hz is not None and abs(meta["rate_hz"] - contract.rate_hz) > 1.0:
        reasons.append(
            f"rate_mismatch: expected {contract.rate_hz} Hz, found {meta['rate_hz']} Hz"
        )
    if meta["frames"] < contract.min_frames:
        reasons.append(
            f"insufficient_frames: expected >= {contract.min_frames}, found {meta['frames']}"
        )
    if contract.frozen_sha256 and meta["digest"] != contract.frozen_sha256:
        reasons.append(
            f"sha256_mismatch: expected {contract.frozen_sha256}, found {meta['digest']}"
        )

    raw_labels = meta["labels"]
    mapped_labels = tuple(effective_label_map.get(lbl, lbl) for lbl in raw_labels)
    if contract.required_labels:
        missing = [
            r
            for r in contract.required_labels
            if r not in mapped_labels and r not in raw_labels
        ]
        if missing:
            reasons.append(f"missing_required_labels: {sorted(missing)}")

    if contract.required_segments:
        for seg in contract.required_segments:
            seg_markers = MARKER_SEGMENTS.get(seg, ())
            has_seg = any(lbl in seg_markers for lbl in mapped_labels) or any(
                lbl in seg_markers for lbl in raw_labels
            )
            if not has_seg:
                reasons.append(f"missing_required_segment: {seg}")

    xyz = np.transpose(meta["points"][:3], (2, 1, 0))
    valid_mask = np.isfinite(xyz).all(axis=2) & (meta["residuals"].T >= 0)
    check_labels = contract.required_labels or mapped_labels
    frames = meta["frames"]
    for lbl in check_labels:
        if lbl in mapped_labels:
            idx = mapped_labels.index(lbl)
        elif lbl in raw_labels:
            idx = raw_labels.index(lbl)
        else:
            continue
        valid_cnt = int(np.count_nonzero(valid_mask[:, idx]))
        gap_frac = (frames - valid_cnt) / frames if frames > 0 else 1.0
        if gap_frac > contract.max_gap_fraction:
            reasons.append(
                f"excessive_gap_fraction: marker '{lbl}' has {gap_frac:.2%} missing frames (max {contract.max_gap_fraction:.2%})"
            )

    if contract.static_calibration_required:
        reasons.append(
            "missing_static_calibration: static address calibration required"
        )

    return reasons


@precondition(
    lambda path, contract=None, label_map=None: Path(path).is_file(),
    "capture file must exist",
)
@postcondition(
    lambda r: isinstance(r, CaptureValidationReport),
    "must return CaptureValidationReport",
)
def validate_capture_contract(
    path: Path | str,
    contract: CaptureContract | None = None,
    label_map: Mapping[str, str] | None = None,
) -> CaptureValidationReport:
    """Validate a C3D capture against a CaptureContract, returning a structured report."""
    p = Path(path)
    active_contract = contract or CaptureContract()
    try:
        meta = _extract_c3d_metadata(p)
    except Exception as err:
        return CaptureValidationReport(
            is_valid=False,
            reasons=(f"c3d_read_error: {err}",),
            metadata=MappingProxyType({"path": str(p)}),
        )

    effective_map = {
        **dict(active_contract.label_map or {}),
        **dict(label_map or {}),
    }
    reasons = _check_contract_reasons(active_contract, meta, effective_map)
    metadata: dict[str, Any] = {
        "digest": meta["digest"],
        "rate_hz": meta["rate_hz"],
        "frames": meta["frames"],
        "units": meta["units"],
        "labels_count": len(meta["labels"]),
    }
    return CaptureValidationReport(
        is_valid=len(reasons) == 0,
        reasons=tuple(reasons),
        metadata=MappingProxyType(metadata),
    )


@precondition(
    lambda path, contract=None, label_map=None: Path(path).is_file(),
    "capture file must exist",
)
@postcondition(
    lambda r: isinstance(r, TourCapture),
    "must return TourCapture",
)
def load_capture(
    path: Path | str,
    contract: CaptureContract | None = None,
    label_map: Mapping[str, str] | None = None,
) -> TourCapture:
    """Read any conforming C3D according to a CaptureContract with label mapping and scaling."""
    p = Path(path)
    active_contract = contract or CaptureContract()
    report = validate_capture_contract(p, contract=active_contract, label_map=label_map)
    report.raise_for_status()

    meta = _extract_c3d_metadata(p)
    effective_map = {
        **dict(active_contract.label_map or {}),
        **dict(label_map or {}),
    }
    raw_labels = meta["labels"]
    mapped_labels = tuple(effective_map.get(lbl, lbl) for lbl in raw_labels)

    points = meta["points"]
    xyz = np.transpose(points[:3], (2, 1, 0)).copy()
    if meta["units"].lower() in ("mm", "millimeter", "millimeters"):
        xyz = xyz * 1e-3

    residuals = meta["residuals"]
    valid = np.isfinite(xyz).all(axis=2) & (residuals.T >= 0)
    xyz[~valid] = np.nan
    time = np.arange(meta["frames"]) / meta["rate_hz"]

    return TourCapture(time, mapped_labels, xyz, valid, meta["digest"])
