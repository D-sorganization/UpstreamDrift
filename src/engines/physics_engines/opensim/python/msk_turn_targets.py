"""Pelvis and upper-trunk turn targets and capture-planted feet (OSV-9, #12042).

Slice 4 of issue #12042. The OSV-9 swing tracking (``msk_club_tracking``)
followed only six joint-centre landmarks of its generated source swing, so
two hip centres 0.18 m apart left the pelvis yaw almost free against the
lumbar and the Rajagopal pelvis stayed nearly square while the thorax
over-turned. This module supplies the two missing pieces:

* :class:`TurnTargets`: per-frame pelvis and upper-trunk *turn* targets in
  degrees, sampled from the shared turn lines of ``swing_comparison.turn``
  (capture markers: ``WaistLeft``/``WaistRight`` and ``BackLeft``/
  ``BackRight``). Turn is relative to address and positive in the backswing.
  The tracker turns them into yaw residuals about the model's own calibrated
  address heading, so a constant marker-to-model offset at address cancels.
* :class:`PlantedFeet`: the feet planted where the capture's foot markers
  stand at address (ankle centre and foot long axis from ``*AnkleOut``,
  ``*ToeIn`` and ``*ToeOut``, through the shared ``foot_progression``
  estimator) instead of a square synthetic stance under the hips.

Model lines (Rajagopal bodies, measured in the native world, Z up, golfer
facing -X): the pelvis line joins the hip centres (``femur_r`` to ``femur_l``
origins), the upper-trunk line is the ``torso`` lateral axis (``-z``, right
to left) and the shoulder-girdle line joins the shoulder centres
(``humerus_r`` to ``humerus_l`` origins). Rajagopal has no shoulder girdle,
so its shoulder line rides the thorax.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.engines.physics_engines.opensim.python import msk_club_calibration as cal
from src.shared.python.motion_matching.foot_progression import (
    ANKLE_LATERAL_OFFSET_M,
    address_window,
    marker_long_axis,
)
from src.shared.python.motion_matching.ground_support import capture_to_native_world
from src.shared.python.motion_matching.hip_calibration import ANKLE_OUT_LATERAL_M
from src.shared.python.swing_comparison.turn import LineTurn, TurnMetrics

#: Weight (rad) of a segment-yaw residual: about one degree, as strong as the
#: lead-hand rotation term, so the markers' turn overrules the source swing's
#: joint-centre landmarks (4 cm) where they disagree.
TURN_WEIGHT_RAD = 0.02
#: Segments carrying a turn target, in residual order.
TARGET_SEGMENTS: tuple[str, ...] = ("pelvis", "upper_trunk")
HIP_BODIES = ("femur_l", "femur_r")
SHOULDER_BODIES = ("humerus_l", "humerus_r")
TORSO_BODY = "torso"
#: Half-distance of the two thorax-fixed points along the torso lateral axis
#: (reporting only; the line direction does not depend on it).
THORAX_HALF_WIDTH_M = 0.1
NATIVE_UP = np.array([0.0, 0.0, 1.0])
FOOT_MARKERS = ("AnkleOut", "ToeIn", "ToeOut")
_SIDES = {"l": "L", "r": "R"}


class BodyProbe(Protocol):
    """What the target helpers need from an OpenSim pose probe."""

    def body(self, name: str) -> np.ndarray: ...


# ------------------------------------------------------------- turn targets
@dataclass(frozen=True)
class TurnTargets:
    """Per-frame pelvis and upper-trunk turn targets (degrees, + backswing).

    ``pelvis_deg`` and ``upper_trunk_deg`` are aligned with the tracked rows;
    NaN means "no target on this frame" (never zero). ``weight_rad`` is the
    residual weight of both segment-yaw terms.
    """

    pelvis_deg: np.ndarray
    upper_trunk_deg: np.ndarray
    weight_rad: float = TURN_WEIGHT_RAD
    source: str = "capture markers"

    def __post_init__(self) -> None:
        pelvis = np.asarray(self.pelvis_deg, dtype=float)
        trunk = np.asarray(self.upper_trunk_deg, dtype=float)
        if pelvis.ndim != 1 or trunk.shape != pelvis.shape:
            raise ValueError(
                "pelvis_deg and upper_trunk_deg must be 1-D with equal length, got "
                f"{pelvis.shape} and {trunk.shape}"
            )
        if np.isinf(pelvis).any() or np.isinf(trunk).any():
            raise ValueError("turn targets must be finite or NaN (no target)")
        if not (math.isfinite(self.weight_rad) and self.weight_rad > 0.0):
            raise ValueError(
                f"weight_rad must be finite and > 0, got {self.weight_rad}"
            )
        object.__setattr__(self, "pelvis_deg", pelvis)
        object.__setattr__(self, "upper_trunk_deg", trunk)

    def __len__(self) -> int:
        return int(self.pelvis_deg.shape[0])

    def at(self, frame: int) -> dict[str, float]:
        """Targets of row ``frame`` keyed by :data:`TARGET_SEGMENTS`."""
        return {
            "pelvis": float(self.pelvis_deg[frame]),
            "upper_trunk": float(self.upper_trunk_deg[frame]),
        }


def _sample(line: LineTurn, times_s: np.ndarray) -> np.ndarray:
    """``line`` linearly interpolated at ``times_s``; NaN outside its valid span."""
    good = np.isfinite(line.turn_deg)
    out = np.full(times_s.shape, np.nan)
    if not good.any():
        return out
    t, turn = line.t[good], line.turn_deg[good]
    inside = (times_s >= t[0]) & (times_s <= t[-1])
    out[inside] = np.interp(times_s[inside], t, turn)
    return out


def turn_targets_from_lines(
    metrics: TurnMetrics,
    times_s: Any,
    *,
    weight_rad: float = TURN_WEIGHT_RAD,
    source: str = "capture markers",
) -> TurnTargets:
    """Turn targets at the tracked rows' ``times_s`` from shared turn lines.

    ``metrics`` is a :class:`swing_comparison.turn.TurnMetrics` (normally
    ``marker_turn_lines`` of the capture). ``times_s`` are the rows' times on
    the same clock, seconds. Raises ``ValueError`` for a non-1-D or
    non-finite ``times_s``.
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
        raise ValueError("times_s must be a non-empty finite 1-D array")
    return TurnTargets(
        pelvis_deg=_sample(metrics.pelvis, times),
        upper_trunk_deg=_sample(metrics.upper_trunk, times),
        weight_rad=weight_rad,
        source=source,
    )


# ------------------------------------------------------------- model lines
def _native(point_os: np.ndarray, floor_native_z: float) -> np.ndarray:
    return cal.opensim_to_native_vector(point_os) + np.array([0.0, 0.0, floor_native_z])


def model_turn_points(
    probe: BodyProbe, floor_native_z: float = 0.0
) -> dict[str, np.ndarray]:
    """Native-world model points for ``swing_comparison.turn.model_turn_lines``.

    ``hip_l``/``hip_r`` are the femur origins, ``shoulder_l``/``shoulder_r``
    the humerus origins and ``thorax_l``/``thorax_r`` two torso-fixed points
    on its lateral axis. ``probe`` must be posed already.
    """
    torso = probe.body(TORSO_BODY)
    lateral = torso[:3, 2] * THORAX_HALF_WIDTH_M  # Rajagopal z: toward the right
    points_os = {
        "hip_l": probe.body(HIP_BODIES[0])[:3, 3],
        "hip_r": probe.body(HIP_BODIES[1])[:3, 3],
        "shoulder_l": probe.body(SHOULDER_BODIES[0])[:3, 3],
        "shoulder_r": probe.body(SHOULDER_BODIES[1])[:3, 3],
        "thorax_l": torso[:3, 3] - lateral,
        "thorax_r": torso[:3, 3] + lateral,
    }
    return {k: _native(v, floor_native_z) for k, v in points_os.items()}


def _yaw(left: np.ndarray, right: np.ndarray) -> float:
    """Native-world yaw (rad) of the right-to-left line (``turn.line_turn``)."""
    vec = np.asarray(left) - np.asarray(right)
    return float(math.atan2(vec[1], vec[0]))


def segment_yaws(probe: BodyProbe) -> dict[str, float]:
    """Native yaw (rad) of the pelvis and upper-trunk lines of a posed probe."""
    pts = model_turn_points(probe)
    return {
        "pelvis": _yaw(pts["hip_l"], pts["hip_r"]),
        "upper_trunk": _yaw(pts["thorax_l"], pts["thorax_r"]),
    }


def wrap_angle(angle_rad: float) -> float:
    """``angle_rad`` wrapped to [-pi, pi)."""
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def turn_residuals(
    yaws: Mapping[str, float],
    reference: Mapping[str, float],
    targets: Mapping[str, float],
    weight_rad: float,
) -> np.ndarray:
    """Weighted segment-yaw residuals, one per :data:`TARGET_SEGMENTS` entry.

    The target yaw is the reference (address) yaw minus the target turn
    (turn is positive in the backswing, ``turn = -(yaw - yaw_address)``). A
    NaN target gives a zero residual so the residual length stays fixed.
    """
    out = np.zeros(len(TARGET_SEGMENTS))
    for i, segment in enumerate(TARGET_SEGMENTS):
        turn = targets[segment]
        if math.isfinite(turn):
            goal = reference[segment] - math.radians(turn)
            out[i] = wrap_angle(yaws[segment] - goal) / weight_rad
    return out


# ------------------------------------------------------------- planted feet
@dataclass(frozen=True)
class PlantedFeet:
    """Capture foot placement at address in the native world (metres).

    ``ankle[side]`` is the ankle centre (lateral malleolus moved medially by
    ``hip_calibration.ANKLE_OUT_LATERAL_M``), ``axis[side]`` the horizontal
    unit long axis (heel toward the toes) and ``lateral[side]`` the horizontal
    unit axis pointing away from the other foot. ``side`` is ``l`` or ``r``.
    """

    ankle: dict[str, np.ndarray]
    axis: dict[str, np.ndarray]
    lateral: dict[str, np.ndarray]
    frames: int

    def __post_init__(self) -> None:
        for name in ("ankle", "axis", "lateral"):
            values = getattr(self, name)
            if set(values) != set(_SIDES):
                raise ValueError(f"{name} needs the 'l' and 'r' feet")
            for vec in values.values():
                arr = np.asarray(vec, dtype=float)
                if arr.shape != (3,) or not np.isfinite(arr).all():
                    raise ValueError(f"{name} entries must be finite 3-vectors")

    def targets(self, floor_native_z: float) -> dict[str, np.ndarray]:
        """Rajagopal foot-body targets (``calcn_r``, ``toes_l``, ...).

        The ``msk_club_calibration.FOOT_TEMPLATE`` offsets (x along the foot,
        z outward, y height with the foot flat) are laid along the measured
        axes; heights stay the template's (feet flat on the floor at y = 0).
        """
        out: dict[str, np.ndarray] = {}
        for suffix in _SIDES:
            ankle = cal.NATIVE_TO_OPENSIM @ (
                self.ankle[suffix] - np.array([0.0, 0.0, floor_native_z])
            )
            axis = cal.NATIVE_TO_OPENSIM @ self.axis[suffix]
            lateral = cal.NATIVE_TO_OPENSIM @ self.lateral[suffix]
            for body, (x, y, z) in cal.FOOT_TEMPLATE.items():
                point = ankle + x * axis + z * lateral
                point[1] = y
                out[f"{body}_{suffix}"] = point
        return out


def _address_median(
    native: np.ndarray, valid: np.ndarray, column: int, window: np.ndarray
) -> np.ndarray:
    good = window[valid[window, column]]
    if good.size == 0:
        return np.full(3, np.nan)
    return np.median(native[good, column], axis=0)


def _horizontal_unit(vec: np.ndarray, what: str) -> np.ndarray:
    flat = np.array([vec[0], vec[1], 0.0])
    norm = float(np.linalg.norm(flat))
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError(f"{what} has no horizontal direction")
    return flat / norm


def planted_feet_from_capture(capture: Any) -> PlantedFeet:
    """Feet at the capture's address from its foot markers.

    ``capture`` is a ``TourCapture`` (Y-up metres, ``valid`` mask). Each foot
    uses the median of its markers over the shared address window
    (``foot_progression.address_window``) and the shared corrected long-axis
    estimator (``marker_long_axis`` with ``ANKLE_LATERAL_OFFSET_M``). Raises
    ``ValueError`` when a foot marker is absent or never valid at address.
    """
    labels = tuple(str(label) for label in capture.labels)
    names = [f"{p}{m}" for p in _SIDES.values() for m in FOOT_MARKERS]
    missing = [n for n in names if n not in labels]
    if missing:
        raise ValueError(f"capture lacks foot markers: {missing}")
    native = capture_to_native_world(np.asarray(capture.points_m, dtype=float))
    valid = np.asarray(capture.valid, dtype=bool)
    window = address_window(native, valid, labels)
    mean = {n: _address_median(native, valid, labels.index(n), window) for n in names}
    bad = [n for n, v in mean.items() if not np.isfinite(v).all()]
    if bad:
        raise ValueError(f"foot markers never valid in the address window: {bad}")
    centre = {
        s: np.mean([mean[f"{p}{m}"] for m in FOOT_MARKERS], axis=0)
        for s, p in _SIDES.items()
    }
    ankle, axis, lateral = {}, {}, {}
    for suffix, prefix in _SIDES.items():
        other = "r" if suffix == "l" else "l"
        out_dir = _horizontal_unit(centre[suffix] - centre[other], "stance line")
        heel = mean[f"{prefix}AnkleOut"]
        axis[suffix] = marker_long_axis(
            heel,
            mean[f"{prefix}ToeIn"],
            mean[f"{prefix}ToeOut"],
            up=NATIVE_UP,
            out_dir=out_dir,
            ankle_lateral_offset_m=ANKLE_LATERAL_OFFSET_M,
        )
        side_axis = np.cross(NATIVE_UP, axis[suffix])
        lateral[suffix] = side_axis if side_axis @ out_dir > 0 else -side_axis
        ankle[suffix] = heel - ANKLE_OUT_LATERAL_M * lateral[suffix]
    return PlantedFeet(ankle=ankle, axis=axis, lateral=lateral, frames=int(window.size))
