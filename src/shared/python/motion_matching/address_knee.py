"""Knee flexion at address: owner prior, capture override, no hyperextension.

R4 of the realism epic #12678 (#12684). The address stage had no knee target:
the multi-start fit kept whichever knee angle fitted the markers best, and the
lower-limb range of motion allows 10 deg of hyperextension. This module is the
one place that defines, measures and resolves the address knee flexion so the
capture, the address solve and the receipt quote the same number.

Owner ruling (#12684, 2026-10-15)
---------------------------------
* The address knee flexion is a **prior of about 15 deg with a +-5 deg band**,
  for both knees (``KNEE_FLEX_PRIOR_DEG``, ``KNEE_FLEX_BAND_DEG``). These are
  owner-set values, not literature numbers.
* **Capture wins** wherever the markers determine the knee flexion reliably
  (``capture`` mode); the prior fills in where they do not.
* **Hyperextension is forbidden** as a hard bound on the knee coordinate (never
  a soft penalty) plus a postcondition on the solved address.
* Every knee's target carries its source into the receipt with R3's
  ``target_source`` vocabulary (``foot_progression.TARGET_SOURCES``).

Definitions
-----------
* Geometric knee flexion: the angle between the hip -> knee and knee -> ankle
  joint-centre vectors (0 deg = straight leg). It is unsigned, so the knee
  *coordinate* decides hyperextension: the coordinate's flexion direction is
  found from forward kinematics (sign-safe), and the coordinate may not pass
  its anatomical zero on the other side.
* Model joint centres: the ``femur_*``, ``tibia_*`` and ``talus_*`` body origins.
* Capture joint centres: the hip centre from the functional (sphere-fit) hip
  calibration, carried per frame by the rigid waist-marker pelvis pose; the knee
  and ankle centres are the lateral ``*KneeOut``/``*AnkleOut`` markers moved
  medially along the knee flexion axis by the leg-seed lateral offsets
  (``hip_calibration.knee_flexion_axis``). Capture value: the median over the
  address window, reliable when the per-frame spread and coverage meet R3's
  address-window criteria (``foot_progression``).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.motion_matching.foot_progression import (
    MAX_SPREAD_DEG,
    MIN_VALID_FRACTION,
    MIN_WINDOW_FRAMES,
    TARGET_SOURCES,
    address_window,
    foot_role,
)
from src.shared.python.motion_matching.hip_calibration import (
    LATERAL_SEEDS_M,
    functional_hip_calibration,
    knee_flexion_axis,
)
from src.shared.python.motion_matching.marker_calibration import (
    rigid_pose_from_markers,
)

__all__ = [
    "CANONICAL_KNEE_FLEXION",
    "KNEE_FLEX_BAND_DEG",
    "KNEE_FLEX_PRIOR_DEG",
    "KNEE_FLEXION_PRIOR_WEIGHT",
    "KNEE_TARGET_SOURCES",
    "LATERAL_MARKER_OFFSETS_M",
    "MODES",
    "KneeFlexion",
    "KneeTargets",
    "assert_no_hyperextension",
    "capture_knee_flexion",
    "flexion_sign",
    "knee_coordinate_for_flexion",
    "knee_flexion_deg",
    "knee_flexion_from_markers",
    "knee_flexion_report",
    "knee_posture_prior",
    "model_knee_flexion_deg",
    "no_hyperextension_bounds",
    "record_knee_flexion",
    "requested_mode",
    "resolve_knee_target",
]

#: Owner prior for the address knee flexion of both knees (#12684, 2026-10-15).
KNEE_FLEX_PRIOR_DEG: float = 15.0
#: Owner band about the address knee target (#12684, 2026-10-15).
KNEE_FLEX_BAND_DEG: float = 5.0
#: Weight (per rad^2) of the soft knee-flexion residual in the address solve.
#: Re-exported by ``pipeline.constants``; see the R4 evidence README.
KNEE_FLEXION_PRIOR_WEIGHT: float = 1.0
#: ``target_source`` values a knee target can carry (a subset of R3's vocabulary).
KNEE_TARGET_SOURCES: tuple[str, ...] = ("capture", "prior")
#: ``--knee-flexion`` choices: ``off`` (legacy: no knee target, no bound),
#: ``prior`` (the owner prior for both knees) and ``capture`` (a reliable
#: capture measurement wins, the prior fills in).
MODES: tuple[str, ...] = ("off", "prior", "capture")
#: Mode of the canonical address when no ``--knee-flexion`` is passed.
CANONICAL_KNEE_FLEXION: str = "capture"
#: Lateral knee and ankle marker distances from the joint centres (leg seeds).
LATERAL_MARKER_OFFSETS_M: tuple[float, float] = LATERAL_SEEDS_M

if not set(KNEE_TARGET_SOURCES) <= set(TARGET_SOURCES):
    raise ImportError("knee target sources must reuse foot_progression.TARGET_SOURCES")

_SIDES = ("left", "right")
_SFX = {"left": "l", "right": "r"}
_PREFIX = {"left": "L", "right": "R"}
#: ``hip_calibration`` side index: 0 is the right leg, 1 the left.
_SIDE_INDEX = {"right": 0, "left": 1}
_PROBE_RAD = float(np.radians(10.0))
_BISECTION_STEPS = 60
_FLEXION_SPAN_RAD = float(np.radians(120.0))


# ------------------------------------------------------------- measurement


def _point(value: Any, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{what} must be a finite 3-vector")
    return arr


def knee_flexion_deg(hip: Any, knee: Any, ankle: Any) -> float:
    """Geometric knee flexion (deg) from the hip, knee and ankle joint centres.

    Preconditions: finite 3-vectors; thigh and shank segments longer than 1 mm.
    Postcondition: a value in [0, 180]; 0 for a straight (collinear) leg.
    """
    h, k, a = _point(hip, "hip"), _point(knee, "knee"), _point(ankle, "ankle")
    thigh, shank = k - h, a - k
    lengths = float(np.linalg.norm(thigh)), float(np.linalg.norm(shank))
    if min(lengths) < 1e-3:
        raise ValueError(f"degenerate leg segment (lengths {lengths} m)")
    cosine = float(thigh @ shank) / (lengths[0] * lengths[1])
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def knee_flexion_from_markers(
    hip: Any,
    knee_out: Any,
    ankle_out: Any,
    *,
    side: str,
    lateral: tuple[float, float] = LATERAL_MARKER_OFFSETS_M,
) -> float:
    """Knee flexion (deg) from the hip centre and the lateral knee/ankle markers.

    The markers are moved medially along the knee flexion axis
    (:func:`hip_calibration.knee_flexion_axis`) onto the joint centres. Raises
    ``ValueError`` for a bad side, non-finite input or a leg too straight to
    define the flexion plane.
    """
    if side not in _SIDE_INDEX:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    h = _point(hip, "hip")
    k = _point(knee_out, "knee_out")
    a = _point(ankle_out, "ankle_out")
    axis = knee_flexion_axis(
        h.tolist(), k.tolist(), a.tolist(), side=_SIDE_INDEX[side], lateral=lateral
    )
    return knee_flexion_deg(h, k - lateral[0] * axis, a - lateral[1] * axis)


@dataclass(frozen=True)
class KneeFlexion:
    """One knee's capture-derived address flexion, with its reliability."""

    side: str
    role: str
    angle_deg: float
    frames: int
    spread_deg: float
    reliable: bool
    reason: str

    def to_receipt(self) -> dict[str, object]:
        """JSON-safe receipt entry (NaN becomes ``None``)."""

        def clean(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        return {
            "side": self.side,
            "role": self.role,
            "angle_deg": clean(self.angle_deg),
            "window_frames": self.frames,
            "spread_deg": clean(self.spread_deg),
            "reliable": self.reliable,
            "reason": self.reason,
        }


def _hip_centres(
    points: np.ndarray,
    valid: np.ndarray,
    labels: tuple[str, ...],
    waist_offsets: Mapping[str, Sequence[float]],
    window: np.ndarray,
) -> dict[str, np.ndarray]:
    """World hip centres per window frame (NaN where the pelvis is not seen)."""
    calibration = functional_hip_calibration(points, valid, labels, waist_offsets)
    waist = [label for label in waist_offsets if label in labels]
    cols = [labels.index(label) for label in waist]
    offsets = np.asarray([waist_offsets[label] for label in waist], dtype=float)
    local = {"right": calibration.centre_r, "left": calibration.centre_l}
    out = {side: np.full((len(window), 3), np.nan) for side in _SIDES}
    for row, frame in enumerate(window):
        if not valid[frame, cols].all():
            continue
        rotation, translation = rigid_pose_from_markers(offsets, points[frame, cols])
        for side in _SIDES:
            out[side][row] = rotation @ np.asarray(local[side]) + translation
    return out


def _measure_knee(
    side: str,
    role: str,
    hip: np.ndarray,
    knee_out: np.ndarray,
    ankle_out: np.ndarray,
) -> KneeFlexion:
    """Median flexion over the window rows where every point is finite."""
    angles = []
    for h, k, a in zip(hip, knee_out, ankle_out, strict=True):
        try:
            angles.append(knee_flexion_from_markers(h, k, a, side=side))
        except ValueError:
            continue  # pelvis or leg marker missing, or no flexion plane
    n = len(angles)
    if n < MIN_VALID_FRACTION * len(hip) or n < MIN_WINDOW_FRAMES:
        nan = float("nan")
        reason = f"knee markers usable in only {n} of {len(hip)} window frames"
        return KneeFlexion(side, role, nan, n, nan, False, reason)
    values = np.asarray(angles)
    spread = float(np.subtract(*np.percentile(values, [75, 25])))
    reliable = spread <= MAX_SPREAD_DEG
    reason = "measured over the address window"
    if not reliable:
        reason = f"unstable knee markers (IQR {spread:.1f} deg > {MAX_SPREAD_DEG} deg)"
    return KneeFlexion(
        side, role, float(np.median(values)), n, spread, reliable, reason
    )


def capture_knee_flexion(
    points: Any,
    valid: Any,
    labels: Sequence[str],
    waist_offsets: Mapping[str, Sequence[float]],
    *,
    handedness: str = "right",
) -> dict[str, KneeFlexion]:
    """Per-knee address flexion of a capture (``left``/``right`` keys).

    ``points`` (frames, markers, 3) are world metres with validity ``valid``;
    ``waist_offsets`` are the pelvis-frame offsets of the waist markers.
    Preconditions: the waist, ``*KneeOut``, ``*AnkleOut`` and wrist markers are
    capture labels (``ValueError`` otherwise). A knee whose markers are missing
    or unstable is returned with ``reliable=False``, never a silent value.
    """
    pts = np.asarray(points, dtype=float)
    ok = np.asarray(valid, dtype=bool)
    names = tuple(labels)
    needed = [f"{p}{m}" for p in "LR" for m in ("KneeOut", "AnkleOut")]
    missing = [n for n in needed if n not in names]
    if missing:
        raise ValueError(f"capture lacks knee/ankle markers: {missing}")
    window = address_window(pts, ok, names)
    hips = _hip_centres(pts, ok, names, waist_offsets, window)
    out: dict[str, KneeFlexion] = {}
    for side in _SIDES:
        series = []
        for marker in ("KneeOut", "AnkleOut"):
            col = names.index(f"{_PREFIX[side]}{marker}")
            series.append(np.where(ok[window, col, None], pts[window, col], np.nan))
        role = foot_role(side, handedness)
        out[side] = _measure_knee(side, role, hips[side], series[0], series[1])
    return out


# ------------------------------------------------------- target resolution


def resolve_knee_target(
    measured: KneeFlexion | None, *, mode: str
) -> tuple[float, str]:
    """``(target_deg, source)`` for one knee under ``mode``.

    * ``prior``: the owner prior ``KNEE_FLEX_PRIOR_DEG`` (source ``prior``).
    * ``capture``: a reliable capture measurement (source ``capture``), else
      the prior (source ``prior``).

    Preconditions: ``mode`` is ``prior`` or ``capture`` (``off`` sets no
    target). Postcondition: a finite flexion in [0, 120) deg; a capture value
    that would hyperextend the knee raises ``ValueError`` instead.
    """
    if mode not in MODES[1:]:
        raise ValueError(
            f"mode must be one of {MODES[1:]}, got {mode!r} "
            "('off' sets no knee target; do not resolve one)"
        )
    if mode == "capture" and measured is not None and measured.reliable:
        angle, source = float(measured.angle_deg), "capture"
    else:
        angle, source = KNEE_FLEX_PRIOR_DEG, "prior"
    if not (np.isfinite(angle) and 0.0 <= angle < 120.0):
        raise ValueError(
            f"knee flexion target {angle!r} deg is outside [0, 120): a negative "
            "flexion is hyperextension, which the address forbids"
        )
    return angle, source


@dataclass(frozen=True)
class KneeTargets:
    """Requested address knee flexion per knee, with provenance.

    ``sources`` tags each knee ``capture`` or ``prior``; ``measured`` carries
    the capture report when there is one; ``weight`` is the soft residual
    weight of the address solve.
    """

    target_deg: Mapping[str, float]
    sources: Mapping[str, str]
    mode: str = "capture"
    handedness: str = "right"
    weight: float = KNEE_FLEXION_PRIOR_WEIGHT
    notes: str = ""
    measured: Mapping[str, KneeFlexion] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.target_deg) != set(_SIDES):
            raise ValueError("target_deg must map exactly 'left' and 'right'")
        bad = {s: v for s, v in self.sources.items() if v not in KNEE_TARGET_SOURCES}
        if bad or set(self.sources) != set(_SIDES):
            raise ValueError(
                f"sources must map 'left' and 'right' to one of "
                f"{KNEE_TARGET_SOURCES}, got {dict(self.sources)!r}"
            )
        for side, deg in self.target_deg.items():
            if not (np.isfinite(deg) and 0.0 <= deg < 120.0):
                raise ValueError(
                    f"{side} knee target {deg!r} deg: below 0 is hyperextension"
                )
        if self.mode not in MODES[1:]:
            raise ValueError(f"mode must be one of {MODES[1:]}, got {self.mode!r}")
        if not (np.isfinite(self.weight) and self.weight > 0.0):
            raise ValueError("weight must be finite and positive")


def requested_mode(args: argparse.Namespace) -> str:
    """``--knee-flexion`` of ``args``; ``off`` for a Namespace without it."""
    return str(getattr(args, "knee_flexion", "off"))


# -------------------------------------------------------------- model side


def model_knee_flexion_deg(kin: Any, q: np.ndarray) -> dict[str, float]:
    """Geometric knee flexion per knee at ``q`` from the model joint centres."""
    names = [f"{b}_{sfx}" for sfx in _SFX.values() for b in ("femur", "tibia", "talus")]
    poses = kin.body_poses(np.asarray(q, dtype=float), names)
    return {
        side: knee_flexion_deg(
            poses[f"femur_{sfx}"][1], poses[f"tibia_{sfx}"][1], poses[f"talus_{sfx}"][1]
        )
        for side, sfx in _SFX.items()
    }


def _knee_index(kin: Any, side: str) -> int:
    name = f"knee_angle_{_SFX[side]}"
    order = list(kin.coordinate_order)
    if name not in order:
        raise ValueError(f"model has no {name} coordinate")
    return order.index(name)


def _flexion_at(kin: Any, q: np.ndarray, side: str, coord: float) -> float:
    moved = np.asarray(q, dtype=float).copy()
    moved[_knee_index(kin, side)] = coord
    return model_knee_flexion_deg(kin, moved)[side]


def flexion_sign(kin: Any, q: np.ndarray, side: str) -> float:
    """+1 or -1: the knee coordinate direction that flexes the knee (sign-safe)."""
    plus = _flexion_at(kin, q, side, _PROBE_RAD)
    minus = _flexion_at(kin, q, side, -_PROBE_RAD)
    if abs(plus - minus) < 1.0:
        raise ValueError(f"the {side} knee coordinate barely moves the shank")
    return 1.0 if plus > minus else -1.0


def knee_coordinate_for_flexion(
    kin: Any, q: np.ndarray, side: str, flexion_deg: float
) -> float:
    """Knee coordinate (rad) giving ``flexion_deg`` of geometric flexion.

    Searched on the flexion side only, so the result never hyperextends; a
    target below the flexion at the coordinate's zero returns 0. Precondition:
    ``flexion_deg`` in [0, 120). Postcondition: ``sign * coord >= 0``.
    """
    if not (np.isfinite(flexion_deg) and 0.0 <= flexion_deg < 120.0):
        raise ValueError(f"flexion_deg must be in [0, 120), got {flexion_deg!r}")
    sign = flexion_sign(kin, q, side)
    if _flexion_at(kin, q, side, 0.0) >= flexion_deg:
        return 0.0
    lo, hi = 0.0, _FLEXION_SPAN_RAD
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if _flexion_at(kin, q, side, sign * mid) < flexion_deg:
            lo = mid
        else:
            hi = mid
    return float(sign * 0.5 * (lo + hi))


def no_hyperextension_bounds(
    kin: Any, q: np.ndarray, bounds: Mapping[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    """Copy of ``bounds`` whose knee ranges stop at the anatomical zero.

    Hard bound (owner ruling): the extension side of each knee coordinate is
    cut at 0. Only ever tightens; a knee without a range gets the flexion half.
    """
    out = dict(bounds)
    for side in _SIDES:
        name = f"knee_angle_{_SFX[side]}"
        lo, hi = out.get(name, (-np.inf, np.inf))
        if flexion_sign(kin, q, side) < 0:
            out[name] = (lo, min(hi, 0.0))
        else:
            out[name] = (max(lo, 0.0), hi)
    return out


def assert_no_hyperextension(kin: Any, q: np.ndarray) -> None:
    """Postcondition of the address: neither knee coordinate is hyperextended."""
    values = np.asarray(q, dtype=float)
    for side in _SIDES:
        coord = float(values[_knee_index(kin, side)])
        if flexion_sign(kin, values, side) * coord < -1e-9:
            raise ValueError(
                f"{side} knee hyperextended at address "
                f"(knee_angle_{_SFX[side]} = {np.degrees(coord):.2f} deg)"
            )


def knee_posture_prior(
    kin: Any, q: np.ndarray, targets: KneeTargets
) -> dict[str, tuple[float, float]]:
    """Solve-pose ``posture_prior`` pulling each knee to its target flexion."""
    return {
        f"knee_angle_{_SFX[side]}": (
            knee_coordinate_for_flexion(kin, q, side, float(targets.target_deg[side])),
            float(targets.weight),
        )
        for side in _SIDES
    }


# ----------------------------------------------------------------- receipt


def knee_flexion_report(
    kin: Any, q: np.ndarray | None, targets: KneeTargets | None
) -> dict[str, Any]:
    """Receipt block ``knee_flexion_deg``: per-knee model vs target and source."""
    if targets is None:
        return {"enabled": False}
    values = np.asarray(q, dtype=float)
    model = model_knee_flexion_deg(kin, values)
    block: dict[str, Any] = {
        "enabled": True,
        "mode": targets.mode,
        "prior_deg": KNEE_FLEX_PRIOR_DEG,
        "band_deg": KNEE_FLEX_BAND_DEG,
        "weight": float(targets.weight),
        "notes": targets.notes,
        "knees": {},
    }
    for side in _SIDES:
        coord = float(values[_knee_index(kin, side)])
        error = float(model[side] - targets.target_deg[side])
        entry: dict[str, Any] = {
            "role": foot_role(side, targets.handedness),
            "target_deg": float(targets.target_deg[side]),
            "target_source": targets.sources[side],
            "model_deg": float(model[side]),
            "coordinate_deg": float(np.degrees(coord)),
            "error_deg": error,
            "within_band": abs(error) <= KNEE_FLEX_BAND_DEG,
            "hyperextended": flexion_sign(kin, values, side) * coord < -1e-9,
        }
        measured = targets.measured.get(side)
        if measured is not None:
            entry["capture"] = measured.to_receipt()
        block["knees"][side] = entry
    return block


def record_knee_flexion(
    address_report: dict[str, Any],
    kin: Any,
    q: np.ndarray | None,
    targets: KneeTargets | None,
) -> None:
    """Add the ``knee_flexion_deg`` block to ``address_report`` when enabled."""
    if targets is not None:
        address_report["knee_flexion_deg"] = knee_flexion_report(kin, q, targets)
