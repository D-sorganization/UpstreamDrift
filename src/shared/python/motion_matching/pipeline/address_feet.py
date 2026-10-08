"""Address foot-progression residual for the ground-support pipeline (OSV-4, #11730).

The models have no foot-yaw coordinate: foot yaw is pelvis yaw plus the leg's
``hip_rotation_*`` (and a little knee/hip-flexion coupling). The marker fit
cannot see it once the foot marker offsets are calibrated at the fitted pose,
so it is decided by whichever multi-start seed happens to fit best. This module
makes it an explicit, receipted choice: after the address fit, ``hip_rotation``
is moved until each foot's calcn -> toes axis has the requested toe-out, with a
high prior weight (a soft constraint, the markers can still overrule it) and
the stance spheres still pinned flat by the solver.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.motion_matching.foot_progression import (
    DEFAULT_TOE_OUT_DEG,
    FootProgression,
    capture_foot_progression,
    foot_role,
    model_long_axis,
    progression_angle_deg,
    resolve_toe_out_target,
)

__all__ = [
    "FOOT_PRIOR_WEIGHT",
    "FOOT_TOLERANCE_DEG",
    "FootTargets",
    "NATIVE_TARGET_AXIS",
    "NATIVE_UP_AXIS",
    "build_foot_targets",
    "foot_progression_report",
    "foot_progression_series",
    "model_feet_deg",
    "refine_foot_progression",
]

#: Capture/native world (Z up): the target is toward -Y (the lead foot's side).
NATIVE_TARGET_AXIS = np.array([0.0, -1.0, 0.0])
NATIVE_UP_AXIS = np.array([0.0, 0.0, 1.0])
#: Prior weight on ``hip_rotation_*`` while the foot-progression refit runs.
FOOT_PRIOR_WEIGHT: float = 1.0e3
#: Convergence tolerance of the refit; the issue asks for 2 degrees.
FOOT_TOLERANCE_DEG: float = 0.5
MAX_REFINE_PASSES: int = 4
MODES = ("off", "capture", "default")
_SIDE_SUFFIX = {"left": "l", "right": "r"}


@dataclass(frozen=True)
class FootTargets:
    """Requested address toe-out per foot, with provenance.

    ``is_default`` marks a flagged default (``DEFAULT_TOE_OUT_DEG``), never a
    measurement. ``measured`` carries the capture report when there is one.
    """

    target_deg: Mapping[str, float]
    is_default: Mapping[str, bool]
    source: str
    handedness: str = "right"
    weight: float = FOOT_PRIOR_WEIGHT
    notes: str = ""
    measured: Mapping[str, FootProgression] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for side in ("left", "right"):
            if side not in self.target_deg:
                raise ValueError(f"target_deg needs a '{side}' entry")
        if self.weight <= 0.0:
            raise ValueError("weight must be positive")


def build_foot_targets(
    lane: Any, mode: str, *, handedness: str = "right"
) -> FootTargets | None:
    """Targets for ``mode``: ``off`` -> None, ``default`` -> flagged 20 deg,
    ``capture`` -> the capture's address median, with a flagged default for any
    foot the markers cannot measure reliably."""
    if mode not in MODES:
        raise ValueError(f"foot_progression must be one of {MODES}, got {mode!r}")
    if mode == "off":
        return None
    default = {"left": DEFAULT_TOE_OUT_DEG, "right": DEFAULT_TOE_OUT_DEG}
    if mode == "default":
        return FootTargets(
            default,
            {"left": True, "right": True},
            "default",
            handedness,
            notes="forced default, not a measurement",
        )
    try:
        measured = capture_foot_progression(
            lane.points,
            lane.valid,
            tuple(lane.labels),
            up=NATIVE_UP_AXIS,
            target_axis=NATIVE_TARGET_AXIS,
            handedness=handedness,
        )
    except ValueError as exc:
        return FootTargets(
            default,
            {"left": True, "right": True},
            "default",
            handedness,
            notes=f"capture foot markers unusable ({exc}); wrist or foot markers",
        )
    chosen = {s: resolve_toe_out_target(m) for s, m in measured.items()}
    return FootTargets(
        {s: v[0] for s, v in chosen.items()},
        {s: v[1] for s, v in chosen.items()},
        "capture",
        handedness,
        notes="capture address median (ankle-toe axis, malleolus-corrected)",
        measured=measured,
    )


def model_feet_deg(kin: Any, q: np.ndarray, targets: FootTargets) -> dict[str, float]:
    """Model toe-out per foot at ``q`` from the calcn -> toes axis."""
    names = [f"{b}_{sfx}" for sfx in ("r", "l") for b in ("calcn", "toes")]
    poses = kin.body_poses(q, names)
    out: dict[str, float] = {}
    for side, sfx in _SIDE_SUFFIX.items():
        axis = model_long_axis(
            poses[f"calcn_{sfx}"][1], poses[f"toes_{sfx}"][1], NATIVE_UP_AXIS
        )
        out[side] = progression_angle_deg(
            axis,
            target_axis=NATIVE_TARGET_AXIS,
            up=NATIVE_UP_AXIS,
            foot_role=foot_role(side, targets.handedness),
            handedness=targets.handedness,
        )
    return out


def _sensitivity(
    kin: Any, q: np.ndarray, targets: FootTargets, side: str, index: int
) -> float:
    """d(toe-out)/d(hip_rotation) in deg/deg by central difference (sign-safe)."""
    step = np.radians(1.0)
    hi, lo = q.copy(), q.copy()
    hi[index] += step
    lo[index] -= step
    d = model_feet_deg(kin, hi, targets)[side] - model_feet_deg(kin, lo, targets)[side]
    return float(d / np.degrees(2.0 * step))


def refine_foot_progression(
    kin: Any,
    fit: Any,
    targets: FootTargets,
    resolve: Callable[[np.ndarray, Mapping[str, float]], Any],
) -> Any:
    """Move ``hip_rotation_*`` until each foot has its target toe-out.

    ``resolve(q_start, prior_weights)`` re-solves the address pose from
    ``q_start`` with the given extra prior weights (the caller keeps its own
    bounds, stance pinning and marker weights). Returns the last fit; ``fit``
    itself when both feet are already within ``FOOT_TOLERANCE_DEG``.
    """
    index = {}
    for side, sfx in _SIDE_SUFFIX.items():
        name = f"hip_rotation_{sfx}"
        if name not in kin.coordinate_order:
            raise ValueError(f"model has no {name} coordinate")
        index[side] = list(kin.coordinate_order).index(name)
    prior = {f"hip_rotation_{sfx}": targets.weight for sfx in _SIDE_SUFFIX.values()}
    current = fit
    for _ in range(MAX_REFINE_PASSES):
        angles = model_feet_deg(kin, current.q, targets)
        errors = {s: targets.target_deg[s] - angles[s] for s in angles}
        if max(abs(e) for e in errors.values()) <= FOOT_TOLERANCE_DEG:
            break
        q_start = np.asarray(current.q, dtype=float).copy()
        for side, err in errors.items():
            gain = _sensitivity(kin, q_start, targets, side, index[side])
            if abs(gain) < 0.1:
                raise ValueError(
                    f"hip_rotation barely turns the {side} foot (gain {gain:.2f})"
                )
            q_start[index[side]] += np.radians(err / gain)
        current = resolve(q_start, prior)
    return current


def foot_progression_report(
    kin: Any,
    q: np.ndarray,
    targets: FootTargets | None,
    *,
    before_q: np.ndarray | None = None,
) -> dict[str, Any]:
    """Receipt block: per-foot model vs capture/target toe-out at ``q``."""
    if targets is None:
        return {"enabled": False}
    model = model_feet_deg(kin, q, targets)
    block: dict[str, Any] = {
        "enabled": True,
        "source": targets.source,
        "notes": targets.notes,
        "tolerance_deg": FOOT_TOLERANCE_DEG,
        "feet": {},
    }
    for side in ("left", "right"):
        entry: dict[str, Any] = {
            "role": foot_role(side, targets.handedness),
            "target_deg": float(targets.target_deg[side]),
            "target_is_default": bool(targets.is_default[side]),
            "model_deg": float(model[side]),
            "error_deg": float(model[side] - targets.target_deg[side]),
        }
        if before_q is not None:
            entry["model_before_deg"] = float(
                model_feet_deg(kin, before_q, targets)[side]
            )
        measured = targets.measured.get(side)
        if measured is not None:
            entry["capture"] = measured.to_receipt()
        block["feet"][side] = entry
    return block


def foot_progression_series(
    kin: Any, q_traj: np.ndarray, targets: FootTargets, stride: int = 1
) -> dict[str, list[float]]:
    """Model toe-out time series (every ``stride`` frames) for the receipt."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    series: dict[str, list[float]] = {"left": [], "right": []}
    for q in np.asarray(q_traj)[::stride]:
        for side, deg in model_feet_deg(kin, q, targets).items():
            series[side].append(round(float(deg), 3))
    return series
