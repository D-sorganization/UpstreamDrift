"""Foot progression (toe-out) angle at address, from markers and from model frames.

OSV-4 (#11730). The models have no foot-yaw coordinate: foot yaw is a by-product
of pelvis yaw, ``hip_rotation_*`` and the knee. This module is the single place
that defines and measures the angle so a capture, an address seed and a fitted
receipt all quote the same number.

Definitions (binding, from the issue)
-------------------------------------
* ``target_axis`` (``x_t``) is the horizontal direction toward the target and
  ``up`` the world up axis. The golfer-forward axis (toward the ball) is
  ``f = x_t x up`` for a right-handed golfer. NOTE: the issue text writes
  ``up x x_t``; with the lead (left) foot on the target side that vector points
  *behind* the golfer in both the capture world and the model world (checked
  against the C3D toe markers and the model toes), so the sign here is chosen to
  mean "toward the ball". A left-handed golfer is the mirror image: ``f`` flips
  and the right foot becomes the lead foot.
* Foot progression angle: the signed angle from ``f`` to the foot long axis
  about ``up``, positive = toes turned out (lead foot toward the target, trail
  foot away from it). Lead and trail are reported separately.
* Marker long axis: ``normalise(proj_ground(toe - heel))`` with the heel proxy
  the ``*AnkleOut`` marker and the toe point the midpoint of ``*ToeIn`` and
  ``*ToeOut``. A model foot frame uses the calcn -> toes axis instead.
* Capture value: median over the address window (the first static window before
  takeaway).
* When the capture has no reliable foot markers the value is
  ``DEFAULT_TOE_OUT_DEG``, flagged ``is_default`` (a default, not a measurement).

Estimator note
--------------
``*AnkleOut`` is the lateral malleolus, about 0.04 m outside the heel centre
line, so the raw ankle -> toe axis reads roughly 13 degrees toe-in on a straight
foot. :func:`capture_foot_progression` therefore reports three numbers per foot:
``raw_angle_deg`` (the binding definition, unmodified), ``angle_deg`` (the same
axis with the heel proxy moved medially by ``ANKLE_LATERAL_OFFSET_M``; this is
the headline value) and ``forefoot_angle_deg`` (the normal of the
``ToeIn``-``ToeOut`` line, an independent cross-check).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ANKLE_LATERAL_OFFSET_M",
    "DEFAULT_TOE_OUT_DEG",
    "FootProgression",
    "address_window",
    "capture_foot_progression",
    "foot_role",
    "forward_axis",
    "marker_long_axis",
    "model_long_axis",
    "progression_angle_deg",
    "resolve_toe_out_target",
]

#: Toe-out per foot used when the capture has no reliable foot markers.
DEFAULT_TOE_OUT_DEG: float = 20.0
#: Lateral malleolus to heel-centre-line distance (adult male, documented
#: anthropometric value; the raw binding estimator uses 0).
ANKLE_LATERAL_OFFSET_M: float = 0.04
#: Wrist displacement from the opening pose that ends the address window.
TAKEAWAY_DISPLACEMENT_M: float = 0.03
#: Frames used to define the opening wrist pose.
OPENING_FRAMES: int = 5
MIN_WINDOW_FRAMES: int = 5
#: Per-frame spread (degrees) above which the capture value is not reliable.
MAX_SPREAD_DEG: float = 3.0
#: Fraction of window frames that must carry all three foot markers.
MIN_VALID_FRACTION: float = 0.8

_HANDEDNESS = ("right", "left")
_ROLES = ("lead", "trail")


def _unit(vec: np.ndarray, what: str) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"{what} must be a non-zero finite vector")
    return vec / norm


def _check_handedness(handedness: str) -> None:
    if handedness not in _HANDEDNESS:
        raise ValueError(f"handedness must be one of {_HANDEDNESS}, got {handedness!r}")


def _ground(vec: np.ndarray, up: np.ndarray, what: str) -> np.ndarray:
    """Project ``vec`` onto the ground plane (normal ``up``) and normalise."""
    up_u = _unit(np.asarray(up, dtype=float), "up")
    v = np.asarray(vec, dtype=float)
    flat = v - (v @ up_u) * up_u
    if np.linalg.norm(flat) < 1e-9 * max(1.0, np.linalg.norm(v)):
        raise ValueError(f"{what} is vertical: no ground-plane direction")
    return _unit(flat, what)


def forward_axis(
    target_axis: np.ndarray, up: np.ndarray, handedness: str = "right"
) -> np.ndarray:
    """Golfer-forward unit axis (toward the ball), horizontal.

    Postcondition: unit length, orthogonal to ``up`` and ``target_axis``.
    """
    _check_handedness(handedness)
    up_u = _unit(np.asarray(up, dtype=float), "up")
    target = _ground(target_axis, up_u, "target_axis")
    fwd = np.cross(target, up_u)
    return fwd if handedness == "right" else -fwd


def foot_role(side: str, handedness: str = "right") -> str:
    """``lead`` or ``trail`` for ``left``/``right`` foot and golfer handedness."""
    _check_handedness(handedness)
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    lead_side = "left" if handedness == "right" else "right"
    return "lead" if side == lead_side else "trail"


def progression_angle_deg(
    long_axis: np.ndarray,
    *,
    target_axis: np.ndarray,
    up: np.ndarray,
    foot_role: str,  # noqa: A002 - mirrors the issue vocabulary
    handedness: str = "right",
) -> float:
    """Signed toe-out angle in degrees; positive = toes turned out.

    Preconditions: ``long_axis`` is not vertical; ``foot_role`` is ``lead`` or
    ``trail``. Postcondition: the ground projection removes any foot pitch.
    """
    if foot_role not in _ROLES:
        raise ValueError(f"foot_role must be one of {_ROLES}, got {foot_role!r}")
    up_u = _unit(np.asarray(up, dtype=float), "up")
    axis = _ground(long_axis, up_u, "long_axis")
    target = _ground(target_axis, up_u, "target_axis")
    fwd = forward_axis(target, up_u, handedness)
    out = target if foot_role == "lead" else -target
    return float(np.degrees(np.arctan2(axis @ out, axis @ fwd)))


def model_long_axis(calcn: np.ndarray, toes: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Ground-projected calcn -> toes unit axis of a model foot."""
    return _ground(
        np.asarray(toes, float) - np.asarray(calcn, float), up, "calcn->toes"
    )


def marker_long_axis(
    heel: np.ndarray,
    toe_in: np.ndarray,
    toe_out: np.ndarray,
    *,
    up: np.ndarray,
    out_dir: np.ndarray | None = None,
    ankle_lateral_offset_m: float = 0.0,
) -> np.ndarray:
    """Ground-projected heel -> toe-midpoint unit axis from the foot markers.

    With ``ankle_lateral_offset_m == 0`` this is exactly the issue definition.
    A positive offset moves the heel proxy medially (opposite ``out_dir``,
    perpendicular to the axis, iterated to convergence) before forming the axis.
    """
    if ankle_lateral_offset_m < 0.0:
        raise ValueError("ankle_lateral_offset_m must be non-negative")
    up_u = _unit(np.asarray(up, dtype=float), "up")
    heel_a = np.asarray(heel, float)
    toe_mid = 0.5 * (np.asarray(toe_in, float) + np.asarray(toe_out, float))
    axis = _ground(toe_mid - heel_a, up_u, "heel->toe")
    if ankle_lateral_offset_m == 0.0:
        return axis
    if out_dir is None:
        raise ValueError("out_dir is required when ankle_lateral_offset_m > 0")
    out_g = _ground(out_dir, up_u, "out_dir")
    for _ in range(4):
        lateral = np.cross(up_u, axis)
        lateral = lateral if lateral @ out_g > 0 else -lateral
        axis = _ground(
            toe_mid - (heel_a - ankle_lateral_offset_m * lateral), up_u, "axis"
        )
    return axis


def _forefoot_axis(
    toe_in: np.ndarray, toe_out: np.ndarray, up: np.ndarray, forward: np.ndarray
) -> np.ndarray:
    """Normal of the ToeIn-ToeOut line in the ground plane, pointing forward."""
    up_u = _unit(np.asarray(up, dtype=float), "up")
    line = _ground(
        np.asarray(toe_out, float) - np.asarray(toe_in, float), up_u, "toe line"
    )
    normal = np.cross(up_u, line)
    return normal if normal @ forward > 0 else -normal


@dataclass(frozen=True)
class FootProgression:
    """One foot's address toe-out, with its provenance.

    ``angle_deg`` is the headline value (measurement or flagged default);
    ``raw_angle_deg``/``forefoot_angle_deg`` are NaN when no measurement exists.
    """

    side: str
    role: str
    angle_deg: float
    raw_angle_deg: float
    forefoot_angle_deg: float
    frames: int
    spread_deg: float
    reliable: bool
    is_default: bool
    reason: str

    def to_receipt(self) -> dict[str, object]:
        """JSON-safe receipt entry (NaN becomes ``None``)."""

        def clean(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        return {
            "side": self.side,
            "role": self.role,
            "angle_deg": clean(self.angle_deg),
            "raw_angle_deg": clean(self.raw_angle_deg),
            "forefoot_angle_deg": clean(self.forefoot_angle_deg),
            "window_frames": self.frames,
            "spread_deg": clean(self.spread_deg),
            "reliable": self.reliable,
            "is_default": self.is_default,
            "reason": self.reason,
        }


def resolve_toe_out_target(measured: FootProgression | None) -> tuple[float, bool]:
    """``(target_deg, is_default)``: the measurement when reliable, else the default."""
    if measured is not None and measured.reliable and not measured.is_default:
        return float(measured.angle_deg), False
    return DEFAULT_TOE_OUT_DEG, True


def address_window(
    points: np.ndarray,
    valid: np.ndarray,
    labels: Sequence[str],
    *,
    takeaway_m: float = TAKEAWAY_DISPLACEMENT_M,
) -> np.ndarray:
    """Frame indices of the first static window before takeaway.

    The window runs from frame 0 until a wrist marker has moved ``takeaway_m``
    from its opening pose (median of the first ``OPENING_FRAMES`` valid frames).
    Preconditions: ``points`` (frames, markers, 3), ``valid`` (frames, markers),
    at least one of ``LWristTop``/``RWristTop`` present, and a window of at
    least ``MIN_WINDOW_FRAMES`` frames.
    """
    pts = np.asarray(points, float)
    ok = np.asarray(valid, bool)
    if pts.ndim != 3 or pts.shape[2] != 3 or ok.shape != pts.shape[:2]:
        raise ValueError("points must be (frames, markers, 3) matching valid")
    if len(labels) != pts.shape[1]:
        raise ValueError("labels length must match the marker axis")
    cols = [labels.index(m) for m in ("LWristTop", "RWristTop") if m in labels]
    if not cols:
        raise ValueError("address window needs a wrist marker (LWristTop/RWristTop)")
    wrist = np.where(ok[:, cols, None], pts[:, cols], np.nan)
    opening = np.nanmedian(wrist[:OPENING_FRAMES], axis=0)
    if not np.isfinite(opening).all():
        raise ValueError("wrist markers are missing at the start of the capture")
    moved = np.nanmax(np.linalg.norm(wrist - opening, axis=2), axis=1)
    exceeded = np.flatnonzero(moved > takeaway_m)
    end = int(exceeded[0]) if exceeded.size else pts.shape[0]
    if end < MIN_WINDOW_FRAMES:
        raise ValueError("no static address window of at least 5 frames")
    return np.arange(end)


def _stance_target_axis(
    mean_points: dict[str, np.ndarray], up: np.ndarray, handedness: str
) -> np.ndarray:
    """Trail -> lead mid-foot direction (square-stance assumption)."""
    lead_side = "L" if handedness == "right" else "R"
    trail_side = "R" if handedness == "right" else "L"

    def centre(side: str) -> np.ndarray:
        stack = np.array(
            [mean_points[f"{side}{m}"] for m in ("ToeIn", "ToeOut", "AnkleOut")]
        )
        if np.isnan(stack).all(axis=1).all():
            raise ValueError("cannot derive the stance line: foot markers missing")
        return np.nanmean(stack, axis=0)

    return _ground(centre(lead_side) - centre(trail_side), up, "stance line")


def _default_entry(side: str, role: str, frames: int, reason: str) -> FootProgression:
    nan = float("nan")
    return FootProgression(
        side, role, DEFAULT_TOE_OUT_DEG, nan, nan, frames, nan, False, True, reason
    )


def _measure_foot(
    side: str,
    role: str,
    frames_pts: dict[str, np.ndarray],
    ok: np.ndarray,
    window: np.ndarray,
    *,
    axes: tuple[np.ndarray, np.ndarray],
    handedness: str,
    lateral_m: float,
) -> FootProgression:
    target, up = axes
    prefix = "L" if side == "left" else "R"
    n_valid = int(ok.sum())
    if n_valid < MIN_VALID_FRACTION * len(window) or n_valid < MIN_WINDOW_FRAMES:
        return _default_entry(
            side,
            role,
            n_valid,
            f"foot markers missing in {len(window) - n_valid} frames",
        )
    out = target if role == "lead" else -target
    fwd = forward_axis(target, up, handedness)
    angles = np.empty((n_valid, 3))
    for k, f in enumerate(np.flatnonzero(ok)):
        heel = frames_pts[f"{prefix}AnkleOut"][f]
        t_in = frames_pts[f"{prefix}ToeIn"][f]
        t_out = frames_pts[f"{prefix}ToeOut"][f]
        kw = {
            "target_axis": target,
            "up": up,
            "foot_role": role,
            "handedness": handedness,
        }
        raw = marker_long_axis(heel, t_in, t_out, up=up)
        cor = marker_long_axis(
            heel, t_in, t_out, up=up, out_dir=out, ankle_lateral_offset_m=lateral_m
        )
        fore = _forefoot_axis(t_in, t_out, up, fwd)
        angles[k] = [
            progression_angle_deg(raw, **kw),
            progression_angle_deg(cor, **kw),
            progression_angle_deg(fore, **kw),
        ]
    med = np.median(angles, axis=0)
    spread = float(np.subtract(*np.percentile(angles[:, 1], [75, 25])))
    reliable = bool(spread <= MAX_SPREAD_DEG)
    reason = "measured over the address window"
    if not reliable:
        reason = f"unstable foot markers (IQR {spread:.1f} deg > {MAX_SPREAD_DEG} deg)"
    return FootProgression(
        side,
        role,
        float(med[1]),
        float(med[0]),
        float(med[2]),
        n_valid,
        spread,
        reliable,
        False,
        reason,
    )


def capture_foot_progression(
    points: np.ndarray,
    valid: np.ndarray,
    labels: Sequence[str],
    *,
    up: np.ndarray,
    target_axis: np.ndarray | None = None,
    handedness: str = "right",
    ankle_lateral_offset_m: float = ANKLE_LATERAL_OFFSET_M,
) -> dict[str, FootProgression]:
    """Per-foot address toe-out of a capture (``left``/``right`` keys).

    ``points`` are world coordinates in metres with ``up`` the vertical axis.
    ``target_axis`` defaults to the trail -> lead stance line (assumes a square
    stance; pass the real target direction when it is known). A foot whose
    markers are missing or unstable gets the flagged 20 degree default, never a
    silent measurement.
    """
    _check_handedness(handedness)
    up_u = _unit(np.asarray(up, dtype=float), "up")
    pts = np.asarray(points, float)
    ok_all = np.asarray(valid, bool)
    window = address_window(pts, ok_all, labels)
    names = [f"{s}{m}" for s in "LR" for m in ("AnkleOut", "ToeIn", "ToeOut")]
    missing = [n for n in names if n not in labels]
    if missing:
        raise ValueError(f"capture lacks foot markers: {missing}")
    cols = {n: labels.index(n) for n in names}
    series = {n: pts[:, c] for n, c in cols.items()}
    ok_foot = {
        side: ok_all[window][
            :, [cols[f"{p}{m}"] for m in ("AnkleOut", "ToeIn", "ToeOut")]
        ].all(axis=1)
        for side, p in (("left", "L"), ("right", "R"))
    }
    if target_axis is None:
        mean = {}
        for n in names:
            good = ok_all[window][:, cols[n]]
            mean[n] = (
                np.median(series[n][window][good], axis=0)
                if good.any()
                else np.full(3, np.nan)
            )
        target = _stance_target_axis(mean, up_u, handedness)
    else:
        target = _ground(target_axis, up_u, "target_axis")
    result: dict[str, FootProgression] = {}
    for side in ("left", "right"):
        role = foot_role(side, handedness)
        windowed = {n: series[n][window] for n in names}
        # frame index inside the window, so pad the validity mask to window length
        result[side] = _measure_foot(
            side,
            role,
            windowed,
            ok_foot[side],
            window,
            axes=(target, up_u),
            handedness=handedness,
            lateral_m=ankle_lateral_offset_m,
        )
    return result
