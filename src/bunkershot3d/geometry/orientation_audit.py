"""Clubhead orientation and displacement-derived forward travel audit (#15805 WP-1)."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .delivery import DeliveryCondition, delivered_rotation
from .wedge import WedgeGeometry

__all__ = [
    "DEFAULT_ENTRY_ATTACK_ANGLE_DEG",
    "ClubheadOrientationSample",
    "OrientationAuditResult",
    "OrthographicVerificationFrame",
    "audit_clubhead_orientation",
    "build_orthographic_verification_frame",
    "derive_world_travel_vectors",
]

DEFAULT_ENTRY_ATTACK_ANGLE_DEG: float = -6.0
_NORM_TOL: float = 1e-12
_BG, _GRID, _SOLE = (20, 24, 32, 255), (68, 76, 92, 255), (180, 196, 214, 255)
_LEAD, _TRAIL = (46, 229, 157, 255), (245, 158, 11, 255)
_FACE, _TRAVEL = (96, 165, 250, 255), (244, 114, 182, 255)


def _freeze(
    val: ArrayLike, n: int, name: str, *, unit: bool = False
) -> NDArray[np.float64]:
    """Validate finite 1-D vector (optionally normalizing) and freeze it."""
    arr = np.array(val, dtype=np.float64, copy=True).reshape(-1)
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite with shape ({n},)")
    if unit:
        norm = math.sqrt(float(np.vdot(arr, arr)))
        if norm <= _NORM_TOL:
            raise ValueError(f"{name} must have positive norm")
        arr = np.array(arr / norm, dtype=np.float64, copy=False).reshape(-1)
    arr.flags.writeable = False
    return arr


def _finite(name: str, val: float) -> float:
    """Validate a finite scalar and return float."""
    out = float(val)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return out


def _freeze_dc(obj: object) -> None:
    """Validate and freeze registered vector fields on a dataclass instance."""
    for f in fields(obj):  # type: ignore[arg-type]
        if f.name.endswith(("_m", "_world", "_unit")):
            object.__setattr__(obj, f.name, _freeze(getattr(obj, f.name), 3, f.name))
        elif f.name.endswith("_2d") and f.name != "bounds_2d":
            object.__setattr__(obj, f.name, _freeze(getattr(obj, f.name), 2, f.name))


def _attack_deg(vec: NDArray[np.float64]) -> float:
    """Return elevation/attack angle in degrees (negative for descending)."""
    return math.degrees(
        math.atan2(float(vec[2]), math.hypot(float(vec[0]), float(vec[1])))
    )


def _vec3_seq(arr_like: ArrayLike, lo: int, hi: int, name: str) -> NDArray[np.float64]:
    """Validate a finite (N, 3) array with row count in [lo, hi]."""
    arr = np.asarray(arr_like, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or not lo <= arr.shape[0] <= hi:
        raise ValueError(f"{name} bad shape {arr.shape!r}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


@dataclass(frozen=True, slots=True, eq=False)
class ClubheadOrientationSample:
    """Per-step clubhead orientation audit record in world coordinates."""

    step_index: int
    position_world_m: NDArray[np.float64]
    leading_edge_world_m: NDArray[np.float64]
    trailing_edge_world_m: NDArray[np.float64]
    face_normal_world: NDArray[np.float64]
    travel_unit_world: NDArray[np.float64]
    horizontal_travel_unit: NDArray[np.float64]
    velocity_attack_angle_deg: float | None = None

    def __post_init__(self) -> None:
        """Validate invariants and freeze vector attributes."""
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        _freeze_dc(self)
        if self.velocity_attack_angle_deg is not None:
            va = _finite("velocity_attack_angle_deg", self.velocity_attack_angle_deg)
            object.__setattr__(self, "velocity_attack_angle_deg", va)

    @property
    def leading_ahead_margin_m(self) -> float:
        """Signed projection of (leading - trailing) along horizontal travel."""
        delta = self.leading_edge_world_m - self.trailing_edge_world_m
        return float(np.dot(delta, self.horizontal_travel_unit))

    @property
    def face_forward_alignment(self) -> float:
        """Dot product of striking-face normal and horizontal travel unit."""
        return float(np.dot(self.face_normal_world, self.horizontal_travel_unit))

    @property
    def face_upward_component(self) -> float:
        """Vertical (+z) component of the world striking-face normal."""
        return float(self.face_normal_world[2])

    @property
    def attack_angle_deg(self) -> float:
        """Displacement-derived attack angle in degrees (negative descending)."""
        return _attack_deg(self.travel_unit_world)


@dataclass(frozen=True, slots=True, eq=False)
class OrientationAuditResult:
    """Aggregate result of auditing a clubhead trajectory."""

    samples: tuple[ClubheadOrientationSample, ...]
    mean_travel_unit_world: NDArray[np.float64]
    mean_horizontal_travel_unit: NDArray[np.float64]
    expected_attack_angle_deg: float
    attack_angle_tolerance_deg: float
    passed: bool
    is_backward_travel: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate audit result invariants and freeze mean vectors."""
        if not self.samples:
            raise ValueError("OrientationAuditResult requires at least one sample")
        _freeze_dc(self)
        tol = _finite("attack_angle_tolerance_deg", self.attack_angle_tolerance_deg)
        if tol <= 0.0:
            raise ValueError("attack_angle_tolerance_deg must be positive")
        att = _finite("expected_attack_angle_deg", self.expected_attack_angle_deg)
        object.__setattr__(self, "attack_angle_tolerance_deg", tol)
        object.__setattr__(self, "expected_attack_angle_deg", att)

    def require_passed(self) -> None:
        """Raise ValueError if any orientation or travel check failed."""
        if not self.passed:
            raise ValueError("; ".join(self.failures) or "orientation audit failed")


@dataclass(frozen=True, slots=True, eq=False)
class OrthographicVerificationFrame:
    """Headless 2D orthographic verification payload and optional RGBA buffer."""

    plane: str
    u_axis_world: NDArray[np.float64]
    w_axis_world: NDArray[np.float64]
    leading_edge_2d: NDArray[np.float64]
    trailing_edge_2d: NDArray[np.float64]
    face_normal_2d: NDArray[np.float64]
    travel_vector_2d: NDArray[np.float64]
    bounds_2d: tuple[float, float, float, float]
    image_buffer: NDArray[np.uint8] | None = None

    def __post_init__(self) -> None:
        """Validate projected 2D geometry and freeze arrays."""
        if not self.plane.strip() or len(self.bounds_2d) != 4:
            raise ValueError("plane must be non-empty and bounds_2d length 4")
        _freeze_dc(self)
        if self.image_buffer is not None:
            buf = np.array(self.image_buffer, dtype=np.uint8, copy=True)
            if buf.ndim != 3 or buf.shape[2] not in (3, 4):
                raise ValueError("image_buffer must have shape (H, W, 3|4)")
            buf.flags.writeable = False
            object.__setattr__(self, "image_buffer", buf)


def derive_world_travel_vectors(
    positions_m: ArrayLike,
    *,
    horizontal_only: bool = False,
    min_displacement_m: float = _NORM_TOL,
) -> NDArray[np.float64]:
    """Derive unit travel vectors strictly from successive world-space positions."""
    if _finite("min_displacement_m", min_displacement_m) <= 0.0:
        raise ValueError("min_displacement_m must be positive")
    positions = _vec3_seq(positions_m, 2, 10_000_000, "positions_m")
    deltas = np.array(positions[1:] - positions[:-1], dtype=np.float64, copy=True)
    if horizontal_only:
        deltas[:, 2] = 0.0
    norms = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
    bad = np.flatnonzero(norms <= min_displacement_m)
    if bad.size > 0:
        raise ValueError(f"step {int(bad[0])} has degenerate displacement norm")
    units = deltas / norms[:, None]
    units.flags.writeable = False
    return units


def audit_clubhead_orientation(
    geometry: WedgeGeometry,
    positions_m: ArrayLike,
    *,
    delivery: DeliveryCondition | None = None,
    velocities_m_s: ArrayLike | None = None,
    orientations: ArrayLike | None = None,
    expected_attack_angle_deg: float = DEFAULT_ENTRY_ATTACK_ANGLE_DEG,
    attack_angle_tolerance_deg: float = 0.5,
) -> OrientationAuditResult:
    """Audit clubhead orientation against displacement-derived world travel."""
    if not isinstance(geometry, WedgeGeometry):
        raise TypeError("geometry must be a WedgeGeometry")
    cond = DeliveryCondition() if delivery is None else delivery
    if not isinstance(cond, DeliveryCondition):
        raise TypeError("delivery must be a DeliveryCondition")
    exp_att = _finite("expected_attack_angle_deg", expected_attack_angle_deg)
    tol = _finite("attack_angle_tolerance_deg", attack_angle_tolerance_deg)
    if tol <= 0.0:
        raise ValueError("attack_angle_tolerance_deg must be positive")
    pos = _vec3_seq(positions_m, 2, 10_000_000, "positions_m")
    n_p = pos.shape[0]
    vels = (
        None
        if velocities_m_s is None
        else _vec3_seq(velocities_m_s, n_p - 1, n_p, "velocities_m_s")
    )
    rots = None if orientations is None else np.asarray(orientations, dtype=np.float64)
    if rots is not None and (rots.ndim != 3 or rots.shape[1:] != (3, 3)):
        raise ValueError("orientations must have shape (N, 3, 3)")
    def_rot = delivered_rotation(
        lie_deg=geometry.lie_deg,
        face_open_deg=cond.face_open_deg,
        shaft_lean_deg=cond.shaft_lean_deg,
    )
    t3d = derive_world_travel_vectors(pos)
    thz = derive_world_travel_vectors(pos, horizontal_only=True)
    x_fp, loft = geometry.face_progression_m, geometry.loft_rad
    lead_b = np.array([x_fp, 0.0, 0.0])
    trail_b = np.array(
        [x_fp + geometry.sole_width_m, 0.0, -geometry.trailing_contact_drop_m]
    )
    face_b = np.array([-math.cos(loft), 0.0, math.sin(loft)])
    samples: list[ClubheadOrientationSample] = []
    failures: list[str] = []
    backward = False
    for i in range(t3d.shape[0]):
        rot = def_rot if rots is None else rots[i]
        vel = None if vels is None else vels[i]
        n_f = _freeze(rot @ face_b, 3, "face_normal_world", unit=True)
        va = _attack_deg(_freeze(vel, 3, "vel", unit=True)) if vel is not None else None
        pl, pt = pos[i] + rot @ lead_b, pos[i] + rot @ trail_b
        s = ClubheadOrientationSample(i, pos[i], pl, pt, n_f, t3d[i], thz[i], va)
        samples.append(s)
        if s.leading_ahead_margin_m <= 0.0:
            backward = True
            failures.append(f"step {i}: backward/inverted (leading behind)")
        if s.face_forward_alignment <= 0.0:
            backward = True
            failures.append(f"step {i}: backward/inverted (face opposes)")
        if s.face_upward_component <= 0.0:
            failures.append(f"step {i}: face normal not upward")
        if float(np.dot(thz[i], thz[0])) <= 0.0:
            backward = True
            failures.append(f"step {i}: travel reversed")
        if abs(s.attack_angle_deg - exp_att) > tol:
            failures.append(f"step {i}: attack mismatch")
        if va is not None and abs(va - exp_att) > tol:
            failures.append(f"step {i}: velocity attack mismatch")
        if vel is not None and float(np.dot(vel, t3d[i])) <= 0.0:
            backward = True
            failures.append(f"step {i}: velocity opposes")
    m3, mh = np.mean(t3d, axis=0), np.mean(thz, axis=0)
    u3 = _freeze(m3, 3, "m3", unit=True) if np.vdot(m3, m3) > _NORM_TOL else t3d[0]
    uh = _freeze(mh, 3, "mh", unit=True) if np.vdot(mh, mh) > _NORM_TOL else thz[0]
    return OrientationAuditResult(
        tuple(samples), u3, uh, exp_att, tol, not failures, backward, tuple(failures)
    )


def _render_buffer(
    pts: tuple[NDArray[np.float64], ...],
    bounds: tuple[float, float, float, float],
    arrow: float,
    w: int,
    h: int,
) -> NDArray[np.uint8]:
    """Render 2D orthographic vectors into a deterministic RGBA uint8 buffer."""
    lead, trail, face, travel = pts
    u0, u1, w0, w1 = bounds
    sx, sy = (w - 1) / (u1 - u0), (h - 1) / (w1 - w0)
    canvas = np.empty((h, w, 4), dtype=np.uint8)
    canvas[:, :] = _BG
    mid = 0.5 * (lead + trail)
    segs = (
        (np.array([u0, trail[1]]), np.array([u1, trail[1]]), _GRID),
        (trail, lead, _SOLE),
        (mid, mid + arrow * face, _FACE),
        (lead, lead + arrow * travel, _TRAVEL),
    )
    for a, b, rgba in segs:
        c0, r0 = round((float(a[0]) - u0) * sx), round((w1 - float(a[1])) * sy)
        c1, r1 = round((float(b[0]) - u0) * sx), round((w1 - float(b[1])) * sy)
        n = max(abs(c1 - c0), abs(r1 - r0), 1)
        xs = np.rint(np.linspace(c0, c1, n + 1)).astype(np.int64)
        ys = np.rint(np.linspace(r0, r1, n + 1)).astype(np.int64)
        m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        canvas[ys[m], xs[m]] = rgba
    yy, xx = np.ogrid[:h, :w]
    for pt, rgba in ((trail, _TRAIL), (lead, _LEAD)):
        cx, cy = round((float(pt[0]) - u0) * sx), round((w1 - float(pt[1])) * sy)
        canvas[(xx - cx) ** 2 + (yy - cy) ** 2 <= 9] = rgba
    canvas.flags.writeable = False
    return canvas


def build_orthographic_verification_frame(
    sample: ClubheadOrientationSample,
    *,
    render: bool = True,
    width_px: int = 160,
    height_px: int = 120,
    margin_ratio: float = 0.25,
) -> OrthographicVerificationFrame:
    """Build a headless 2D orthographic verification frame from an audit sample."""
    if not isinstance(sample, ClubheadOrientationSample):
        raise TypeError("sample must be a ClubheadOrientationSample")
    if width_px < 16 or height_px < 16 or _finite("margin_ratio", margin_ratio) < 0.01:
        raise ValueError("invalid frame dimensions or margin_ratio")
    u_ax, w_ax = sample.horizontal_travel_unit, np.array([0.0, 0.0, 1.0])
    proj = np.vstack([u_ax, w_ax])
    lead = proj @ sample.leading_edge_world_m
    trail = proj @ sample.trailing_edge_world_m
    face, travel = proj @ sample.face_normal_world, proj @ sample.travel_unit_world
    arrow = max(float(np.linalg.norm(lead - trail)), 0.02)
    pts = np.vstack(
        [lead, trail, 0.5 * (lead + trail) + arrow * face, lead + arrow * travel]
    )
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-3)
    half = 0.5 * span * (1.0 + 2.0 * margin_ratio)
    um = 0.5 * float(pts[:, 0].min() + pts[:, 0].max())
    wm = 0.5 * float(pts[:, 1].min() + pts[:, 1].max())
    bounds = (um - half, um + half, wm - half, wm + half)
    buf = (
        _render_buffer((lead, trail, face, travel), bounds, arrow, width_px, height_px)
        if render
        else None
    )
    return OrthographicVerificationFrame(
        "travel_vertical", u_ax, w_ax, lead, trail, face, travel, bounds, buf
    )
