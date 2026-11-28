"""Biomechanical swing planes and WSCG quiver overlay (Slice 3, #15664)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-9
_TIE = 1e-12
_Z_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
Vec3 = tuple[float, float, float]
Arr = NDArray[np.float64]
VecIn = Sequence[float] | NDArray[Any]
PtsIn = Sequence[Sequence[float]] | Sequence[NDArray[Any]] | NDArray[Any]
_OV_SEQS = ("origins_m", "quiver_vectors", "work_rate_vectors") + (
    "active_mask",
    "segment_work_rates_w",
    "segment_kinetic_energy_j",
)
__all__ = [
    "SwingPlaneResult",
    "WscgQuiverOverlay",
    "build_wscg_quiver_overlay",
    "compute_hogan_glass_plane",
    "compute_mackenzie_net_force_plane",
    "compute_trackman_swing_plane",
    "fit_kwon_fsp_plane",
]


def _v3(a: Arr) -> Vec3:
    """Convert a 3-element float64 array into a 3-tuple."""
    return (float(a[0]), float(a[1]), float(a[2]))


def _as_vec3(values: VecIn, name: str) -> Vec3:
    """Validate and convert a 3D vector into a finite float 3-tuple."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,) or not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be a finite 3-vector")
    return _v3(arr)


def _as_pts(values: VecIn | PtsIn, name: str, n_min: int = 1) -> Arr:
    """Validate a (3,) or (N, 3) array and return a finite float64 (N, 3) array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1 and arr.shape == (3,):
        arr = arr.reshape(1, 3)
    elif arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (N, 3)")
    if arr.shape[0] < n_min or not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must have >= {n_min} finite 3D points")
    return arr


def _as_1d(vals: VecIn, n: int, name: str, lo: float, strict: bool) -> Arr:
    """Validate a 1D finite array of length n against a lower bound."""
    arr = np.asarray(vals, dtype=np.float64)
    bad = bool(np.any(arr <= lo)) if strict else bool(np.any(arr < lo))
    if arr.shape != (n,) or not bool(np.all(np.isfinite(arr))) or bad:
        raise ValueError(f"{name} invalid")
    return arr


def _norm(vec: NDArray[Any], name: str) -> Arr:
    """Return unit-length float64 copy of a 3D vector, raising ValueError if zero."""
    arr = np.asarray(vec, dtype=np.float64)
    mag = float(np.linalg.norm(arr))
    if not math.isfinite(mag) or mag <= _EPS:
        raise ValueError(f"{name} is degenerate")
    return arr / mag


def _make_plane(
    kind: str,
    orig: Arr,
    nrm: Arr,
    prim: Arr,
    rmse: float,
    opts: tuple[VecIn | None, VecIn, int],
) -> SwingPlaneResult:
    """Orient normal, build right-handed axes, and return SwingPlaneResult."""
    ref_in, target_line, n_samples = opts
    un = _norm(nrm, "normal")
    if ref_in is not None:
        ref = _norm(np.asarray(_as_vec3(ref_in, "reference_normal")), "ref")
        un, tag = (
            (-un if float(np.dot(un, ref)) < 0.0 else un),
            "aligned_with_reference",
        )
    else:
        nz, ny, nx = float(un[2]), float(un[1]), float(un[0])
        if nz < -_TIE or (
            abs(nz) <= _TIE and (ny < -_TIE or (abs(ny) <= _TIE and nx < 0.0))
        ):
            un = -un
        tag = "world_Zup_upper_hemisphere"
    tv = _norm(np.asarray(_as_vec3(target_line, "target_line")), "target_line")
    proj = prim - float(np.dot(prim, un)) * un
    if float(np.linalg.norm(proj)) <= _EPS:
        fb = np.array([1.0, 0.0, 0.0] if abs(float(un[0])) < 0.9 else [0.0, 1.0, 0.0])
        proj = fb - float(np.dot(fb, un)) * un
    ix = _norm(proj, "in_plane_x")
    iy = _norm(np.cross(un, ix), "in_plane_y")
    h, dir_deg = np.cross(_Z_UP, un), 0.0
    if float(np.linalg.norm(h)) > _EPS:
        hu = _norm(h, "h")
        t2 = (
            _norm(np.array([float(tv[0]), float(tv[1]), 0.0]), "t")
            if math.hypot(tv[0], tv[1]) > _EPS
            else np.array([1.0, 0.0, 0.0])
        )
        hu = -hu if float(np.dot(hu, t2)) < 0.0 else hu
        dir_deg = float(
            math.degrees(math.atan2(float(np.cross(t2, hu)[2]), float(np.dot(t2, hu))))
        )
    slope = float(math.degrees(math.acos(min(1.0, abs(float(un[2]))))))
    axes: tuple[Vec3, Vec3, Vec3, Vec3] = (_v3(orig), _v3(un), _v3(ix), _v3(iy))
    return SwingPlaneResult(
        kind, *axes, float(rmse), slope, dir_deg, tag, {"n_samples": float(n_samples)}
    )


@dataclass(frozen=True)
class SwingPlaneResult:
    """Immutable geometric description of a fitted or instantaneous swing plane."""

    plane_kind: str
    origin_m: Vec3
    normal: Vec3
    in_plane_x: Vec3
    in_plane_y: Vec3
    plane_rmse_m: float
    slope_deg: float
    direction_deg: float
    sign_convention: str
    metadata: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate unit axes, finite coordinates, non-negative RMSE, and metadata."""
        if not self.plane_kind.strip() or not self.sign_convention.strip():
            raise ValueError("plane_kind and sign_convention must be non-empty")
        object.__setattr__(self, "origin_m", _as_vec3(self.origin_m, "origin_m"))
        for attr in ("normal", "in_plane_x", "in_plane_y"):
            vec = _as_vec3(getattr(self, attr), attr)
            if abs(math.hypot(*vec) - 1.0) > 1e-6:
                raise ValueError(f"{attr} must be unit length")
            object.__setattr__(self, attr, vec)
        if not math.isfinite(self.plane_rmse_m) or self.plane_rmse_m < 0.0:
            raise ValueError("plane_rmse_m must be >= 0")
        if not math.isfinite(self.slope_deg) or not (0.0 <= self.slope_deg <= 90.0001):
            raise ValueError("slope_deg must be in [0, 90]")
        if not math.isfinite(self.direction_deg) or not isinstance(
            self.metadata, Mapping
        ):
            raise ValueError("invalid direction_deg or metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class WscgQuiverOverlay:
    """Immutable per-segment KE, work-rate, and WSCG quiver overlay snapshot."""

    segment_names: tuple[str, ...]
    origins_m: tuple[Vec3, ...]
    quiver_vectors: tuple[Vec3, ...]
    work_rate_vectors: tuple[Vec3, ...]
    active_mask: tuple[bool, ...]
    segment_work_rates_w: tuple[float, ...]
    segment_kinetic_energy_j: tuple[float, ...]
    total_kinetic_energy_j: float
    wscg_position_m: Vec3
    wscg_velocity_m_s: Vec3
    wscg_quiver_vector: Vec3
    wscg_work_rate_w: float
    metadata: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate WSCG overlay shapes, finite scalars, and freeze metadata."""
        n = len(self.segment_names)
        if n == 0 or any(len(getattr(self, a)) != n for a in _OV_SEQS):
            raise ValueError("overlay segment sequences invalid")
        tke, wwr = self.total_kinetic_energy_j, self.wscg_work_rate_w
        if (
            not math.isfinite(tke)
            or tke < 0.0
            or not math.isfinite(wwr)
            or not isinstance(self.metadata, Mapping)
        ):
            raise ValueError("invalid scalar overlay field or metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def fit_kwon_fsp_plane(
    clubhead_positions_m: PtsIn,
    *,
    reference_normal: VecIn | None = None,
    target_line: VecIn = (1.0, 0.0, 0.0),
) -> SwingPlaneResult:
    """Fit Kwon's Functional Swing Plane (FSP) via SVD to a 3D clubhead arc."""
    pts = _as_pts(clubhead_positions_m, "clubhead_positions_m", n_min=3)
    c = pts.mean(axis=0)
    ctr = pts - c
    _, s, vh = np.linalg.svd(ctr, full_matrices=False)
    if float(s[0]) <= _EPS or float(s[1]) <= _EPS * max(1.0, float(s[0])):
        raise ValueError("clubhead_positions_m are collinear or degenerate")
    nrm = vh[2, :]
    dev = ctr @ nrm
    rmse = float(np.sqrt(np.mean(dev * dev)))
    opts = (reference_normal, target_line, pts.shape[0])
    return _make_plane("kwon_fsp", c, nrm, vh[0, :], rmse, opts)


def _shaft_plane(
    kind: str,
    tri: tuple[VecIn | PtsIn, VecIn | PtsIn, VecIn | PtsIn],
    on_head: bool,
    opts: tuple[VecIn | None, VecIn],
) -> SwingPlaneResult:
    """Fit a plane spanned by club shaft axes and paired physical vectors."""
    g, h, p = _as_pts(tri[0], "grip"), _as_pts(tri[1], "head"), _as_pts(tri[2], "pair")
    if not (g.shape == h.shape == p.shape):
        raise ValueError("grip, clubhead, and paired shapes must match")
    shafts = h - g
    sh = np.vstack([_norm(shafts[i], "shaft") for i in range(shafts.shape[0])])
    ph = np.vstack([_norm(p[i], "paired") for i in range(p.shape[0])])
    orig = h.mean(axis=0) if on_head else g.mean(axis=0)
    if shafts.shape[0] == 1:
        nrm, rmse = _norm(np.cross(sh[0], ph[0]), kind), 0.0
    else:
        _, s, vh = np.linalg.svd(np.vstack([sh, ph]), full_matrices=False)
        if float(s[1]) <= _EPS * max(1.0, float(s[0])):
            raise ValueError(f"{kind} vectors are collinear")
        nrm = _norm(vh[2, :], kind)
        dev = (np.vstack([g, h]) - orig) @ nrm
        rmse = float(np.sqrt(np.mean(dev * dev)))
    prim = p.mean(axis=0) if on_head else shafts.mean(axis=0)
    return _make_plane(kind, orig, nrm, prim, rmse, (opts[0], opts[1], g.shape[0]))


def compute_mackenzie_net_force_plane(
    grip_positions_m: VecIn | PtsIn,
    clubhead_positions_m: VecIn | PtsIn,
    net_forces_n: VecIn | PtsIn,
    *,
    reference_normal: VecIn | None = None,
    target_line: VecIn = (1.0, 0.0, 0.0),
) -> SwingPlaneResult:
    """Compute MacKenzie's net force plane spanned by club shaft and net force."""
    tri, o = (
        (grip_positions_m, clubhead_positions_m, net_forces_n),
        (reference_normal, target_line),
    )
    return _shaft_plane("mackenzie_net_force", tri, False, o)


def compute_trackman_swing_plane(
    clubhead_velocity_m_s: VecIn | PtsIn,
    grip_position_m: VecIn | PtsIn,
    clubhead_position_m: VecIn | PtsIn,
    *,
    reference_normal: VecIn | None = None,
    target_line: VecIn = (1.0, 0.0, 0.0),
) -> SwingPlaneResult:
    """Compute TrackMan's swing plane spanned by clubhead velocity and shaft axis."""
    tri, o = (
        (grip_position_m, clubhead_position_m, clubhead_velocity_m_s),
        (reference_normal, target_line),
    )
    return _shaft_plane("trackman_swing_plane", tri, True, o)


def compute_hogan_glass_plane(
    shoulder_position_m: VecIn | PtsIn,
    ball_position_m: VecIn,
    *,
    target_line: VecIn = (1.0, 0.0, 0.0),
    reference_normal: VecIn | None = None,
) -> SwingPlaneResult:
    """Compute Ben Hogan's glass plane anchored on shoulder line and ball target line."""
    sh = _as_pts(shoulder_position_m, "shoulder_position_m")
    ball = np.asarray(_as_vec3(ball_position_m, "ball_position_m"), dtype=np.float64)
    th = _norm(np.asarray(_as_vec3(target_line, "target_line")), "target_line")
    rel = sh - ball
    rp = rel - np.outer(rel @ th, th)
    if sh.shape[0] == 1:
        nrm, rmse = _norm(np.cross(th, _norm(rp[0], "shoulder offset")), "hogan"), 0.0
    else:
        if float(np.max(np.linalg.norm(rp, axis=1))) <= _EPS:
            raise ValueError("shoulder offsets collinear with target_line")
        _, s, vh = np.linalg.svd(rp, full_matrices=False)
        if float(s[0]) <= _EPS:
            raise ValueError("shoulder offsets degenerate")
        nrm = _norm(np.cross(th, _norm(vh[0, :], "hogan_shoulder")), "hogan")
        dev = rel @ nrm
        rmse = float(np.sqrt(np.mean(dev * dev)))
    return _make_plane(
        "hogan_glass_plane", ball, nrm, th, rmse, (reference_normal, th, sh.shape[0])
    )


def _floor_vec(vec: Arr, floor: float, scale: float) -> tuple[Vec3, bool]:
    """Scale vec and clamp to zero when raw magnitude is below floor."""
    if float(np.linalg.norm(vec)) < floor:
        return (0.0, 0.0, 0.0), False
    return _v3(vec * scale), True


def build_wscg_quiver_overlay(
    segment_names: Sequence[str],
    positions_m: PtsIn,
    velocities_m_s: PtsIn,
    forces_n: PtsIn,
    masses_kg: VecIn,
    rotational_ke_j: VecIn | None = None,
    min_vector_magnitude: float = 1e-6,
    scale_factor: float = 1.0,
) -> WscgQuiverOverlay:
    """Compute WSCG state, segmental kinetic energy, work rates, and quiver vectors."""
    fl, sc = float(min_vector_magnitude), float(scale_factor)
    if not math.isfinite(fl) or fl < 0.0 or not math.isfinite(sc) or sc <= 0.0:
        raise ValueError("min_vector_magnitude must be >= 0 and scale_factor > 0")
    names = tuple(segment_names)
    if not names or any(not s.strip() for s in names) or len(set(names)) != len(names):
        raise ValueError("segment_names must be non-empty unique strings")
    n = len(names)
    pos = _as_pts(positions_m, "pos")
    vel = _as_pts(velocities_m_s, "vel")
    frc = _as_pts(forces_n, "frc")
    if any(a.shape != (n, 3) for a in (pos, vel, frc)):
        raise ValueError("positions, velocities, forces shape mismatch")
    m = _as_1d(masses_kg, n, "masses_kg", 0.0, strict=True)
    rke: Arr = np.zeros(n)
    if rotational_ke_j is not None:
        rke = _as_1d(rotational_ke_j, n, "rot", 0.0, False)
    tm = float(np.sum(m))
    wp, wv, wf = (m @ pos) / tm, (m @ vel) / tm, np.sum(frc, axis=0)
    wq, _ = _floor_vec(wf, fl, sc)
    ke, wr = 0.5 * m * np.sum(vel * vel, axis=1) + rke, np.sum(frc * vel, axis=1)
    qv, wrv, act = [], [], []
    for i in range(n):
        q, ok = _floor_vec(frc[i], fl, sc)
        qv.append(q)
        act.append(ok)
        uv, vok = _floor_vec(vel[i], fl, 1.0 / max(float(np.linalg.norm(vel[i])), _EPS))
        w = _v3(np.asarray(uv) * (float(wr[i]) * sc))
        wrv.append(w if vok and abs(float(wr[i])) >= fl else (0.0, 0.0, 0.0))
    meta = {"total_mass_kg": tm, "min_vector_magnitude": fl, "scale_factor": sc}
    o_t, w_t, k_t = (
        tuple(_v3(p) for p in pos),
        tuple(float(w) for w in wr),
        tuple(float(k) for k in ke),
    )
    seqs = (names, o_t, tuple(qv), tuple(wrv), tuple(act), w_t, k_t)
    tke, wwr = float(np.sum(ke)), float(np.dot(wf, wv))
    return WscgQuiverOverlay(*seqs, tke, _v3(wp), _v3(wv), wq, wwr, meta)
