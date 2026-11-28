"""Viewer Slice 2 contracts: visibility, traces, club strobes, torque curls."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Final, Literal, Self

import numpy as np

from .mujoco_segment_group_catalog import infer_segment_group_catalog
from .series_validation import validate_strictly_increasing_times

CurlDirection = Literal["ccw", "cw"]
SelectionMode = Literal["phase_indices", "stride"]
Vec3 = tuple[float, float, float]
RGBA = tuple[float, float, float, float]

DEFAULT_BODY_SEGMENTS: Final[tuple[str, ...]] = (
    "head", "torso", "pelvis", "thigh_l", "thigh_r", "shin_l", "shin_r", "foot_l", "foot_r",
    "upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r", "hand_l", "hand_r", "club_shaft", "club_head",
)  # fmt: skip
DEFAULT_TRACE_RGBA: Final[RGBA] = (0.12, 0.47, 0.71, 1.0)
CCW_TORQUE_RGBA: Final[RGBA] = (0.0, 0.45, 0.70, 1.0)
CW_TORQUE_RGBA: Final[RGBA] = (0.84, 0.37, 0.0, 1.0)
_MAX_SWEEP: Final[float] = 1.75 * math.pi
_TWO_PI: Final[float] = 2.0 * math.pi

__all__ = [
    "CCW_TORQUE_RGBA", "CW_TORQUE_RGBA", "DEFAULT_BODY_SEGMENTS", "DEFAULT_TRACE_RGBA",
    "BodyPartVisibilityState", "ClubStrobeSample", "CurlDirection", "CurledTorqueArrowSpec",
    "FrozenClubStrobeSeries", "SelectionMode", "TraceLineSeries", "build_curled_torque_arrows",
    "build_frozen_club_strobes", "preset_club_and_arms_only", "preset_full", "preset_ghost_body",
]  # fmt: skip


def _str(v: object, name: str = "value") -> str:
    """Validate non-empty string."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return v


def _int(v: object, name: str = "value", min_v: int = 0) -> int:
    """Validate integer (excluding bool) >= ``min_v``."""
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise TypeError(f"{name} must be an integer")
    if int(v) < min_v:
        raise ValueError(f"{name} must be >= {min_v}")
    return int(v)


def _real(v: object, name: str = "value", lo: float | None = None, hi: float | None = None, pos: bool = False) -> float:  # fmt: skip
    """Validate finite real number within bounds."""
    if isinstance(v, bool) or not isinstance(v, Real):
        raise TypeError(f"{name} must be a real number")
    val = float(v)
    bad = not math.isfinite(val) or (pos and val <= 0.0)
    if bad or (lo is not None and val < lo) or (hi is not None and val > hi):
        raise ValueError(f"{name} out of valid finite range")
    return val


def _vec(v: object, name: str = "vec", n: int = 3, unit: bool = False) -> Any:
    """Validate fixed-length finite float tuple."""
    if isinstance(v, (str, bytes)) or not isinstance(v, Sequence) or len(v) != n:
        raise ValueError(f"{name} must have length {n}")
    lo, hi = (0.0, 1.0) if unit else (None, None)
    return tuple(_real(v[i], name, lo, hi) for i in range(n))


def _norm_axis(axis: object) -> Vec3:
    """Validate and normalize non-zero 3D axis vector."""
    arr = np.asarray(_vec(axis, "axis_unit"), dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        raise ValueError("joint axis must have non-zero norm")
    u = arr / n
    return (float(u[0]), float(u[1]), float(u[2]))


def _to_json(val: Any) -> Any:
    """Convert frozen dataclasses, tuples, and mappings into deterministic JSON."""
    if is_dataclass(val) and not isinstance(val, type):
        return {f.name: _to_json(getattr(val, f.name)) for f in fields(val)}
    if isinstance(val, Mapping):
        return {k: _to_json(val[k]) for k in sorted(val)}
    return [_to_json(i) for i in val] if isinstance(val, (tuple, list)) else val


class _JsonContract:
    """Shared deterministic JSON serialization mixin for Slice 2 contracts."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize instance to a deterministic JSON dictionary."""
        return dict(_to_json(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Reconstruct a validated instance from JSON."""
        if not isinstance(data, Mapping):
            raise TypeError(f"{cls.__name__} data must be a mapping")
        if set(data) != {f.name for f in fields(cls)}:  # type: ignore[arg-type]
            raise ValueError(f"{cls.__name__} keys mismatch")
        return cls(**data)


@dataclass(frozen=True)
class BodyPartVisibilityState(_JsonContract):
    """Per-segment visibility toggles and [0.0, 1.0] opacities for 3D viewers."""

    segment_visible: Mapping[str, bool] = field(default_factory=dict)
    segment_opacity: Mapping[str, float] = field(default_factory=dict)
    default_visible: bool = True
    default_opacity: float = 1.0

    def __post_init__(self) -> None:
        """Validate visibility flags, opacity bounds, and freeze mappings."""
        sv, so = self.segment_visible, self.segment_opacity
        if not isinstance(self.default_visible, bool) or not isinstance(sv, Mapping) or not isinstance(so, Mapping):  # fmt: skip
            raise TypeError("invalid visibility state types")
        vis = {_str(k, "seg"): sv[k] for k in sorted(sv) if isinstance(sv[k], bool)}
        if len(vis) != len(sv):
            raise TypeError("segment_visible values must be bool")
        op = {_str(k, "seg"): _real(so[k], "op", 0.0, 1.0) for k in sorted(so)}
        d_op = _real(self.default_opacity, "default_opacity", 0.0, 1.0)
        for k, v in (("default_opacity", d_op), ("segment_visible", MappingProxyType(vis)), ("segment_opacity", MappingProxyType(op))):  # fmt: skip
            object.__setattr__(self, k, v)

    def is_visible(self, segment: str) -> bool:
        """Return whether ``segment`` is visible."""
        return self.segment_visible.get(_str(segment), self.default_visible)

    def opacity_for(self, segment: str) -> float:
        """Return configured opacity for ``segment`` in [0.0, 1.0]."""
        return self.segment_opacity.get(_str(segment), self.default_opacity)

    def effective_opacity(self, segment: str) -> float:
        """Return 0.0 when ``segment`` is hidden, else its configured opacity."""
        return self.opacity_for(segment) if self.is_visible(segment) else 0.0

    def with_segment(self, segment: str, *, visible: bool | None = None, opacity: float | None = None) -> BodyPartVisibilityState:  # fmt: skip
        """Return a copy with updated visibility and/or opacity for ``segment``."""
        key, vis, op = (
            _str(segment),
            dict(self.segment_visible),
            dict(self.segment_opacity),
        )
        if visible is not None:
            vis[key] = visible
        if opacity is not None:
            op[key] = opacity
        return BodyPartVisibilityState(vis, op, self.default_visible, self.default_opacity)  # fmt: skip


def preset_full(segments: Sequence[str] = DEFAULT_BODY_SEGMENTS) -> BodyPartVisibilityState:  # fmt: skip
    """Build preset with all body and club segments fully visible and opaque."""
    names = tuple(_str(s) for s in segments)
    return BodyPartVisibilityState(dict.fromkeys(names, True), dict.fromkeys(names, 1.0))  # fmt: skip


def preset_ghost_body(segments: Sequence[str] = DEFAULT_BODY_SEGMENTS, *, body_opacity: float = 0.25, club_opacity: float = 1.0) -> BodyPartVisibilityState:  # fmt: skip
    """Build preset with translucent body segments and opaque club segments."""
    b, c = _real(body_opacity, "b", 0.0, 1.0), _real(club_opacity, "c", 0.0, 1.0)
    names = tuple(_str(s) for s in segments)
    club = infer_segment_group_catalog(names).club
    return BodyPartVisibilityState(dict.fromkeys(names, True), {s: (c if s in club else b) for s in names}, True, b)  # fmt: skip


def preset_club_and_arms_only(segments: Sequence[str] = DEFAULT_BODY_SEGMENTS) -> BodyPartVisibilityState:  # fmt: skip
    """Build preset showing only club and arm segments."""
    names = tuple(_str(s) for s in segments)
    cat = infer_segment_group_catalog(names)
    kept = cat.club | cat.arms
    return BodyPartVisibilityState({s: (s in kept) for s in names}, {s: (1.0 if s in kept else 0.0) for s in names}, False, 0.0)  # fmt: skip


@dataclass(frozen=True)
class TraceLineSeries(_JsonContract):
    """Time-aligned 3D polyline trace with validated RGBA color and width."""

    trace_id: str
    points_m: tuple[Vec3, ...]
    time_s: tuple[float, ...]
    color_rgba: RGBA = DEFAULT_TRACE_RGBA
    width_px: float = 2.0

    def __post_init__(self) -> None:
        """Validate trace identifier, 3D points, timestamps, color, and width."""
        _str(self.trace_id, "trace_id")
        times = validate_strictly_increasing_times(self.time_s)
        pts = tuple(_vec(pt, f"points_m[{i}]") for i, pt in enumerate(self.points_m))
        if len(pts) != len(times):
            raise ValueError("points_m and time_s must have identical length")
        for k, v in (("points_m", pts), ("time_s", times), ("color_rgba", _vec(self.color_rgba, "rgba", 4, True)), ("width_px", _real(self.width_px, "width", pos=True))):  # fmt: skip
            object.__setattr__(self, k, v)


@dataclass(frozen=True)
class ClubStrobeSample(_JsonContract):
    """Single frozen club shaft+head strobe pose at a specific frame."""

    frame_idx: int
    time_s: float
    grip_pos_m: Vec3
    hosel_pos_m: Vec3
    head_pos_m: Vec3
    opacity: float
    phase_label: str = ""

    def __post_init__(self) -> None:
        """Validate frame index, timestamp, club endpoints, and opacity."""
        if not isinstance(self.phase_label, str):
            raise TypeError("phase_label must be a string")
        for k, v in (("frame_idx", _int(self.frame_idx, "frame_idx")), ("time_s", _real(self.time_s, "time_s")), ("grip_pos_m", _vec(self.grip_pos_m, "grip")), ("hosel_pos_m", _vec(self.hosel_pos_m, "hosel")), ("head_pos_m", _vec(self.head_pos_m, "head")), ("opacity", _real(self.opacity, "opacity", 0.0, 1.0))):  # fmt: skip
            object.__setattr__(self, k, v)


@dataclass(frozen=True)
class FrozenClubStrobeSeries(_JsonContract):
    """Chronological collection of non-overlapping frozen club strobe poses."""

    samples: tuple[ClubStrobeSample, ...]
    club_id: str = "club"
    selection_mode: SelectionMode = "stride"

    def __post_init__(self) -> None:
        """Validate club identifier, selection mode, and chronological ordering."""
        _str(self.club_id, "club_id")
        if self.selection_mode not in ("phase_indices", "stride"):
            raise ValueError("selection_mode must be 'phase_indices' or 'stride'")
        samples = tuple(self.samples)
        for i, s in enumerate(samples):
            if not isinstance(s, ClubStrobeSample):
                raise TypeError("samples must contain ClubStrobeSample instances")
            if i > 0 and s.time_s <= samples[i - 1].time_s:
                raise ValueError("strobe samples must have strictly increasing time_s")
        object.__setattr__(self, "samples", samples)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrozenClubStrobeSeries:
        """Reconstruct a validated ``FrozenClubStrobeSeries`` from JSON."""
        parsed = tuple(ClubStrobeSample.from_dict(s) for s in data.get("samples", ()))
        return super().from_dict({**data, "samples": parsed})


def _strobe_candidates(n: int, phase_indices: Sequence[int] | Mapping[str, int] | None, stride: int | None) -> tuple[SelectionMode, list[tuple[int, str]]]:  # fmt: skip
    """Select sorted unique candidate (frame_idx, label) pairs."""
    if phase_indices is None:
        step = 1 if stride is None else _int(stride, "stride", 1)
        return "stride", [(i, "") for i in range(0, n, step)]
    raw = sorted((_int(v, "idx"), _str(k, "lbl")) for k, v in phase_indices.items()) if isinstance(phase_indices, Mapping) else [(_int(r, "idx"), "") for r in phase_indices]  # fmt: skip
    by_idx: dict[int, str] = {}
    for idx, lbl in raw:
        if idx >= n:
            raise IndexError(f"phase index {idx} out of range")
        by_idx.setdefault(idx, lbl)
    return "phase_indices", sorted(by_idx.items(), key=lambda p: p[0])


def build_frozen_club_strobes(
    time_s: Sequence[float],
    grip_positions_m: Sequence[Sequence[float]],
    hosel_positions_m: Sequence[Sequence[float]],
    head_positions_m: Sequence[Sequence[float]],
    phase_indices: Sequence[int] | Mapping[str, int] | None = None,
    stride: int | None = None,
    min_separation_m: float = 0.02,
    opacity_range: tuple[float, float] = (0.2, 1.0),
) -> FrozenClubStrobeSeries:
    """Extract non-overlapping club shaft+head strobes with monotonic opacity decay."""
    times = validate_strictly_increasing_times(time_s)
    n = len(times)
    if not (len(grip_positions_m) == len(hosel_positions_m) == len(head_positions_m) == n):  # fmt: skip
        raise ValueError("position sequences must match time_s length")
    sep = _real(min_separation_m, "min_separation_m", 0.0)
    min_op, max_op = _vec(opacity_range, "opacity_range", 2, True)
    if min_op > max_op:
        raise ValueError("opacity_range[0] must be <= opacity_range[1]")
    mode, candidates = _strobe_candidates(n, phase_indices, stride)
    kept: list[tuple[int, str, tuple[Vec3, Vec3, Vec3]]] = []
    for idx, lbl in candidates:
        pose = (_vec(grip_positions_m[idx]), _vec(hosel_positions_m[idx]), _vec(head_positions_m[idx]))  # fmt: skip
        if kept and sep > 0.0 and max(math.dist(p, q) for p, q in zip(pose, kept[-1][2], strict=True)) < sep:  # fmt: skip
            continue
        kept.append((idx, lbl, pose))
    m = len(kept)
    samples = tuple(
        ClubStrobeSample(
            idx,
            times[idx],
            *pose,
            max_op if m == 1 else min_op + (r / (m - 1)) * (max_op - min_op),
            lbl,
        )  # fmt: skip
        for r, (idx, lbl, pose) in enumerate(kept)
    )
    return FrozenClubStrobeSeries(samples=samples, selection_mode=mode)


@dataclass(frozen=True)
class CurledTorqueArrowSpec(_JsonContract):
    """Curved 3D torque arc around a joint axis with bounded sweep and curl sign."""

    joint_name: str
    center_m: Vec3
    axis_unit: Vec3
    radius_m: float
    signed_torque_nm: float
    sweep_angle_rad: float
    curl_direction: CurlDirection
    arc_points_m: tuple[Vec3, ...]
    head_tip_m: Vec3
    head_base_m: Vec3
    color_rgba: RGBA = CCW_TORQUE_RGBA

    def __post_init__(self) -> None:
        """Validate joint torque arrow geometry, sweep bounds, and curl direction."""
        _str(self.joint_name, "joint_name")
        sweep = _real(self.sweep_angle_rad, "sweep_angle_rad", 0.0)
        if sweep >= _TWO_PI or self.curl_direction not in ("ccw", "cw"):
            raise ValueError("invalid sweep_angle_rad or curl_direction")
        pts = tuple(_vec(p, f"arc[{i}]") for i, p in enumerate(self.arc_points_m))
        if len(pts) < 2:
            raise ValueError("arc_points_m must contain at least 2 points")
        for k, v in (("center_m", _vec(self.center_m, "c")), ("axis_unit", _norm_axis(self.axis_unit)), ("radius_m", _real(self.radius_m, "r", pos=True)), ("signed_torque_nm", _real(self.signed_torque_nm, "tq")), ("sweep_angle_rad", sweep), ("arc_points_m", pts), ("head_tip_m", _vec(self.head_tip_m, "tip")), ("head_base_m", _vec(self.head_base_m, "base")), ("color_rgba", _vec(self.color_rgba, "rgba", 4, True))):  # fmt: skip
            object.__setattr__(self, k, v)


def build_curled_torque_arrows(
    joint_names: Sequence[str],
    joint_centers_m: Sequence[Sequence[float]],
    joint_axes: Sequence[Sequence[float]],
    torques_nm: Sequence[float],
    radius_m: float = 0.08,
    torque_scale_nm: float = 100.0,
    magnitude_floor_nm: float = 0.5,
    n_segments: int = 24,
) -> tuple[CurledTorqueArrowSpec, ...]:
    """Build curled 3D torque arc specs around joint axes with floor cutoff."""
    if not (len(joint_names) == len(joint_centers_m) == len(joint_axes) == len(torques_nm)):  # fmt: skip
        raise ValueError("joint arrays must have identical length")
    rad, scale = _real(radius_m, "r", pos=True), _real(torque_scale_nm, "s", pos=True)
    floor_nm, segs = _real(magnitude_floor_nm, "floor", 0.0), _int(n_segments, "n", 4)
    out: list[CurledTorqueArrowSpec] = []
    for i, raw_name in enumerate(joint_names):
        torque = _real(torques_nm[i], f"torques_nm[{i}]")
        if abs(torque) == 0.0 or abs(torque) < floor_nm:
            continue
        sweep = min((abs(torque) / scale) * _MAX_SWEEP, _MAX_SWEEP)
        sign, direction = (1.0, "ccw") if torque >= 0.0 else (-1.0, "cw")
        ax = _norm_axis(joint_axes[i])
        a = np.asarray(ax, dtype=float)
        seed = np.array([1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        u = seed - np.dot(seed, a) * a
        u /= float(np.linalg.norm(u))
        v = np.cross(a, u)
        th = sign * sweep * np.linspace(0.0, 1.0, segs + 1)
        ctr = _vec(joint_centers_m[i], "c")
        arc = np.asarray(ctr, dtype=float) + rad * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))  # fmt: skip
        pts = tuple((float(p[0]), float(p[1]), float(p[2])) for p in arc)
        b = arc[-1] - min(0.35 * rad, 0.45 * rad * sweep) * sign * (-math.sin(th[-1]) * u + math.cos(th[-1]) * v)  # fmt: skip
        base: Vec3 = (float(b[0]), float(b[1]), float(b[2]))
        rgba = CCW_TORQUE_RGBA if direction == "ccw" else CW_TORQUE_RGBA
        out.append(CurledTorqueArrowSpec(_str(raw_name, "j"), ctr, ax, rad, torque, sweep, direction, pts, pts[-1], base, rgba))  # type: ignore[arg-type]  # fmt: skip
    return tuple(out)
