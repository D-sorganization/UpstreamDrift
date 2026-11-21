"""Headless angular-momentum arrow glyphs (S4, #15398)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.force_overlay.glyphs import (
    SCHEMA_VERSION,
    ArrowGlyph,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
    _compute_arc_basis,
    _hex_to_rgba,
)

from .segment_angular_momentum import (
    REFERENCE_POINT,
    FRAME_WORLD,
    UNITS_ANGMOM,
    SegmentAngularMomentumFrame,
    SegmentGroupCatalog,
    aggregate_group,
)

DisplayMode = Literal["groups", "per_segment", "spin_only"]

# Fixed visual scale, held constant for the whole swing (not per-frame autoscaled).
# Metres of arrow length per (N·m·s) of |H|; 4000 N·m·s maps to 1.0 m.
DEFAULT_SCALE_M_PER_NMS: float = 2.5e-4
DEFAULT_MAX_LENGTH_M: float = 1.2
DEFAULT_MIN_LENGTH_M: float = 0.04
DEFAULT_SHAFT_RADIUS_M: float = 0.01
DEFAULT_PER_SEGMENT_THRESHOLD_NMS: float = 2.0
DEFAULT_PER_SEGMENT_TOP_N: int = 8
DEFAULT_PER_SEGMENT_MAX_COUNT: int = 24

GROUP_PALETTE: Mapping[str, str] = MappingProxyType(
    {
        "whole_body": "#E69F00",
        "legs": "#56B4E9",
        "pelvis": "#009E73",
        "torso": "#F0E442",
        "arms": "#0072B2",
        "club": "#CC79A7",
        "spin_component": "#7F7F7F",
    }
)

GROUP_DISPLAY_ORDER: tuple[str, ...] = (
    "whole_body",
    "legs",
    "pelvis",
    "torso",
    "arms",
    "club",
)


@dataclass(frozen=True)
class AngularMomentumArrowStyle:
    """Fixed-length scale and caps for angular-momentum arrows."""

    scale_m_per_nms: float = DEFAULT_SCALE_M_PER_NMS
    min_length_m: float = DEFAULT_MIN_LENGTH_M
    max_length_m: float = DEFAULT_MAX_LENGTH_M
    shaft_radius_m: float = DEFAULT_SHAFT_RADIUS_M
    head_length_ratio: float = 0.22
    head_radius_ratio: float = 2.4
    display_mode: DisplayMode = "groups"
    show_spin_component: bool = False
    per_segment_threshold_nms: float = DEFAULT_PER_SEGMENT_THRESHOLD_NMS
    per_segment_top_n: int = DEFAULT_PER_SEGMENT_TOP_N
    per_segment_max_count: int = DEFAULT_PER_SEGMENT_MAX_COUNT
    spin_arc_sweep_rad: float = 1.25 * math.pi
    spin_arc_segments: int = 24

    def __post_init__(self) -> None:
        for name in (
            "scale_m_per_nms",
            "min_length_m",
            "max_length_m",
            "shaft_radius_m",
            "head_length_ratio",
            "head_radius_ratio",
            "per_segment_threshold_nms",
            "spin_arc_sweep_rad",
        ):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_length_m >= self.max_length_m:
            raise ValueError("min_length_m must be < max_length_m")
        if self.display_mode not in ("groups", "per_segment", "spin_only"):
            raise ValueError(f"invalid display_mode {self.display_mode!r}")
        if not isinstance(self.per_segment_top_n, int) or self.per_segment_top_n < 1:
            raise ValueError("per_segment_top_n must be a positive int")
        if (
            not isinstance(self.per_segment_max_count, int)
            or self.per_segment_max_count < 1
        ):
            raise ValueError("per_segment_max_count must be a positive int")
        if self.spin_arc_segments < 3:
            raise ValueError("spin_arc_segments must be >= 3")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale_m_per_nms": self.scale_m_per_nms,
            "min_length_m": self.min_length_m,
            "max_length_m": self.max_length_m,
            "shaft_radius_m": self.shaft_radius_m,
            "head_length_ratio": self.head_length_ratio,
            "head_radius_ratio": self.head_radius_ratio,
            "display_mode": self.display_mode,
            "show_spin_component": self.show_spin_component,
            "per_segment_threshold_nms": self.per_segment_threshold_nms,
            "per_segment_top_n": self.per_segment_top_n,
            "per_segment_max_count": self.per_segment_max_count,
            "spin_arc_sweep_rad": self.spin_arc_sweep_rad,
            "spin_arc_segments": self.spin_arc_segments,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AngularMomentumArrowStyle:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown AngularMomentumArrowStyle keys: {sorted(unknown)}")
        return cls(**{k: data[k] for k in allowed if k in data})


def angular_momentum_legend_line(style: AngularMomentumArrowStyle) -> str:
    """Reference frame label and fixed scale for the status bar / legend."""
    ref_h = 1.0 / style.scale_m_per_nms
    mode = style.display_mode.replace("_", " ")
    return (
        f"H about {REFERENCE_POINT.replace('_', ' ')}, {FRAME_WORLD} frame; "
        f"{UNITS_ANGMOM}; scale {ref_h:.0f} {UNITS_ANGMOM} = 1 m ({mode})"
    )


def _vector_magnitude(vec: Sequence[float]) -> float:
    return float(math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2))


def _build_straight_arrow(
    *,
    label: str,
    anchor_m: tuple[float, float, float],
    vector_nm_s: tuple[float, float, float],
    style: AngularMomentumArrowStyle,
    color_hex: str,
) -> ArrowGlyph | None:
    mag = _vector_magnitude(vector_nm_s)
    if mag <= 0.0 or not math.isfinite(mag):
        return None
    vx, vy, vz = vector_nm_s
    h_hat = (vx / mag, vy / mag, vz / mag)
    raw_len = mag * style.scale_m_per_nms
    clamped = raw_len > style.max_length_m
    length = max(style.min_length_m, min(raw_len, style.max_length_m))
    px, py, pz = anchor_m
    tip = (px + length * h_hat[0], py + length * h_hat[1], pz + length * h_hat[2])
    head_len = style.head_length_ratio * length
    head_base = (
        tip[0] - head_len * h_hat[0],
        tip[1] - head_len * h_hat[1],
        tip[2] - head_len * h_hat[2],
    )
    shaft = style.shaft_radius_m
    rgba = _hex_to_rgba(color_hex)
    return ArrowGlyph(
        label=label,
        kind=WrenchKind.EXTERNAL,
        tail_m=anchor_m,
        tip_m=tip,
        head_base_m=head_base,
        shaft_radius_m=shaft,
        head_radius_m=shaft * style.head_radius_ratio,
        rgba=rgba,
        magnitude=mag,
        units=UNITS_ANGMOM,
        clamped=clamped,
    )


def _build_spin_arc(
    *,
    label: str,
    center_m: tuple[float, float, float],
    spin_nm_s: tuple[float, float, float],
    style: AngularMomentumArrowStyle,
    color_hex: str,
) -> TorqueArcGlyph | None:
    mag = _vector_magnitude(spin_nm_s)
    if mag <= 0.0:
        return None
    axis = (spin_nm_s[0] / mag, spin_nm_s[1] / mag, spin_nm_s[2] / mag)
    radius = min(style.max_length_m * 0.35, max(style.min_length_m, mag * style.scale_m_per_nms * 0.5))
    u, v = _compute_arc_basis(axis)
    cx, cy, cz = center_m
    pts: list[tuple[float, float, float]] = []
    sweep = style.spin_arc_sweep_rad
    n_seg = style.spin_arc_segments
    for i in range(n_seg + 1):
        theta = sweep * (i / n_seg)
        pts.append(
            (
                cx + radius * (math.cos(theta) * u[0] + math.sin(theta) * v[0]),
                cy + radius * (math.cos(theta) * u[1] + math.sin(theta) * v[1]),
                cz + radius * (math.cos(theta) * u[2] + math.sin(theta) * v[2]),
            )
        )
    head_tip = pts[-1]
    cos_last = math.cos(sweep)
    sin_last = math.sin(sweep)
    tangent = (
        -sin_last * u[0] + cos_last * v[0],
        -sin_last * u[1] + cos_last * v[1],
        -sin_last * u[2] + cos_last * v[2],
    )
    head_len = style.head_length_ratio * radius * 2.0
    head_base = (
        head_tip[0] - head_len * tangent[0],
        head_tip[1] - head_len * tangent[1],
        head_tip[2] - head_len * tangent[2],
    )
    rgba = _hex_to_rgba(color_hex)
    return TorqueArcGlyph(
        label=label,
        kind=WrenchKind.EXTERNAL,
        center_m=center_m,
        axis_unit=axis,
        radius_m=radius,
        polyline_m=tuple(pts),
        head_tip_m=head_tip,
        head_base_m=head_base,
        rgba=rgba,
        magnitude=mag,
        units=UNITS_ANGMOM,
        clamped=False,
    )


def _select_per_segment(
    frame: SegmentAngularMomentumFrame,
    style: AngularMomentumArrowStyle,
) -> list[tuple[str, tuple[float, float, float]]]:
    ranked: list[tuple[float, str, tuple[float, float, float]]] = []
    floor = style.per_segment_threshold_nms
    for name, vec in frame.contribution_nm_s.items():
        if vec is None:
            continue
        mag = _vector_magnitude(vec)
        if mag < floor:
            continue
        ranked.append((mag, name, vec))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    cap = min(style.per_segment_top_n, style.per_segment_max_count, len(ranked))
    return [(name, vec) for _, name, vec in ranked[:cap]]


def _targets_for_mode(
    frame: SegmentAngularMomentumFrame,
    style: AngularMomentumArrowStyle,
) -> list[tuple[str, str, tuple[float, float, float] | None]]:
    if style.display_mode == "groups":
        out: list[tuple[str, str, tuple[float, float, float] | None]] = []
        if frame.whole_body_nm_s is not None:
            out.append(("whole_body", "whole_body", frame.whole_body_nm_s))
        for key in GROUP_DISPLAY_ORDER:
            if key == "whole_body":
                continue
            out.append((key, key, frame.groups_nm_s.get(key)))
        return out
    if style.display_mode == "spin_only":
        return [
            (name, name, vec)
            for name, vec in sorted(frame.spin_only_nm_s.items())
            if vec is not None
        ]
    return [(name, name, vec) for name, vec in _select_per_segment(frame, style)]


def build_angular_momentum_glyphs(
    frame: SegmentAngularMomentumFrame | None,
    *,
    style: AngularMomentumArrowStyle,
    anchors_m: Mapping[str, tuple[float, float, float]],
    enabled: bool,
    catalog: SegmentGroupCatalog | None = None,
) -> GlyphSet | None:
    """Build straight H arrows (and spin arcs in spin-only mode)."""
    if not enabled or frame is None:
        return None
    targets = _targets_for_mode(frame, style)
    arrows: list[ArrowGlyph] = []
    arcs: list[TorqueArcGlyph] = []
    clamped: list[str] = []
    for anchor_key, label, vec in targets:
        if vec is None:
            continue
        anchor = anchors_m.get(anchor_key)
        if anchor is None:
            continue
        color = GROUP_PALETTE.get(label, "#888888")
        arrow = _build_straight_arrow(
            label=f"angmom:{label}",
            anchor_m=anchor,
            vector_nm_s=vec,
            style=style,
            color_hex=color,
        )
        if arrow is None:
            continue
        arrows.append(arrow)
        if arrow.clamped:
            clamped.append(arrow.label)
        if style.show_spin_component and style.display_mode != "spin_only":
            spin = _spin_for_target(frame, label, style.display_mode, catalog)
            if spin is not None:
                spin_arrow = _build_straight_arrow(
                    label=f"angmom:spin:{label}",
                    anchor_m=anchor,
                    vector_nm_s=spin,
                    style=style,
                    color_hex=GROUP_PALETTE["spin_component"],
                )
                if spin_arrow is not None:
                    arrows.append(spin_arrow)
        if style.display_mode == "spin_only":
            arc = _build_spin_arc(
                label=f"angmom:arc:{label}",
                center_m=anchor,
                spin_nm_s=vec,
                style=style,
                color_hex=color,
            )
            if arc is not None:
                arcs.append(arc)
    if not arrows and not arcs:
        return None
    legend = LegendSpec(
        engine="angular_momentum",
        source_labels=(frame.source,),
        scale_mode="fixed",
        clamped_labels=tuple(clamped),
    )
    return GlyphSet(
        time_s=frame.time_s,
        arrows=tuple(arrows),
        torque_arcs=tuple(arcs),
        legend=legend,
        schema_version=SCHEMA_VERSION,
    )


def _spin_for_target(
    frame: SegmentAngularMomentumFrame,
    label: str,
    mode: DisplayMode,
    catalog: SegmentGroupCatalog | None,
) -> tuple[float, float, float] | None:
    if mode == "per_segment":
        return frame.spin_only_nm_s.get(label)
    if catalog is None:
        return None
    spins: dict[str, np.ndarray | None] = {
        name: None if vec is None else np.asarray(vec, dtype=float)
        for name, vec in frame.spin_only_nm_s.items()
    }
    if label == "whole_body":
        parts = [v for v in spins.values() if v is not None]
    elif label in catalog.as_mapping():
        members = catalog.as_mapping()[label]
        vec = aggregate_group(spins, tuple(members))
        return None if vec is None else (float(vec[0]), float(vec[1]), float(vec[2]))
    else:
        return None
    if not parts:
        return None
    total = sum(parts, start=np.zeros(3))
    return (float(total[0]), float(total[1]), float(total[2]))


def merge_glyph_sets(
    primary: GlyphSet | None, secondary: GlyphSet | None
) -> GlyphSet | None:
    """Concatenate arrows and arcs; ``primary`` supplies ``time_s`` when present."""
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    time_s = primary.time_s if primary.time_s == secondary.time_s else secondary.time_s
    clamped = primary.legend.clamped_labels + secondary.legend.clamped_labels
    legend = LegendSpec(
        engine=primary.legend.engine,
        source_labels=primary.legend.source_labels + secondary.legend.source_labels,
        scale_mode=primary.legend.scale_mode,
        clamped_labels=clamped,
    )
    return GlyphSet(
        time_s=time_s,
        arrows=primary.arrows + secondary.arrows,
        torque_arcs=primary.torque_arcs + secondary.torque_arcs,
        legend=legend,
        ellipsoids=primary.ellipsoids + secondary.ellipsoids,
    )