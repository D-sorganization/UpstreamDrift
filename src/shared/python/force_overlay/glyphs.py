"""Glyph builder and renderer-neutral geometry (ADR-0052, #11288).

Pure, deterministic builder converting ForceTorqueFrame to GlyphSet wire geometry.
Zero rendering or GUI dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Final

import numpy as np

from .contracts import ForceTorqueFrame, OverlayWrench, WrenchKind
from .palette import FORCE_KIND_PALETTE, hex_to_rgba

__all__ = [
    "ArrowGlyph",
    "ForceGlyphStyle",
    "GlyphSet",
    "LegendSpec",
    "TorqueArcGlyph",
    "build_glyphs",
    "scale_for_view",
]

GLYPH_SCHEMA_VERSION: Final[str] = "glyph-set-v1"


def _closest_nice_number(val: float) -> float:
    """Find the nice number (1, 2, or 5 * 10^n) closest to val."""
    if val <= 0.0 or not math.isfinite(val):
        return 1.0
    p = math.floor(math.log10(val))
    candidates = []
    for exp in (p - 1, p, p + 1):
        for mult in (1.0, 2.0, 5.0):
            candidates.append(mult * (10.0**exp))
    return min(candidates, key=lambda c: (abs(c - val), c))


@dataclass(frozen=True)
class ForceGlyphStyle:
    """Styling and filtering configuration for force and torque glyph generation."""

    force_scale_m_per_n: float = 1.0 / 1000.0
    torque_scale_m_per_nm: float = 1.0 / 200.0
    min_length_m: float = 0.02
    max_length_m: float = 0.6
    shaft_radius_m: float = 0.006
    head_length_ratio: float = 0.22
    head_radius_ratio: float = 2.4
    torque_style: str = "arc"  # "arc" or "axis_double_head"
    arc_sweep_rad: float = 1.5 * math.pi
    arc_segments: int = 32
    kinds: tuple[str, ...] = tuple(k.value for k in WrenchKind)
    magnitude_floor_n: float = 1.0
    magnitude_floor_nm: float = 0.1
    show_labels: bool = False
    palette: dict[str, str] = field(default_factory=lambda: dict(FORCE_KIND_PALETTE))

    def __post_init__(self) -> None:
        for name, val in [
            ("force_scale_m_per_n", self.force_scale_m_per_n),
            ("torque_scale_m_per_nm", self.torque_scale_m_per_nm),
            ("min_length_m", self.min_length_m),
            ("max_length_m", self.max_length_m),
            ("shaft_radius_m", self.shaft_radius_m),
            ("head_length_ratio", self.head_length_ratio),
            ("head_radius_ratio", self.head_radius_ratio),
            ("arc_sweep_rad", self.arc_sweep_rad),
            ("magnitude_floor_n", self.magnitude_floor_n),
            ("magnitude_floor_nm", self.magnitude_floor_nm),
        ]:
            if (
                not isinstance(val, (int, float))
                or not math.isfinite(val)
                or val <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite, got {val}")

        if self.min_length_m >= self.max_length_m:
            raise ValueError(
                f"min_length_m must be strictly less than max_length_m: {self.min_length_m} >= {self.max_length_m}"
            )
        if self.torque_style not in ("arc", "axis_double_head"):
            raise ValueError(
                f"torque_style must be 'arc' or 'axis_double_head', got {self.torque_style!r}"
            )
        if self.arc_segments < 3:
            raise ValueError(f"arc_segments must be >= 3, got {self.arc_segments}")

        # Normalize kinds to string tuple
        norm_kinds = tuple(
            k.value if hasattr(k, "value") else str(k) for k in self.kinds
        )
        object.__setattr__(self, "kinds", norm_kinds)


@dataclass(frozen=True)
class ArrowGlyph:
    """Renderer-neutral geometry for a 3D arrow glyph."""

    label: str
    kind: str
    tail_m: tuple[float, float, float]
    tip_m: tuple[float, float, float]
    head_base_m: tuple[float, float, float]
    shaft_radius_m: float
    head_radius_m: float
    rgba: tuple[float, float, float, float]
    magnitude: float
    units: str
    clamped: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "tail_m": list(self.tail_m),
            "tip_m": list(self.tip_m),
            "head_base_m": list(self.head_base_m),
            "shaft_radius_m": self.shaft_radius_m,
            "head_radius_m": self.head_radius_m,
            "rgba": list(self.rgba),
            "magnitude": self.magnitude,
            "units": self.units,
            "clamped": self.clamped,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ArrowGlyph:
        return cls(
            label=str(d["label"]),
            kind=str(d["kind"]),
            tail_m=(
                float(d["tail_m"][0]),
                float(d["tail_m"][1]),
                float(d["tail_m"][2]),
            ),
            tip_m=(float(d["tip_m"][0]), float(d["tip_m"][1]), float(d["tip_m"][2])),
            head_base_m=(
                float(d["head_base_m"][0]),
                float(d["head_base_m"][1]),
                float(d["head_base_m"][2]),
            ),
            shaft_radius_m=float(d["shaft_radius_m"]),
            head_radius_m=float(d["head_radius_m"]),
            rgba=(
                float(d["rgba"][0]),
                float(d["rgba"][1]),
                float(d["rgba"][2]),
                float(d["rgba"][3]),
            ),
            magnitude=float(d["magnitude"]),
            units=str(d["units"]),
            clamped=bool(d["clamped"]),
        )


@dataclass(frozen=True)
class TorqueArcGlyph:
    """Renderer-neutral geometry for a 3D torque arc polyline glyph."""

    label: str
    kind: str
    center_m: tuple[float, float, float]
    axis_unit: tuple[float, float, float]
    radius_m: float
    polyline_m: tuple[tuple[float, float, float], ...]
    head_tip_m: tuple[float, float, float]
    head_base_m: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    magnitude: float
    units: str
    clamped: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "center_m": list(self.center_m),
            "axis_unit": list(self.axis_unit),
            "radius_m": self.radius_m,
            "polyline_m": [list(pt) for pt in self.polyline_m],
            "head_tip_m": list(self.head_tip_m),
            "head_base_m": list(self.head_base_m),
            "rgba": list(self.rgba),
            "magnitude": self.magnitude,
            "units": self.units,
            "clamped": self.clamped,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TorqueArcGlyph:
        return cls(
            label=str(d["label"]),
            kind=str(d["kind"]),
            center_m=(
                float(d["center_m"][0]),
                float(d["center_m"][1]),
                float(d["center_m"][2]),
            ),
            axis_unit=(
                float(d["axis_unit"][0]),
                float(d["axis_unit"][1]),
                float(d["axis_unit"][2]),
            ),
            radius_m=float(d["radius_m"]),
            polyline_m=tuple(
                (float(pt[0]), float(pt[1]), float(pt[2])) for pt in d["polyline_m"]
            ),
            head_tip_m=(
                float(d["head_tip_m"][0]),
                float(d["head_tip_m"][1]),
                float(d["head_tip_m"][2]),
            ),
            head_base_m=(
                float(d["head_base_m"][0]),
                float(d["head_base_m"][1]),
                float(d["head_base_m"][2]),
            ),
            rgba=(
                float(d["rgba"][0]),
                float(d["rgba"][1]),
                float(d["rgba"][2]),
                float(d["rgba"][3]),
            ),
            magnitude=float(d["magnitude"]),
            units=str(d["units"]),
            clamped=bool(d["clamped"]),
        )


@dataclass(frozen=True)
class LegendSpec:
    """Reference scaling and metadata for viewport overlay legends."""

    force_reference_n: float | None
    force_reference_length_m: float | None
    torque_reference_nm: float | None
    torque_reference_radius_m: float | None
    kinds_present: tuple[str, ...]
    unavailable_labels: tuple[str, ...]
    engine: str
    source_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_reference_n": self.force_reference_n,
            "force_reference_length_m": self.force_reference_length_m,
            "torque_reference_nm": self.torque_reference_nm,
            "torque_reference_radius_m": self.torque_reference_radius_m,
            "kinds_present": list(self.kinds_present),
            "unavailable_labels": list(self.unavailable_labels),
            "engine": self.engine,
            "source_labels": list(self.source_labels),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LegendSpec:
        return cls(
            force_reference_n=(
                float(d["force_reference_n"])
                if d.get("force_reference_n") is not None
                else None
            ),
            force_reference_length_m=(
                float(d["force_reference_length_m"])
                if d.get("force_reference_length_m") is not None
                else None
            ),
            torque_reference_nm=(
                float(d["torque_reference_nm"])
                if d.get("torque_reference_nm") is not None
                else None
            ),
            torque_reference_radius_m=(
                float(d["torque_reference_radius_m"])
                if d.get("torque_reference_radius_m") is not None
                else None
            ),
            kinds_present=tuple(str(k) for k in d.get("kinds_present", ())),
            unavailable_labels=tuple(
                str(lbl) for lbl in d.get("unavailable_labels", ())
            ),
            engine=str(d.get("engine", "")),
            source_labels=tuple(str(lbl) for lbl in d.get("source_labels", ())),
        )


@dataclass(frozen=True)
class GlyphSet:
    """Collection of renderable glyphs for one instant in time."""

    time_s: float
    arrows: tuple[ArrowGlyph, ...]
    torque_arcs: tuple[TorqueArcGlyph, ...]
    legend: LegendSpec
    schema_version: str = GLYPH_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "time_s": self.time_s,
            "arrows": [a.to_dict() for a in self.arrows],
            "torque_arcs": [t.to_dict() for t in self.torque_arcs],
            "legend": self.legend.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GlyphSet:
        expected_keys = {"schema_version", "time_s", "arrows", "torque_arcs", "legend"}
        unknown = set(d.keys()) - expected_keys
        if unknown:
            raise ValueError(f"Unknown keys in GlyphSet: {unknown}")

        version = d.get("schema_version")
        if version != GLYPH_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be '{GLYPH_SCHEMA_VERSION}', got {version!r}"
            )

        return cls(
            time_s=float(d["time_s"]),
            arrows=tuple(ArrowGlyph.from_dict(a) for a in d.get("arrows", ())),
            torque_arcs=tuple(
                TorqueArcGlyph.from_dict(t) for t in d.get("torque_arcs", ())
            ),
            legend=LegendSpec.from_dict(d["legend"]),
            schema_version=version,
        )


def _build_torque_arc_basis(a_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pick world axis e with smallest |a_hat . e| (breaking ties x, y, z) and build u, v."""
    dx = abs(float(a_hat[0]))
    dy = abs(float(a_hat[1]))
    dz = abs(float(a_hat[2]))

    if dx <= dy and dx <= dz:
        e = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif dy <= dz:
        e = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        e = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    proj = e - float(np.dot(e, a_hat)) * a_hat
    u = proj / float(np.linalg.norm(proj))
    v = np.cross(a_hat, u)
    return u, v


def _build_arrow_glyph(
    w: OverlayWrench,
    cfg: ForceGlyphStyle,
    rgba: tuple[float, float, float, float],
) -> ArrowGlyph | None:
    if w.force_n is None:
        return None
    f_arr = np.array(w.force_n, dtype=np.float64)
    f_mag = float(np.linalg.norm(f_arr))
    if f_mag < cfg.magnitude_floor_n:
        return None

    f_hat = f_arr / f_mag
    l_des = f_mag * cfg.force_scale_m_per_n
    l_eff = min(max(l_des, cfg.min_length_m), cfg.max_length_m)
    clamped = not math.isclose(l_des, l_eff, abs_tol=1e-7)

    pt = np.array(w.point_m, dtype=np.float64)
    tip = pt + l_eff * f_hat
    head_len = cfg.head_length_ratio * l_eff
    head_base = tip - head_len * f_hat
    kind_str = w.kind.value if hasattr(w.kind, "value") else str(w.kind)

    return ArrowGlyph(
        label=w.label,
        kind=kind_str,
        tail_m=(float(pt[0]), float(pt[1]), float(pt[2])),
        tip_m=(float(tip[0]), float(tip[1]), float(tip[2])),
        head_base_m=(float(head_base[0]), float(head_base[1]), float(head_base[2])),
        shaft_radius_m=cfg.shaft_radius_m,
        head_radius_m=cfg.shaft_radius_m * cfg.head_radius_ratio,
        rgba=rgba,
        magnitude=f_mag,
        units="N",
        clamped=clamped,
    )


def _build_torque_arc_glyph(
    w: OverlayWrench,
    cfg: ForceGlyphStyle,
    rgba: tuple[float, float, float, float],
) -> TorqueArcGlyph | None:
    if w.torque_nm is None:
        return None
    t_arr = np.array(w.torque_nm, dtype=np.float64)
    t_mag = float(np.linalg.norm(t_arr))
    if t_mag < cfg.magnitude_floor_nm:
        return None

    a_hat = t_arr / t_mag
    l_des = t_mag * cfg.torque_scale_m_per_nm
    l_eff = min(max(l_des, cfg.min_length_m), cfg.max_length_m)
    clamped = not math.isclose(l_des, l_eff, abs_tol=1e-7)
    radius_m = l_eff / 2.0

    center = np.array(w.point_m, dtype=np.float64)
    u, v = _build_torque_arc_basis(a_hat)

    thetas = np.linspace(0.0, cfg.arc_sweep_rad, cfg.arc_segments + 1)
    pts = [center + radius_m * (math.cos(th) * u + math.sin(th) * v) for th in thetas]
    poly = tuple((float(p[0]), float(p[1]), float(p[2])) for p in pts)

    last_th = float(thetas[-1])
    tangent = -math.sin(last_th) * u + math.cos(last_th) * v
    t_norm = float(np.linalg.norm(tangent))
    if t_norm > 0:
        tangent = tangent / t_norm

    head_tip = pts[-1]
    head_len = cfg.head_length_ratio * l_eff
    head_base = head_tip - head_len * tangent
    kind_str = w.kind.value if hasattr(w.kind, "value") else str(w.kind)

    return TorqueArcGlyph(
        label=w.label,
        kind=kind_str,
        center_m=(float(center[0]), float(center[1]), float(center[2])),
        axis_unit=(float(a_hat[0]), float(a_hat[1]), float(a_hat[2])),
        radius_m=radius_m,
        polyline_m=poly,
        head_tip_m=(float(head_tip[0]), float(head_tip[1]), float(head_tip[2])),
        head_base_m=(float(head_base[0]), float(head_base[1]), float(head_base[2])),
        rgba=rgba,
        magnitude=t_mag,
        units="N*m",
        clamped=clamped,
    )


def _build_legend_spec(
    frame: ForceTorqueFrame,
    cfg: ForceGlyphStyle,
    force_mags: list[float],
    torque_mags: list[float],
    kinds_present_set: set[str],
    unavailable: list[str],
    source_labels: list[str],
) -> LegendSpec:
    f_ref_n = _closest_nice_number(float(np.median(force_mags))) if force_mags else None
    f_ref_l = (
        min(max(f_ref_n * cfg.force_scale_m_per_n, cfg.min_length_m), cfg.max_length_m)
        if f_ref_n is not None
        else None
    )

    t_ref_nm = (
        _closest_nice_number(float(np.median(torque_mags))) if torque_mags else None
    )
    t_ref_r = (
        min(
            max(t_ref_nm * cfg.torque_scale_m_per_nm, cfg.min_length_m),
            cfg.max_length_m,
        )
        / 2.0
        if t_ref_nm is not None
        else None
    )

    return LegendSpec(
        force_reference_n=f_ref_n,
        force_reference_length_m=f_ref_l,
        torque_reference_nm=t_ref_nm,
        torque_reference_radius_m=t_ref_r,
        kinds_present=tuple(sorted(kinds_present_set)),
        unavailable_labels=tuple(sorted(set(unavailable))),
        engine=frame.engine,
        source_labels=tuple(sorted(set(source_labels))),
    )


def build_glyphs(
    frame: ForceTorqueFrame,
    style: ForceGlyphStyle | None = None,
) -> GlyphSet:
    """Deterministic generator converting ForceTorqueFrame to GlyphSet wire geometry."""
    cfg = style or ForceGlyphStyle()
    arrows: list[ArrowGlyph] = []
    torque_arcs: list[TorqueArcGlyph] = []
    unavailable: list[str] = []
    source_labels: list[str] = [w.label for w in frame.wrenches]
    force_mags: list[float] = []
    torque_mags: list[float] = []
    kinds_present_set: set[str] = set()

    for w in sorted(frame.wrenches, key=lambda x: x.label):
        kind_str = w.kind.value if hasattr(w.kind, "value") else str(w.kind)
        if kind_str not in cfg.kinds:
            continue

        hex_col = cfg.palette.get(kind_str, "#000000")
        rgba = hex_to_rgba(hex_col)

        if w.force_n is None:
            unavailable.append(w.label)
        else:
            arrow = _build_arrow_glyph(w, cfg, rgba)
            if arrow is not None:
                arrows.append(arrow)
                force_mags.append(arrow.magnitude)
                kinds_present_set.add(kind_str)

        if w.torque_nm is None:
            unavailable.append(w.label)
        else:
            arc = _build_torque_arc_glyph(w, cfg, rgba)
            if arc is not None:
                torque_arcs.append(arc)
                torque_mags.append(arc.magnitude)
                kinds_present_set.add(kind_str)

    arrows.sort(key=lambda a: a.label)
    torque_arcs.sort(key=lambda t: t.label)

    legend = _build_legend_spec(
        frame=frame,
        cfg=cfg,
        force_mags=force_mags,
        torque_mags=torque_mags,
        kinds_present_set=kinds_present_set,
        unavailable=unavailable,
        source_labels=source_labels,
    )

    return GlyphSet(
        time_s=frame.time_s,
        arrows=tuple(arrows),
        torque_arcs=tuple(torque_arcs),
        legend=legend,
    )


def scale_for_view(glyphs: GlyphSet, scale_factor: float) -> GlyphSet:
    """Rescale glyph visual display dimensions while keeping physical magnitudes unchanged."""
    if (
        not isinstance(scale_factor, (int, float))
        or scale_factor <= 0.0
        or not math.isfinite(scale_factor)
    ):
        raise ValueError(
            f"scale_factor must be positive and finite, got {scale_factor}"
        )

    s = float(scale_factor)
    scaled_arrows: list[ArrowGlyph] = []
    for a in glyphs.arrows:
        tail = np.array(a.tail_m, dtype=np.float64)
        tip = tail + (np.array(a.tip_m, dtype=np.float64) - tail) * s
        hb = (
            tip
            + (
                np.array(a.head_base_m, dtype=np.float64)
                - np.array(a.tip_m, dtype=np.float64)
            )
            * s
        )
        scaled_arrows.append(
            ArrowGlyph(
                label=a.label,
                kind=a.kind,
                tail_m=a.tail_m,
                tip_m=(float(tip[0]), float(tip[1]), float(tip[2])),
                head_base_m=(float(hb[0]), float(hb[1]), float(hb[2])),
                shaft_radius_m=a.shaft_radius_m * s,
                head_radius_m=a.head_radius_m * s,
                rgba=a.rgba,
                magnitude=a.magnitude,
                units=a.units,
                clamped=a.clamped,
            )
        )

    scaled_arcs: list[TorqueArcGlyph] = []
    for t in glyphs.torque_arcs:
        center = np.array(t.center_m, dtype=np.float64)
        scaled_poly = tuple(
            (
                float(center[0] + (pt[0] - center[0]) * s),
                float(center[1] + (pt[1] - center[1]) * s),
                float(center[2] + (pt[2] - center[2]) * s),
            )
            for pt in t.polyline_m
        )
        ht = center + (np.array(t.head_tip_m, dtype=np.float64) - center) * s
        hb = center + (np.array(t.head_base_m, dtype=np.float64) - center) * s
        scaled_arcs.append(
            TorqueArcGlyph(
                label=t.label,
                kind=t.kind,
                center_m=t.center_m,
                axis_unit=t.axis_unit,
                radius_m=t.radius_m * s,
                polyline_m=scaled_poly,
                head_tip_m=(float(ht[0]), float(ht[1]), float(ht[2])),
                head_base_m=(float(hb[0]), float(hb[1]), float(hb[2])),
                rgba=t.rgba,
                magnitude=t.magnitude,
                units=t.units,
                clamped=t.clamped,
            )
        )

    old_leg = glyphs.legend
    new_leg = LegendSpec(
        force_reference_n=old_leg.force_reference_n,
        force_reference_length_m=(
            old_leg.force_reference_length_m * s
            if old_leg.force_reference_length_m is not None
            else None
        ),
        torque_reference_nm=old_leg.torque_reference_nm,
        torque_reference_radius_m=(
            old_leg.torque_reference_radius_m * s
            if old_leg.torque_reference_radius_m is not None
            else None
        ),
        kinds_present=old_leg.kinds_present,
        unavailable_labels=old_leg.unavailable_labels,
        engine=old_leg.engine,
        source_labels=old_leg.source_labels,
    )

    return GlyphSet(
        time_s=glyphs.time_s,
        arrows=tuple(scaled_arrows),
        torque_arcs=tuple(scaled_arcs),
        legend=new_leg,
        schema_version=glyphs.schema_version,
    )
