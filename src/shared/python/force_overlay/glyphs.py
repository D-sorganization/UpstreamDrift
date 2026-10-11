"""Renderer-neutral force and torque glyph generation and serialization (ADR-0052, #11288)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
    validate_vec3,
)
from src.shared.python.force_overlay.palette import label_variant_hex
from src.shared.python.plot_style import FORCE_KIND_PALETTE

__all__ = [
    "ArrowGlyph",
    "ForceGlyphStyle",
    "GlyphSet",
    "LegendSpec",
    "TorqueArcGlyph",
    "ALL_GROUPS",
    "GROUP_LABELS",
    "DEFAULT_GROUPS",
    "SCALE_MODES",
    "build_glyphs",
    "clamped_tip_shift",
    "label_group",
    "scale_for_view",
]

ScaleMode = Literal["fixed", "body_weight", "peak"]
SCALE_MODES: tuple[str, ...] = ("fixed", "body_weight", "peak")

#: Overlay groups selected by label prefix (GCV-1 ``contact:`` / GCV-2 ``grip:``).
ALL_GROUPS: frozenset[str] = frozenset(
    {
        "per_foot",
        "net",
        "free_moment",
        "moment_about_com",
        "contact_points",
        "grip_per_hand",
        "grip_net",
        "grip_couple",
        "grip_mof",
    }
)
#: Raw per-sphere contacts and moment-about-CoM arcs are opt-in.
DEFAULT_GROUPS: frozenset[str] = ALL_GROUPS - {"contact_points", "moment_about_com"}

#: Display labels shared by the PyQt Visualization tab and the web panel.
GROUP_LABELS: dict[str, str] = {
    "per_foot": "Per-Foot GRF",
    "net": "Net GRF",
    "free_moment": "Free Moment",
    "moment_about_com": "Moment About CoM",
    "contact_points": "Contact Points",
    "grip_per_hand": "Grip Per Hand",
    "grip_net": "Grip Net",
    "grip_couple": "Grip Couple",
    "grip_mof": "Grip MOF",
}

# Order matters: the first matching prefix wins (``grf_net`` before ``grf_``).
_GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("contact:grf_net", "net"),
    ("contact:grf_", "per_foot"),
    ("contact:free_moment", "free_moment"),
    ("contact:moment_com", "moment_about_com"),
    ("grip:hand_", "grip_per_hand"),
    ("grip:net_midpoint", "grip_net"),
    ("grip:couple_midpoint", "grip_couple"),
    ("grip:mof_", "grip_mof"),
    ("contact:", "contact_points"),
)


def label_group(label: str) -> str | None:
    """Return the overlay group of a wrench label, or ``None`` if ungrouped.

    Matching is by label prefix only, so providers need not import this module.
    Ungrouped labels (joint torques, gravity, ...) are never filtered by groups.
    """
    for prefix, group in _GROUP_PREFIXES:
        if label.startswith(prefix):
            return group
    return None


SCHEMA_VERSION = "glyph-set-v1"


def _hex_to_rgba(hex_code: str) -> tuple[float, float, float, float]:
    """Parse #RGB, #RRGGBB, or #RRGGBBAA hex color into float RGBA in [0, 1]."""
    s = hex_code.strip().lstrip("#")
    if len(s) == 3:
        r, g, b = (int(c * 2, 16) / 255.0 for c in s)
        return (r, g, b, 1.0)
    if len(s) == 6:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b, 1.0)
    if len(s) == 8:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        a = int(s[6:8], 16) / 255.0
        return (r, g, b, a)
    raise ValueError(f"Invalid hex color code: {hex_code!r}")


def _nice_number(val: float) -> float:
    """Return the nice number (1, 2 or 5 * 10^n) closest to val."""
    if val <= 0.0 or not math.isfinite(val):
        return 1.0
    exp = math.floor(math.log10(val))
    scale = 10.0**exp
    candidates = (1.0 * scale, 2.0 * scale, 5.0 * scale, 10.0 * scale)
    return min(candidates, key=lambda c: abs(c - val))


@dataclass(frozen=True)
class ForceGlyphStyle:
    """Configuration style for generating force and torque glyphs."""

    force_scale_m_per_n: float = 1.0 / 1000.0
    torque_scale_m_per_nm: float = 1.0 / 200.0
    min_length_m: float = 0.02
    max_length_m: float = 0.6
    shaft_radius_m: float = 0.012
    head_length_ratio: float = 0.22
    head_radius_ratio: float = 2.4
    torque_style: str = "arc"
    arc_sweep_rad: float = 1.5 * math.pi
    arc_segments: int = 32
    kinds: frozenset[WrenchKind] = field(default_factory=lambda: frozenset(WrenchKind))
    magnitude_floor_n: float = 1.0
    magnitude_floor_nm: float = 0.1
    show_labels: bool = False
    palette: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType(dict(FORCE_KIND_PALETTE))
    )
    scale_mode: ScaleMode = "fixed"
    reference_force_n: float | None = None
    reference_length_m: float = 0.5
    kind_scale: Mapping[WrenchKind, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    groups: frozenset[str] = DEFAULT_GROUPS
    max_torque_length_m: float | None = None

    def force_scale_for(self, kind: WrenchKind) -> float:
        """Effective arrow length in metres per newton for ``kind``.

        ``fixed`` uses ``force_scale_m_per_n``; ``body_weight`` and ``peak`` map
        ``reference_force_n`` to ``reference_length_m``. ``kind_scale`` then
        multiplies the result for that kind. Postcondition: finite and positive.
        """
        if self.scale_mode == "fixed":
            base = self.force_scale_m_per_n
        else:
            assert self.reference_force_n is not None  # validated in __post_init__
            base = self.reference_length_m / self.reference_force_n
        return base * self.kind_scale.get(kind, 1.0)

    def _validate_scaling(self) -> None:
        if self.scale_mode not in SCALE_MODES:
            raise ValueError(
                f"scale_mode must be one of {SCALE_MODES}, got {self.scale_mode!r}"
            )
        ref = self.reference_force_n
        if ref is not None:
            if not isinstance(ref, (int, float)) or isinstance(ref, bool):
                raise TypeError("reference_force_n must be numeric or None")
            if not math.isfinite(ref) or ref <= 0.0:
                raise ValueError(
                    f"reference_force_n must be finite and positive, got {ref}"
                )
        if self.scale_mode != "fixed" and ref is None:
            raise ValueError(
                f"scale_mode {self.scale_mode!r} requires reference_force_n "
                "(body weight in N for body_weight; series peak in N for peak)"
            )
        for kind, factor in self.kind_scale.items():
            if not isinstance(kind, WrenchKind):
                raise TypeError("kind_scale keys must be WrenchKind")
            if (
                not isinstance(factor, (int, float))
                or isinstance(factor, bool)
                or not math.isfinite(factor)
                or factor <= 0.0
            ):
                raise ValueError(
                    f"kind_scale[{kind.value}] must be finite and positive, got {factor!r}"
                )
        unknown = set(self.groups) - ALL_GROUPS
        if unknown:
            raise ValueError(
                f"Unknown groups {sorted(unknown)}; valid: {sorted(ALL_GROUPS)}"
            )

    def __post_init__(self) -> None:
        for name in (
            "force_scale_m_per_n",
            "torque_scale_m_per_nm",
            "min_length_m",
            "max_length_m",
            "shaft_radius_m",
            "head_length_ratio",
            "head_radius_ratio",
            "arc_sweep_rad",
            "magnitude_floor_n",
            "magnitude_floor_nm",
            "reference_length_m",
        ):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {val}")

        self._validate_scaling()

        if self.min_length_m >= self.max_length_m:
            raise ValueError(
                f"min_length_m ({self.min_length_m}) must be < max_length_m ({self.max_length_m})"
            )

        if self.max_torque_length_m is not None:
            if not isinstance(self.max_torque_length_m, (int, float)) or isinstance(
                self.max_torque_length_m, bool
            ):
                raise TypeError("max_torque_length_m must be numeric")
            if (
                not math.isfinite(self.max_torque_length_m)
                or self.max_torque_length_m <= 0.0
            ):
                raise ValueError("max_torque_length_m must be finite and positive")
            if self.min_length_m >= self.max_torque_length_m:
                raise ValueError(
                    f"min_length_m ({self.min_length_m}) must be < max_torque_length_m ({self.max_torque_length_m})"
                )

        if not isinstance(self.arc_segments, int) or isinstance(
            self.arc_segments, bool
        ):
            raise TypeError("arc_segments must be int")
        if self.arc_segments < 3:
            raise ValueError(f"arc_segments must be >= 3, got {self.arc_segments}")

        if self.torque_style not in ("arc", "axis_double_head"):
            raise ValueError(
                f"torque_style must be 'arc' or 'axis_double_head', got {self.torque_style!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize style configuration to JSON-safe dictionary."""
        return {
            "force_scale_m_per_n": self.force_scale_m_per_n,
            "torque_scale_m_per_nm": self.torque_scale_m_per_nm,
            "min_length_m": self.min_length_m,
            "max_length_m": self.max_length_m,
            "shaft_radius_m": self.shaft_radius_m,
            "head_length_ratio": self.head_length_ratio,
            "head_radius_ratio": self.head_radius_ratio,
            "torque_style": self.torque_style,
            "arc_sweep_rad": self.arc_sweep_rad,
            "arc_segments": self.arc_segments,
            "kinds": sorted(k.value for k in self.kinds),
            "magnitude_floor_n": self.magnitude_floor_n,
            "magnitude_floor_nm": self.magnitude_floor_nm,
            "show_labels": self.show_labels,
            "palette": dict(self.palette),
            "scale_mode": self.scale_mode,
            "reference_force_n": self.reference_force_n,
            "reference_length_m": self.reference_length_m,
            "kind_scale": {
                k.value: v
                for k, v in sorted(self.kind_scale.items(), key=lambda kv: kv[0].value)
            },
            "groups": sorted(self.groups),
            "max_torque_length_m": self.max_torque_length_m,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ForceGlyphStyle:
        """Construct style from dictionary, rejecting unknown keys."""
        valid_keys = {
            "force_scale_m_per_n",
            "torque_scale_m_per_nm",
            "min_length_m",
            "max_length_m",
            "shaft_radius_m",
            "head_length_ratio",
            "head_radius_ratio",
            "torque_style",
            "arc_sweep_rad",
            "arc_segments",
            "kinds",
            "magnitude_floor_n",
            "magnitude_floor_nm",
            "show_labels",
            "palette",
            "scale_mode",
            "reference_force_n",
            "reference_length_m",
            "kind_scale",
            "groups",
            "max_torque_length_m",
        }
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown fields in ForceGlyphStyle: {sorted(unknown)}")

        kinds_raw = data.get("kinds")
        kinds = (
            frozenset(WrenchKind(k) for k in kinds_raw)
            if kinds_raw is not None
            else frozenset(WrenchKind)
        )
        palette_raw = data.get("palette")
        palette = (
            MappingProxyType(dict(palette_raw))
            if palette_raw is not None
            else MappingProxyType(dict(FORCE_KIND_PALETTE))
        )

        kwargs: dict[str, Any] = {}
        for key in valid_keys - {"kinds", "palette", "kind_scale", "groups"}:
            if key in data:
                kwargs[key] = data[key]
        kwargs["kinds"] = kinds
        kwargs["palette"] = palette
        if "kind_scale" in data:
            kwargs["kind_scale"] = MappingProxyType(
                {WrenchKind(k): v for k, v in dict(data["kind_scale"]).items()}
            )
        if "groups" in data:
            kwargs["groups"] = frozenset(data["groups"])
        return cls(**kwargs)


@dataclass(frozen=True)
class ArrowGlyph:
    """Renderer-neutral arrow specification for force or axis vectors."""

    label: str
    kind: WrenchKind
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
        """Convert arrow glyph to JSON dictionary."""
        return {
            "label": self.label,
            "kind": self.kind.value,
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
        """Reconstruct arrow glyph from JSON dictionary."""
        return cls(
            label=str(d["label"]),
            kind=WrenchKind(d["kind"]),
            tail_m=validate_vec3(d["tail_m"], "tail_m"),
            tip_m=validate_vec3(d["tip_m"], "tip_m"),
            head_base_m=validate_vec3(d["head_base_m"], "head_base_m"),
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
    """Renderer-neutral circular arc specification for torque visualization."""

    label: str
    kind: WrenchKind
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
        """Convert torque arc glyph to JSON dictionary."""
        return {
            "label": self.label,
            "kind": self.kind.value,
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
        """Reconstruct torque arc glyph from JSON dictionary."""
        return cls(
            label=str(d["label"]),
            kind=WrenchKind(d["kind"]),
            center_m=validate_vec3(d["center_m"], "center_m"),
            axis_unit=validate_vec3(d["axis_unit"], "axis_unit"),
            radius_m=float(d["radius_m"]),
            polyline_m=tuple(
                validate_vec3(pt, f"polyline_m[{i}]")
                for i, pt in enumerate(d["polyline_m"])
            ),
            head_tip_m=validate_vec3(d["head_tip_m"], "head_tip_m"),
            head_base_m=validate_vec3(d["head_base_m"], "head_base_m"),
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
    """Renderer-neutral legend specification containing scales and metadata."""

    force_reference_n: float | None = None
    force_reference_length_m: float | None = None
    torque_reference_nm: float | None = None
    torque_reference_radius_m: float | None = None
    kinds_present: tuple[Any, ...] = ()
    unavailable_labels: tuple[str, ...] = ()
    engine: str = ""
    source_labels: tuple[str, ...] = ()
    scale_mode: str = "fixed"
    clamped_labels: tuple[str, ...] = ()
    #: How the left/right grip split was obtained (``None`` without grip data).
    grip_split_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert legend spec to JSON dictionary."""
        extra = (
            {"grip_split_method": self.grip_split_method}
            if self.grip_split_method is not None
            else {}
        )
        return {
            **extra,
            "force_reference_n": self.force_reference_n,
            "force_reference_length_m": self.force_reference_length_m,
            "torque_reference_nm": self.torque_reference_nm,
            "torque_reference_radius_m": self.torque_reference_radius_m,
            "kinds_present": [
                k.value if hasattr(k, "value") else str(k) for k in self.kinds_present
            ],
            "unavailable_labels": list(self.unavailable_labels),
            "engine": self.engine,
            "source_labels": list(self.source_labels),
            "scale_mode": self.scale_mode,
            "clamped_labels": list(self.clamped_labels),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LegendSpec:
        """Reconstruct legend spec from JSON dictionary."""
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
            kinds_present=tuple(WrenchKind(k) for k in d["kinds_present"]),
            unavailable_labels=tuple(str(lbl) for lbl in d["unavailable_labels"]),
            engine=str(d["engine"]),
            source_labels=tuple(str(src) for src in d["source_labels"]),
            scale_mode=str(d.get("scale_mode", "fixed")),
            clamped_labels=tuple(str(c) for c in d.get("clamped_labels", ())),
            grip_split_method=(
                str(d["grip_split_method"])
                if d.get("grip_split_method") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class GlyphSet:
    """Complete collection of generated glyphs for a single frame."""

    time_s: float
    arrows: tuple[ArrowGlyph, ...]
    torque_arcs: tuple[TorqueArcGlyph, ...]
    legend: LegendSpec
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize glyph set to canonical JSON dictionary."""
        return {
            "schema_version": self.schema_version,
            "time_s": self.time_s,
            "arrows": [a.to_dict() for a in self.arrows],
            "torque_arcs": [t.to_dict() for t in self.torque_arcs],
            "legend": self.legend.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GlyphSet:
        """Construct glyph set from dictionary, rejecting invalid schemas/keys."""
        valid_keys = {"schema_version", "time_s", "arrows", "torque_arcs", "legend"}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown keys in GlyphSet: {sorted(unknown)}")

        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}"
            )

        time_s = float(data["time_s"])
        arrows = tuple(ArrowGlyph.from_dict(a) for a in data["arrows"])
        torque_arcs = tuple(TorqueArcGlyph.from_dict(t) for t in data["torque_arcs"])
        legend = LegendSpec.from_dict(data["legend"])
        return cls(time_s=time_s, arrows=arrows, torque_arcs=torque_arcs, legend=legend)


def _compute_arc_basis(
    axis_unit: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Determine deterministic orthonormal basis (u, v) in plane normal to axis_unit."""
    ax, ay, az = axis_unit
    dots = (abs(ax), abs(ay), abs(az))
    min_dot = min(dots)

    if abs(dots[0] - min_dot) <= 1e-12:
        e = (1.0, 0.0, 0.0)
    elif abs(dots[1] - min_dot) <= 1e-12:
        e = (0.0, 1.0, 0.0)
    else:
        e = (0.0, 0.0, 1.0)

    e_dot_a = e[0] * ax + e[1] * ay + e[2] * az
    proj = (e[0] - e_dot_a * ax, e[1] - e_dot_a * ay, e[2] - e_dot_a * az)
    norm = math.sqrt(proj[0] ** 2 + proj[1] ** 2 + proj[2] ** 2)
    u = (proj[0] / norm, proj[1] / norm, proj[2] / norm)

    v = (
        ay * u[2] - az * u[1],
        az * u[0] - ax * u[2],
        ax * u[1] - ay * u[0],
    )
    return u, v


def clamped_tip_shift(
    tip: Sequence[float], base_center: Sequence[float], ratio: float = 0.6
) -> tuple[float, ...]:
    """Translation vector for a second, trailing tip marking a clamped glyph.

    ADR-0052 draws force arrows clamped at ``max_length_m`` with a double
    chevron: a second head, identical in shape, shifted backward
    along the shaft by ``ratio`` of the tip-to-base-center distance. Every
    renderer applies this single offset to its own head representation (a 2D
    pixel polygon for OpenCV/QPainter, a 3D cone apex/base for matplotlib)
    instead of re-deriving the geometry (DRY).

    Precondition: ``tip`` and ``base_center`` have equal, nonzero length and
    ``ratio`` is finite and positive.
    Postcondition: the returned vector has the same dimensionality as the
    inputs and points from ``tip`` toward ``base_center``.
    """
    if len(tip) != len(base_center) or len(tip) == 0:
        raise ValueError(
            "tip and base_center must have equal, nonzero length; got "
            f"{len(tip)} and {len(base_center)}"
        )
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"ratio must be finite and positive, got {ratio}")
    return tuple((base_center[i] - tip[i]) * ratio for i in range(len(tip)))


def _build_force_arrow(
    wrench: OverlayWrench,
    style: ForceGlyphStyle,
    color_hex: str,
) -> ArrowGlyph:
    """Build ArrowGlyph from a force vector according to style specification."""
    assert wrench.force_n is not None
    fx, fy, fz = wrench.force_n
    mag = math.sqrt(fx * fx + fy * fy + fz * fz)
    f_hat = (fx / mag, fy / mag, fz / mag)

    raw_len = mag * style.force_scale_for(wrench.kind)
    clamped = raw_len > style.max_length_m
    length = max(style.min_length_m, min(raw_len, style.max_length_m))

    px, py, pz = wrench.point_m
    tip = (px + length * f_hat[0], py + length * f_hat[1], pz + length * f_hat[2])
    head_len = style.head_length_ratio * length
    head_base = (
        tip[0] - head_len * f_hat[0],
        tip[1] - head_len * f_hat[1],
        tip[2] - head_len * f_hat[2],
    )

    shaft_radius = style.shaft_radius_m
    head_radius = shaft_radius * style.head_radius_ratio
    rgba = _hex_to_rgba(color_hex)

    return ArrowGlyph(
        label=wrench.label,
        kind=wrench.kind,
        tail_m=wrench.point_m,
        tip_m=tip,
        head_base_m=head_base,
        shaft_radius_m=shaft_radius,
        head_radius_m=head_radius,
        rgba=rgba,
        magnitude=mag,
        units="N",
        clamped=clamped,
    )


def _build_torque_arc(
    wrench: OverlayWrench,
    style: ForceGlyphStyle,
    color_hex: str,
) -> TorqueArcGlyph:
    """Build TorqueArcGlyph from a torque vector according to style specification."""
    assert wrench.torque_nm is not None
    tx, ty, tz = wrench.torque_nm
    mag = math.sqrt(tx * tx + ty * ty + tz * tz)
    a_hat = (tx / mag, ty / mag, tz / mag)

    raw_len = mag * style.torque_scale_m_per_nm
    max_len = (
        style.max_torque_length_m
        if style.max_torque_length_m is not None
        else style.max_length_m
    )
    clamped = raw_len > max_len
    eff_len = max(style.min_length_m, min(raw_len, max_len))
    radius = eff_len / 2.0

    u, v = _compute_arc_basis(a_hat)
    cx, cy, cz = wrench.point_m
    n_seg = style.arc_segments
    sweep = style.arc_sweep_rad

    pts: list[tuple[float, float, float]] = []
    for i in range(n_seg + 1):
        theta = sweep * (i / n_seg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x = cx + radius * (cos_t * u[0] + sin_t * v[0])
        y = cy + radius * (cos_t * u[1] + sin_t * v[1])
        z = cz + radius * (cos_t * u[2] + sin_t * v[2])
        pts.append((x, y, z))

    head_tip = pts[-1]
    cos_last = math.cos(sweep)
    sin_last = math.sin(sweep)
    tangent = (
        -sin_last * u[0] + cos_last * v[0],
        -sin_last * u[1] + cos_last * v[1],
        -sin_last * u[2] + cos_last * v[2],
    )
    head_len = style.head_length_ratio * eff_len
    head_base = (
        head_tip[0] - head_len * tangent[0],
        head_tip[1] - head_len * tangent[1],
        head_tip[2] - head_len * tangent[2],
    )
    rgba = _hex_to_rgba(color_hex)

    return TorqueArcGlyph(
        label=wrench.label,
        kind=wrench.kind,
        center_m=wrench.point_m,
        axis_unit=a_hat,
        radius_m=radius,
        polyline_m=tuple(pts),
        head_tip_m=head_tip,
        head_base_m=head_base,
        rgba=rgba,
        magnitude=mag,
        units="N*m",
        clamped=clamped,
    )


def _group_visible(label: str, groups: frozenset[str], has_aggregate: bool) -> bool:
    """Whether ``label`` passes the group toggles.

    Raw per-sphere contacts are shown regardless of the toggle when the frame has
    no aggregated GRF labels, so engines that predate GCV-1 keep their arrows.
    """
    group = label_group(label)
    if group is None:
        return True
    if group == "contact_points" and not has_aggregate:
        return True
    return group in groups


def _grip_annotations(
    frame: ForceTorqueFrame, style: ForceGlyphStyle, has_aggregate: bool
) -> tuple[str | None, list[str]]:
    """Grip split label and visible-group unavailable labels from frame metadata.

    ``grip_frame`` records ``grip_split_method`` and
    ``grip_unavailable_labels`` (labels it could not emit).  They are shown only
    while the grip kind is enabled, and only for groups that are toggled on.
    """
    meta = frame.metadata
    if WrenchKind.GRIP not in style.kinds or "grip_split_method" not in meta:
        return None, []
    missing = [
        str(lbl)
        for lbl in meta.get("grip_unavailable_labels", ())
        if _group_visible(str(lbl), style.groups, has_aggregate)
    ]
    return str(meta["grip_split_method"]), missing


def build_glyphs(frame: ForceTorqueFrame, style: ForceGlyphStyle) -> GlyphSet:
    """Pure, deterministic builder turning a ForceTorqueFrame into renderer-neutral glyphs."""
    sorted_wrenches = sorted(frame.wrenches, key=lambda w: w.label)
    has_aggregate = any(
        label_group(w.label) in ("per_foot", "net") for w in sorted_wrenches
    )
    arrows: list[ArrowGlyph] = []
    torque_arcs: list[TorqueArcGlyph] = []
    unavailable: list[str] = []
    kinds_present_set: set[WrenchKind] = set()
    source_labels_set: set[str] = set()

    for w in sorted_wrenches:
        if w.kind not in style.kinds:
            continue
        if not _group_visible(w.label, style.groups, has_aggregate):
            continue
        source_labels_set.add(w.source)
        color = label_variant_hex(
            w.label,
            style.palette.get(w.kind.value, style.palette.get(str(w.kind), "#888888")),
        )

        if w.force_n is None:
            unavailable.append(w.label)
        else:
            mag = math.sqrt(sum(c * c for c in w.force_n))
            if mag >= style.magnitude_floor_n:
                arrow = _build_force_arrow(w, style, color)
                arrows.append(arrow)
                kinds_present_set.add(w.kind)

        if w.torque_nm is None:
            if w.label not in unavailable:
                unavailable.append(w.label)
        else:
            mag = math.sqrt(sum(c * c for c in w.torque_nm))
            if mag >= style.magnitude_floor_nm:
                arc = _build_torque_arc(w, style, color)
                torque_arcs.append(arc)
                kinds_present_set.add(w.kind)

    grip_split, grip_missing = _grip_annotations(frame, style, has_aggregate)
    unavailable.extend(x for x in grip_missing if x not in unavailable)

    # Reference values for legend
    force_ref_n: float | None = None
    force_ref_len: float | None = None
    if arrows:
        median_f = float(np.median([a.magnitude for a in arrows]))
        if style.scale_mode == "fixed":
            force_ref_n = _nice_number(median_f)
            force_ref_len = force_ref_n * style.force_scale_m_per_n
        else:
            force_ref_n = style.reference_force_n
            force_ref_len = style.reference_length_m

    torque_ref_nm: float | None = None
    torque_ref_rad: float | None = None
    if torque_arcs:
        median_t = float(np.median([t.magnitude for t in torque_arcs]))
        torque_ref_nm = _nice_number(median_t)
        torque_ref_rad = (torque_ref_nm * style.torque_scale_m_per_nm) / 2.0

    legend = LegendSpec(
        force_reference_n=force_ref_n,
        force_reference_length_m=force_ref_len,
        torque_reference_nm=torque_ref_nm,
        torque_reference_radius_m=torque_ref_rad,
        kinds_present=tuple(sorted(kinds_present_set, key=lambda k: k.value)),
        unavailable_labels=tuple(sorted(unavailable)),
        engine=frame.engine,
        source_labels=tuple(sorted(source_labels_set)),
        scale_mode=style.scale_mode,
        clamped_labels=tuple(
            sorted(
                {a.label for a in arrows if a.clamped}
                | {t.label for t in torque_arcs if t.clamped}
            )
        ),
        grip_split_method=grip_split,
    )

    return GlyphSet(
        time_s=frame.time_s,
        arrows=tuple(arrows),
        torque_arcs=tuple(torque_arcs),
        legend=legend,
    )


def scale_for_view(glyphs: GlyphSet, scale_factor: float) -> GlyphSet:
    """Rescale all coordinates and spatial dimensions while preserving magnitudes."""
    if scale_factor <= 0.0 or not math.isfinite(scale_factor):
        raise ValueError(
            f"scale_factor must be positive and finite, got {scale_factor}"
        )

    s = scale_factor
    scaled_arrows = tuple(
        ArrowGlyph(
            label=a.label,
            kind=a.kind,
            tail_m=(a.tail_m[0] * s, a.tail_m[1] * s, a.tail_m[2] * s),
            tip_m=(a.tip_m[0] * s, a.tip_m[1] * s, a.tip_m[2] * s),
            head_base_m=(
                a.head_base_m[0] * s,
                a.head_base_m[1] * s,
                a.head_base_m[2] * s,
            ),
            shaft_radius_m=a.shaft_radius_m * s,
            head_radius_m=a.head_radius_m * s,
            rgba=a.rgba,
            magnitude=a.magnitude,
            units=a.units,
            clamped=a.clamped,
        )
        for a in glyphs.arrows
    )

    scaled_arcs = tuple(
        TorqueArcGlyph(
            label=t.label,
            kind=t.kind,
            center_m=(t.center_m[0] * s, t.center_m[1] * s, t.center_m[2] * s),
            axis_unit=t.axis_unit,
            radius_m=t.radius_m * s,
            polyline_m=tuple((pt[0] * s, pt[1] * s, pt[2] * s) for pt in t.polyline_m),
            head_tip_m=(t.head_tip_m[0] * s, t.head_tip_m[1] * s, t.head_tip_m[2] * s),
            head_base_m=(
                t.head_base_m[0] * s,
                t.head_base_m[1] * s,
                t.head_base_m[2] * s,
            ),
            rgba=t.rgba,
            magnitude=t.magnitude,
            units=t.units,
            clamped=t.clamped,
        )
        for t in glyphs.torque_arcs
    )

    old_leg = glyphs.legend
    scaled_legend = LegendSpec(
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
        scale_mode=old_leg.scale_mode,
        clamped_labels=old_leg.clamped_labels,
    )

    return GlyphSet(
        time_s=glyphs.time_s,
        arrows=scaled_arrows,
        torque_arcs=scaled_arcs,
        legend=scaled_legend,
        schema_version=glyphs.schema_version,
    )
