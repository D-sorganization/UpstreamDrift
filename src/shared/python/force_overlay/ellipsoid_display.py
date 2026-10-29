"""Authoritative ellipsoid display builder, scaling, legend, and HUD formatting (CAP-5a, #14038, #14044)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Iterable, Literal, Sequence

import numpy as np

from src.shared.python.force_overlay.palette import hand_ellipsoid_rgba

__all__ = [
    "EllipsoidDisplayBuilder",
    "EllipsoidDisplayGlyph",
    "EllipsoidDisplayScale",
    "LegendScaleBar",
    "format_hud_capability",
]

_TARGET_RADIUS_M: Final[float] = 0.09
_SINGULAR_THRESH: Final[float] = 1e-9
_KIND_UNITS: Final[dict[str, str]] = {
    "mobility": "kg^-1/2",
    "effective-mass": "kg^1/2",
    "acceleration-capacity": "m/s^2",
    "force-capacity": "N",
}
_CAPACITY_KINDS: Final[frozenset[str]] = frozenset(
    {"acceleration-capacity", "force-capacity"}
)


@dataclass(frozen=True)
class EllipsoidDisplayScale:
    """Frozen linear scale mapping physical semi-axes to display metres."""

    scale_m_per_unit: float
    kind: str
    mode: str
    unit: str
    p95_semi_axis: float | None = None
    target_radius_m: float = _TARGET_RADIUS_M

    def __post_init__(self) -> None:
        if isinstance(self.scale_m_per_unit, bool) or not isinstance(
            self.scale_m_per_unit, (int, float)
        ):
            raise TypeError("scale_m_per_unit must be numeric")
        if not math.isfinite(self.scale_m_per_unit) or self.scale_m_per_unit <= 0.0:
            raise ValueError(
                f"scale_m_per_unit must be finite and positive, got {self.scale_m_per_unit!r}"
            )
        for attr in ("kind", "mode", "unit"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{attr} must be a non-empty string")

    @classmethod
    def compute(
        cls,
        semi_axes_series: Iterable[Sequence[float]],
        target_radius_m: float = _TARGET_RADIUS_M,
        kind: str = "force-capacity",
        mode: str = "auto",
    ) -> EllipsoidDisplayScale:
        unit = _KIND_UNITS.get(kind, "N")
        if not (math.isfinite(target_radius_m) and target_radius_m > 0.0):
            raise ValueError(
                f"target_radius_m must be finite and positive, got {target_radius_m!r}"
            )
        if mode.startswith("fixed:"):
            try:
                scale_val = float(mode.split(":", 1)[1])
            except (ValueError, IndexError):
                raise ValueError(
                    f"fixed: mode requires numeric scale, got {mode!r}"
                ) from None
            if not math.isfinite(scale_val) or scale_val <= 0.0:
                raise ValueError(
                    f"fixed: mode requires positive finite scale, got {scale_val!r}"
                )
            return cls(scale_val, kind, mode, unit, target_radius_m=target_radius_m)
        if mode != "auto":
            raise ValueError(f"mode must be 'auto' or 'fixed:<scale>', got {mode!r}")

        valid = [
            float(v)
            for f in semi_axes_series
            for v in f
            if math.isfinite(float(v)) and float(v) > _SINGULAR_THRESH
        ]
        if not valid:
            return cls(
                target_radius_m, kind, mode, unit, target_radius_m=target_radius_m
            )
        p95 = float(np.percentile(valid, 95))
        scale = target_radius_m / p95 if p95 > _SINGULAR_THRESH else target_radius_m
        return cls(
            scale,
            kind,
            mode,
            unit,
            p95_semi_axis=p95,
            target_radius_m=target_radius_m,
        )


@dataclass(frozen=True)
class LegendScaleBar:
    bar_length_m: float
    physical_value: float
    unit: str
    text: str

    @classmethod
    def from_scale(
        cls, scale: EllipsoidDisplayScale, bar_length_m: float = 0.01
    ) -> LegendScaleBar:
        if not (math.isfinite(bar_length_m) and bar_length_m > 0.0):
            raise ValueError("bar_length_m must be finite and positive")
        phys = bar_length_m / scale.scale_m_per_unit
        bar_cm = bar_length_m * 100.0
        cm_lbl = (
            f"{int(round(bar_cm))} cm"
            if math.isclose(bar_cm, round(bar_cm), abs_tol=1e-5)
            else f"{bar_cm:.1f} cm"
        )
        return cls(
            bar_length_m, phys, scale.unit, f"{cm_lbl} = {phys:.1f} {scale.unit}"
        )


@dataclass(frozen=True)
class EllipsoidDisplayGlyph:
    centre: tuple[float, float, float]
    axes: tuple[tuple[float, float, float], ...]
    radii: tuple[float, float, float]
    radii_physical: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    style: Literal["shell", "wireframe", "dashed"]
    clamped: bool
    hand: str
    kind: str
    label: str = ""


@dataclass(frozen=True)
class EllipsoidDisplayBuilder:
    scale: EllipsoidDisplayScale
    min_radius_m: float = 0.005
    max_radius_m: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.scale, EllipsoidDisplayScale):
            raise TypeError("scale must be an EllipsoidDisplayScale")
        if not (0.0 < self.min_radius_m < self.max_radius_m):
            raise ValueError("min_radius_m must be > 0 and < max_radius_m")

    def build_glyph(
        self,
        centre: Sequence[float],
        axes: Sequence[Sequence[float]] | np.ndarray,
        radii_physical: Sequence[float],
        hand: str,
        label: str = "",
    ) -> EllipsoidDisplayGlyph:
        raw = [float(r) * self.scale.scale_m_per_unit for r in radii_physical]
        is_clamped = any(r < self.min_radius_m or r > self.max_radius_m for r in raw)
        clamped = tuple(
            float(np.clip(r, self.min_radius_m, self.max_radius_m)) for r in raw
        )
        style: Literal["shell", "wireframe", "dashed"] = (
            "dashed"
            if is_clamped
            else ("wireframe" if self.scale.kind in _CAPACITY_KINDS else "shell")
        )
        rows = np.asarray(axes, dtype=float)
        ax: tuple[tuple[float, float, float], ...] = (
            (float(rows[0, 0]), float(rows[0, 1]), float(rows[0, 2])),
            (float(rows[1, 0]), float(rows[1, 1]), float(rows[1, 2])),
            (float(rows[2, 0]), float(rows[2, 1]), float(rows[2, 2])),
        )
        c = (float(centre[0]), float(centre[1]), float(centre[2]))
        rp = (
            float(radii_physical[0]),
            float(radii_physical[1]),
            float(radii_physical[2]),
        )
        return EllipsoidDisplayGlyph(
            c,
            ax,
            (clamped[0], clamped[1], clamped[2]),
            rp,
            hand_ellipsoid_rgba(hand),
            style,
            is_clamped,
            hand,
            self.scale.kind,
            label,
        )


def format_hud_capability(
    hand: str,
    kind: str,
    radii_physical: Sequence[float],
    unit: str,
    cond_num: float | None = None,
) -> str:
    r = [float(x) for x in radii_physical]
    if cond_num is None:
        min_r = min(r)
        cond_num = max(r) / min_r if min_r > _SINGULAR_THRESH else math.inf
    c_str = f"{cond_num:.1f}" if math.isfinite(cond_num) else "inf"
    return f"{hand.capitalize()} {kind}: [{r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}] {unit} (cond={c_str})"
