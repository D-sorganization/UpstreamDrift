"""Display scale grouping, nested visibility and a minimum semi-axis (#15150).

The export draws a mobility and a force (effective-mass) ellipsoid at each
point. One display scale per kind lets the largest point (the clubhead) shrink
every other point, and the larger glyph of two at one centre hides the smaller.
Options: ``ellipsoid_visibility_settings.json`` (``--ellipsoid-visibility-settings``).

* ``scale_grouping`` ``per_kind``: one scale per kind over all points (the
  original rule, on-screen sizes comparable between points). ``per_point``: each
  point has its own scale, so its largest radius over the swing maps to
  ``target_radius_m``. Sizes are then **not comparable between points**; the
  panel says so (``per_point_note``).
* ``outer_alpha``: alpha of the larger of two ellipsoids at one centre (1 = off).
* ``min_semi_axis_m``: displayed semi-axis floor (0 = off). A floored axis is
  **not to scale**: that glyph is a visibility aid along the floored axis.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

from src.shared.python.force_overlay.ellipsoid_display import (
    EllipsoidDisplayBuilder,
    EllipsoidDisplayScale,
)
from src.shared.python.force_overlay.glyphs import EllipsoidGlyph

SETTINGS_PATH = Path(__file__).with_name("ellipsoid_visibility_settings.json")
GROUPINGS = ("per_kind", "per_point")
#: the display builder's upper clamp (m); a floor must stay below it
MAX_FLOOR_M = EllipsoidDisplayBuilder.max_radius_m
_CENTRE_DECIMALS = 6


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class VisibilitySettings:
    """Validated ``ellipsoid_visibility_settings.json`` (``ValueError``/``TypeError``)."""

    scale_grouping: str
    target_radius_m: float
    outer_alpha: float
    min_semi_axis_m: float
    per_point_note: str

    def __post_init__(self) -> None:
        if self.scale_grouping not in GROUPINGS:
            raise ValueError(f"scale_grouping must be in {GROUPINGS}")
        if not (
            math.isfinite(_number("target_radius_m", self.target_radius_m))
            and self.target_radius_m > 0.0
        ):
            raise ValueError("target_radius_m must be finite and positive")
        if not 0.0 < _number("outer_alpha", self.outer_alpha) <= 1.0:
            raise ValueError("outer_alpha must lie in (0, 1]")
        if not 0.0 <= _number("min_semi_axis_m", self.min_semi_axis_m) < MAX_FLOOR_M:
            raise ValueError(f"min_semi_axis_m must lie in [0, {MAX_FLOOR_M})")
        if not isinstance(self.per_point_note, str):
            raise TypeError("per_point_note must be a string")

    def builder(self, scale: EllipsoidDisplayScale) -> EllipsoidDisplayBuilder:
        """Display builder of ``scale``, flooring semi-axes at ``min_semi_axis_m``."""
        if self.min_semi_axis_m == 0.0:
            return EllipsoidDisplayBuilder(scale)
        return EllipsoidDisplayBuilder(scale, min_radius_m=self.min_semi_axis_m)


def load_visibility_settings(path: Path | None = None) -> VisibilitySettings:
    """Read the settings file (default: beside this module)."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    raw.pop("sources", None)
    unknown = set(raw) - set(VisibilitySettings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown visibility settings: {sorted(unknown)}")
    return VisibilitySettings(**raw)


def group_scales(
    series: Mapping[str, Sequence[Sequence[float]]],
    kind: str,
    mode: str,
    settings: VisibilitySettings,
) -> dict[str, EllipsoidDisplayScale]:
    """Display scale of ``kind`` per point name of ``series`` (physical semi-axes).

    ``per_kind``: every point shares one scale over all of ``series``;
    ``per_point``: a point's scale is computed from its own series alone.
    """
    target = settings.target_radius_m

    def compute(rows: Any) -> EllipsoidDisplayScale:
        return EllipsoidDisplayScale.compute(
            rows, target_radius_m=target, kind=kind, mode=mode
        )

    if settings.scale_grouping == "per_point":
        return {name: compute(rows) for name, rows in series.items()}
    shared = compute([r for rows in series.values() for r in rows])
    return dict.fromkeys(series, shared)


def fade_outer(
    glyphs: Sequence[EllipsoidGlyph], outer_alpha: float
) -> tuple[EllipsoidGlyph, ...]:
    """``glyphs`` with the largest glyph of each shared centre at ``outer_alpha``.

    Order is kept; a glyph alone at its centre is unchanged, and
    ``outer_alpha == 1`` changes nothing.
    """
    out = list(glyphs)
    if outer_alpha >= 1.0:
        return tuple(out)
    groups: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for i, g in enumerate(glyphs):
        groups[tuple(round(c, _CENTRE_DECIMALS) for c in g.centre)].append(i)
    for members in groups.values():
        if len(members) > 1:
            i = max(members, key=lambda k: math.prod(glyphs[k].radii))
            out[i] = replace(glyphs[i], rgba=(*glyphs[i].rgba[:3], outer_alpha))
    return tuple(out)


def panel_note(settings: VisibilitySettings) -> str:
    """Panel footer: the note when sizes are scaled per point, else empty."""
    return settings.per_point_note if settings.scale_grouping == "per_point" else ""
