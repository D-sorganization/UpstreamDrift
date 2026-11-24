"""Tip-to-tail drift (ZTCF) and control (ZVCF) force arrows (#15685, K5).

A force that splits pointwise into a drift part (the zero-torque
counterfactual, ``tau = 0``) and a control part (the zero-velocity
counterfactual: the applied torques alone) is drawn as two arrows placed tip
to tail, then the total from the application point to the last tip. The
picture reads directly as drift + control = total. The split itself is
:func:`src.shared.python.simulation_backends.constrained_ztcf_zvcf.drift_and_control_reactions`;
this module only draws it. Engine-free.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.force_overlay.glyphs import ArrowGlyph, ForceGlyphStyle
from src.shared.python.force_overlay.palette import hex_to_rgba

__all__ = [
    "PART_VIEWS",
    "QUANTITIES",
    "TipToTailStyle",
    "colour_key",
    "part_arrows",
    "tip_to_tail_arrows",
]

#: What one render shows of the split (#15150 Track Z, Z3): drift and control
#: tip to tail with the total, the ZTCF (drift) alone, the ZVCF (control)
#: alone, or the total alone.
PART_VIEWS = ("tip_to_tail", "ztcf", "zvcf", "total")
#: The force (N, at the point) or the torque (N m, an axial vector).
QUANTITIES = ("force", "torque")
_PARTS_OF = {
    "tip_to_tail": ("drift", "control", "total"),
    "ztcf": ("drift",),
    "zvcf": ("control",),
    "total": ("total",),
}

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class TipToTailStyle:
    """Colours, labels and the total arrow's relative thickness.

    Defaults are Okabe-Ito blue (drift), vermillion (control) and near-white
    (total), distinct from the force-kind palette.
    """

    drift_hex: str = "#0072B2"
    control_hex: str = "#D55E00"
    total_hex: str = "#F2F2F2"
    drift_label: str = "ztcf:drift"
    control_label: str = "zvcf:control"
    total_label: str = "total"
    total_radius_ratio: float = 0.5
    #: colour key texts; gravity is passive, so it is named in the drift only
    drift_key: str = "ZTCF drift: gravity, momentum, passive"
    control_key: str = "ZVCF control: applied torques alone"
    total_key: str = "Total (ZTCF + ZVCF)"

    def __post_init__(self) -> None:
        for name in ("drift_hex", "control_hex", "total_hex"):
            if not _HEX.match(getattr(self, name)):
                raise ValueError(f"{name} must be a #RRGGBB colour")
        if not 0.0 < self.total_radius_ratio <= 1.0:
            raise ValueError("total_radius_ratio must lie in (0, 1]")


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have length 3")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _point(vec: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (float(c) for c in vec)
    return (x, y, z)


def _arrow(
    label: str,
    tail: np.ndarray,
    force: np.ndarray,
    scale: float,
    style: ForceGlyphStyle,
    rgba: tuple[float, float, float, float],
    radius: float,
    units: str = "N",
) -> ArrowGlyph:
    vector = scale * force
    tip = tail + vector
    head = tip - style.head_length_ratio * vector
    return ArrowGlyph(
        label=label,
        kind=WrenchKind.GRIP,
        tail_m=_point(tail),
        tip_m=_point(tip),
        head_base_m=_point(head),
        shaft_radius_m=radius,
        head_radius_m=radius * style.head_radius_ratio,
        rgba=rgba,
        magnitude=float(np.linalg.norm(force)),
        units=units,
        clamped=False,
    )


def tip_to_tail_arrows(
    point_m: Sequence[float],
    drift_n: Sequence[float],
    control_n: Sequence[float],
    style: ForceGlyphStyle,
    colours: TipToTailStyle | None = None,
) -> tuple[ArrowGlyph, ...]:
    """Drift arrow at ``point_m``, control arrow at its tip, and the total.

    All three use one length scale (``style.force_scale_for(GRIP)``) so their
    lengths add like the forces. A part below ``style.magnitude_floor_n`` is
    left out; the total is always drawn when it is above the floor.

    Raises:
        ValueError: a vector is not finite length 3, or an arrow would exceed
            ``style.max_length_m`` (it cannot be clamped without breaking the
            tip-to-tail sum; use a smaller scale).
    """
    return part_arrows(point_m, drift_n, control_n, style, colours=colours)


def _checked_view(view: str, quantity: str) -> None:
    if view not in PART_VIEWS:
        raise ValueError(f"view must be one of {PART_VIEWS}, got {view!r}")
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}, got {quantity!r}")


def _scale(style: ForceGlyphStyle, quantity: str) -> tuple[float, str]:
    """Length per unit and the unit; the grip-kind gain applies to both."""
    if quantity == "force":
        return style.force_scale_for(WrenchKind.GRIP), "N"
    gain = style.kind_scale.get(WrenchKind.GRIP, 1.0)
    return style.torque_scale_m_per_nm * gain, "N m"


def part_arrows(
    point_m: Sequence[float],
    drift: Sequence[float],
    control: Sequence[float],
    style: ForceGlyphStyle,
    view: str = "tip_to_tail",
    quantity: str = "force",
    colours: TipToTailStyle | None = None,
    prefix: str = "",
) -> tuple[ArrowGlyph, ...]:
    """The arrows of one ``view`` (:data:`PART_VIEWS`) of a drift/control split.

    ``tip_to_tail`` draws drift at ``point_m``, control at its tip and the
    total; the other views draw their one vector from ``point_m``. A torque
    (``quantity="torque"``) is an axial vector on the torque scale, in N m.
    ``prefix`` (a hand) is put before each label. A part below
    ``style.magnitude_floor_n`` is left out.

    Raises:
        ValueError: unknown ``view`` or ``quantity``, a vector that is not
            finite length 3, or an arrow longer than ``style.max_length_m``.
    """
    _checked_view(view, quantity)
    look = colours or TipToTailStyle()
    point = _vec3(point_m, "point_m")
    vectors = {"drift": _vec3(drift, "drift_n"), "control": _vec3(control, "control_n")}
    vectors["total"] = vectors["drift"] + vectors["control"]
    scale, units = _scale(style, quantity)
    parts = _PARTS_OF[view]
    reach = sum(float(np.linalg.norm(vectors[p])) for p in parts if p != "total")
    total = float(np.linalg.norm(vectors["total"])) if "total" in parts else 0.0
    longest = scale * max(reach, total)
    if longest > style.max_length_m:
        raise ValueError(
            f"tip-to-tail arrows reach {longest:.3f} m, over max_length_m "
            f"{style.max_length_m}; lower the force scale"
        )
    tails = {"drift": point, "control": point, "total": point}
    if view == "tip_to_tail":
        tails["control"] = point + scale * vectors["drift"]
    lead = f"{prefix} " if prefix else ""
    arrows = []
    for part in parts:
        vector = vectors[part]
        if float(np.linalg.norm(vector)) < style.magnitude_floor_n:
            continue
        if part == "total" and not math.isfinite(scale):
            continue
        ratio = look.total_radius_ratio if part == "total" and len(parts) > 1 else 1.0
        arrows.append(
            _arrow(
                lead + getattr(look, f"{part}_label"),
                tails[part],
                vector,
                scale,
                style,
                hex_to_rgba(getattr(look, f"{part}_hex")),
                style.shaft_radius_m * ratio,
                units,
            )
        )
    return tuple(arrows)


def colour_key(
    view: str, colours: TipToTailStyle | None = None
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    """``(text, 8-bit RGB)`` of each part ``view`` draws, in drawing order."""
    _checked_view(view, "force")
    look = colours or TipToTailStyle()
    out = []
    for part in _PARTS_OF[view]:
        rgba = hex_to_rgba(getattr(look, f"{part}_hex"))
        rgb = (round(255 * rgba[0]), round(255 * rgba[1]), round(255 * rgba[2]))
        out.append((getattr(look, f"{part}_key"), rgb))
    return tuple(out)
