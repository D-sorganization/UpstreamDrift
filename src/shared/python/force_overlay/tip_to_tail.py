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

__all__ = ["TipToTailStyle", "tip_to_tail_arrows"]

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
        units="N",
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
    look = colours or TipToTailStyle()
    point = _vec3(point_m, "point_m")
    drift, control = _vec3(drift_n, "drift_n"), _vec3(control_n, "control_n")
    scale = style.force_scale_for(WrenchKind.GRIP)
    total = drift + control
    longest = scale * max(
        float(np.linalg.norm(drift)) + float(np.linalg.norm(control)),
        float(np.linalg.norm(total)),
    )
    if longest > style.max_length_m:
        raise ValueError(
            f"tip-to-tail arrows reach {longest:.3f} m, over max_length_m "
            f"{style.max_length_m}; lower the force scale"
        )
    radius = style.shaft_radius_m
    floor = style.magnitude_floor_n
    arrows = []
    if float(np.linalg.norm(drift)) >= floor:
        arrows.append(
            _arrow(
                look.drift_label,
                point,
                drift,
                scale,
                style,
                hex_to_rgba(look.drift_hex),
                radius,
            )
        )
    if float(np.linalg.norm(control)) >= floor:
        arrows.append(
            _arrow(
                look.control_label,
                point + scale * drift,
                control,
                scale,
                style,
                hex_to_rgba(look.control_hex),
                radius,
            )
        )
    if float(np.linalg.norm(total)) >= floor and math.isfinite(scale):
        arrows.append(
            _arrow(
                look.total_label,
                point,
                total,
                scale,
                style,
                hex_to_rgba(look.total_hex),
                radius * look.total_radius_ratio,
            )
        )
    return tuple(arrows)
