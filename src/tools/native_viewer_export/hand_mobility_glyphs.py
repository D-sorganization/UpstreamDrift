"""Hand mobility and effective-mass ellipsoid glyphs for the exporter (#14004).

One FIXED scale per clip, shared by both hands and all frames, so sizes compare
across hands and time: the mobility ellipsoid has radius
``0.10 m * sqrt(lambda / 1 kg^-1)`` and its inertial dual (effective mass) its
own scale, ``0.05 m * sqrt(m_eff / 1 kg)`` (the two never share a scale), both
clamped to [0.01, 0.40] m.  A clamped
glyph is flagged in its label and drawn fainter.  Labels speak of effective
mass (kg), an inertial property, never a force capacity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np

from src.shared.python.force_overlay.glyphs import EllipsoidGlyph
from src.shared.python.force_overlay.palette import hand_ellipsoid_rgba

KINDS: Final = ("mobility", "force", "both")
SCALE_M: Final = 0.10  # mobility radius at 1 kg^-1
MASS_SCALE_M: Final = 0.05  # effective-mass radius at 1 kg
MIN_RADIUS_M: Final = 0.01
MAX_RADIUS_M: Final = 0.40
CLAMPED_ALPHA_FACTOR: Final = 0.5
LABEL_PREFIX: Final = "hand "

__all__ = ["KINDS", "hand_mobility_glyphs", "hand_mobility_legend"]


def _label(hand: str, kind: str, mass: np.ndarray, cond: float, clamped: bool) -> str:
    masses = "/".join(f"{m:.1f}" for m in np.sort(mass)[::-1])
    flag = " [clamped]" if clamped else ""
    return (
        f"{LABEL_PREFIX}{hand} {kind}: effective mass (kg) {masses}, "
        f"cond {cond:.0f}{flag}"
    )


def hand_mobility_glyphs(
    mobilities: Mapping[str, object], kind: str
) -> tuple[EllipsoidGlyph, ...]:
    """Ellipsoid glyphs of ``HandMobility`` values keyed by hand for ``kind``.

    Raises ``ValueError`` on an unknown ``kind``.  Order: per hand, mobility
    then force.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    wanted = ("mobility", "force") if kind == "both" else (kind,)
    out: list[EllipsoidGlyph] = []
    for hand, hm in mobilities.items():
        base = hand_ellipsoid_rgba(hand)
        for name in wanted:
            if name == "mobility":
                radii = SCALE_M * np.sqrt(np.asarray(hm.mobility_eigenvalues))
            else:
                radii = MASS_SCALE_M * np.sqrt(np.asarray(hm.effective_mass_kg))
            clamped = bool((radii < MIN_RADIUS_M).any() or (radii > MAX_RADIUS_M).any())
            alpha = base[3] * (CLAMPED_ALPHA_FACTOR if clamped else 1.0)
            glyph = EllipsoidGlyph(
                hm.centre,
                hm.mobility_axes,
                tuple(np.maximum(radii, 1e-9)),
                (*base[:3], alpha),
                _label(hand, name, hm.effective_mass_kg, hm.condition, clamped),
            ).with_clamped_radii(MIN_RADIUS_M, MAX_RADIUS_M)
            out.append(glyph)
    return tuple(out)


def hand_mobility_legend(glyphs: tuple[EllipsoidGlyph, ...]) -> str:
    """HUD text: each hand's eigen effective masses (kg) and cond, once per hand."""
    seen: dict[str, str] = {}
    for g in glyphs:
        if g.label.startswith(LABEL_PREFIX):
            hand, _, text = g.label[len(LABEL_PREFIX) :].partition(" ")
            seen.setdefault(hand, text.partition(": ")[2])
    return " | ".join(f"{hand}: {text}" for hand, text in seen.items())
