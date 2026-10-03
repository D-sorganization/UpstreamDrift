"""Headless toggle-to-glyph mapping for the MuJoCo GUI force overlays (FTO-10, #11295).

The GUI keeps two checkboxes (force, torque) and two scale sliders. This
module turns those toggles plus a ``ForceTorqueFrame`` from the engine provider
into a renderer-neutral ``GlyphSet`` that the MjvScene renderer (native viewer)
and the MeshCat renderer both consume. It never reads ``data.cfrc_*`` or
``xaxis`` (ADR-0052, Law of Demeter) and has no Qt dependency, so it is fully
testable headless.
"""

from __future__ import annotations

import dataclasses
import math

from src.shared.python.force_overlay.contracts import ForceTorqueFrame, OverlayWrench
from src.shared.python.force_overlay.glyphs import (
    ForceGlyphStyle,
    GlyphSet,
    build_glyphs,
)

__all__ = ["build_overlay_glyphs", "overlay_style", "restrict_frame"]


def _check_scale(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return float(value)


def overlay_style(force_scale: float, torque_scale: float) -> ForceGlyphStyle:
    """Map the GUI scale sliders to a ``ForceGlyphStyle``.

    Args:
        force_scale: Arrow length in metres per newton (slider value).
        torque_scale: Arc radius scale in metres per newton-metre.

    Returns:
        A style with every wrench kind enabled and the given scales.

    Raises:
        TypeError: A scale is not numeric.
        ValueError: A scale is not finite and positive.
    """
    return ForceGlyphStyle(
        force_scale_m_per_n=_check_scale("force_scale", force_scale),
        torque_scale_m_per_nm=_check_scale("torque_scale", torque_scale),
    )


def restrict_frame(
    frame: ForceTorqueFrame,
    *,
    show_force: bool,
    show_torque: bool,
    body_name: str | None = None,
) -> ForceTorqueFrame:
    """Keep only the enabled halves (and optionally one body) of a frame.

    Postcondition: the result contains no force half unless ``show_force`` and no
    torque half unless ``show_torque``; wrenches left with neither half are
    dropped. ``axial_loads`` is preserved.
    """
    kept: list[OverlayWrench] = []
    for wrench in frame.wrenches:
        if body_name is not None and wrench.body != body_name:
            continue
        force = wrench.force_n if show_force else None
        torque = wrench.torque_nm if show_torque else None
        if force is None and torque is None:
            continue
        kept.append(dataclasses.replace(wrench, force_n=force, torque_nm=torque))
    return dataclasses.replace(frame, wrenches=tuple(kept))


def build_overlay_glyphs(
    frame: ForceTorqueFrame | None,
    *,
    show_force: bool,
    show_torque: bool,
    force_scale: float,
    torque_scale: float,
    body_name: str | None = None,
) -> GlyphSet | None:
    """Build the glyphs for the enabled toggles, or ``None`` when nothing to draw.

    Returns ``None`` when both toggles are off or the provider has no frame.
    """
    if not (show_force or show_torque) or frame is None:
        return None
    if not isinstance(frame, ForceTorqueFrame):
        raise TypeError(f"frame must be a ForceTorqueFrame, got {type(frame).__name__}")
    style = overlay_style(force_scale, torque_scale)
    restricted = restrict_frame(
        frame, show_force=show_force, show_torque=show_torque, body_name=body_name
    )
    return build_glyphs(restricted, style)
