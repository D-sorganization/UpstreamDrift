"""Force overlay service bridging engine providers and API/WebSocket routes (#11307, FTO-22).

Supplies renderer-neutral serialized GlyphSet drawing payloads alongside
ForceTorqueFrame inspection payloads, respecting LoD and DbC boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import math
from typing import Any

import numpy as np

from src.api.models.responses import ForceVector3D
from src.shared.python.force_overlay import (
    ForceGlyphStyle,
    ForceTorqueFrame,
    ForceTorqueProvider,
    GlyphSet,
    WrenchKind,
    build_glyphs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "current_force_frame",
    "force_overlay_payload",
    "glyphs_to_legacy_vectors",
    "style_from_request_params",
]

_FORCE_TYPE_MAP: dict[str, set[WrenchKind]] = {
    "applied": {WrenchKind.JOINT_ACTUATOR, WrenchKind.EXTERNAL},
    "gravity": {WrenchKind.GRAVITY},
    "contact": {WrenchKind.CONTACT},
    "reaction": {WrenchKind.JOINT_REACTION},
    "joint_reaction": {WrenchKind.JOINT_REACTION},
    "joint_actuator": {WrenchKind.JOINT_ACTUATOR},
    "grip": {WrenchKind.GRIP},
    "muscle": {WrenchKind.MUSCLE},
    "external": {WrenchKind.EXTERNAL},
}


def current_force_frame(engine: Any) -> ForceTorqueFrame | None:
    """Read instantaneous force/torque frame from a qualified engine provider.

    Args:
        engine: Physics engine instance to query.

    Returns:
        ForceTorqueFrame if engine implements ForceTorqueProvider and provides
        a valid frame, otherwise None.
    """
    if engine is None:
        return None
    if not isinstance(engine, ForceTorqueProvider):
        return None
    try:
        frame = engine.get_force_torque_frame()
    except (ValueError, TypeError, RuntimeError, AttributeError):
        logger.exception("Engine raised exception during get_force_torque_frame")
        return None
    if frame is None or not isinstance(frame, ForceTorqueFrame):
        return None
    return frame


def style_from_request_params(
    force_types: Sequence[str] | None = None,
    scale_factor: float = 0.01,
    show_labels: bool = False,
) -> ForceGlyphStyle:
    """Build a ForceGlyphStyle from API query/request parameters.

    Args:
        force_types: Selected force types (e.g. ['applied', 'contact', 'all']).
        scale_factor: Linear scaling factor from API request.
        show_labels: Whether to attach text labels.

    Returns:
        Configured ForceGlyphStyle.
    """
    kinds: set[WrenchKind] = set()
    if force_types:
        for ft in force_types:
            ft_clean = ft.lower().strip()
            if ft_clean == "all":
                kinds = set(WrenchKind)
                break
            if ft_clean in _FORCE_TYPE_MAP:
                kinds.update(_FORCE_TYPE_MAP[ft_clean])
            else:
                for k in WrenchKind:
                    if k.value == ft_clean:
                        kinds.add(k)
                        break

    if not kinds:
        kinds = set(WrenchKind)

    # Scale factor 0.01 maps to default 1e-3 (1 mm / N) and 5e-3 (5 mm / N*m)
    ratio = max(scale_factor, 1e-6) / 0.01
    force_scale = 0.001 * ratio
    torque_scale = 0.005 * ratio

    return ForceGlyphStyle(
        force_scale_m_per_n=force_scale,
        torque_scale_m_per_nm=torque_scale,
        kinds=frozenset(kinds),
        show_labels=show_labels,
    )


def force_overlay_payload(
    frame: ForceTorqueFrame | None,
    style: ForceGlyphStyle | Mapping[str, Any] | None = None,
    *,
    body_filter: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate the complete force overlay API/WebSocket payload.

    Args:
        frame: Instantaneous ForceTorqueFrame, or None if unavailable.
        style: Glyph styling rules or configuration dict.
        body_filter: Optional sequence of body names to retain.

    Returns:
        Dictionary with serialized 'glyphs', 'frame', and 'style', or
        None values with an 'unavailable_reason'.
    """
    if style is None:
        style_obj = ForceGlyphStyle()
    elif isinstance(style, ForceGlyphStyle):
        style_obj = style
    elif isinstance(style, Mapping):
        kinds_raw = style.get("kinds")
        kinds_set: frozenset[WrenchKind] | None = None
        if kinds_raw is not None:
            kinds_set = frozenset(WrenchKind(k) for k in kinds_raw)
        style_obj = ForceGlyphStyle(
            force_scale_m_per_n=float(style.get("force_scale_m_per_n", 0.001)),
            torque_scale_m_per_nm=float(style.get("torque_scale_m_per_nm", 0.005)),
            show_labels=bool(style.get("show_labels", False)),
            kinds=kinds_set if kinds_set is not None else frozenset(WrenchKind),
        )
    else:
        style_obj = ForceGlyphStyle()

    if frame is None:
        return {
            "glyphs": None,
            "frame": None,
            "style": style_obj.to_dict(),
            "unavailable_reason": "No force/torque frame available from active engine",
        }

    frame_to_use = frame
    if body_filter is not None:
        allowed_bodies = set(body_filter)
        filtered_wrenches = tuple(w for w in frame.wrenches if w.body in allowed_bodies)
        frame_to_use = ForceTorqueFrame(
            time_s=frame.time_s,
            engine=frame.engine,
            wrenches=filtered_wrenches,
            axial_loads=frame.axial_loads,
            world_frame=frame.world_frame,
            units=frame.units,
        )

    glyph_set = build_glyphs(frame_to_use, style_obj)
    return {
        "glyphs": glyph_set.to_dict(),
        "frame": frame_to_use.to_dict(),
        "style": style_obj.to_dict(),
    }


def glyphs_to_legacy_vectors(
    glyphs_data: GlyphSet | dict[str, Any],
    frame_data: ForceTorqueFrame | dict[str, Any] | None = None,
    *,
    show_labels: bool = False,
) -> list[ForceVector3D]:
    """Derive backward-compatible ForceVector3D items from generated glyphs.

    Args:
        glyphs_data: GlyphSet instance or serialized glyph-set-v1 dictionary.
        frame_data: Optional corresponding ForceTorqueFrame to resolve body names.
        show_labels: Whether to format string labels.

    Returns:
        List of ForceVector3D models.
    """
    if isinstance(glyphs_data, GlyphSet):
        glyphs_dict = glyphs_data.to_dict()
    else:
        glyphs_dict = glyphs_data

    # Map wrench labels to body names
    label_to_body: dict[str, str] = {}
    if frame_data is not None:
        if isinstance(frame_data, ForceTorqueFrame):
            for w in frame_data.wrenches:
                label_to_body[w.label] = w.body
        elif isinstance(frame_data, dict):
            for w_dict in frame_data.get("wrenches", []):
                label_to_body[w_dict.get("label", "")] = w_dict.get("body", "unknown")

    vectors: list[ForceVector3D] = []

    # 1. Arrows (Forces)
    for arrow in glyphs_dict.get("arrows", []):
        tail = arrow["tail_m"]
        tip = arrow["tip_m"]
        dx = tip[0] - tail[0]
        dy = tip[1] - tail[1]
        dz = tip[2] - tail[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length > 1e-12:
            direction = [dx / length, dy / length, dz / length]
        else:
            direction = [0.0, 0.0, 1.0]

        mag = float(arrow["magnitude"])
        label_text = f"{mag:.1f} {arrow.get('units', 'N')}" if show_labels else None
        body = label_to_body.get(arrow["label"], "unknown")

        vectors.append(
            ForceVector3D(
                body_name=body,
                force_type=str(arrow["kind"]),
                origin=list(tail),
                direction=direction,
                magnitude=mag,
                color=list(arrow["rgba"]),
                label=label_text,
            )
        )

    # 2. Torque arcs (Torques)
    for arc in glyphs_dict.get("torque_arcs", []):
        center = arc["center_m"]
        axis = arc["axis_unit"]
        mag = float(arc["magnitude"])
        label_text = f"{mag:.1f} {arc.get('units', 'N*m')}" if show_labels else None
        body = label_to_body.get(arc["label"], "unknown")

        vectors.append(
            ForceVector3D(
                body_name=body,
                force_type=str(arc["kind"]),
                origin=list(center),
                direction=list(axis),
                magnitude=mag,
                color=list(arc["rgba"]),
                label=label_text,
            )
        )

    return vectors
