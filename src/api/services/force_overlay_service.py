"""Force overlay service bridging engine providers and API/WebSocket routes (#11307, FTO-22).

Supplies renderer-neutral serialized GlyphSet drawing payloads alongside
ForceTorqueFrame inspection payloads, respecting LoD and DbC boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Any

import numpy as np

from src.shared.python.force_overlay import (
    ForceGlyphStyle,
    ForceTorqueFrame,
    ForceTorqueProvider,
    WrenchKind,
    build_glyphs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "current_force_frame",
    "force_overlay_payload",
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


#: Style keys a mapping payload may pass straight through (GCV-4, #11710).
_SCALE_STYLE_KEYS = frozenset(
    {"scale_mode", "reference_force_n", "reference_length_m", "kind_scale", "groups"}
)


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
    *,
    scale_mode: str = "fixed",
    reference_force_n: float | None = None,
    reference_length_m: float = 0.5,
    kind_scale: Mapping[str, float] | None = None,
    groups: Sequence[str] | None = None,
) -> ForceGlyphStyle:
    """Build a ForceGlyphStyle from API query/request parameters.

    Args:
        force_types: Selected force types (e.g. ['applied', 'contact', 'all']).
        scale_factor: Linear scaling factor from API request.
        show_labels: Whether to attach text labels.
        scale_mode: ``fixed`` (``scale_factor``), ``body_weight`` or ``peak``.
        reference_force_n: Body weight (N) or series peak (N); required for
            ``body_weight`` and ``peak``.
        reference_length_m: Arrow length for one reference force.
        kind_scale: Per-WrenchKind length multipliers keyed by kind value.
        groups: Enabled overlay groups; ``None`` keeps the style default.

    Returns:
        Configured ForceGlyphStyle.

    Raises:
        ValueError: Unknown mode/group/kind, or a non-fixed mode without a
            reference force.
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

    extra: dict[str, Any] = {}
    if groups is not None:
        extra["groups"] = frozenset(groups)
    if kind_scale:
        extra["kind_scale"] = {WrenchKind(k): float(v) for k, v in kind_scale.items()}
    return ForceGlyphStyle(
        force_scale_m_per_n=force_scale,
        torque_scale_m_per_nm=torque_scale,
        kinds=frozenset(kinds),
        show_labels=show_labels,
        scale_mode=scale_mode,  # type: ignore[arg-type]
        reference_force_n=reference_force_n,
        reference_length_m=reference_length_m,
        **extra,
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
        style_obj = ForceGlyphStyle.from_dict(
            {
                **{k: v for k, v in style.items() if k in _SCALE_STYLE_KEYS},
                "force_scale_m_per_n": float(style.get("force_scale_m_per_n", 0.001)),
                "torque_scale_m_per_nm": float(
                    style.get("torque_scale_m_per_nm", 0.005)
                ),
                "show_labels": bool(style.get("show_labels", False)),
                "kinds": sorted(k.value for k in kinds_set)
                if kinds_set is not None
                else sorted(k.value for k in WrenchKind),
            }
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
