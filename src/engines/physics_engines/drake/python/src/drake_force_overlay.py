"""Headless force/torque overlay controller for the Drake GUI (FTO-12, #11297).

Keeps every decision testable without Qt or a browser: the GUI mixin only reads
its checkboxes and delegates here. The controller talks to a frame provider, a
``MeshcatGlyphRenderer`` and one ``MeshcatForceColorSession`` and nothing else.

MeshCat path mapping (segment shading): ``MeshcatVisualizer`` publishes each
illustration geometry at ``<prefix>/<frame name>/<geometry name>`` with ``::``
scope separators turned into ``/``. ``drake_color_bindings`` rebuilds those
leaf paths from ``SceneGraphInspector`` names; the colour adapter requires leaf
objects, so ``/<object>`` is appended. The default prefix is ``visualizer``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.force_overlay.contracts import ForceTorqueFrame, WrenchKind
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
    MeshcatGlyphRenderer,
    MeshcatSink,
    legend_text,
)

__all__ = [
    "ForceOverlayController",
    "drake_color_bindings",
    "illustration_base_rgba",
    "kinds_for_toggles",
]

DEFAULT_VISUALIZER_PREFIX = "visualizer"
_FALLBACK_RGBA = (0.9, 0.9, 0.9, 1.0)


def kinds_for_toggles(
    *, forces: bool, torques: bool, gravity: bool
) -> frozenset[WrenchKind]:
    """Map the three GUI checkboxes to wrench kinds.

    Forces -> {joint_reaction, contact, external}; Torques -> {joint_actuator};
    Gravity -> {gravity}. Raises TypeError for non-bool flags.
    """
    for name, flag in (("forces", forces), ("torques", torques), ("gravity", gravity)):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be bool, got {type(flag).__name__}")
    kinds: set[WrenchKind] = set()
    if forces:
        kinds |= {WrenchKind.JOINT_REACTION, WrenchKind.CONTACT, WrenchKind.EXTERNAL}
    if torques:
        kinds.add(WrenchKind.JOINT_ACTUATOR)
    if gravity:
        kinds.add(WrenchKind.GRAVITY)
    return frozenset(kinds)


class ForceOverlayController:
    """Draws provider frames as MeshCat glyphs and feeds segment shading.

    ``frame_provider(include_gravity)`` returns a ``ForceTorqueFrame`` or None.
    Also a ``ForceColorTarget``: install it next to the colour session in
    ``install_force_color_menu`` so the existing menu stays the only control.
    """

    def __init__(
        self,
        frame_provider: Callable[[bool], ForceTorqueFrame | None],
        sink: MeshcatSink,
        color_session: Any,
        style: ForceGlyphStyle | None = None,
    ) -> None:
        if not callable(frame_provider):
            raise TypeError("frame_provider must be callable")
        self._provider = frame_provider
        self._renderer = MeshcatGlyphRenderer(sink)
        self._session = color_session
        self._style = style or ForceGlyphStyle()
        self._colors_enabled = ForceColorScale().enabled

    def set_axial_color_scale(self, scale: ForceColorScale) -> None:
        """Track the menu's enabled flag and forward the scale to the session."""
        if not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale")
        self._colors_enabled = scale.enabled
        self._session.set_axial_color_scale(scale)

    def clear(self) -> None:
        """Remove every glyph this controller drew (e.g. before a model swap)."""
        self._renderer.clear()

    def tick(self, *, forces: bool, torques: bool, gravity: bool) -> str:
        """Refresh glyphs and shading for one visual tick; return the legend text.

        Postcondition: arrows of unchecked kinds are removed from MeshCat; the
        colour session sees ``set_frame`` only while shading is enabled.
        """
        kinds = kinds_for_toggles(forces=forces, torques=torques, gravity=gravity)
        frame = self._provider(gravity)
        if self._colors_enabled:
            self._session.set_frame(None if frame is None else frame.axial_loads)
        if frame is None or not kinds:
            self._renderer.clear()
            return ""
        style = replace(self._style, kinds=kinds)
        glyphs = build_glyphs(frame, style)
        self._renderer.update(glyphs)
        text = legend_text(glyphs)
        unavailable = glyphs.legend.unavailable_labels
        if unavailable:
            text += " | Unavailable: " + ", ".join(unavailable)
        return text


def drake_color_bindings(
    plant: Any,
    inspector: Any,
    segment_labels: Mapping[Any, str],
    *,
    base_rgba_of: Callable[[Any], tuple[float, ...]] | None = None,
    prefix: str = DEFAULT_VISUALIZER_PREFIX,
) -> dict[str, dict[str, tuple[float, ...]]]:
    """Build ``MeshcatForceColors`` bindings: segment label -> leaf path -> RGBA.

    ``segment_labels`` maps ``BodyIndex`` (``DrakeForceTorqueSource.body_labels``) to the segment ID used by the axial
    load frame. Bodies without illustration geometry are omitted (unbound
    segments simply stay uncoloured). ``base_rgba_of(geometry_id)`` supplies
    the original colour; it defaults to a neutral grey.
    """
    base_of = base_rgba_of or (lambda _gid: _FALLBACK_RGBA)
    bindings: dict[str, dict[str, tuple[float, ...]]] = {}
    for body_index, label in segment_labels.items():
        frame_id = plant.GetBodyFrameIdOrThrow(body_index)
        frame_path = inspector.GetName(frame_id).replace("::", "/")
        for gid in inspector.GetGeometries(frame_id, _illustration_role()):
            geom_path = inspector.GetName(gid).replace("::", "/")
            path = f"{prefix}/{frame_path}/{geom_path}/<object>"
            bindings.setdefault(label, {})[path] = tuple(base_of(gid))
    return bindings


def illustration_base_rgba(inspector: Any) -> Callable[[Any], tuple[float, ...]]:
    """Return a ``geometry_id -> RGBA`` lookup reading the phong diffuse colour."""

    def lookup(gid: Any) -> tuple[float, ...]:
        props = inspector.GetIllustrationProperties(gid)
        try:
            c = props.GetProperty("phong", "diffuse")
            return (float(c.r()), float(c.g()), float(c.b()), float(c.a()))
        except (AttributeError, RuntimeError, TypeError):
            return _FALLBACK_RGBA

    return lookup


def _illustration_role() -> Any:
    try:
        from pydrake.geometry import Role

        return Role.kIllustration
    except ImportError:
        return "illustration"
