"""Pinocchio GUI force and torque overlay helper (ADR-0052, FTO-14, #11299).

Consolidates duplicate visualization mixins onto shared force/torque contracts,
MeshcatGlyphRenderer, and synchronous segment force-color shading.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.body_part_viz.meshcat_force_colors import (
    MeshcatForceColors,
    MeshcatForceColorSession,
)
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
    build_glyphs,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle
from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
    MeshcatGlyphRenderer,
    MeshcatPythonSink,
    MeshcatSink,
)
from src.shared.python.logging_pkg.logging_config import get_logger

__all__ = ["PinocchioForceOverlayView", "meshcat_paths_for_links"]

logger = get_logger(__name__)


def meshcat_paths_for_links(
    visual_model: Any,
    root: str = "pinocchio",
    model: Any = None,
) -> dict[str, list[str]]:
    """Map URDF link names to MeshCat visual paths created by MeshcatVisualizer.

    Preconditions: ``visual_model`` must have a ``geometryObjects`` sequence.
    Postconditions: returns a dictionary mapping each URDF link name to its list
    of MeshCat visual paths under ``<root>/visuals/<geom_name>``.
    """
    if visual_model is None or not hasattr(visual_model, "geometryObjects"):
        raise TypeError("visual_model must have geometryObjects attribute")

    clean_root = root.strip("/")
    prefix = f"{clean_root}/visuals" if clean_root else "visuals"

    paths_by_link: dict[str, list[str]] = {}

    for geom in visual_model.geometryObjects:
        path = f"{prefix}/{geom.name}"
        link_name: str | None = None

        if hasattr(geom, "parent_link") and geom.parent_link:
            link_name = str(geom.parent_link)
        elif hasattr(geom, "parent_link_name") and geom.parent_link_name:
            link_name = str(geom.parent_link_name)
        elif (
            model is not None
            and hasattr(geom, "parentFrame")
            and hasattr(model, "frames")
            and 0 <= geom.parentFrame < len(model.frames)
        ):
            link_name = str(model.frames[geom.parentFrame].name)
        elif (
            model is not None
            and hasattr(geom, "parentJoint")
            and hasattr(model, "names")
            and 0 <= geom.parentJoint < len(model.names)
        ):
            link_name = str(model.names[geom.parentJoint])
        else:
            name = str(geom.name)
            if "/" in name:
                link_name = name.split("/")[0]
            elif ":" in name:
                link_name = name.split(":")[0]
            else:
                link_name = re.sub(r"_\d+$", "", name)

        if link_name:
            paths_by_link.setdefault(link_name, []).append(path)

    return paths_by_link


class PinocchioForceOverlayView:
    """Helper connecting Pinocchio providers to MeshCat glyphs and color session."""

    def __init__(
        self,
        engine_or_source: Any,
        meshcat_visualizer: Any = None,
        color_session: MeshcatForceColorSession | None = None,
        *,
        root: str = "/force_overlay",
    ) -> None:
        self.engine_or_source = engine_or_source
        self.visualizer = meshcat_visualizer
        self.color_session = color_session
        self.root = root
        self._renderer: MeshcatGlyphRenderer | None = None
        self._last_viz: Any = None

    def _ensure_renderer(self) -> MeshcatGlyphRenderer | None:
        """Instantiate or reuse MeshcatGlyphRenderer for the current viewer."""
        viz = self.visualizer
        if viz is None:
            return None

        if viz is self._last_viz and self._renderer is not None:
            return self._renderer

        sink: MeshcatSink
        if isinstance(viz, MeshcatSink):
            sink = viz
        elif hasattr(viz, "viewer") and isinstance(viz.viewer, MeshcatSink):
            sink = viz.viewer
        elif hasattr(viz, "viewer"):
            sink = MeshcatPythonSink(viz.viewer)
        else:
            sink = MeshcatPythonSink(viz)

        self._renderer = MeshcatGlyphRenderer(sink, root=self.root)
        self._last_viz = viz
        return self._renderer

    def _extract_toggles(self, toggles: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Normalize toggles dictionary or object into a uniform dict."""
        out = {
            "forces": True,
            "torques": True,
            "shading": False,
            "ztcf": False,
            "force_scale": 1.0 / 1000.0,
            "torque_scale": 1.0 / 200.0,
        }
        if toggles is None:
            return out

        if isinstance(toggles, Mapping):
            for k in out:
                if k in toggles:
                    out[k] = toggles[k]
        else:
            for k in out:
                if hasattr(toggles, k):
                    val = getattr(toggles, k)
                    out[k] = val() if callable(val) else val
        return out

    def _get_base_frame(self) -> ForceTorqueFrame | None:
        """Retrieve the primary ForceTorqueFrame from engine_or_source."""
        src = self.engine_or_source
        if hasattr(src, "get_force_torque_frame"):
            return src.get_force_torque_frame()
        if hasattr(src, "sample") and hasattr(src, "model"):
            q = getattr(src, "q", None)
            v = getattr(src, "v", None)
            tau = getattr(src, "tau", None)
            if q is not None and v is not None:
                tau_val = np.zeros(src.model.nv) if tau is None else tau
                a = src.acceleration(q, v, tau_val)
                return src.sample(
                    q, v, a, tau_val, time_s=float(getattr(src, "time", 0.0))
                )
        return None

    def _get_ztcf_frame(self) -> ForceTorqueFrame | None:
        """Retrieve or compute the Zero-Torque Counterfactual frame."""
        src = self.engine_or_source
        if hasattr(src, "get_ztcf_frame"):
            return src.get_ztcf_frame()
        if hasattr(src, "sample") and hasattr(src, "model"):
            q = getattr(src, "q", None)
            v = getattr(src, "v", None)
            if q is not None and v is not None:
                tau_zero = np.zeros(src.model.nv)
                a = src.acceleration(q, v, tau_zero)
                return src.sample(
                    q, v, a, tau_zero, time_s=float(getattr(src, "time", 0.0))
                )
        return None

    def update(self, toggles: Mapping[str, Any] | Any = None) -> None:
        """Update MeshCat glyphs and color session with current simulation state."""
        cfg = self._extract_toggles(toggles)
        frame = self._get_base_frame()

        if frame is None:
            renderer = self._ensure_renderer()
            if renderer is not None:
                renderer.clear()
            if self.color_session is not None:
                self.color_session.set_frame(None)
            return

        # Synchronize axial loads to color session
        if self.color_session is not None:
            if cfg["shading"]:
                self.color_session.set_frame(frame.axial_loads)
            else:
                self.color_session.set_frame(None)

        # Build active kinds set
        kinds_present: list[WrenchKind] = []
        if cfg["forces"]:
            kinds_present.extend(
                [
                    WrenchKind.JOINT_REACTION,
                    WrenchKind.CONTACT,
                    WrenchKind.EXTERNAL,
                    WrenchKind.GRAVITY,
                    WrenchKind.MUSCLE,
                ]
            )
        if cfg["torques"]:
            kinds_present.append(WrenchKind.JOINT_ACTUATOR)

        combined_wrenches: list[OverlayWrench] = list(frame.wrenches)

        # Handle explicit ZTCF overlay if requested
        if cfg["ztcf"]:
            ztcf_frame = self._get_ztcf_frame()
            if ztcf_frame is not None:
                for w in ztcf_frame.wrenches:
                    label = w.label if w.label.startswith("cf:") else f"cf:{w.label}"
                    combined_wrenches.append(
                        OverlayWrench(
                            kind=WrenchKind.EXTERNAL,
                            label=label,
                            body=w.body,
                            point_m=w.point_m,
                            force_n=w.force_n,
                            torque_nm=w.torque_nm,
                            source="pinocchio:ztcf",
                        )
                    )

        annotated_frame = ForceTorqueFrame(
            time_s=frame.time_s,
            engine=frame.engine,
            wrenches=tuple(combined_wrenches),
            axial_loads=frame.axial_loads,
            world_frame=frame.world_frame,
            units=frame.units,
        )

        style = ForceGlyphStyle(
            force_scale_m_per_n=float(cfg["force_scale"]),
            torque_scale_m_per_nm=float(cfg["torque_scale"]),
            kinds=frozenset(kinds_present),
        )

        glyphs = build_glyphs(annotated_frame, style)
        renderer = self._ensure_renderer()
        if renderer is not None:
            renderer.update(glyphs)

    def bind_color_session(
        self,
        model: Any,
        visual_model: Any,
        viewer: Any = None,
        root: str = "pinocchio",
    ) -> None:
        """Bind MeshcatForceColorSession to the active visualizer objects."""
        if self.color_session is None or visual_model is None or model is None:
            return

        vis = viewer if viewer is not None else self.visualizer
        if vis is None:
            return

        # Setter callback for meshcat property
        def _set_prop(path: str, prop: str, value: list[float]) -> None:
            clean_path = path.strip("/")
            if hasattr(vis, "viewer"):
                target = vis.viewer
            else:
                target = vis
            node = target[clean_path] if hasattr(target, "__getitem__") else target
            if hasattr(node, "set_property"):
                node.set_property(prop, value)

        paths = meshcat_paths_for_links(visual_model, root=root, model=model)
        bindings: dict[str, dict[str, tuple[float, float, float, float]]] = {}

        for link_name, link_paths in paths.items():
            link_map: dict[str, tuple[float, float, float, float]] = {}
            for p in link_paths:
                leaf_path = f"{p}/<object>"
                link_map[leaf_path] = (0.7, 0.7, 0.7, 1.0)
            if link_map:
                bindings[link_name] = link_map

        if bindings:
            adapter = MeshcatForceColors(_set_prop, bindings)
            self.color_session.bind(adapter, model)
