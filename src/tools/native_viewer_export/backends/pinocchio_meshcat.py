"""Pinocchio native export: pinocchio's own ``MeshcatVisualizer``.

(Gepetto-viewer is a conda-only Qt/OSG GUI; see issue #11680.) The viewer
serves a meshcat-python page that headless Chromium captures.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from importlib.util import find_spec
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
    MeshcatGlyphRenderer,
    MeshcatPythonSink,
)
from src.shared.python.golf_view_presets import meshcat_camera
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.tools.native_viewer_export.backends._club import club_parts
from src.tools.native_viewer_export.backends._head import head_mesh_files
from src.tools.native_viewer_export.backends._meshcat_page import (
    MeshcatPage,
    playwright_unavailable_reason,
)
from src.tools.native_viewer_export.backends._scene import z_axis_frame
from src.tools.native_viewer_export.core import (
    ExportSettings,
    Image8,
    OverlayFeed,
    SwingInput,
)

_CAPSULE_RGBA = [0.75, 0.78, 0.85, 1.0]
_SHAPE_RGBA = [0.7, 0.72, 0.8, 1.0]
_FLOOR_RGBA = [0.35, 0.45, 0.3, 1.0]


def _bvh_mesh(coal: Any, mesh: Any) -> Any:
    """A coal triangle mesh for a closed ``Mesh`` (club-body frame)."""
    bvh = coal.BVHModelOBBRSS()
    bvh.beginModel(0, 0)
    for face in mesh.faces:
        a, b, c = (np.asarray(mesh.vertices[i], dtype=float) for i in face)
        bvh.addTriangle(a, b, c)
    bvh.endModel()
    return bvh


class PinocchioMeshcatBackend:
    """Pinocchio ``MeshcatVisualizer`` captured with headless Chromium."""

    engine = "pinocchio"

    def unavailable_reason(self) -> str | None:
        for module in ("pinocchio", "meshcat", "coal"):
            if find_spec(module) is None:
                return f"{module} is not installed"
        return playwright_unavailable_reason()

    def _build(self, swing: SwingInput) -> tuple[Any, Any, Any]:

        import coal
        import meshcat
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer

        from src.engines.physics_engines.pinocchio.python.native_model import (
            FullBodyPinocchioModel,
        )

        spec = json.loads(swing.bundle.spec_bytes)
        skeleton = derive_visual_skeleton(spec)
        adapter = FullBodyPinocchioModel(spec)
        geometry = pin.GeometryModel()

        def add(
            name: str, body: str, rot: Any, pos: Any, shape: Any, rgba: list[float]
        ) -> None:
            joint, body_pose = adapter._bodies[body]  # noqa: SLF001 - adapter exposes no public accessor
            obj = pin.GeometryObject(name, joint, body_pose * pin.SE3(rot, pos), shape)  # type: ignore[attr-defined]
            obj.meshColor = np.array(rgba)
            geometry.addGeometryObject(obj)  # type: ignore[attr-defined]

        n = 0
        # The shared visual head replaces the head capsule (visual only).
        self._head_dir = tempfile.mkdtemp(prefix="ud_head_")
        heads = head_mesh_files(spec, Path(self._head_dir))
        head_body = heads[0].body if heads else None
        for head in heads:
            joint, body_pose = adapter._bodies[head.body]  # noqa: SLF001
            shape = coal.MeshLoader().load(str(head.path))
            obj = pin.GeometryObject(  # type: ignore[attr-defined]
                f"head_{head.name}", joint, body_pose, shape, str(head.path)
            )
            obj.meshColor = np.array(head.rgba)
            obj.overrideMaterial = True
            geometry.addGeometryObject(obj)  # type: ignore[attr-defined]
        for cap in skeleton.capsules:
            if cap.body == head_body:
                continue
            rot, centre, length = z_axis_frame(cap.start_m, cap.end_m)
            add(
                f"cap{n}",
                cap.body,
                rot,
                centre,
                coal.Capsule(cap.radius_m, length),
                _CAPSULE_RGBA,
            )
            n += 1
        club = club_parts(spec)
        if club is not None:
            club_body, parts = club
            for part in parts:
                add(
                    part.name,
                    club_body,
                    np.eye(3),
                    np.zeros(3),
                    _bvh_mesh(coal, part.mesh),
                    list(part.rgba),
                )
        for shp in skeleton.shapes:
            if club is not None and shp.body == club[0]:
                continue  # the mesh head replaces the ellipsoid hint
            shape = (
                coal.Ellipsoid(*shp.half_size_m)
                if shp.kind == "ellipsoid"
                else coal.Box(*(2.0 * np.asarray(shp.half_size_m)))
            )
            add(
                f"shp{n}",
                shp.body,
                np.eye(3),
                np.array(shp.center_m),
                shape,
                _SHAPE_RGBA,
            )
            n += 1
        floor = pin.GeometryObject(  # type: ignore[attr-defined]
            "floor",
            0,
            pin.SE3(np.eye(3), np.array([1.0, 0.0, skeleton.ground.height_m - 0.005])),
            coal.Box(6.0, 6.0, 0.01),
        )
        floor.meshColor = np.array(_FLOOR_RGBA)
        geometry.addGeometryObject(floor)  # type: ignore[attr-defined]
        viz = MeshcatVisualizer(adapter.model, geometry, geometry)
        viz.initViewer(viewer=meshcat.Visualizer(), open=False, loadModel=False)
        viz.loadViewerModel(rootNodeName="golfer")
        return adapter, viz, meshcat

    def render(
        self,
        swing: SwingInput,
        settings: ExportSettings,
        indices: Sequence[int],
        overlay: OverlayFeed | None,
    ) -> Iterator[dict[str, Image8]]:
        adapter, viz, _ = self._build(swing)
        names = swing.bundle.coordinate_order

        def set_state(index: int) -> None:
            viz.display(
                adapter.configuration(dict(zip(names, swing.q[index], strict=True)))
            )

        set_state(indices[0])
        node = viz.viewer
        node["/Cameras/default"].set_transform(np.eye(4))
        node["/Grid"].set_property("visible", False)
        node["/Axes"].set_property("visible", False)
        glyphs = (
            MeshcatGlyphRenderer(MeshcatPythonSink(node))
            if overlay is not None
            else None
        )
        with MeshcatPage(node.url(), settings.width, settings.height, 3000) as page:
            for k in indices:
                set_state(k)
                if glyphs is not None and overlay is not None:
                    glyphs.update(overlay.glyphs_at(k))
                tiles: dict[str, Image8] = {}
                for view in settings.views:
                    cam = meshcat_camera(view, settings.lookat_m, settings.distance_m)
                    node["/Cameras/default/rotated/<object>"].set_property(
                        "position", list(cam.position_three)
                    )
                    page.look_at(cam.target_three)
                    page.settle()
                    tiles[view] = page.screenshot()
                yield tiles
