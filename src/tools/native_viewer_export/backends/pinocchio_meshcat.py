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
from src.tools.native_viewer_export.backends._ball import (
    MeshcatBall,
    PythonMeshcatBallSink,
    update_meshcat_frame,
)
from src.tools.native_viewer_export.backends._meshcat_scene import (
    VisualItem,
    build_scene_visuals,
    check_bodies,
    default_appearance,
)
from src.tools.native_viewer_export.backends._meshcat_page import (
    MeshcatPage,
    playwright_unavailable_reason,
)
from src.tools.native_viewer_export.core import (
    ExportSettings,
    Image8,
    OverlayFeed,
    SwingInput,
    view_lookats,
)


def _primitive(coal: Any, item: VisualItem) -> Any:
    """The coal primitive of a non-mesh scene item (capsule, ellipsoid, box)."""
    if item.kind == "capsule":
        return coal.Capsule(*item.size)
    if item.kind == "ellipsoid":
        return coal.Ellipsoid(*item.size)
    return coal.Box(*item.size)


class PinocchioMeshcatBackend:
    """Pinocchio ``MeshcatVisualizer`` captured with headless Chromium."""

    engine = "pinocchio"

    def __init__(self, plain: bool = False) -> None:
        """``plain`` draws the grey library skeleton instead of the appearance."""
        self._appearance = None if plain else default_appearance()

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
        adapter = FullBodyPinocchioModel(spec)
        geometry = pin.GeometryModel()
        self._mesh_dir = tempfile.TemporaryDirectory(prefix="ud_visuals_")
        scene = build_scene_visuals(spec, Path(self._mesh_dir.name), self._appearance)
        bodies = adapter._bodies  # noqa: SLF001 - adapter exposes no public accessor
        check_bodies(scene, bodies)
        for item in scene.items:
            joint, body_pose = bodies[item.body]
            pose = body_pose * pin.SE3(item.rotation, item.translation)
            if item.kind == "mesh":
                path = str(item.mesh_path)
                obj = pin.GeometryObject(  # type: ignore[attr-defined]
                    item.name, joint, pose, coal.MeshLoader().load(path), path
                )
                obj.overrideMaterial = True
            else:
                obj = pin.GeometryObject(  # type: ignore[attr-defined]
                    item.name, joint, pose, _primitive(coal, item)
                )
            obj.meshColor = np.array(item.rgba)
            geometry.addGeometryObject(obj)  # type: ignore[attr-defined]
        floor = pin.GeometryObject(  # type: ignore[attr-defined]
            "floor",
            0,
            pin.SE3(np.eye(3), np.array([1.0, 0.0, scene.ground_height_m - 0.005])),
            coal.Box(6.0, 6.0, 0.01),
        )
        floor.meshColor = np.array(scene.floor_rgba)
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
        ball = (
            MeshcatBall(PythonMeshcatBallSink(node), swing.ball)
            if swing.ball is not None
            else None
        )
        with MeshcatPage(node.url(), settings.width, settings.height, 3000) as page:
            looks = view_lookats(settings, indices, overlay)
            for pos, k in enumerate(indices):
                set_state(k)
                update_meshcat_frame(ball, glyphs, overlay, swing, k)
                tiles: dict[str, Image8] = {}
                for view in settings.views:
                    cam = meshcat_camera(view, looks[view][pos], settings.distance_m)
                    node["/Cameras/default/rotated/<object>"].set_property(
                        "position", list(cam.position_three)
                    )
                    page.look_at(cam.target_three)
                    page.settle()
                    tiles[view] = page.screenshot()
                yield tiles
