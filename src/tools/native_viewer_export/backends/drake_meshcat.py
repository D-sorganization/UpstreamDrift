"""Drake MeshCat native export: the spec robot in Drake's own MeshCat viewer.

The parity rollout drives a ``MultibodyPlant`` built from the specification
URDF export with the shared visual skeleton; Drake's ``MeshcatVisualizer``
publishes it and headless Chromium captures each requested view.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from src.engines.physics_engines.drake.python.src.drake_meshed_golfer import (
    register_scene_visuals,
)
from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
    MeshcatGlyphRenderer,
)
from src.shared.python.golf_view_presets import drake_meshcat_camera_pose
from src.tools.native_viewer_export.backends._ball import (
    DrakeBallSink,
    MeshcatBall,
    update_meshcat_frame,
)
from src.tools.native_viewer_export.backends._meshcat_scene import (
    SceneVisuals,
    build_scene_visuals,
    check_bodies,
    default_appearance as default_appearance,
)
from src.tools.native_viewer_export.backends._body import scene_appearance
from src.tools.native_viewer_export.backends._meshcat_page import (
    MeshcatPage,
    playwright_unavailable_reason,
)
from src.tools.native_viewer_export.backends._subprocess import export_urdf
from src.engines.physics_engines.drake.python.src.drake_force_overlay import (
    drake_color_bindings,
    illustration_base_rgba,
)
from src.tools.native_viewer_export.shading import build_meshcat_shading
from src.tools.native_viewer_export.core import (
    ExportSettings,
    Image8,
    OverlayFeed,
    SwingInput,
    view_lookats,
)


class DrakeMeshcatBackend:
    """Drake's MeshCat viewer, captured with headless Chromium."""

    engine = "drake"

    def __init__(self, plain: bool = False) -> None:
        """``plain`` draws the grey library skeleton instead of the appearance."""
        self._plain = plain
        self._appearance = scene_appearance(plain, None)

    def _apply_settings(self, settings: ExportSettings) -> None:
        """Adopt the export's resolved appearance document (HRG-B2, #13167)."""
        self._appearance = scene_appearance(self._plain, settings.appearance)

    def unavailable_reason(self) -> str | None:
        if find_spec("pydrake") is None:
            return "pydrake is not installed"
        return playwright_unavailable_reason()

    def _register_visuals(
        self, plant: Any, inst: Any, links: Mapping[str, str], scene: SceneVisuals
    ) -> None:
        """Translate the shared scene items into Drake visual geometry."""
        register_scene_visuals(plant, inst, dict(links), scene)

    def _build(
        self, swing: SwingInput
    ) -> tuple[Any, Any, Any, Any, Any, list[int], dict[str, Any]]:
        from pydrake.geometry import (
            Box,
            Meshcat,
            MeshcatVisualizer,
            MeshcatVisualizerParams,
            Rgba,
            Role,
        )
        from pydrake.math import RigidTransform
        from pydrake.multibody.parsing import Parser
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph

        spec_bytes = swing.bundle.spec_bytes
        xml, meta = export_urdf(spec_bytes)
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        inst = Parser(plant).AddModelsFromString(xml, "urdf")[0]
        links = meta["body_links"]
        plant.WeldFrames(
            plant.world_frame(), plant.GetBodyByName(links["world"], inst).body_frame()
        )
        spec = json.loads(spec_bytes)
        self._mesh_dir = tempfile.TemporaryDirectory(prefix="ud_visuals_")
        scene = build_scene_visuals(spec, Path(self._mesh_dir.name), self._appearance)
        check_bodies(scene, links)
        self._scene = scene
        self._register_visuals(plant, inst, links, scene)
        plant.RegisterVisualGeometry(
            plant.world_body(),
            RigidTransform(np.array([1.0, 0.0, scene.ground_height_m - 0.005])),  # type: ignore[arg-type]
            Box(6.0, 6.0, 0.01),
            "floor",
            np.array(scene.floor_rgba),
        )
        plant.Finalize()
        meshcat = Meshcat()
        MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration),
        )
        diagram = builder.Build()
        ctx = diagram.CreateDefaultContext()
        pctx = plant.GetMyMutableContextFromRoot(ctx)
        starts = [
            plant.GetJointByName(nm, inst).position_start()
            for nm in swing.bundle.coordinate_order
        ]
        _ = Rgba
        inspector = scene_graph.model_inspector()
        bindings = drake_color_bindings(
            plant,
            inspector,
            {
                plant.GetBodyByName(link, inst).index(): spec_body
                for spec_body, link in links.items()
                if spec_body != "world"
            },
            base_rgba_of=illustration_base_rgba(inspector),
        )
        return plant, diagram, ctx, pctx, meshcat, starts, bindings

    def render(
        self,
        swing: SwingInput,
        settings: ExportSettings,
        indices: Sequence[int],
        overlay: OverlayFeed | None,
    ) -> Iterator[dict[str, Image8]]:
        from src.engines.physics_engines.drake.python.src.drake_meshcat_sink import (
            DrakeMeshcatSink,
        )

        self._apply_settings(settings)
        plant, diagram, ctx, pctx, meshcat, starts, bindings = self._build(swing)

        def set_state(index: int) -> None:
            full = np.zeros(plant.num_positions())
            full[starts] = swing.q[index]
            plant.SetPositions(pctx, full)
            diagram.ForcedPublish(ctx)

        set_state(indices[0])
        for node, prop in (
            ("/Background", "visible"),
            ("/Grid", "visible"),
            ("/Axes", "visible"),
        ):
            meshcat.SetProperty(node, prop, node == "/Background")
        glyphs = (
            MeshcatGlyphRenderer(DrakeMeshcatSink(meshcat))
            if overlay is not None
            else None
        )
        ball = (
            MeshcatBall(DrakeBallSink(meshcat), swing.ball)
            if swing.ball is not None
            else None
        )
        shading = (
            build_meshcat_shading(
                self.engine,
                settings.shading if settings.shading_enabled else None,
                meshcat.SetProperty,
                bindings,
            )
            if overlay is not None
            else None
        )
        try:
            with MeshcatPage(
                meshcat.web_url(), settings.width, settings.height
            ) as page:
                looks = view_lookats(settings, indices, swing)
                for pos, k in enumerate(indices):
                    set_state(k)
                    update_meshcat_frame(ball, glyphs, overlay, swing, k, shading)
                    tiles: dict[str, Image8] = {}
                    for view in settings.views:
                        eye, target = drake_meshcat_camera_pose(
                            view, looks[view][pos], settings.distance_m
                        )
                        meshcat.SetCameraPose(list(eye), list(target))
                        page.settle()
                        tiles[view] = page.screenshot()
                    yield tiles
        finally:
            if shading is not None:
                shading.log_summary()
