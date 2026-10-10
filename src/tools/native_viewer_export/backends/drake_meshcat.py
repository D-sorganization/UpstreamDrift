"""Drake MeshCat native export: the spec robot in Drake's own MeshCat viewer.

The parity rollout drives a ``MultibodyPlant`` built from the specification
URDF export with the shared visual skeleton; Drake's ``MeshcatVisualizer``
publishes it and headless Chromium captures each requested view.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from importlib.util import find_spec
import json
import logging
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
    MeshcatGlyphRenderer,
)
from src.shared.python.golf_view_presets import drake_meshcat_camera_pose
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.tools.native_viewer_export.backends._club import club_parts, write_obj
from src.tools.native_viewer_export.backends._head import head_mesh_files
from src.tools.native_viewer_export.backends._meshcat_page import (
    MeshcatPage,
    playwright_unavailable_reason,
)
from src.tools.native_viewer_export.backends._scene import z_axis_frame
from src.tools.native_viewer_export.backends._subprocess import export_urdf
from src.tools.native_viewer_export.ball import resolve_address_ball
from src.tools.native_viewer_export.core import (
    ExportSettings,
    Image8,
    OverlayFeed,
    SwingInput,
    view_lookats,
)

logger = logging.getLogger(__name__)

_CAPSULE_RGBA = (0.75, 0.78, 0.85, 1.0)
_SHAPE_RGBA = (0.7, 0.72, 0.8, 1.0)
_FLOOR_RGBA = (0.35, 0.45, 0.3, 1.0)
_BALL_RGBA = (0.95, 0.95, 0.95, 1.0)


class DrakeMeshcatBackend:
    """Drake's MeshCat viewer, captured with headless Chromium."""

    engine = "drake"

    def unavailable_reason(self) -> str | None:
        if find_spec("pydrake") is None:
            return "pydrake is not installed"
        return playwright_unavailable_reason()

    def _register_meshes(
        self, plant: Any, inst: Any, links: Mapping[str, str], spec: Any
    ) -> tuple[str | None, str | None]:
        """Register the shared head and club meshes; return their spec bodies.

        The head mesh replaces the head capsule and the club mesh replaces the
        club ellipsoid hint (visual only). ``None`` means no mesh was drawn.
        """
        from pydrake.geometry import Mesh
        from pydrake.math import RigidTransform

        self._head_dir = tempfile.mkdtemp(prefix="ud_head_")
        heads = head_mesh_files(spec, Path(self._head_dir))
        for head in heads:
            plant.RegisterVisualGeometry(
                plant.GetBodyByName(links[head.body], inst),
                RigidTransform(),  # type: ignore[arg-type]
                Mesh(head.path),
                f"head_{head.name}",
                np.array(head.rgba),
            )
        club = club_parts(spec)
        if club is None:
            return (heads[0].body if heads else None), None
        club_body, parts = club
        self._mesh_dir = tempfile.TemporaryDirectory(prefix="club_meshes_")
        for part in parts:
            path = write_obj(part.mesh, self._mesh_dir.name, part.name)
            plant.RegisterVisualGeometry(
                plant.GetBodyByName(links[club_body], inst),
                RigidTransform(),
                Mesh(Path(path)),
                part.name,
                np.array(part.rgba),
            )
        return (heads[0].body if heads else None), club_body

    def _build(
        self, swing: SwingInput, *, ball: bool = True
    ) -> tuple[Any, Any, Any, Any, Any, list[int]]:
        from pydrake.geometry import (
            Box,
            Capsule,
            Ellipsoid,
            Mesh,
            Meshcat,
            MeshcatVisualizer,
            MeshcatVisualizerParams,
            Rgba,
            Role,
            Sphere,
        )
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.multibody.parsing import Parser
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph

        spec_bytes = swing.bundle.spec_bytes
        skeleton = derive_visual_skeleton(json.loads(spec_bytes))
        xml, meta = export_urdf(spec_bytes)
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        inst = Parser(plant).AddModelsFromString(xml, "urdf")[0]
        links = meta["body_links"]
        plant.WeldFrames(
            plant.world_frame(), plant.GetBodyByName(links["world"], inst).body_frame()
        )
        n = 0
        head_body, club_body = self._register_meshes(
            plant, inst, links, json.loads(spec_bytes)
        )
        for cap in skeleton.capsules:
            if cap.body == head_body:
                continue
            rot, centre, length = z_axis_frame(cap.start_m, cap.end_m)
            plant.RegisterVisualGeometry(
                plant.GetBodyByName(links[cap.body], inst),
                RigidTransform(RotationMatrix(rot), centre),  # type: ignore[arg-type]
                Capsule(cap.radius_m, length),
                f"cap{n}",
                np.array(_CAPSULE_RGBA),
            )
            n += 1
        for shp in skeleton.shapes:
            if shp.body == club_body:
                continue  # the mesh head replaces the ellipsoid hint
            shape = (
                Ellipsoid(*shp.half_size_m)
                if shp.kind == "ellipsoid"
                else Box(*(2.0 * np.asarray(shp.half_size_m)))
            )
            plant.RegisterVisualGeometry(
                plant.GetBodyByName(links[shp.body], inst),
                RigidTransform(np.asarray(shp.center_m)),  # type: ignore[arg-type]
                shape,
                f"shp{n}",
                np.array(_SHAPE_RGBA),
            )
            n += 1
        plant.RegisterVisualGeometry(
            plant.world_body(),
            RigidTransform(np.array([1.0, 0.0, skeleton.ground.height_m - 0.005])),  # type: ignore[arg-type]
            Box(6.0, 6.0, 0.01),
            "floor",
            np.array(_FLOOR_RGBA),
        )
        if ball:
            resolved = resolve_address_ball(swing)
            if resolved.position_m is None:
                logger.warning("skipping decorative ball: %s", resolved.reason)
            else:
                plant.RegisterVisualGeometry(
                    plant.world_body(),
                    RigidTransform(resolved.position_m),  # type: ignore[arg-type]
                    Sphere(resolved.radius_m),
                    "visual_ball",
                    np.array(_BALL_RGBA),
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
        return plant, diagram, ctx, pctx, meshcat, starts

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

        plant, diagram, ctx, pctx, meshcat, starts = self._build(
            swing, ball=settings.ball
        )

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
        with MeshcatPage(meshcat.web_url(), settings.width, settings.height) as page:
            looks = view_lookats(settings, indices, overlay)
            for pos, k in enumerate(indices):
                set_state(k)
                if glyphs is not None and overlay is not None:
                    glyphs.update(overlay.glyphs_at(k))
                tiles: dict[str, Image8] = {}
                for view in settings.views:
                    eye, target = drake_meshcat_camera_pose(
                        view, looks[view][pos], settings.distance_m
                    )
                    meshcat.SetCameraPose(list(eye), list(target))
                    page.settle()
                    tiles[view] = page.screenshot()
                yield tiles
