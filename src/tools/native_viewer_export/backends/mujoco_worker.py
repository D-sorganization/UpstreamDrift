"""Render worker: MuJoCo offscreen ``mujoco.Renderer`` with the appearance layer.

The specification is exported once with the shared MuJoCo appearance document
(smooth body segments, garments, shoes, head, club shaft/grip/head meshes,
ground, sky and lights) and rendered with the golf view presets. The rollout
``q`` is mapped to ``qpos`` through the bundle's coordinate order, so the
pose shown is the same-input pose every other engine displays. 3D force and
torque glyphs are added to the MuJoCo scene before each render. With segment
shading on, a body skin is cut into per-segment pieces that take the load
colours (``_skin_shading``); otherwise the model is the exported one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.shared.python.model_appearance import document_from_dict
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.backends._ball import draw_job_ball
from src.tools.native_viewer_export.backends._scene_overlays import (
    apply_scene_overlays,
)
from src.tools.native_viewer_export.backends._scene_lighting import light_scene
from src.tools.native_viewer_export.backends._skin_shading import (
    compile_shaded_model,
)
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.backends.myosuite_worker import make_camera
from src.tools.native_viewer_export.shading import (
    log_coverage,
    mujoco_scene_coverage,
)

#: The exporter's own decorative ball is off: it is placed from the static
#: reference pose, not the address pose, so it landed away from the face and
#: duplicated the worker's ball (``draw_job_ball``, the single shared
#: placement from ``native_viewer_export.ball``; #12955).
APPEARANCE = {**default_document_dict(), "ball": {"enabled": False}}


def build_scene_xml(
    spec_bytes: bytes,
    appearance: Mapping[str, Any] | None = None,
    show_contact_spheres: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Full-body MJCF with the appearance document (body, club, scene).

    ``appearance`` is the resolved document of the job (``None``: the contract
    default); the exporter's own ball stays off either way. Contact spheres
    stay out of the drawing unless ``show_contact_spheres`` is true.

    Postcondition: the returned summary's ``visual_layer`` records the mesh
    count and whether the club assembly was attached.
    """
    document = {**(APPEARANCE if appearance is None else appearance)}
    document["ball"] = {"enabled": False}
    return export_full_body_mjcf(
        spec_bytes,
        appearance=document_from_dict(document),
        show_contact_spheres=show_contact_spheres,
    )


SCENE_EXTENT_PATH = Path(__file__).with_name("scene_extent_settings.json")


@dataclass(frozen=True)
class SceneExtent:
    """Validated ``scene_extent_settings.json`` (``ValueError`` when invalid)."""

    ground_half_extent_m: float

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.ground_half_extent_m) and self.ground_half_extent_m > 0
        ):
            raise ValueError("ground_half_extent_m must be positive and finite")


def load_scene_extent(path: Path | None = None) -> SceneExtent:
    """Read the scene extent settings (``KeyError`` for a missing key)."""
    raw = json.loads((path or SCENE_EXTENT_PATH).read_text(encoding="utf-8"))
    return SceneExtent(float(raw["ground_half_extent_m"]))


def widen_ground(model: Any, half_extent_m: float) -> None:
    """Make the ``visual_floor`` plane ``half_extent_m`` wide each way.

    A model without that plane is left alone. Raises ``ValueError`` for an
    extent that is not positive and finite.
    """
    SceneExtent(half_extent_m)
    floor = model.geom("visual_floor") if "visual_floor" in _geom_names(model) else None
    if floor is not None:
        floor.size[:2] = half_extent_m


def _geom_names(model: Any) -> set[str]:
    return {model.geom(i).name for i in range(model.ngeom)}


def grow_offscreen_buffer(model: Any, width: int, height: int) -> None:
    """Enlarge ``model``'s offscreen framebuffer (default 640x480) to fit a frame."""
    buffer = model.vis.global_
    buffer.offwidth = max(width, int(buffer.offwidth))
    buffer.offheight = max(height, int(buffer.offheight))


def write_depth(renderer: Any, job: WorkerJob, view: str, position: int) -> None:
    """Save the scene depth (metres) of the current scene where the job asks for it."""
    if not job.depth_positions or not job.depth_positions[position]:
        return
    renderer.enable_depth_rendering()
    try:
        np.save(job.depth_path(view, position), renderer.render().copy())
    finally:
        renderer.disable_depth_rendering()


def main(job_path: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    job = WorkerJob.load(Path(job_path))
    bundle = InputBundle.load(Path(job.bundle_path))
    q = np.load(job.q_path)
    xml, _ = build_scene_xml(bundle.spec_bytes, job.appearance)
    xml = light_scene(xml, job.lighting)  # studio lights and sun unless opted in
    axial_frames, shading = job.load_axial_frames(), job.load_shading()
    model, skin = compile_shaded_model(xml, axial_frames, shading)
    # the default offscreen buffer is 640x480; grow it to the requested size
    grow_offscreen_buffer(model, job.width, job.height)
    widen_ground(model, load_scene_extent().ground_half_extent_m)
    data = mujoco.MjData(model)
    adr = [model.joint(n).qposadr[0] for n in bundle.coordinate_order]
    glyph_sets = job.load_glyph_sets()
    if axial_frames is not None and shading is not None:
        log_coverage("mujoco", *mujoco_scene_coverage(model, axial_frames))
    cams = {v: make_camera(mujoco, v, job) for v in job.views}
    renderer = mujoco.Renderer(model, job.height, job.width)
    try:
        for pos, k in enumerate(job.indices):
            data.qpos[adr] = q[k]
            mujoco.mj_forward(model, data)
            if axial_frames is not None and shading is not None:
                skin.colour(model, axial_frames[pos], shading)
            for view in job.views:
                cams[view].lookat[:] = job.lookat_for(view, pos)
                renderer.update_scene(data, camera=cams[view])
                apply_scene_overlays(
                    model,
                    renderer.scene,
                    pos,
                    axial_frames,
                    shading,
                    glyph_sets,
                    keep_geoms=skin.keep_geoms,
                )
                draw_job_ball(renderer.scene, job, pos)
                np.save(job.frame_path(view, pos), renderer.render().copy())
                write_depth(renderer, job, view, pos)
    finally:
        renderer.close()


if __name__ == "__main__":
    main(sys.argv[1])
