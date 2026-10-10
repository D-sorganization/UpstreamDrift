"""Render worker: MuJoCo offscreen ``mujoco.Renderer`` with the appearance layer.

The specification is exported once with the shared MuJoCo appearance document
(smooth body segments, garments, shoes, head, club shaft/grip/head meshes,
ground, sky and lights) and rendered with the golf view presets. The rollout
``q`` is mapped to ``qpos`` through the bundle's coordinate order, so the
pose shown is the same-input pose every other engine displays. 3D force and
torque glyphs are added to the MuJoCo scene before each render. The
decorative address ball (GCV-13, #11719), already resolved by the caller
from the swing's true address frame, is attached with
:func:`backends._ball.set_decorative_ball`, which replaces whatever the
appearance layer's own static-reference-pose fallback may have drawn.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any
from defusedxml import ElementTree as ET

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
    add_glyphs_to_scene,
)
from src.shared.python.model_appearance import document_from_dict
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.backends._ball import set_decorative_ball
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.backends.myosuite_worker import make_camera

APPEARANCE = {"schema_version": "appearance-v1"}


def build_scene_xml(spec_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Full-body MJCF with the default appearance document (body, club, scene).

    Postcondition: the returned summary's ``visual_layer`` records the mesh
    count and whether the club assembly was attached.
    """
    return export_full_body_mjcf(spec_bytes, appearance=document_from_dict(APPEARANCE))


def grow_offscreen_buffer(model: Any, width: int, height: int) -> None:
    """Enlarge ``model``'s offscreen framebuffer (default 640x480) to fit a frame."""
    buffer = model.vis.global_
    buffer.offwidth = max(width, int(buffer.offwidth))
    buffer.offheight = max(height, int(buffer.offheight))


def main(job_path: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    job = WorkerJob.load(Path(job_path))
    bundle = InputBundle.load(Path(job.bundle_path))
    q = np.load(job.q_path)
    xml, _ = build_scene_xml(bundle.spec_bytes)
    root = ET.fromstring(xml)
    set_decorative_ball(root, job.ball_position_m)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # the default offscreen buffer is 640x480; grow it to the requested size
    grow_offscreen_buffer(model, job.width, job.height)
    data = mujoco.MjData(model)
    adr = [model.joint(n).qposadr[0] for n in bundle.coordinate_order]
    glyph_sets = job.load_glyph_sets()
    cams = {v: make_camera(mujoco, v, job) for v in job.views}
    renderer = mujoco.Renderer(model, job.height, job.width)
    try:
        for pos, k in enumerate(job.indices):
            data.qpos[adr] = q[k]
            mujoco.mj_forward(model, data)
            for view in job.views:
                cams[view].lookat[:] = job.lookat_for(view, pos)
                renderer.update_scene(data, camera=cams[view])
                if glyph_sets is not None:
                    add_glyphs_to_scene(renderer.scene, glyph_sets[pos])
                np.save(job.frame_path(view, pos), renderer.render().copy())
    finally:
        renderer.close()


if __name__ == "__main__":
    main(sys.argv[1])
