"""Render worker: MyoSuite's ``MujocoEnv`` / ``MJRenderer`` arena over EGL.

Runs in a Python that can import ``myosuite`` (often a separate virtual
environment). The specification MJCF is merged with the MyoSuite arena scene
assets (floor texture, headlight), loaded into a MyoSuite ``MujocoEnv`` and
rendered with the env's own renderer, with 3D force/torque glyphs added to the
MuJoCo scene before each render. The decorative address ball (GCV-13,
#11719), already resolved by the caller, is attached with
:func:`backends._ball.set_decorative_ball` after the arena scene is merged,
so it never depends on the plain visual layer's own static-reference-pose
fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from defusedxml import ElementTree as ET

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
    add_glyphs_to_scene,
)
from src.shared.python.force_overlay.glyphs import GlyphSet
from src.shared.python.golf_view_presets import mujoco_camera_params
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.tools.native_viewer_export.backends._ball import set_decorative_ball
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.backends.myosuite_compat import (
    find_scene,
    import_mj_renderer,
)


def arena_scene_path() -> Path:
    """The MyoSuite arena scene shipped with the installed package."""
    return find_scene()


def _append_child(parent: Any, tag: str) -> Any:
    """Create and append an empty child element (``SubElement`` without stdlib xml)."""
    child = parent.makeelement(tag, {})
    parent.append(child)
    return child


def merge_arena(spec_xml: str, ground_height_m: float, width: int, height: int) -> str:
    """Specification MJCF with the MyoSuite arena floor, headlight and map."""
    root = ET.fromstring(spec_xml)
    scene_path = arena_scene_path()
    scene = ET.parse(scene_path).getroot()
    scene_asset = scene.find("asset")
    scene_visual = scene.find("visual")
    scene_world = scene.find("worldbody")
    if scene_asset is None or scene_visual is None or scene_world is None:
        raise ValueError(f"{scene_path} lacks asset, visual or worldbody sections")
    asset = root.find("asset")
    if asset is None:
        asset = _append_child(root, "asset")
    for elem in scene_asset:
        if elem.get("file"):
            elem.set(
                "file",
                str((scene_path.parent / Path(str(elem.get("file"))).name).resolve()),
            )
        asset.append(elem)
    visual = root.find("visual")
    if visual is None:
        visual = _append_child(root, "visual")
    for elem in scene_visual:
        if elem.tag in ("headlight", "map", "scale"):
            old = visual.find(elem.tag)
            if old is not None:
                visual.remove(old)
            visual.append(elem)
    glob = visual.find("global")
    if glob is None:
        glob = _append_child(visual, "global")
    glob.set("offwidth", str(max(width, 640)))
    glob.set("offheight", str(max(height, 480)))
    world = root.find("worldbody")
    if world is None:
        raise ValueError("specification MJCF has no worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name") == "visual_floor":
            world.remove(geom)
    names = {e.get("name") for e in root.iter() if e.get("name")}
    ground = None
    for elem in scene_world:
        if elem.tag != "geom" or elem.get("group") == "2":
            continue
        if elem.get("name") in names:
            elem.set("name", "myo_" + str(elem.get("name")))
        world.append(elem)
        if elem.get("name") in ("ground", "myo_ground"):
            ground = elem
    if ground is not None:
        ground.set("pos", f"1 0 {ground_height_m}")
        ground.set("contype", "0")
        ground.set("conaffinity", "0")
    return ET.tostring(root, encoding="unicode")


def make_camera(mujoco: Any, view: str, job: WorkerJob) -> Any:
    """``MjvCamera`` for a golf view preset."""
    params = mujoco_camera_params(view, job.lookat_m, job.distance_m)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = params.lookat
    cam.distance, cam.azimuth, cam.elevation = (
        params.distance,
        params.azimuth,
        params.elevation,
    )
    return cam


def main(job_path: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    mj_renderer_cls = import_mj_renderer()

    job = WorkerJob.load(Path(job_path))
    bundle = InputBundle.load(Path(job.bundle_path))
    q = np.load(job.q_path)
    skeleton = derive_visual_skeleton(json.loads(bundle.spec_bytes))
    xml, _ = export_full_body_mjcf(bundle.spec_bytes, visual=True, with_head=True)
    scene_xml = merge_arena(xml, skeleton.ground.height_m, job.width, job.height)
    scene_root = ET.fromstring(scene_xml)
    set_decorative_ball(scene_root, job.ball_position_m)
    scene_xml = ET.tostring(scene_root, encoding="unicode")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.xml"
        path.write_text(scene_xml, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    myo_renderer = mj_renderer_cls(model, data)
    adr = [model.joint(n).qposadr[0] for n in bundle.coordinate_order]
    # warm up MyoSuite's own renderer, then render through it with glyphs added
    myo_renderer.render_offscreen(width=job.width, height=job.height, camera_id=-1)
    renderer = myo_renderer._renderer  # noqa: SLF001 - the env's mujoco.Renderer
    option = myo_renderer._scene_option  # noqa: SLF001
    glyph_sets = job.load_glyph_sets()
    cams = {v: make_camera(mujoco, v, job) for v in job.views}
    for pos, k in enumerate(job.indices):
        data.qpos[adr] = q[k]
        mujoco.mj_forward(model, data)
        for view in job.views:
            cams[view].lookat[:] = job.lookat_for(view, pos)
            renderer.update_scene(data, camera=cams[view], scene_option=option)
            if glyph_sets is not None:
                add_glyphs_to_scene(renderer.scene, glyph_sets[pos])
            np.save(job.frame_path(view, pos), renderer.render().copy())


if __name__ == "__main__":
    main(sys.argv[1])
