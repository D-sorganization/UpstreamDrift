"""Render worker: MyoSuite's ``MujocoEnv`` / ``MJRenderer`` arena over EGL.

Runs in a Python that can import ``myosuite`` (often a separate virtual
environment). The specification MJCF is merged with the MyoSuite arena scene
assets (floor texture, headlight), loaded into a MyoSuite ``MujocoEnv`` and
rendered with the env's own renderer, with 3D force/torque glyphs added to the
MuJoCo scene before each render.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET  # noqa: S405 - trusted, locally generated MJCF

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
from src.tools.native_viewer_export.backends._worker_job import WorkerJob

SCENE_FILE = ("simhive", "myo_sim", "scene", "myosuite_quad.xml")


def arena_scene_path() -> Path:
    """The MyoSuite arena scene shipped with the installed package."""
    import myosuite

    return Path(myosuite.__file__).parent.joinpath(*SCENE_FILE)


def merge_arena(spec_xml: str, ground_height_m: float, width: int, height: int) -> str:
    """Specification MJCF with the MyoSuite arena floor, headlight and map."""
    root = ET.fromstring(spec_xml)  # noqa: S314 - generated locally
    scene_path = arena_scene_path()
    scene = ET.parse(scene_path).getroot()  # noqa: S314
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for elem in scene.find("asset") or []:
        if elem.get("file"):
            elem.set(
                "file", str((scene_path.parent / Path(elem.get("file")).name).resolve())
            )
        asset.append(elem)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    for elem in scene.find("visual") or []:
        if elem.tag in ("headlight", "map", "scale"):
            old = visual.find(elem.tag)
            if old is not None:
                visual.remove(old)
            visual.append(elem)
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", str(max(width, 640)))
    glob.set("offheight", str(max(height, 480)))
    world = root.find("worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name") == "visual_floor":
            world.remove(geom)
    names = {e.get("name") for e in root.iter() if e.get("name")}
    ground = None
    for elem in scene.find("worldbody"):
        if elem.tag != "geom" or elem.get("group") == "2":
            continue
        if elem.get("name") in names:
            elem.set("name", "myo_" + elem.get("name"))
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
    from myosuite.envs.env_base import MujocoEnv

    job = WorkerJob.load(Path(job_path))
    bundle = InputBundle.load(Path(job.bundle_path))
    q = np.load(job.q_path)
    skeleton = derive_visual_skeleton(json.loads(bundle.spec_bytes))
    xml, _ = export_full_body_mjcf(bundle.spec_bytes, visual=True)
    scene_xml = merge_arena(xml, skeleton.ground.height_m, job.width, job.height)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.xml"
        path.write_text(scene_xml, encoding="utf-8")
        env = MujocoEnv(str(path))
    model, data = env.mj_model, env.mj_data
    adr = [model.joint(n).qposadr[0] for n in bundle.coordinate_order]
    # warm up MyoSuite's own renderer, then render through it with glyphs added
    env.mj_renderer.render_offscreen(width=job.width, height=job.height, camera_id=-1)
    renderer = env.mj_renderer._renderer  # noqa: SLF001 - the env's mujoco.Renderer
    option = env.mj_renderer._scene_option  # noqa: SLF001
    glyph_sets = None
    if job.glyphs_path:
        glyph_sets = [
            GlyphSet.from_dict(d)
            for d in json.loads(Path(job.glyphs_path).read_text(encoding="utf-8"))
        ]
    cams = {v: make_camera(mujoco, v, job) for v in job.views}
    for pos, k in enumerate(job.indices):
        data.qpos[adr] = q[k]
        mujoco.mj_forward(model, data)
        for view in job.views:
            renderer.update_scene(data, camera=cams[view], scene_option=option)
            if glyph_sets is not None:
                add_glyphs_to_scene(renderer.scene, glyph_sets[pos])
            np.save(job.frame_path(view, pos), renderer.render().copy())


if __name__ == "__main__":
    main(sys.argv[1])
