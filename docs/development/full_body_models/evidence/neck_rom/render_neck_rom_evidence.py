"""SHOW-A4 (#14553): before/after stills of the neck range-of-motion limit.

"Before" poses the head at an over-range command (axial 120, lateral 70 deg)
on a model whose neck joints are unlimited, as the export was before A4;
"after" clamps the same command to the exported joint range, which is the
anatomical envelope in ``neck_rom.py``.  Offscreen MuJoCo; set
``MUJOCO_GL=egl``/``osmesa`` when the default backend is unavailable.
Stills are visual evidence only, not physics validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from src.engines.physics_engines.mujoco.python.full_body_mjcf import (  # noqa: E402
    export_full_body_mjcf,
)
from src.shared.python.motion_matching.neck_rom import NECK_ROM  # noqa: E402

SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
OUT = Path(__file__).resolve().parent
COMMAND_DEG = {"NeckInputX": 70.0, "NeckInputY": -30.0, "NeckInputZ": 120.0}


def _pose(model: mujoco.MjModel, clamp: bool) -> mujoco.MjData:
    data = mujoco.MjData(model)
    for name, deg in COMMAND_DEG.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q = np.radians(deg)
        if clamp and model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q = float(np.clip(q, lo, hi))
        data.qpos[model.jnt_qposadr[jid]] = q
    mujoco.mj_forward(model, data)
    return data


def main() -> None:
    import imageio.v3 as iio

    xml, _ = export_full_body_mjcf(SPEC.read_bytes(), visual=True)
    after_model = mujoco.MjModel.from_xml_string(xml)
    before_model = mujoco.MjModel.from_xml_string(xml)
    before_model.jnt_limited[:] = 0  # the pre-A4 export left every joint free
    head = mujoco.mj_name2id(after_model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.2, 150.0, -10.0
    frames = []
    for model, clamp in ((before_model, False), (after_model, True)):
        data = _pose(model, clamp)
        cam.lookat[:] = data.xpos[head]
        with mujoco.Renderer(model, 480, 480) as r:
            r.update_scene(data, cam)
            frames.append(r.render())
    iio.imwrite(OUT / "neck_rom_before_after.png", np.concatenate(frames, axis=1))
    for name, rom in NECK_ROM.items():
        print(name, rom.motion, rom.lower_deg, rom.upper_deg)


if __name__ == "__main__":
    main()
