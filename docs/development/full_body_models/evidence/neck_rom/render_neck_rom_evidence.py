"""SHOW-A4 (#14553): before/after stills of the neck range-of-motion limit.

"Before" poses the head at an over-range command (axial 120, lateral 70 deg)
on a model whose neck joints are unlimited, as the export was before A4;
"after" clamps the same command to the exported joint range, which is the
anatomical envelope in ``neck_rom.py``.  Offscreen MuJoCo; set
``MUJOCO_GL=egl``/``osmesa`` when the default backend is unavailable.
Stills are visual evidence only, not physics validation.
"""

from __future__ import annotations

import json
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


def _driver_excursion() -> dict[str, dict[str, object]]:
    """Min/max neck angle of the matched driver swing against the limits."""
    run = ROOT / "docs/development/full_body_models/evidence/ground_support"
    run = run / "anthro_driver_drake"
    spec = json.loads((run / "full_body_spec_hipcal_scaled.json").read_text("utf-8"))
    q = np.load(run / "dynamics_record.npz")["q_track"]
    out: dict[str, dict[str, object]] = {}
    for name, rom in NECK_ROM.items():
        deg = np.degrees(q[:, spec["coordinate_order"].index(name)])
        out[name] = {
            "motion": rom.motion,
            "min_deg": round(float(deg.min()), 1),
            "max_deg": round(float(deg.max()), 1),
            "limit_deg": [rom.lower_deg, rom.upper_deg],
        }
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text((8, 8), text, fill=(255, 255, 255))
    return np.asarray(pil)


def main() -> None:
    import imageio.v3 as iio

    xml, _ = export_full_body_mjcf(SPEC.read_bytes(), visual=True)
    after_model = mujoco.MjModel.from_xml_string(xml)
    before_model = mujoco.MjModel.from_xml_string(xml)
    before_model.jnt_limited[:] = 0  # the pre-A4 export left every joint free
    head = mujoco.mj_name2id(
        after_model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.2, 150.0, -10.0
    frames = []
    for model, clamp in ((before_model, False), (after_model, True)):
        data = _pose(model, clamp)
        cam.lookat[:] = data.xpos[head]
        with mujoco.Renderer(model, 480, 480) as r:
            r.update_scene(data, cam)
            cmd = ", ".join(f"{k[-1]}={v:g}" for k, v in COMMAND_DEG.items())
            shown = "unlimited (before)" if not clamp else "clamped to ROM (after)"
            frames.append(_label(r.render(), f"{shown}: command deg {cmd}"))
    iio.imwrite(OUT / "neck_rom_before_after.png", np.concatenate(frames, axis=1))
    excursion = _driver_excursion()
    (OUT / "driver_swing_neck_excursion.json").write_text(
        json.dumps(excursion, indent=2) + chr(10), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
