"""SHOW-A3 (#14552): before/after stills of the thoracic segment (headless MuJoCo).

BEFORE (``before_v1_rigid_*``) = the v1 driver, one rigid upper trunk
(``TorsoInput`` twist). AFTER (``after_v2_segmented_*``) = the v2 lumbar +
thoracic trunk derived from it. Both turn the SAME 55 degrees at the top; v2
splits it 17 % lumbar (9.4 deg) / 83 % thoracic (45.6 deg, inside the +-50 deg
soft ROM), Bruno 2015 (``src/config/spine_coupling_laws.json``). Synthetic
illustrations of the added DOF, not a fitted swing, not physics validation.

Run (offscreen; ``MUJOCO_GL=glfw`` on Windows)::

    python docs/development/full_body_models/evidence/trunk_segments/render_trunk_segments.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
TOTAL_TURN_DEG = 55.0  # top-of-backswing turn; keeps thoracic < the +-50 deg ROM
LUMBAR_SHARE = 0.17  # Bruno 2015 regional split, axial rotation
POSES = {"address": 0.0, "top": TOTAL_TURN_DEG}  # pose -> total axial turn (deg)


def _render(xml: str, qpos: dict[str, float], path: Path, azimuth: float) -> None:
    import mujoco
    import numpy as np
    from PIL import Image

    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth = model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    for name, value in qpos.items():
        data.qpos[model.joint(name).qposadr[0]] = math.radians(value)
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.subtree_com[0] + [0.0, 0.0, 0.25]
    cam.distance, cam.azimuth, cam.elevation = 1.9, azimuth, 5.0
    with mujoco.Renderer(model, 720, 720) as renderer:
        renderer.update_scene(data, cam)
        Image.fromarray(np.asarray(renderer.render())).save(path)


def main() -> int:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.full_body_v2_geometry import (
        derive_landmark_v2_spec,
    )

    from src.shared.python.motion_matching.full_body_v2_showpiece import SHOWPIECE_V1

    v1 = json.loads(SHOWPIECE_V1.read_text(encoding="utf-8"))
    v2 = derive_landmark_v2_spec(v1, provenance="SHOW-A3 before/after stills")
    docs = {"before_v1_rigid": v1, "after_v2_segmented": v2}
    for label, doc in docs.items():
        xml, _ = export_full_body_mjcf(json.dumps(doc).encode(), visual=True)
        for pose, turn in POSES.items():
            if doc is v1:
                q = {"TorsoInput": turn}
            else:
                q = {
                    "lumbar_axial_rotation": turn * LUMBAR_SHARE,
                    "thoracic_axial_rotation": turn * (1.0 - LUMBAR_SHARE),
                }
            for view, az in (("face_on", 90.0), ("dtl", 180.0)):
                _render(xml, q, OUT / f"{label}_{pose}_{view}.png", az)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    raise SystemExit(main())
