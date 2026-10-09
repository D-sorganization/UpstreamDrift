"""Render driver/iron address, top and finish stills and a 0.5x clip (MuJoCo).

Headless (EGL): ``MUJOCO_GL=egl PYTHONPATH=.:src python3 \
scripts/render_head_visuals.py --bundle driver.npz --out DIR``. Uses the
appearance layer with the visible head, replaying the bundle reference
trajectory; the clip is sampled at 60 Hz of simulation time and played at
30 fps, i.e. half speed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import document_from_dict
from src.shared.python.motion_matching.same_input import InputBundle

logger = logging.getLogger(__name__)
WIDTH, HEIGHT = 960, 720
SAMPLE_HZ, PLAY_FPS = 60, 30  # 30 fps playback of 60 Hz samples = 0.5x


def _poses(model, data, bundle, adr, club_body):
    import mujoco

    heights = []
    for q in bundle.reference_q:
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        heights.append(data.xpos[club_body][2])
    top = int(np.argmax(heights[: len(heights) * 2 // 3]))
    return {"address": 0, "top": top, "finish": len(heights) - 1}


def main() -> None:
    import imageio
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="driver")
    parser.add_argument("--headwear", default="hair", choices=("none", "hair", "cap"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bundle = InputBundle.load(args.bundle)
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"headwear": args.headwear}}
    )
    xml, _ = exporter.export_full_body_mjcf(bundle.spec_bytes, appearance=doc)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    adr = [model.joint(n).qposadr[0] for n in bundle.coordinate_order]
    club = next(
        i
        for i in range(model.nbody)
        if "clubface"
        in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
    )
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 2.5, -5.0, 330.0

    def frame(q: np.ndarray) -> np.ndarray:
        data.qpos[adr] = q
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.subtree_com[1]
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    try:
        for label, k in _poses(model, data, bundle, adr, club).items():
            imageio.imwrite(
                args.out / f"{args.name}_mujoco_{label}.png",
                frame(bundle.reference_q[k]),
            )
        step = max(1, round(1.0 / (bundle.dt_s * SAMPLE_HZ)))
        frames = [frame(q) for q in bundle.reference_q[::step]]
        imageio.mimsave(
            args.out / f"{args.name}_mujoco_half_speed.mp4", frames, fps=PLAY_FPS
        )
    finally:
        renderer.close()
    logger.info("wrote %d clip frames to %s", len(frames), args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
