"""Side-by-side 0.5x head-gaze clips (OSV-3, #11729), MuJoCo, headless.

Left: gaze weight 0 (marker-faithful). Right: gaze on. Each panel draws a red
head-forward glyph from the eye point along the gaze axis, so head orientation
is visible without a face mesh, and the ball at address (white). Inputs are two
pipeline run directories (``ik_trajectory.npz`` + ``full_body_spec_hipcal_scaled.json``).

    MUJOCO_GL=egl python3 -m scripts.render_head_gaze_clips RUN_W0 RUN_ON OUT.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SLOWDOWN = 0.5
HEIGHT, WIDTH = 480, 480
GLYPH_LENGTH_M = 0.6


def _load(run: Path):  # noqa: ANN202
    spec_bytes = (run / "full_body_spec_hipcal_scaled.json").read_bytes()
    q = np.load(run / "ik_trajectory.npz")
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    return spec_bytes, q["q_ref"], q["time_s"], receipt["head_gaze"]


def _connector(scene, p0, p1, radius, rgba) -> None:  # noqa: ANN001
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(p0, dtype=float),
        np.asarray(p1, dtype=float),
    )
    scene.ngeom += 1


def _panel_frames(run: Path, label: str, azimuth: float, stride: int):  # noqa: ANN202
    import cv2
    import mujoco

    from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
    from src.engines.physics_engines.mujoco.python.visual_layer import (
        add_scene_marker,
    )
    from src.shared.python.motion_matching import gaze
    from src.shared.python.motion_matching.pipeline import gaze_residual as gr
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    spec_bytes, q, times, block = _load(run)
    spec = json.loads(spec_bytes)
    plant = get_plant("mujoco", spec)
    att = {
        k: (v["body"], tuple(v["offset_m"]))
        for k, v in spec["marker_attachments"].items()
        if v["offset_m"] is not None
    }
    kin = plant.create_ik(dict(list(att.items())[:5]), ik_backend="lm")
    head_r, head_t = gr.frame_poses(kin, q, gr.HEAD_FRAME)
    eyes = gaze.eye_point(head_r, head_t)
    forward = gaze.gaze_direction(head_r)
    ball = np.asarray(block["plan"]["ball_at_address_m"])
    impact = int(block["plan"]["impact_index"])

    xml, _ = exporter.export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    names = tuple(kin.coordinate_order)
    addresses = [model.joint(n).qposadr[0] for n in names]
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = 0.5 * (eyes[0] + ball) - np.array([0.0, 0.0, 0.25])
    cam.distance, cam.azimuth, cam.elevation = 2.6, azimuth, -10.0
    out = []
    for k in range(0, q.shape[0], stride):
        data.qpos[addresses] = q[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        add_scene_marker(renderer.scene, ball, 0.021335, (1, 1, 1, 1))
        tip = eyes[k] + GLYPH_LENGTH_M * forward[k]
        _connector(renderer.scene, eyes[k], tip, 0.008, (0.95, 0.1, 0.1, 1))
        add_scene_marker(renderer.scene, eyes[k], 0.014, (1.0, 0.8, 0.1, 1))
        img = np.ascontiguousarray(renderer.render())
        phase = "ball" if k <= impact + 3 else "release"
        text = f"{label}  t={times[k] - times[impact]:+.2f}s  ({phase})"
        cv2.putText(
            img,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        out.append(img)
    return out, 1.0 / float(np.median(np.diff(times)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_w0", type=Path)
    parser.add_argument("run_on", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--azimuth", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    import imageio

    left, rate = _panel_frames(args.run_w0, "gaze weight 0", args.azimuth, args.stride)
    right, _ = _panel_frames(args.run_on, "gaze on", args.azimuth, args.stride)
    fps = rate / args.stride * SLOWDOWN
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.out, fps=fps, codec="libx264", quality=7) as writer:
        for a, b in zip(left, right, strict=True):
            writer.append_data(np.hstack([a, b]))


if __name__ == "__main__":
    main()
