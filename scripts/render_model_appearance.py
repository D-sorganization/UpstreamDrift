"""Render before/after stills and a clip of the MuJoCo appearance layer.

Headless (EGL): ``MUJOCO_GL=egl PYTHONPATH=.:src python3 \
scripts/render_model_appearance.py --out DIR``. Renders the full-body model at
a hand-set address-like pose with the legacy capsule layer ("before") and with
the default appearance document ("after") at 960x720.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import document_from_dict

logger = logging.getLogger(__name__)
SPEC = Path("docs/development/full_body_models/full_body_spec_v1.json")
WIDTH, HEIGHT = 960, 720
# Visual demonstration pose (radians) from a least-squares reach target
# (feet apart, hands together in front); not a fitted or qualified posture.
DEMO_POSE = {
    "TranslationInputX": 0.003,
    "TranslationInputY": 0.644,
    "TranslationInputZ": 0.647,
    "HipInputX": -0.328,
    "HipInputY": -0.437,
    "HipInputZ": 0.548,
    "TorsoInput": 0.021,
    "SpineInputX": 0.111,
    "SpineInputY": 0.094,
    "LScapInputX": -0.131,
    "LScapInputY": -0.969,
    "LSInputX": 0.028,
    "LSInputY": 0.673,
    "LSInputZ": 0.148,
    "LEInput": -0.274,
    "RScapInputX": 0.156,
    "RScapInputY": 0.84,
    "RSInputX": -0.043,
    "RSInputY": 0.739,
    "RSInputZ": -0.176,
    "REInput": -0.354,
    "hip_flexion_r": 0.113,
    "hip_adduction_r": 0.422,
    "hip_rotation_r": 0.042,
    "knee_angle_r": 0.128,
    "ankle_angle_r": -0.517,
    "subtalar_angle_r": 0.138,
    "hip_flexion_l": 0.117,
    "hip_adduction_l": 0.003,
    "hip_rotation_l": -0.145,
    "knee_angle_l": 0.112,
    "ankle_angle_l": -0.226,
    "subtalar_angle_l": -0.054,
}


def _render(xml: str, azimuths: list[float], pose: dict[str, float]):
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for name, value in pose.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.subtree_com[1]
    camera.distance, camera.elevation = 3.4, -10.0
    frames = []
    try:
        for azimuth in azimuths:
            camera.azimuth = azimuth
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return frames


def main() -> None:
    import imageio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=SPEC)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.spec.read_bytes()
    before_xml, _ = exporter.export_full_body_mjcf(raw, visual=True)
    # The legacy layer keeps MuJoCo's 640x480 buffer; enlarge it for the render.
    before_xml = before_xml.replace(
        "</mujoco>",
        f'<visual><global offwidth="{WIDTH}" offheight="{HEIGHT}"/></visual></mujoco>',
    )
    doc = document_from_dict({"schema_version": "appearance-v1"})
    after_xml, meta = exporter.export_full_body_mjcf(raw, appearance=doc)
    for label, xml in (("before", before_xml), ("after", after_xml)):
        for azimuth in (135.0, 45.0):
            (frame,) = _render(xml, [azimuth], DEMO_POSE)
            imageio.imwrite(args.out / f"{label}_az{int(azimuth)}.png", frame)
    clip = _render(
        after_xml, list(np.linspace(0.0, 360.0, 72, endpoint=False)), DEMO_POSE
    )
    imageio.mimsave(args.out / "after_orbit.gif", clip, duration=60, loop=0)
    logger.info("appearance layer: %s", meta["visual_layer"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
