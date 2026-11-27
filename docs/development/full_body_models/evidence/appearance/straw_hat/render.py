"""Render controlled native before/after headwear stills at desktop and phone sizes.

Usage: MUJOCO_GL=egl PYTHONPATH=.:src python render.py --spec MODEL.json
       --appearance MODEL.appearance.json --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Render the same model and camera with its original and straw hats."""
    import imageio.v3 as iio
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.model_appearance import document_from_dict

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--appearance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.spec.read_bytes()
    for mode in ("before", "after"):
        doc = json.loads(args.appearance.read_text(encoding="utf-8"))
        if mode == "after":
            doc["head"]["cap_style"], doc["head"]["cap_material"] = (
                "straw_hat",
                "hat_straw",
            )
        xml, _ = export_full_body_mjcf(raw, appearance=document_from_dict(doc))
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 960
        camera = mujoco.MjvCamera()
        camera.lookat[:] = data.geom_xpos[model.geom("visual_head_skin_mesh").id]
        camera.distance, camera.azimuth, camera.elevation = 0.95, 180, -4
        for width, height in ((1280, 960), (390, 844)):
            renderer = mujoco.Renderer(model, height, width)
            try:
                renderer.update_scene(data, camera=camera)
                iio.imwrite(
                    args.out / f"{mode}_{width}x{height}.png", renderer.render().copy()
                )
            finally:
                renderer.close()


if __name__ == "__main__":
    main()
