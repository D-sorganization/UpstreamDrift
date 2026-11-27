"""Render native period-hat head comparisons in three views at two sizes."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.shared.python.model_appearance import document_from_dict
from src.shared.python.model_appearance.head import (
    head_frame_in_body,
    resolve_head_anchor,
)
from src.shared.python.model_appearance.player_stature import spec_body_poses
from src.shared.python.motion_matching.execution.golfer_reference_specs import (
    reference_spec,
)


def main() -> None:
    """Render selected hats on a common neutral reference golfer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--styles", nargs="+", default=["straw_hat", "flat_cap", "newsboy_cap"]
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = reference_spec("driver")
    raw = json.dumps(spec).encode()
    for style in args.styles:
        doc = document_from_dict(
            {
                "schema_version": "appearance-v1",
                "head": {
                    "cap_style": style,
                    "cap_material": "hat_straw" if style == "straw_hat" else "cap_navy",
                },
            }
        )
        xml, _ = export_full_body_mjcf(raw, appearance=doc)
        model = mujoco.MjModel.from_xml_string(xml)
        model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 960
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        heads = [
            i
            for i in range(model.ngeom)
            if (model.geom(i).name or "").startswith("visual_head_")
        ]
        anchor = resolve_head_anchor(spec)
        if anchor is None or not heads:
            raise ValueError("reference model has no head geometry or anchor")
        forward = (
            spec_body_poses(spec, {})[anchor.body]
            @ head_frame_in_body(anchor, doc.head)
        )[:3, 0]
        azimuth = float(np.degrees(np.arctan2(-forward[1], -forward[0])) % 360)
        for width, height in ((1280, 960), (390, 844)):
            renderer = mujoco.Renderer(model, height, width)
            try:
                for view, angle in (("front", 0), ("threequarter", 35), ("side", 90)):
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = np.mean(data.geom_xpos[heads], axis=0)
                    camera.distance, camera.azimuth, camera.elevation = (
                        (0.85 if width == 1280 else 1.25),
                        azimuth + angle,
                        -5,
                    )
                    renderer.update_scene(data, camera=camera)
                    imageio.imwrite(
                        args.out / f"{style}_{view}_{width}.png", renderer.render()
                    )
            finally:
                renderer.close()


if __name__ == "__main__":
    main()
