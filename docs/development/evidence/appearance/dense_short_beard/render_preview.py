"""Render dense short-beard front/profile evidence at desktop and phone sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import mujoco
from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.shared.python.motion_matching.execution.golfer_reference_specs import (
    reference_spec,
)
from src.shared.python.model_appearance import document_from_dict


def main() -> None:
    """Render reproducible desktop and phone views of the dense beard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(reference_spec("driver")).encode()
    appearance = document_from_dict(
        {
            "schema_version": "appearance-v1",
            "skin_tone": "skin_2",
            "head": {
                "enabled": True,
                "headwear": "hair",
                "headwear_material": "hair_dark_brown",
                "model": "cc0_mesh",
                "hair_style": "short_crop",
                "hair_part": "left",
                "cap_style": "baseball_cap",
                "cap_material": "cap_white",
                "cap_brim": 0.3,
                "facial_hair": "dense_short_beard",
            },
        }
    )
    xml, _ = export_full_body_mjcf(raw, appearance=appearance)
    xml = xml.replace(
        "</mujoco>",
        '<visual><global offwidth="1280" offheight="844"/></visual></mujoco>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    head = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.xpos[head]
    camera.lookat[2] += 0.045
    camera.elevation = 0.0
    for width, height, size in ((1280, 720, "desktop"), (390, 844, "phone")):
        renderer = mujoco.Renderer(model, height, width)
        try:
            for angle, view in ((180.0, "front"), (90.0, "profile")):
                camera.azimuth = angle
                camera.distance = 1.5 if size == "phone" and view == "profile" else 0.65
                renderer.update_scene(data, camera=camera)
                iio.imwrite(out / f"{size}_{view}.png", renderer.render().copy())
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
