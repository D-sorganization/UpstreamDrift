"""R7 (#12687) evidence: close-up head stills, parametric versus CC0 mesh head.

Renders the head of the repository's anthropometric driver spec in MuJoCo at the
zero pose, front and three-quarter, for the default parametric head and for
``head.model = "cc0_mesh"``. Visual evidence only; appearance never feeds
dynamics. Usage, from the repository root::

    PYTHONPATH=. python3 docs/development/full_body_models/evidence/appearance/\
cc0_head/render_cc0_head.py --out OUT_DIR
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SPEC = Path("docs/development/full_body_models/full_body_spec_anthro_driver.json")
HEAD = "GolfSwing3D_Kinetic/Head"
VIEWS = {"front": 0.0, "three_quarter": 35.0}  # degrees off the face normal
SIZE, DISTANCE_M = 640, 0.75


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
    import imageio.v2 as imageio
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.model_appearance import document_from_dict

    args.out.mkdir(parents=True, exist_ok=True)
    raw = SPEC.read_bytes()
    for label, head in (("parametric", {}), ("cc0_mesh", {"model": "cc0_mesh"})):
        doc = document_from_dict({"schema_version": "appearance-v1", "head": head})
        model = mujoco.MjModel.from_xml_string(
            export_full_body_mjcf(raw, appearance=doc)[0]
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body = model.body(HEAD).id
        # Face direction: the spec head frame looks along the body +x axis.
        fwd = np.asarray(data.xmat[body]).reshape(3, 3)[:, 0]
        centre = data.xpos[body] + np.array([0.0, 0.0, 0.12])
        base_az = np.degrees(np.arctan2(fwd[1], fwd[0]))
        renderer = mujoco.Renderer(model, SIZE, SIZE)
        try:
            for view, off in VIEWS.items():
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = centre
                cam.distance, cam.elevation = DISTANCE_M, -5.0
                cam.azimuth = base_az + 180.0 + off  # camera looks at the face
                renderer.update_scene(data, camera=cam)
                imageio.imwrite(args.out / f"r7_{label}_{view}.png", renderer.render())
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
