"""Render address before and after the arm refit on the moved wrist joints (#15255).

Usage, from the repository root, headless::

    MUJOCO_GL=egl PYTHONPATH=. python3 \\
        docs/development/full_body_models/evidence/hands_on_club/render_address_refit.py \\
        <swing bundle .npz> <output .png>

The bundle is a matched-swing bundle with ``spec`` (the spec's JSON bytes) and
``reference_q`` (frames by coordinates, radians and metres). Frame 0 is drawn.
Rows: the unmoved model; the moved wrist joints with the same angles; the
arms refitted alone; the arms refitted with the trunk free to straighten. Columns: whole golfer from two sides, then the
hands from the side, the front and above.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.engines.physics_engines.mujoco.python.skinned_arm_refit import (  # noqa: E402
    refit_arms_to_club,
)
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (  # noqa: E402
    apply_skinned_hand_wrists,
)
from src.tools.native_viewer_export.backends.mujoco_worker import (  # noqa: E402
    build_scene_xml,
)

#: (azimuth, elevation, distance m, aimed at the hands)
VIEWS = (
    (90, -10, 3.2, False),
    (0, -10, 3.2, False),
    (90, -15, 0.6, True),
    (0, -20, 0.6, True),
    (90, -75, 0.5, True),
)
WIDTH, HEIGHT = 800, 1000
#: Trunk coordinates the last row lets move with the arms; the feet stay put.
TRUNK = ("lumbar_flexion", "thoracic_flexion")


def _row(spec: dict, order: list[str], q: np.ndarray) -> np.ndarray:
    xml, _ = build_scene_xml(json.dumps(spec).encode())
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    data = mujoco.MjData(model)
    for name, value in zip(order, q, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    hands = np.mean(
        [data.site(n).xpos for n in ("native_closure_a", "native_closure_b")], axis=0
    )
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    images = []
    for azimuth, elevation, distance, at_hands in VIEWS:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = hands if at_hands else data.subtree_com[0]
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        renderer.update_scene(data, camera=camera)
        images.append(renderer.render().copy())
    return np.hstack(images)


def main(bundle_path: str, out_path: str) -> None:
    bundle = np.load(bundle_path, allow_pickle=True)
    spec = json.loads(bytes(bundle["spec"]))
    order = list(spec["coordinate_order"])
    q = np.asarray(bundle["reference_q"][0], dtype=float)
    moved = apply_skinned_hand_wrists(spec)
    arms_only = refit_arms_to_club(spec, moved, q)
    refit = refit_arms_to_club(spec, moved, q, also_free=TRUNK)
    poses = ((spec, q), (moved, q), (moved, arms_only.q), (moved, refit.q))
    rows = [_row(document, order, pose) for document, pose in poses]
    Image.fromarray(np.vstack(rows)).save(out_path)
    report = {
        "arms_only_elbows_deg": [
            round(float(np.degrees(arms_only.q[order.index(n)])), 1)
            for n in ("LEInput", "REInput")
        ],
        "club_offset_before_m": refit.club_offset_before_m,
        "closure_before_m": refit.closure_before_m,
        "club_offset_m": refit.club_offset_m,
        "club_turn_deg": refit.club_turn_deg,
        "closure_m": refit.closure_m,
        "closure_deg": refit.closure_deg,
        "forearm_to_cuff_deg": refit.forearm_to_cuff_deg,
        "moved_deg": {
            name: [round(float(np.degrees(a)), 1), round(float(np.degrees(b)), 1)]
            for name, a, b in zip(order, q, refit.q, strict=True)
            if a != b
        },
    }
    sys.stdout.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
