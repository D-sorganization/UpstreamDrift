"""SHOW-A1 (#14550): face-on address grip roll of a full-body spec.

Poses the MuJoCo model of ``--spec`` at the first frame of ``--q`` (``ik_trajectory.npz``
``q`` in ``coordinate_order``), reads the native club frame from the ``LW`` site and
reports ``motion_matching.grip_roll`` (0 deg = vertical, + = strong). Writes
``receipt_<tag>.json`` and a face-on PNG with the hand normals (lead red, trail blue) and
a dial of the roll as the golfer sees it down the shaft. Headless: ``MUJOCO_GL=egl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (  # noqa: E402
    export_full_body_mjcf,
)
from src.shared.python.motion_matching.grip_roll import (  # noqa: E402
    address_grip_roll_deg,
    grip_pose_state_from_spec,
)

MODELS = ROOT / "docs/development/full_body_models"
SPEC = MODELS / "full_body_spec_anthro_driver.json"
Q = MODELS / "evidence/ground_support/anthro_driver_drake/ik_trajectory.npz"
HERE = Path(__file__).resolve().parent
COLORS = {"lead": (255, 60, 60), "trail": (80, 120, 255)}
WHITE = (255, 255, 255)


def measure(spec_bytes: bytes, q: np.ndarray) -> dict[str, Any]:
    """Roll of both hands plus the posed model, hand normals and hand centres."""
    xml, meta = export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 1000
    data = mujoco.MjData(model)
    for value, name in zip(q, meta["coordinate_order"], strict=True):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    site = {
        k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"][k])
        for k in ("LW", "RW")
    }
    rot = data.site_xmat[site["LW"]].reshape(3, 3)
    state = grip_pose_state_from_spec(json.loads(spec_bytes), rot)
    lead, trail = address_grip_roll_deg(state)
    return {
        "lead": lead,
        "trail": trail,
        "model": model,
        "data": data,
        "normals": {
            "lead": -(rot @ state.lead_dorsal_club),
            "trail": -(rot @ state.trail_dorsal_club),
        },
        "centres": {
            "lead": data.site_xpos[site["LW"]].copy(),
            "trail": data.site_xpos[site["RW"]].copy(),
        },
    }


def _draw_dial(draw: Any, result: dict[str, Any], cx: int, cy: int) -> None:
    """Roll dial in the golfer's view down the shaft (target left, up on top)."""
    r = 80

    def at(deg: float, scale: float = 1.0) -> tuple[float, float]:
        rad = math.radians(deg)
        return cx - scale * r * math.cos(rad), cy - scale * r * math.sin(rad)

    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(200, 200, 200))
    draw.line((cx - r, cy, cx + r, cy), fill=(90, 90, 90))
    draw.line((cx, cy - r, cx, cy + r), fill=(90, 90, 90))
    draw.line((at(15.0, 0.8), at(15.0)), fill=(60, 220, 60), width=4)
    draw.text((cx - r - 8, cy + 4), "target", fill=WHITE)
    draw.text((cx - 8, cy - r - 14), "up", fill=WHITE)
    for key, rgb in COLORS.items():
        draw.line((cx, cy, *at(result[key])), fill=rgb, width=4)
    draw.text((cx - r, cy + r + 6), "roll down the shaft (green = 15 deg)", fill=WHITE)


def render_face_on(result: dict[str, Any], path: Path, tag: str) -> bool:
    """Face-on still with arrows and dial; ``False`` when there is no GL context."""
    from PIL import Image, ImageDraw

    try:
        renderer = mujoco.Renderer(result["model"], height=900, width=700)
    except Exception as exc:  # noqa: BLE001 - no GL context is an environment fact
        print(f"no GL context: {exc}", file=sys.stderr)  # noqa: T201
        return False
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.9, 0.0, 0.95]
    camera.distance, camera.elevation = 2.6, 0.0
    camera.azimuth = 0.0  # camera on the -x side looking down +x: face-on
    renderer.update_scene(result["data"], camera)
    arrow = mujoco.mjtGeom.mjGEOM_ARROW
    for key, rgb in COLORS.items():
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        rgba = np.array([*(c / 255 for c in rgb), 1.0], dtype=np.float32)
        mujoco.mjv_initGeom(geom, arrow, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        start = result["centres"][key]
        end = start + 0.3 * result["normals"][key]
        mujoco.mjv_connector(geom, arrow, 0.012, start, end)
        renderer.scene.ngeom += 1
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), f"address grip roll ({tag}), face-on", fill=WHITE)
    draw.text(
        (12, 28),
        f"lead {result['lead']:+.1f} deg (red)   trail {result['trail']:+.1f} deg (blue)",
        fill=WHITE,
    )
    _draw_dial(draw, result, 580, 760)
    image.save(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--q", type=Path, default=Q)
    parser.add_argument("--tag", default="before")
    args = parser.parse_args()
    spec_bytes = args.spec.read_bytes()
    result = measure(spec_bytes, np.load(args.q)["q"][0])
    drawn = render_face_on(result, HERE / f"face_on_{args.tag}.png", args.tag)
    receipt = {
        "issue": 14550,
        "tag": args.tag,
        "spec": args.spec.name,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "convention": "face-on; 0 = vertical, + = strong; lead = back of hand, "
        "trail = palm (motion_matching/grip_roll.py)",
        "lead_roll_deg": round(result["lead"], 2),
        "trail_roll_deg": round(result["trail"], 2),
        "png": f"face_on_{args.tag}.png" if drawn else None,
    }
    (HERE / f"receipt_{args.tag}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
