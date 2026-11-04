"""SHOW-A1 (#14550): measure the face-on address grip roll of a full-body spec.

Builds the MuJoCo model of ``--spec``, poses it at the first frame of ``--q``
(an ``ik_trajectory.npz`` ``q`` array in ``coordinate_order``), reads the native
club frame from the ``LW`` frame site and reports the lead/trail roll of
``motion_matching.grip_roll`` (0 deg = vertical, + = strong). It writes
``receipt.json`` and, when a GL context exists, a face-on PNG with the
back-of-lead-hand (red) and trail-palm (blue) normals drawn as arrows.

Headless::

    MUJOCO_GL=egl MPLBACKEND=Agg python3 \\
        docs/development/full_body_models/evidence/grip_roll/measure_address_grip_roll.py \\
        --tag before

Stills are visual evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
DEFAULT_SPEC = MODELS / "full_body_spec_anthro_driver.json"
DEFAULT_Q = MODELS / "evidence/ground_support/anthro_driver_drake/ik_trajectory.npz"
OUT = Path(__file__).resolve().parent
ARROW_LEN_M = 0.3


def pose_model(spec_bytes: bytes, q: np.ndarray):
    """MuJoCo model and data of the spec, posed at ``q`` (coordinate order)."""
    xml, meta = export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 1000
    data = mujoco.MjData(model)
    for value, name in zip(q, meta["coordinate_order"], strict=True):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    return model, data, meta


def site_id(model: Any, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def measure(spec_bytes: bytes, q: np.ndarray) -> dict[str, Any]:
    model, data, meta = pose_model(spec_bytes, q)
    sites = meta["frame_sites"]
    lw = site_id(model, sites["LW"])
    club_rotation = data.site_xmat[lw].reshape(3, 3)
    state = grip_pose_state_from_spec(json.loads(spec_bytes), club_rotation)
    lead, trail = address_grip_roll_deg(state)
    normals = {
        "lead": -(club_rotation @ state.lead_dorsal_club),
        "trail": -(club_rotation @ state.trail_dorsal_club),
    }
    centres = {
        "lead": data.site_xpos[lw].copy(),
        "trail": data.site_xpos[site_id(model, sites["RW"])].copy(),
    }
    return {
        "lead_roll_deg": lead,
        "trail_roll_deg": trail,
        "normals": normals,
        "centres": centres,
        "model": model,
        "data": data,
    }


def _draw_dial(draw: Any, result: dict[str, Any], origin: tuple[int, int]) -> None:
    """Roll dial in the golfer's view down the shaft (target left, up on top).

    Positive roll runs from the target side up and over: clockwise, i.e. strong.
    The green tick marks the owner's 15 deg target.
    """
    import math

    cx, cy = origin
    radius = 80

    def point(deg: float, scale: float = 1.0) -> tuple[float, float]:
        rad = math.radians(deg)
        return cx - scale * radius * math.cos(rad), cy - scale * radius * math.sin(rad)

    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius), outline=(200, 200, 200)
    )
    draw.line((cx - radius, cy, cx + radius, cy), fill=(90, 90, 90))
    draw.line((cx, cy - radius, cx, cy + radius), fill=(90, 90, 90))
    draw.line((point(15.0, 0.8), point(15.0, 1.0)), fill=(60, 220, 60), width=4)
    draw.text((cx - radius - 8, cy + 4), "target", fill=(255, 255, 255))
    draw.text((cx - 8, cy - radius - 14), "up", fill=(255, 255, 255))
    for key, rgb in (
        ("lead_roll_deg", (255, 60, 60)),
        ("trail_roll_deg", (80, 120, 255)),
    ):
        draw.line((cx, cy, *point(result[key])), fill=rgb, width=4)
    draw.text(
        (cx - radius, cy + radius + 6), "roll, down the shaft", fill=(255, 255, 255)
    )
    draw.text(
        (cx - radius, cy + radius + 20),
        "green tick = 15 deg target",
        fill=(60, 220, 60),
    )


def render_face_on(result: dict[str, Any], path: Path, label: str) -> bool:
    """Face-on still with the hand-normal arrows; ``False`` when no GL context."""
    from PIL import Image, ImageDraw

    model, data = result["model"], result["data"]
    try:
        renderer = mujoco.Renderer(model, height=900, width=700)
    except Exception as exc:  # noqa: BLE001 - no GL context is an environment fact
        print(f"no GL context: {exc}", file=sys.stderr)  # noqa: T201
        return False
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.9, 0.0, 0.95]
    camera.distance = 2.6
    camera.azimuth = 0.0  # camera on the -x side, looking down +x: face-on
    camera.elevation = 0.0
    renderer.update_scene(data, camera)
    scene = renderer.scene
    for key, rgba in (("lead", [1, 0.1, 0.1, 1]), ("trail", [0.1, 0.3, 1, 1])):
        start = result["centres"][key]
        end = start + ARROW_LEN_M * result["normals"][key]
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, start, end)
        scene.ngeom += 1
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), label, fill=(255, 255, 255))
    draw.text(
        (12, 28),
        f"face-on roll  lead {result['lead_roll_deg']:+.1f} deg (red)   "
        f"trail {result['trail_roll_deg']:+.1f} deg (blue)",
        fill=(255, 255, 255),
    )
    _draw_dial(draw, result, (580, 760))
    image.save(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--q", type=Path, default=DEFAULT_Q)
    parser.add_argument("--tag", default="before")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    spec_bytes = args.spec.read_bytes()
    q = np.load(args.q)["q"][0]
    result = measure(spec_bytes, q)
    png = args.out / f"face_on_{args.tag}.png"
    drawn = render_face_on(result, png, f"address grip roll ({args.tag})")
    receipt = {
        "issue": 14550,
        "tag": args.tag,
        "spec": args.spec.relative_to(ROOT).as_posix()
        if args.spec.is_relative_to(ROOT)
        else args.spec.name,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "convention": "face-on; 0 = vertical, + = strong; lead=back of hand, "
        "trail=palm; see motion_matching/grip_roll.py",
        "lead_roll_deg": round(result["lead_roll_deg"], 2),
        "trail_roll_deg": round(result["trail_roll_deg"], 2),
        "png": png.name if drawn else None,
    }
    (args.out / f"receipt_{args.tag}.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
