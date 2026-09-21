"""Render the qualified address pose from three views with the capture markers.

Uses the full-body MuJoCo export with the visual layer, sets the 27 native
upper-body coordinates to the returned81 candidate's original q0 (the Simscape
qualified starting pose) with the lower limbs at zero, and overlays the
frame-0 capture markers (black) and the model's native marker predictions
(blue) as world spheres. Front, side and top views plus a segment-length table
are written beside this script. Kinematic only; no dynamics.

With ``--spec DOC --trajectory ik_trajectory.npz --suffix NAME`` the same
views are rendered for another full-body document at frame 0 of a driver IK
trajectory (all 41 coordinates), with the capture markers mapped to the
native world; outputs carry the suffix.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET  # nosec B405 - construction only
from defusedxml import ElementTree as SafeET
from pathlib import Path

import imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter  # noqa: E402
from src.engines.physics_engines.mujoco.python.visual_layer import (  # noqa: E402
    add_com_markers,
)
from src.shared.python.motion_matching.ground_support import (  # noqa: E402
    capture_to_native_world,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    load_tour_capture,
)
from src.shared.python.motion_matching.visual_skeleton import (  # noqa: E402
    derive_visual_skeleton,
)

C3D = ROOT / "data/C3D_TA_Driver.c3d"

NATIVE = Path(
    "C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native/docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81"
)
HERE = Path(__file__).resolve().parent
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v2.json"


def add_marker_spheres(xml: str, points: np.ndarray, rgba: str, prefix: str) -> str:
    root = SafeET.fromstring(xml)
    world = root.find("worldbody")
    for i, p in enumerate(points):
        if np.isfinite(p).all():
            ET.SubElement(
                world,
                "geom",
                name=f"{prefix}_{i}",
                type="sphere",
                size="0.012",
                pos=" ".join(f"{v:.6f}" for v in p),
                rgba=rgba,
                contype="0",
                conaffinity="0",
                group="2",
                mass="0",
            )
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--trajectory", type=Path, help="driver ik_trajectory.npz")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    suffix = f"_{args.suffix}" if args.suffix else ""
    spec_path: Path = args.spec
    xml, _ = exporter.export_full_body_mjcf(spec_path.read_bytes(), visual=True)
    if args.trajectory is None:
        replay = np.load(NATIVE / "returned-replay.npz")
        candidate = json.loads((NATIVE / "returned-candidate.json").read_text())
        target0 = replay["target_m"][0]
        model0 = replay["markers_m"][0]
        valid0 = replay["valid"][0]
        names = candidate["coordinate_names"]
        q0 = np.asarray(candidate["q0"], dtype=float)
        pose = (
            "original q0 of the native candidate (Simscape qualified start), legs zero"
        )
        xml = add_marker_spheres(xml, model0, "0.1 0.4 1 1", "native")
    else:
        capture = load_tour_capture(C3D)
        target0 = capture_to_native_world(capture.points_m)[0]
        valid0 = capture.valid[0]
        model0 = None
        names = json.loads(spec_path.read_text())["coordinate_order"]
        q0 = np.load(args.trajectory)["q"][0]
        pose = f"frame 0 of {args.trajectory.name} on {spec_path.name}"
    xml = add_marker_spheres(xml, target0, "0 0 0 1", "capture")
    ground_height = float(
        json.loads(spec_path.read_text())["contact"].get("ground_height_m") or 0.0
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for name, value in zip(names, q0, strict=True):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 360, 480)
    centre = np.nanmean(target0, axis=0)
    views = {"front": (0.0, -8.0), "side": (90.0, -8.0), "top": (0.0, -85.0)}
    for view, (azimuth, elevation) in views.items():
        cam = mujoco.MjvCamera()
        cam.lookat[:] = centre
        cam.distance, cam.azimuth, cam.elevation = 3.0, azimuth, elevation
        renderer.update_scene(data, camera=cam)
        com = add_com_markers(renderer.scene, model, data, ground_height)
        imageio.imwrite(HERE / f"address_{view}{suffix}.png", renderer.render().copy())
    spec = json.loads(spec_path.read_text())
    skeleton = derive_visual_skeleton(spec)
    table = sorted(
        (
            {
                "body": c.body.rsplit("/", 1)[-1],
                "length_m": round(c.length_m(), 4),
                "radius_m": round(c.radius_m, 4),
            }
            for c in skeleton.capsules
        ),
        key=lambda r: -r["length_m"],
    )
    (HERE / f"address_segments{suffix}.json").write_text(
        json.dumps(table, indent=2) + "\n"
    )
    rms = (
        None
        if model0 is None
        else float(np.sqrt(np.mean(np.sum((model0 - target0)[valid0] ** 2, axis=1))))
    )
    (HERE / f"address_receipt{suffix}.json").write_text(
        json.dumps(
            {
                "spec": spec_path.name,
                "pose": pose,
                "frame0_valid_marker_rms_m": rms,
                "views": list(views),
                "centre_of_mass_m": [float(v) for v in com],
                "com_markers": "red sphere: whole body plus club; yellow: its ground projection",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
