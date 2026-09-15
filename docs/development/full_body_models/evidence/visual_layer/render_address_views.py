"""Render the qualified address pose from three views with the capture markers.

Uses the full-body MuJoCo export with the visual layer, sets the 27 native
upper-body coordinates to the returned81 candidate's original q0 (the Simscape
qualified starting pose) with the lower limbs at zero, and overlays the
frame-0 capture markers (black) and the model's native marker predictions
(blue) as world spheres. Front, side and top views plus a segment-length table
are written beside this script. Kinematic only; no dynamics.
"""

from __future__ import annotations

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
from src.shared.python.motion_matching.visual_skeleton import (  # noqa: E402
    derive_visual_skeleton,
)

NATIVE = Path(
    "C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native/docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81"
)
HERE = Path(__file__).resolve().parent
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


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
    xml, _ = exporter.export_full_body_mjcf(SPEC.read_bytes(), visual=True)
    replay = np.load(NATIVE / "returned-replay.npz")
    candidate = json.loads((NATIVE / "returned-candidate.json").read_text())
    target0 = replay["target_m"][0]
    model0 = replay["markers_m"][0]
    xml = add_marker_spheres(xml, target0, "0 0 0 1", "capture")
    xml = add_marker_spheres(xml, model0, "0.1 0.4 1 1", "native")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    names = candidate["coordinate_names"]
    q0 = np.asarray(candidate["q0"], dtype=float)
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
        imageio.imwrite(HERE / f"address_{view}.png", renderer.render().copy())
    spec = json.loads(SPEC.read_text())
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
    (HERE / "address_segments.json").write_text(json.dumps(table, indent=2) + "\n")
    rms = float(
        np.sqrt(np.mean(np.sum((model0 - target0)[replay["valid"][0]] ** 2, axis=1)))
    )
    (HERE / "address_receipt.json").write_text(
        json.dumps(
            {
                "candidate_sha256": "dfafdff1cdec1a7fa15c41a34d41f054ef45d8a7df0a1faf8fab7898ab855776",
                "pose": "original q0 of the native candidate (Simscape qualified start), legs zero",
                "frame0_valid_marker_rms_m": rms,
                "views": list(views),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
