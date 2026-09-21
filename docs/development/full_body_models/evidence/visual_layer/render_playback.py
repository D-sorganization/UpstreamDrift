"""Regenerate the visual-layer evidence: default pose PNG and returned81 playback GIF.

Kinematic playback only: the returned81 upper-body trajectory is applied by
joint name, lower-limb coordinates stay at zero, no dynamics. The receipt is
rewritten with the current spec and MJCF hashes and the visual-layer counts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter  # noqa: E402

NATIVE = Path(
    "C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native/docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81"
)
HERE = Path(__file__).resolve().parent
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v2.json"
SIZE = (240, 320)
STRIDE = 3


def camera(lookat: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -12.0
    return cam


def main() -> None:
    spec_bytes = SPEC.read_bytes()
    xml, meta = exporter.export_full_body_mjcf(spec_bytes, visual=True)
    (HERE / "full_body_visual.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    replay = np.load(NATIVE / "returned-replay.npz")
    candidate = json.loads((NATIVE / "returned-candidate.json").read_text())
    addresses = [model.joint(n).qposadr[0] for n in candidate["coordinate_names"]]
    q = replay["native_state"][:, : len(addresses)]  # positions precede velocities
    lookat = np.nanmean(replay["target_m"][0], axis=0)
    renderer = mujoco.Renderer(model, *SIZE)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera(lookat))
    imageio.imwrite(HERE / "default_pose.png", renderer.render().copy())
    frames = []
    for k in range(0, q.shape[0], STRIDE):
        data.qpos[addresses] = q[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera(lookat))
        frames.append(renderer.render().copy())
    imageio.mimsave(
        HERE / "returned81_playback.gif", frames, duration=1000 * STRIDE / 360, loop=0
    )
    receipt = {
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "mjcf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "visual_layer": meta["visual_layer"],
        "ngeom": int(model.ngeom),
        "nlight": int(model.nlight),
        "render_size": list(SIZE),
        "playback": "kinematic only: returned81 upper-body q applied by joint name, lower-limb coordinates zero, no dynamics",
        "frames": len(frames),
        "source_replay": "two_window_fit_9967_81/returned-replay.npz",
        "radius_policy": "uniform-density cylinder sqrt(m / (pi rho L)), rho 1500 kg/m^3, clamped 6-50 mm",
    }
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
