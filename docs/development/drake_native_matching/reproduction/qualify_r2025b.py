"""Compare native Drake to recorded R2025b states, efforts and coefficient replay."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate

parser = argparse.ArgumentParser(description=__doc__)
for name in ("assets", "reference", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
a = parser.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
a.output.mkdir()
raw = (a.assets / "native_geometry_spec_9967.json").read_bytes()
urdf = (a.assets / "native-golf-9967-01.urdf").read_bytes()
sidecar = (a.assets / "native-golf-9967-01.sidecar.json").read_bytes()
ref = json.loads(a.reference.read_bytes())
if ref["release"] != "2025b":
    raise ValueError("R2025b acceptance reference required")
engine = NativeDrakeModel(urdf, sidecar, raw)
names = ref["coordinate_names"]
if tuple(names) != engine.names:
    raise ValueError("Reference native coordinate order differs")
actual = []
poses = []
for q, v, effort in zip(ref["q"], ref["qd"], ref["primitive_efforts"], strict=True):
    positions = dict(zip(names, q, strict=True))
    acc = engine.accelerations(
        positions,
        dict(zip(names, v, strict=True)),
        dict(zip(names, effort, strict=True)),
    )
    actual.append([acc[name] for name in names])
    frames = engine.frame_poses(positions)
    poses.append([frames[name] for name in ref["frame_names"]])
error = np.asarray(actual) - ref["qdd"]
report = {
    "execution_mode": "drake-tree-custom-rigid-kkt",
    "reference_release": ref["release"],
    "reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
    "model_sha256": hashlib.sha256(raw).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "time_s": ref["time_s"],
    "acceleration_max_abs": float(np.max(np.abs(error))),
    "acceleration_max_scaled": float(np.max(np.abs(error) / (1 + np.abs(ref["qdd"])))),
    "frame_max_abs": float(np.max(np.abs(np.asarray(poses) - ref["poses"]))),
}
report["moving_passed"] = (
    report["acceleration_max_scaled"] < 1e-7 and report["frame_max_abs"] < 1e-10
)
(a.output / "moving.json").write_text(json.dumps(report, indent=2))
if not report["moving_passed"]:
    raise ValueError(report)
doc = json.loads(
    (a.assets / "native-root-force-9967-02/returned-candidate.json").read_text()
)
doc.update(
    q0=ref["q"][0],
    qd0=ref["qd"][0],
    coefficients=ref["native_coefficients"],
    duration_s=ref["time_s"][-1],
)
candidate = NativeReplayCandidate.from_document(
    doc, names, hashlib.sha256(raw).hexdigest()
)
(a.output / "reference-specific-candidate.json").write_text(
    json.dumps(candidate.document, indent=2)
)


def factory(_: dict) -> NativeDrakeModel:
    return NativeDrakeModel(urdf, sidecar, raw)


result = replay_candidate(
    raw,
    candidate,
    np.asarray(ref["time_s"]),
    model_factory=factory,
    rtol=1e-11,
    atol=1e-13,
    max_step=0.00025,
)
report["reference_specific_candidate_sha256"] = candidate.sha256
report["continuous_q_max_abs"] = float(
    np.max(np.abs(result.integration.state[:, :27] - ref["q"]))
)
report["continuous_qd_max_abs"] = float(
    np.max(np.abs(result.integration.state[:, 27:] - ref["qd"]))
)
replay_poses = []
for state in result.integration.state:
    frames = engine.frame_poses(dict(zip(names, state[:27], strict=True)))
    replay_poses.append([frames[name] for name in ref["frame_names"]])
report["continuous_frame_position_max_abs_m"] = float(
    np.max(
        np.abs(
            np.asarray(replay_poses)[:, :, :3, 3]
            - np.asarray(ref["poses"])[:, :, :3, 3]
        )
    )
)
report["gates"] = {
    "continuous_q_max_abs": 1e-4,
    "continuous_qd_max_abs": 0.05,
    "continuous_frame_position_max_abs_m": 1e-5,
    "acceleration_max_scaled": 1e-7,
    "frame_max_abs": 1e-10,
}
report["passed"] = all(report[key] < limit for key, limit in report["gates"].items())
report["scope"] = (
    "Six recorded R2025b states and same-input continuous0.8s baseline; not full capture fit"
)
report["closure_pose_max_abs"] = result.closure_pose_max_abs
report["closure_rate_max_abs"] = result.closure_velocity_max_abs
report["replay_elapsed_s"] = result.integration.elapsed_s
np.savez_compressed(
    a.output / "replay.npz",
    time_s=result.integration.time,
    state=result.integration.state,
    markers=result.markers_m,
    poses=replay_poses,
)
(a.output / "report.json").write_text(json.dumps(report, indent=2))
sys.stdout.write(json.dumps(report) + "\n")
if not report["passed"]:
    raise ValueError("Native R2025b equivalence gates failed")
