"""Same-candidate native replay parity on actual capture samples and masks."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate

p = argparse.ArgumentParser(description=__doc__)
for name in ("model", "candidate", "target", "output"):
    p.add_argument("--" + name, type=Path, required=True)
p.add_argument("--reference", type=Path)
p.add_argument("--urdf", type=Path)
p.add_argument("--sidecar", type=Path)
p.add_argument("--expected-candidate", required=True)
p.add_argument("--rtol", type=float, default=1e-11)
p.add_argument("--atol", type=float, default=1e-13)
p.add_argument("--max-step", type=float, default=0.00025)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
raw = a.model.read_bytes()
spec = json.loads(raw)
candidate = NativeReplayCandidate.from_document(
    json.loads(a.candidate.read_bytes()),
    spec["coordinate_order"],
    hashlib.sha256(raw).hexdigest(),
)
if candidate.sha256 != a.expected_candidate:
    raise ValueError("Candidate identity differs from requested experiment")
doc = candidate.document
target = json.loads(a.target.read_bytes())
if target["source_sha256"] != doc["capture_sha256"]:
    raise ValueError("Capture identity differs from candidate")
indices = [target["labels"].index(name) for name in doc["marker_labels"]]
clock = np.asarray(target["time_s"])
rows = clock <= doc["duration_s"]
clock = clock[rows]
if clock[0] != 0 or clock[-1] != doc["duration_s"] or np.any(np.diff(clock) <= 0):
    raise ValueError("Actual capture samples must include complete candidate coverage")
points = np.asarray(target["points_world_m"])[rows][:, indices]
valid = np.asarray(target["valid"], dtype=bool)[rows][:, indices]
valid &= np.isfinite(points).all(axis=2)
if not valid.any():
    raise ValueError("No observed markers")
factory = NativePinocchioModel
if a.reference:
    from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel

    def factory(_: dict) -> NativeDrakeModel:
        return NativeDrakeModel(a.urdf.read_bytes(), a.sidecar.read_bytes(), raw)

    ref_report = json.loads((a.reference / "report.json").read_bytes())
    if (
        ref_report["candidate_sha256"] != candidate.sha256
        or ref_report["model_sha256"] != hashlib.sha256(raw).hexdigest()
        or ref_report["target_sha256"]
        != hashlib.sha256(a.target.read_bytes()).hexdigest()
    ):
        raise ValueError("Reference input identities differ")
    ref = np.load(a.reference / "trajectory.npz")
    if not np.array_equal(ref["time_s"], clock) or not np.array_equal(
        ref["valid"], valid
    ):
        raise ValueError("Reference clock or observed masks differ")
a.output.mkdir()
result = replay_candidate(
    raw,
    candidate,
    clock,
    model_factory=factory,
    rtol=a.rtol,
    atol=a.atol,
    max_step=a.max_step,
)
report = {
    "execution_mode": "drake-tree-custom-rigid-kkt"
    if a.reference
    else "pinocchio-rigid-constraint-dynamics",
    "candidate_sha256": candidate.sha256,
    "candidate_raw_sha256": hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
    "model_sha256": hashlib.sha256(raw).hexdigest(),
    "target_sha256": hashlib.sha256(a.target.read_bytes()).hexdigest(),
    "capture_sha256": doc["capture_sha256"],
    "duration_s": float(clock[-1]),
    "samples": len(clock),
    "coordinate_order": spec["coordinate_order"],
    "marker_labels": doc["marker_labels"],
    "observed_count": int(valid.sum()),
    "terminal_observed_count": int(valid[-1].sum()),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "integration": {
        "method": "DOP853",
        "rtol": a.rtol,
        "atol": a.atol,
        "max_step_s": a.max_step,
        "initial_state_resets": 0,
    },
    "closure_pose_max_abs": result.closure_pose_max_abs,
    "closure_rate_max_abs": result.closure_velocity_max_abs,
    "elapsed_s": result.integration.elapsed_s,
    "scope": "Unaccepted fitted candidate cross-engine parity; no new fit acceptance or MATLAB extended-horizon qualification",
    "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
}
squared = np.sum((result.markers_m - points) ** 2, axis=2)
report["c3d_whole_rms_m"] = float(np.sqrt(np.mean(squared[valid])))
report["c3d_terminal_rms_m"] = (
    float(np.sqrt(np.mean(squared[-1, valid[-1]]))) if valid[-1].any() else None
)
if a.reference:
    delta = result.markers_m - ref["markers_m"]
    report.update(
        state_q_max_abs=float(
            np.max(np.abs(result.integration.state[:, :27] - ref["state"][:, :27]))
        ),
        state_v_max_abs=float(
            np.max(np.abs(result.integration.state[:, 27:] - ref["state"][:, 27:]))
        ),
        marker_coordinate_max_abs_m=float(np.max(np.abs(delta))),
        observed_marker_max_distance_m=float(
            np.max(np.linalg.norm(delta, axis=2)[valid])
        ),
        observed_marker_rms_distance_m=float(
            np.sqrt(np.mean(np.sum(delta**2, axis=2)[valid]))
        ),
    )
    report["gates"] = {
        "state_q_max_abs": 1e-6,
        "state_v_max_abs": 1e-4,
        "observed_marker_max_distance_m": 1e-7,
        "closure_pose_max_abs": 1e-7,
        "closure_rate_max_abs": 1e-7,
    }
    report["parity_passed"] = all(
        report[key] < limit for key, limit in report["gates"].items()
    )
    report["versions"]["drake"] = importlib.metadata.version("drake")
    report["urdf_sha256"] = hashlib.sha256(a.urdf.read_bytes()).hexdigest()
    report["sidecar_sha256"] = hashlib.sha256(a.sidecar.read_bytes()).hexdigest()
import src.engines.physics_engines.pinocchio.python.native_replay as replay_module

runtime = Path(replay_module.__file__).parents[5]
report["runtime_source_sha256"] = {
    str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((runtime / "src").rglob("*.py"))
}
np.savez_compressed(
    a.output / "trajectory.npz",
    time_s=clock,
    state=result.integration.state,
    markers_m=result.markers_m,
    target_m=points,
    valid=valid,
    candidate_sha256=candidate.sha256,
)
(a.output / "report.json").write_text(json.dumps(report, indent=2))
sys.stdout.write(
    json.dumps(
        {key: value for key, value in report.items() if key != "runtime_source_sha256"}
    )
    + "\n"
)
if a.reference and not report["parity_passed"]:
    raise ValueError("Candidate parity gates failed")
