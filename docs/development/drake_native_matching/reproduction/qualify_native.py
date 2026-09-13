"""Native Drake qualification against an independently saved Pinocchio replay."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import sys
import numpy as np
from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate

parser = argparse.ArgumentParser()
for key in ("model", "candidate", "output"):
    parser.add_argument("--" + key, type=Path, required=True)
parser.add_argument("--urdf", type=Path)
parser.add_argument("--sidecar", type=Path)
parser.add_argument("--reference", type=Path)
a = parser.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
a.output.mkdir()
raw = a.model.read_bytes()
spec = json.loads(raw)
names = spec["coordinate_order"]
candidate = NativeReplayCandidate.from_document(
    json.loads(a.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
)
clock = np.linspace(0, 0.8, 81)
start = time.perf_counter()
if a.reference:
    from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel

    def factory(_: dict) -> NativeDrakeModel:
        return NativeDrakeModel(a.urdf.read_bytes(), a.sidecar.read_bytes(), raw)

    reference_report = json.loads((a.reference / "report.json").read_text())
    if (
        reference_report["model_sha256"] != hashlib.sha256(raw).hexdigest()
        or reference_report["candidate_sha256"] != candidate.sha256
    ):
        raise ValueError("Reference source/candidate identity mismatch")
    ref = np.load(a.reference / "reference.npz")
    states = ref["state"]
else:
    factory = NativePinocchioModel
    result = replay_candidate(
        raw,
        candidate,
        clock,
        model_factory=factory,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.00025,
    )
    states = result.integration.state
engine = factory(spec)
indices = [0, 40, 60, 70, 75, 80]
pulses = np.vstack((np.zeros(27), np.eye(27)))
accelerations = []
frames = []
for i in indices:
    q = dict(zip(names, states[i, :27], strict=True))
    v = dict(zip(names, states[i, 27:], strict=True))
    frame = engine.frame_poses(q)
    frames.append([frame[key] for key in sorted(frame)])
    rows = []
    for pulse in pulses:
        acc = engine.accelerations(q, v, dict(zip(names, pulse, strict=True)))
        rows.append([acc[key] for key in names])
    accelerations.append(rows)
accelerations = np.asarray(accelerations)
frames = np.asarray(frames)
report = {
    "model_sha256": hashlib.sha256(raw).hexdigest(),
    "candidate_sha256": candidate.sha256,
    "coordinate_order": names,
    "pulse_times_s": clock[indices].tolist(),
    "engine": "drake" if a.reference else "pinocchio",
}
if a.reference:
    report["pulse_max_abs"] = float(
        np.max(np.abs(accelerations - ref["accelerations"]))
    )
    report["pulse_max_scaled"] = float(
        np.max(
            np.abs(accelerations - ref["accelerations"])
            / (1 + np.abs(ref["accelerations"]))
        )
    )
    report["frame_max_abs"] = float(np.max(np.abs(frames - ref["frames"])))
    (a.output / "pulse-report.json").write_text(json.dumps(report, indent=2))
    assert report["pulse_max_scaled"] < 1e-7, report
    assert report["frame_max_abs"] < 1e-10, report
    result = replay_candidate(
        raw,
        candidate,
        clock,
        model_factory=factory,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.00025,
    )
    report["state_q_max_abs"] = float(
        np.max(np.abs(result.integration.state[:, :27] - states[:, :27]))
    )
    report["state_v_max_abs"] = float(
        np.max(np.abs(result.integration.state[:, 27:] - states[:, 27:]))
    )
    report["marker_max_abs_m"] = float(
        np.max(np.abs(result.markers_m - ref["markers"]))
    )
    report["runtime_version"] = importlib.metadata.version("drake")
    report["passed"] = (
        report["state_q_max_abs"] < 1e-6
        and report["state_v_max_abs"] < 1e-4
        and report["marker_max_abs_m"] < 1e-7
    )
    report["urdf_sha256"] = hashlib.sha256(a.urdf.read_bytes()).hexdigest()
    report["sidecar_sha256"] = hashlib.sha256(a.sidecar.read_bytes()).hexdigest()
report["execution_mode"] = (
    "drake-tree-custom-rigid-kkt"
    if a.reference
    else "pinocchio-rigid-constraint-dynamics"
)
report["integration"] = {
    "method": "DOP853",
    "rtol": 1e-11,
    "atol": 1e-13,
    "max_step_s": 0.00025,
    "initial_state_resets": 0,
    "closure_tolerance": 1e-7,
}
report["scope"] = (
    "native baseline through 0.8 seconds; not a C3D match or stock Drake SAP qualification"
)
report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report["candidate_raw_sha256"] = hashlib.sha256(a.candidate.read_bytes()).hexdigest()
report["versions"] = {
    key: importlib.metadata.version(key) for key in ("numpy", "scipy")
}
report["gates"] = {
    "pulse_max_scaled": 1e-7,
    "frame_max_abs": 1e-10,
    "state_q_max_abs": 1e-6,
    "state_v_max_abs": 1e-4,
    "marker_max_abs_m": 1e-7,
}
if a.reference:
    import src.engines.physics_engines.drake.python.native_model as adapter

    runtime_root = Path(adapter.__file__).parents[5]
    report["runtime_source_sha256"] = {
        str(path.relative_to(runtime_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((runtime_root / "src").rglob("*.py"))
    }
report.update(
    closure_pose_max_abs=result.closure_pose_max_abs,
    closure_rate_max_abs=result.closure_velocity_max_abs,
    replay_elapsed_s=result.integration.elapsed_s,
    total_elapsed_s=time.perf_counter() - start,
)
np.savez_compressed(
    a.output / "reference.npz",
    time_s=clock,
    state=result.integration.state,
    markers=result.markers_m,
    accelerations=accelerations,
    frames=frames,
)
(a.output / "report.json").write_text(json.dumps(report, indent=2))
sys.stdout.write(json.dumps(report) + "\n")

if a.reference and not report["passed"]:
    raise ValueError("Continuous native parity gate failed")
