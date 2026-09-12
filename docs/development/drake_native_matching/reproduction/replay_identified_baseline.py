"""One continuous replay of the identified known-baseline polynomial."""

from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate

base = Path("/mnt/c/Users/diete")
output = base / "native-reaction-replay-10022-01"
if output.exists():
    raise FileExistsError(output)
raw = (base / "native_geometry_spec_9967.json").read_bytes()
spec = json.loads(raw)
doc = json.loads(
    (base / "native-root-force-9967-02/returned-candidate.json").read_bytes()
)
original = NativeReplayCandidate.from_document(
    doc, spec["coordinate_order"], hashlib.sha256(raw).hexdigest()
)
identified = np.load(base / "native-reaction-identification-10022-01/identified.npz")
doc["coefficients"] = identified["coefficients"].tolist()
candidate = NativeReplayCandidate.from_document(
    doc, spec["coordinate_order"], hashlib.sha256(raw).hexdigest()
)
reference = np.load(base / "drake-native-reference-10022-01/reference.npz")
output.mkdir()
(output / "identified-candidate.json").write_text(
    json.dumps(candidate.document, indent=2)
)
result = replay_candidate(
    raw, candidate, reference["time_s"], rtol=1e-11, atol=1e-13, max_step=0.00025
)
report = {
    "scope": "Known baseline reconstruction only, not C3D fit or general identifiability evidence",
    "original_candidate_sha256": original.sha256,
    "identified_candidate_sha256": candidate.sha256,
    "model_sha256": hashlib.sha256(raw).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "degree": 6,
    "initial_state_unchanged": candidate.document["q0"] == original.document["q0"]
    and candidate.document["qd0"] == original.document["qd0"],
    "duration_s": float(reference["time_s"][-1]),
    "samples": len(reference["time_s"]),
    "q_max_abs": float(
        np.max(np.abs(result.integration.state[:, :27] - reference["state"][:, :27]))
    ),
    "rate_max_abs": float(
        np.max(np.abs(result.integration.state[:, 27:] - reference["state"][:, 27:]))
    ),
    "marker_max_vector_difference_m": float(
        np.max(np.linalg.norm(result.markers_m - reference["markers"], axis=2))
    ),
    "closure_pose_max_abs": result.closure_pose_max_abs,
    "closure_rate_max_abs": result.closure_velocity_max_abs,
    "elapsed_s": result.integration.elapsed_s,
    "integration": {
        "method": "DOP853",
        "rtol": 1e-11,
        "atol": 1e-13,
        "max_step_s": 0.00025,
        "initial_state_resets": 0,
    },
    "gates": {
        "q_max_abs": 1e-6,
        "rate_max_abs": 1e-4,
        "marker_max_vector_difference_m": 1e-7,
        "closure_pose_max_abs": 1e-7,
        "closure_rate_max_abs": 1e-7,
    },
}
report["baseline_reconstruction_passed"] = all(
    report[key] < limit for key, limit in report["gates"].items()
)
np.savez_compressed(
    output / "trajectory.npz",
    time_s=result.integration.time,
    state=result.integration.state,
    markers_m=result.markers_m,
)
(output / "report.json").write_text(json.dumps(report, indent=2))
sys.stdout.write(json.dumps(report) + "\n")
if not report["baseline_reconstruction_passed"]:
    raise ValueError("Identified baseline reconstruction gate failed")
