from pathlib import Path
import json
import hashlib
import time
import sys
import numpy as np
import pinocchio
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate

raw = Path("model.bin").read_bytes()
doc = json.loads(Path("candidate.json").read_text())
candidate = NativeReplayCandidate.from_document(
    doc, json.loads(raw)["coordinate_order"], hashlib.sha256(raw).hexdigest()
)
ref = np.load("reference.npz", allow_pickle=False)
start = time.perf_counter()
result = replay_candidate(
    raw, candidate, ref["time_s"], rtol=1e-11, atol=1e-13, max_step=6.25e-5
)
marker_max = float(np.max(np.linalg.norm(result.markers_m - ref["markers_m"], axis=2)))
receipt = {
    "qualification": "clean source snapshot baseline replay; not new fitting",
    "pinocchio": pinocchio.__version__,
    "python": sys.version,
    "elapsed_s": time.perf_counter() - start,
    "candidate_sha256": result.candidate_sha256,
    "model_sha256": hashlib.sha256(raw).hexdigest(),
    "max_marker_difference_m": marker_max,
    "closure_pose_max_abs": result.closure_pose_max_abs,
    "closure_velocity_max_abs": result.closure_velocity_max_abs,
    "marker_gate_m": 1e-6,
    "passed": marker_max <= 1e-6,
}
Path("receipt_restored_verified.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)  # noqa: T201 - standalone audit receipt
