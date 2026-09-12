"""Physical body angular velocities for one explicit reconstructed-seed pair."""

from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from pydrake.multibody.tree import JacobianWrtVariable
from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel

base = Path("/mnt/c/Users/diete")
output = base / "drake-restart-angular-10022-01"
if output.exists():
    raise FileExistsError(output)
raw = (base / "native_geometry_spec_9967.json").read_bytes()
spec = json.loads(raw)
meta = json.loads((base / "native-golf-9967-01.sidecar.json").read_bytes())
engine = NativeDrakeModel(
    (base / "native-golf-9967-01.urdf").read_bytes(),
    (base / "native-golf-9967-01.sidecar.json").read_bytes(),
    raw,
)
names = spec["coordinate_order"]
body_names = list(meta["body_links"])
frames = [
    engine.plant.GetBodyByName(meta["body_links"][name]).body_frame()
    for name in body_names
]
v_columns = [engine.plant.GetJointByName(name).velocity_start() for name in names]
world = engine.plant.world_frame()
source = np.load(base / "drake-restart-source-10022-01/trajectory.npz")
reconstructed = np.load(base / "drake-restart-reconstructed-10022-01/trajectory.npz")
for key in ("time_s", "valid"):
    if not np.array_equal(source[key], reconstructed[key]):
        raise ValueError("Clock or marker mask differs")
if not np.array_equal(source["state"][0], reconstructed["state"][0]):
    raise ValueError("Initial state changed")
clock = source["time_s"]
angular = []
for data in (source, reconstructed):
    values = []
    for state in data["state"]:
        engine.frame_poses(dict(zip(names, state[:27], strict=True)))
        v = np.empty(27)
        v[v_columns] = state[27:]
        values.append(
            [
                engine.plant.CalcJacobianSpatialVelocity(
                    engine.context,
                    JacobianWrtVariable.kV,
                    frame,
                    np.zeros(3),
                    world,
                    world,
                )[:3]
                @ v
                for frame in frames
            ]
        )
    angular.append(np.asarray(values))
difference = np.linalg.norm(angular[1] - angular[0], axis=2)
body_results = []
for j, name in enumerate(body_names):
    i = int(np.argmax(difference[:, j]))
    magnitude = float(np.linalg.norm(angular[0][i, j]))
    body_results.append(
        {
            "body": name,
            "time_of_max_difference_s": float(clock[i]),
            "max_vector_difference_rad_s": float(difference[i, j]),
            "reference_speed_at_max_rad_s": magnitude,
            "relative_difference_at_max": float(difference[i, j] / magnitude)
            if magnitude
            else None,
        }
    )
joint = next(
    j
    for j in spec["joints"]
    if any(p["coordinate"] == "LSInputX" for p in j["primitives"])
)
parent, child = [body_names.index(joint[key]) for key in ("parent", "child")]
relative = [value[:, child] - value[:, parent] for value in angular]
error = np.linalg.norm(relative[1] - relative[0], axis=1)
i = int(np.argmax(error))
rate_delta = np.abs(reconstructed["state"][:, 27:] - source["state"][:, 27:])
ri, rj = np.unravel_index(np.argmax(rate_delta), rate_delta.shape)
report = {
    "scope": "One Pinocchio replay pair; actual Drake world-expressed frame angular Jacobians; diagnostic only, no exact-restart or fit acceptance",
    "source_candidate_sha256": str(source["candidate_sha256"]),
    "reconstructed_candidate_sha256": str(reconstructed["candidate_sha256"]),
    "model_sha256": hashlib.sha256(raw).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "samples": len(clock),
    "duration_s": float(clock[-1]),
    "body_count": len(body_names),
    "all_body_max_vector_difference_rad_s": float(np.max(difference)),
    "per_body": body_results,
    "shoulder_relative": {
        "definition": "omega_child_world minus omega_parent_world, both expressed in world",
        "max_vector_difference_rad_s": float(error[i]),
        "time_s": float(clock[i]),
        "reference_angular_speed_rad_s": float(np.linalg.norm(relative[0][i])),
        "relative_difference": float(error[i] / np.linalg.norm(relative[0][i])),
        "source_vector_rad_s": relative[0][i].tolist(),
        "reconstructed_vector_rad_s": relative[1][i].tolist(),
    },
    "scalar_rate_peak": {
        "coordinate": names[rj],
        "time_s": float(clock[ri]),
        "max_abs_difference": float(rate_delta[ri, rj]),
        "source_rate": float(source["state"][ri, rj + 27]),
        "reconstructed_rate": float(reconstructed["state"][ri, rj + 27]),
    },
    "q_max_abs_difference": float(
        np.max(np.abs(reconstructed["state"][:, :27] - source["state"][:, :27]))
    ),
    "observed_marker_max_vector_difference_m": float(
        np.max(
            np.linalg.norm(reconstructed["markers_m"] - source["markers_m"], axis=2)[
                source["valid"]
            ]
        )
    ),
    "initial_states_identical": True,
    "full_state_restart_gate_passed": False,
    "state_rate_gate": 1e-4,
}
output.mkdir()
np.savez_compressed(
    output / "angular-velocities.npz",
    time_s=clock,
    body_names=body_names,
    source_world_rad_s=angular[0],
    reconstructed_world_rad_s=angular[1],
)
(output / "report.json").write_text(json.dumps(report, indent=2))
sys.stdout.write(
    json.dumps({k: v for k, v in report.items() if k != "per_body"}) + "\n"
)
