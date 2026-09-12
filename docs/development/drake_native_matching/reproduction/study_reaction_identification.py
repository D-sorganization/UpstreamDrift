"""Bounded native feasibility study; no trajectory optimization or integration."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile
from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape

p = argparse.ArgumentParser()
p.add_argument("--phase", choices=("reference", "identify"), required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--reference", type=Path)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
a.output.mkdir()
base = Path("/mnt/c/Users/diete")
raw = (base / "native_geometry_spec_9967.json").read_bytes()
spec = json.loads(raw)
names = spec["coordinate_order"]
doc = json.loads(
    (base / "native-root-force-9967-02/returned-candidate.json").read_bytes()
)
known = np.load(base / "drake-native-reference-10022-01/reference.npz")
clock, states = known["time_s"], known["state"]
root = next(j for j in spec["joints"] if j["parent"] == "world")
rotation = np.asarray(root["parent_to_base"])[:3, :3].T
profile = NativeEffortProfile(names, doc["coefficients"], rotation)
if a.phase == "reference":
    engine = NativePinocchioModel(spec)
    qdd = []
    for t, state in zip(clock, states, strict=True):
        result = engine.accelerations(
            dict(zip(names, state[:27], strict=True)),
            dict(zip(names, state[27:], strict=True)),
            profile.evaluate(float(t)),
        )
        qdd.append([result[name] for name in names])
    np.savez_compressed(a.output / "reference.npz", time_s=clock, state=states, qdd=qdd)
    (a.output / "report.json").write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(raw).hexdigest(),
                "scope": "Analytic Pinocchio accelerations at existing qualified baseline states, no new integration",
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    sys.exit(0)
from pydrake.multibody.tree import JacobianWrtVariable
from src.engines.physics_engines.drake.python.native_model import NativeDrakeModel
from reaction_identification import projected_system

reference = np.load(a.reference / "reference.npz")
if not np.array_equal(reference["state"], states) or not np.array_equal(
    reference["time_s"], clock
):
    raise ValueError("Baseline source states differ")
engine = NativeDrakeModel(
    (base / "native-golf-9967-01.urdf").read_bytes(),
    (base / "native-golf-9967-01.sidecar.json").read_bytes(),
    raw,
)
plant = engine.plant
indices = [plant.GetJointByName(name).velocity_start() for name in names]
frame_a = plant.GetFrameByName("native_closure_a")
frame_b = plant.GetFrameByName("native_closure_b")
matrices = []
biases = []
jacobians = []
gammas = []
efforts = []
closure = []
for t, state in zip(clock, states, strict=True):
    coordinates = dict(zip(names, state[:27], strict=True))
    rates = dict(zip(names, state[27:], strict=True))
    engine.accelerations(coordinates, rates, dict.fromkeys(names, 0.0))
    closure.append(max(np.max(np.abs(v)) for v in engine.closure_errors()))
    matrices.append(plant.CalcMassMatrix(engine.context)[np.ix_(indices, indices)])
    biases.append(
        (
            plant.CalcBiasTerm(engine.context)
            - plant.CalcGravityGeneralizedForces(engine.context)
        )[indices]
    )
    jacobians.append(
        plant.CalcJacobianSpatialVelocity(
            engine.context,
            JacobianWrtVariable.kV,
            frame_b,
            np.zeros(3),
            frame_a,
            frame_a,
        )[:, indices]
    )
    gammas.append(
        plant.CalcBiasSpatialAcceleration(
            engine.context,
            JacobianWrtVariable.kV,
            frame_b,
            np.zeros(3),
            frame_a,
            frame_a,
        ).get_coeffs()
    )
    efforts.append(
        profile.bernstein_control_jacobian(
            float(t), basis_duration_s=0.8, first_control=0
        )
    )
arrays = list(
    map(np.asarray, (matrices, biases, reference["qdd"], jacobians, gammas, efforts))
)
compatibility = float(
    np.max(np.abs(np.einsum("sij,sj->si", arrays[3], arrays[2]) + arrays[4]))
)
report = {
    "model_sha256": hashlib.sha256(raw).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "Loop reaction-eliminated global sextic identification feasibility only; no C3D optimization or identified-profile forward replay",
    "state_closure_max_abs": float(max(closure)),
    "acceleration_closure_max_abs": compatibility,
    "samples": len(clock),
    "controls": 189,
    "basis": "Global Bernstein degree6 over0.8s; coordinate-major controls; exact shared native force map",
    "rank_relative_cutoff": 1e-10,
}
(a.output / "preflight.json").write_text(json.dumps(report, indent=2))
if max(closure) > 1e-7:
    raise ValueError("Baseline position/rate closure failed")
matrix, rhs = projected_system(*arrays)
rows_per_sample = 21
train = np.concatenate(
    [
        np.arange(i * rows_per_sample, (i + 1) * rows_per_sample)
        for i in range(0, len(clock), 2)
    ]
)
validation = np.setdiff1d(np.arange(len(rhs)), train)
scale = np.linalg.norm(matrix[train], axis=0)
if np.any(scale == 0):
    raise ValueError("A polynomial coefficient is completely unobserved")
normalized = matrix[train] / scale
scaled, _, rank, singular = np.linalg.lstsq(normalized, rhs[train], rcond=1e-10)
controls = (scaled / scale).reshape(27, 7)
coefficients = bernstein_to_simscape(controls, duration_s=0.8)
identified = NativeEffortProfile(names, coefficients, rotation)
acceleration_error = []
for i in range(1, len(clock), 2):
    state = states[i]
    actual = engine.accelerations(
        dict(zip(names, state[:27], strict=True)),
        dict(zip(names, state[27:], strict=True)),
        identified.evaluate(float(clock[i])),
    )
    acceleration_error.append(
        np.asarray([actual[name] for name in names]) - reference["qdd"][i]
    )
error = np.asarray(acceleration_error)
report.update(
    training_samples=len(range(0, len(clock), 2)),
    held_out_samples=len(range(1, len(clock), 2)),
    normalized_rank=int(rank),
    nullity=int(189 - rank),
    normalized_singular_max=float(singular[0]),
    normalized_singular_min=float(singular[-1]),
    training_projected_max_abs=float(
        np.max(np.abs(matrix[train] @ controls.ravel() - rhs[train]))
    ),
    held_out_projected_max_abs=float(
        np.max(np.abs(matrix[validation] @ controls.ravel() - rhs[validation]))
    ),
    held_out_acceleration_max_abs=float(np.max(np.abs(error))),
    held_out_acceleration_max_scaled=float(
        np.max(np.abs(error) / (1 + np.abs(reference["qdd"][1::2])))
    ),
    control_max_abs=float(np.max(np.abs(controls))),
)
np.savez_compressed(
    a.output / "identified.npz",
    controls=controls,
    coefficients=coefficients,
    matrix=matrix,
    rhs=rhs,
    time_s=clock,
    training_rows=train,
    held_out_rows=validation,
    singular_values=singular,
)
(a.output / "report.json").write_text(json.dumps(report, indent=2))
sys.stdout.write(json.dumps(report) + "\n")
