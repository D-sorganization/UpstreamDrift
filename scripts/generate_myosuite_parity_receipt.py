"""Generate the MyoSuite same-input parity receipt (#11612, epic #11605).

Run from the repository root with the offscreen environment::

    MUJOCO_GL=egl MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen \
        python3 -m scripts.generate_myosuite_parity_receipt

The receipt records the L0 (mass, M(q)), L1 (pointwise accelerations) and L2
(30 ms open-loop replay) numbers of the spec model in the MyoSuite runtime
against MuJoCo, plus the max coordinate error over a 0.2 s open-loop replay.
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

import numpy as np

from src.shared.python.engine_core.mujoco_compat import full_mass_matrix
from src.shared.python.motion_matching.same_input import (
    VectorPlant,
    generate_reference_bundle,
    open_loop,
    project_to_closure,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/ground_support"
RECEIPT = (
    ROOT
    / "docs/development/full_body_models/evidence/same_input"
    / "myosuite_parity_receipt.json"
)
FRAMES = (0, 400, 420)
DECISION = (
    "spec model in MyoSuite runtime; MyoHub body excluded from torque parity "
    "(unrelated topology, 11/44 coords)"
)


def _l0(plants: dict[str, VectorPlant], record) -> dict[str, float]:
    mass_err, matrix_err = 0.0, 0.0
    for frame in FRAMES:
        mats, masses = {}, {}
        for engine, plant in plants.items():
            adapter = plant._adapter
            adapter.generalized_forces(
                plant._named(record["q"][frame]), plant._named(record["v"][frame])
            )
            mats[engine] = full_mass_matrix(adapter._mj, adapter.model, adapter.data)
            masses[engine] = float(np.sum(adapter.model.body_mass))
        mass_err = max(mass_err, abs(masses["myosuite"] - masses["mujoco"]))
        matrix_err = max(
            matrix_err, float(np.abs(mats["myosuite"] - mats["mujoco"]).max())
        )
    return {"max_abs_mass_error_kg": mass_err, "max_abs_mass_matrix_error": matrix_err}


def _l1(plants: dict[str, VectorPlant], record) -> dict[str, dict[str, float]]:
    out = {}
    for frame in FRAMES:
        state = project_to_closure(
            plants["mujoco"], record["q"][frame], record["v"][frame]
        )
        tau = record["tau"][frame]
        ref = plants["mujoco"].acceleration(state.q, state.v, tau)
        got = plants["myosuite"].acceleration(state.q, state.v, tau)
        out[str(frame)] = {
            "max_abs_acceleration_error": float(np.abs(got - ref).max()),
            "max_abs_acceleration": float(np.abs(ref).max()),
            "bound": float(1e-6 + 1e-8 * np.abs(ref).max()),
        }
    return out


def _replay(spec: bytes, record, duration_s: float) -> dict[str, object]:
    bundle = generate_reference_bundle(
        spec, record["time_s"], record["q"], duration_s=duration_s
    )
    rollout = open_loop(
        VectorPlant("myosuite", spec),
        bundle.q0,
        bundle.v0,
        bundle.efforts,
        dt_s=bundle.dt_s,
    )
    dq = np.abs(rollout.q - bundle.reference_q).max(axis=1)
    dv = np.abs(rollout.v - bundle.reference_v).max(axis=1)
    return {
        "duration_s": duration_s,
        "steps": int(dq.size - 1),
        "dt_s": float(bundle.dt_s),
        "max_abs_q_error_rad": float(dq.max()),
        "max_abs_v_error_rad_s": float(dv.max()),
        "bit_exact": bool(
            np.array_equal(rollout.q, bundle.reference_q)
            and np.array_equal(rollout.v, bundle.reference_v)
        ),
        "q_error_history": {
            "time_s": [round(float(i * bundle.dt_s), 9) for i in range(dq.size)][::10],
            "max_abs_q_error_rad": [float(x) for x in dq][::10],
        },
    }


def main() -> None:
    spec = (EVIDENCE / "full_body_spec_hipcal_scaled.json").read_bytes()
    with np.load(EVIDENCE / "dynamics_record.npz") as data:
        record = {key: data[key] for key in ("time_s", "q", "v", "tau")}
    plants = {e: VectorPlant(e, spec) for e in ("mujoco", "myosuite")}
    receipt = {
        "issue": 11612,
        "epic": 11605,
        "decision": DECISION,
        "versions": {
            "myosuite": metadata.version("myosuite"),
            "mujoco": metadata.version("mujoco"),
            "numpy": np.__version__,
        },
        "spec": "full_body_spec_hipcal_scaled.json",
        "spec_sha256": hashlib.sha256(spec).hexdigest(),
        "myosuite_loader": "myosuite.envs.env_base.MujocoEnv (MjSpec.from_file -> compile)",
        "kkt_regularization": 0.0,
        "L0": _l0(plants, record),
        "L1": _l1(plants, record),
        "L2_30ms": _replay(spec, record, 0.03),
        "extended_200ms": _replay(spec, record, 0.2),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k != "extended_200ms"}, indent=2
        )
    )
    print(json.dumps(receipt["extended_200ms"], indent=2))


if __name__ == "__main__":
    main()
