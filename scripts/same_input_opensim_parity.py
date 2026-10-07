"""Write the OpenSim same-input parity receipt (#11611, epic #11605).

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_opensim_parity

Reports the L0 (mass, mass matrix, frame kinematics), L1 (pointwise
constrained accelerations on the closure manifold) and L2 (30 ms open-loop
replay against the MuJoCo reference) gates of the OpenSim/Simbody adapter,
comparing against Pinocchio (L0, L1) and MuJoCo (L1, L2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.same_input import (
    VectorPlant,
    generate_reference_bundle,
    open_loop,
    project_to_closure,
)

EVIDENCE = Path("docs/development/full_body_models/evidence")
DEFAULT_SPEC = EVIDENCE / "ground_support/full_body_spec_hipcal_scaled.json"
DEFAULT_RECORD = EVIDENCE / "ground_support/dynamics_record.npz"
DEFAULT_OUT = EVIDENCE / "same_input/opensim_parity_receipt.json"
COMMAND = "MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_opensim_parity"
MASS_RTOL = 1e-9
ACCELERATION_ATOL = 1e-6
ACCELERATION_RTOL = 1e-8
L2_Q_ATOL = 1e-10
L2_V_ATOL = 1e-8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinocchio_mass_matrix(plant: VectorPlant, q: np.ndarray) -> np.ndarray:
    model = plant._adapter
    raw = model.mass_matrix(plant._named(q))
    idx = [model._velocity_indices[name] for name in plant.coordinate_order]
    raw = raw[np.ix_(idx, idx)]
    return np.triu(raw) + np.triu(raw, 1).T  # crba fills the upper triangle


def level_zero(opensim: VectorPlant, pinocchio: VectorPlant, seeds: int) -> dict:
    """Mass, mass matrix and frame poses at random configurations."""
    pin_model = pinocchio._adapter.model
    pin_mass = sum(pin_model.inertias[i].mass for i in range(1, pin_model.njoints))
    mass_o = opensim._adapter.mass_kg
    rows = []
    for seed in range(seeds):
        q = np.random.default_rng(seed).uniform(-0.6, 0.6, opensim.nv)
        mo = opensim._adapter.mass_matrix(opensim._named(q))
        mp = _pinocchio_mass_matrix(pinocchio, q)
        fo, fp = opensim.kinematic_frames(q), pinocchio.kinematic_frames(q)
        rows.append(
            {
                "seed": seed,
                "mass_matrix_relative": float(np.abs(mo - mp).max() / np.abs(mp).max()),
                "frame_pose_max_abs": max(
                    float(np.abs(fo[k] - fp[k]).max()) for k in fo
                ),
            }
        )
    worst = max(r["mass_matrix_relative"] for r in rows)
    return {
        "total_mass_opensim_kg": mass_o,
        "total_mass_pinocchio_kg": pin_mass,
        "total_mass_relative": abs(mass_o - pin_mass) / pin_mass,
        "mass_matrix_rtol": MASS_RTOL,
        "worst_mass_matrix_relative": worst,
        "worst_frame_pose_max_abs": max(r["frame_pose_max_abs"] for r in rows),
        "samples": rows,
        "passed": worst <= MASS_RTOL and abs(mass_o - pin_mass) / pin_mass <= MASS_RTOL,
    }


def level_one(plants: dict[str, VectorPlant], record_path: Path, stride: int) -> dict:
    """Pointwise accelerations at closure-projected recorded states."""
    with np.load(record_path) as data:
        q_all, v_all, tau_all = data["q"], data["v"], data["tau"]
    rows = []
    for frame in range(0, q_all.shape[0], stride):
        state = project_to_closure(plants["pinocchio"], q_all[frame], v_all[frame])
        own = project_to_closure(plants["opensim"], q_all[frame], v_all[frame])
        acc = {
            e: p.acceleration(state.q, state.v, tau_all[frame])
            for e, p in plants.items()
        }
        scale = float(np.abs(acc["pinocchio"]).max())
        diff = {
            e: float(np.abs(acc["opensim"] - acc[e]).max())
            for e in ("pinocchio", "mujoco", "drake")
            if e in acc
        }
        rows.append(
            {
                "frame": frame,
                "max_abs_acceleration": scale,
                "opensim_minus": diff,
                "projection_spread_q": float(np.abs(own.q - state.q).max()),
                "projection_spread_v": float(np.abs(own.v - state.v).max()),
                "passed": diff["pinocchio"]
                <= ACCELERATION_ATOL + ACCELERATION_RTOL * scale,
            }
        )
    return {
        "acceleration_atol": ACCELERATION_ATOL,
        "acceleration_rtol": ACCELERATION_RTOL,
        "frames_checked": len(rows),
        "worst_difference_to_pinocchio": max(
            r["opensim_minus"]["pinocchio"] for r in rows
        ),
        "worst_relative_difference_to_pinocchio": max(
            r["opensim_minus"]["pinocchio"] / max(r["max_abs_acceleration"], 1.0)
            for r in rows
        ),
        "worst_projection_spread": max(
            max(r["projection_spread_q"], r["projection_spread_v"]) for r in rows
        ),
        "passed": all(r["passed"] for r in rows),
        "frames": rows,
    }


def level_two(spec: bytes, record_path: Path) -> dict:
    """30 ms open-loop replay of the MuJoCo reference bundle."""
    with np.load(record_path) as record:
        bundle = generate_reference_bundle(
            spec, record["time_s"], record["q"], duration_s=0.03
        )
    rollout = open_loop(
        VectorPlant("opensim", bundle.spec_bytes),
        bundle.q0,
        bundle.v0,
        bundle.efforts,
        dt_s=bundle.dt_s,
    )
    dq = float(np.abs(rollout.q - bundle.reference_q).max())
    dv = float(np.abs(rollout.v - bundle.reference_v).max())
    return {
        "duration_s": 0.03,
        "steps": int(rollout.q.shape[0] - 1),
        "max_abs_q_difference_rad": dq,
        "max_abs_v_difference_rad_s": dv,
        "q_atol": L2_Q_ATOL,
        "v_atol": L2_V_ATOL,
        "passed": dq <= L2_Q_ATOL and dv <= L2_V_ATOL,
    }


def build_receipt(spec_path: Path, record_path: Path, stride: int) -> dict:
    """Return the receipt; ``stride`` must be a positive integer."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    import opensim

    spec = spec_path.read_bytes()
    plants = {
        e: VectorPlant(e, spec) for e in ("opensim", "pinocchio", "mujoco", "drake")
    }
    levels = {
        "L0": level_zero(plants["opensim"], plants["pinocchio"], seeds=3),
        "L1": level_one(plants, record_path, stride),
        "L2": level_two(spec, record_path),
    }
    return {
        "schema": "same-input-opensim-parity/v1",
        "issue": 11611,
        "epic": 11605,
        "generated_by": COMMAND,
        "opensim_version": opensim.GetVersionAndDate(),
        "spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "record": {"path": str(record_path), "sha256": _sha256(record_path)},
        "adapter": "src/engines/physics_engines/opensim/python/full_body_parity.py",
        "method": (
            "Simbody mass matrix, zero-udot bias and station Jacobians; shared "
            "contact law; exact unregularised weld KKT (no OpenSim constraint or "
            "HuntCrossleyForce)"
        ),
        "levels": levels,
        "passed": all(level["passed"] for level in levels.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    receipt = build_receipt(args.spec, args.record, args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    levels = receipt["levels"]
    print(  # noqa: T201 - CLI summary
        f"L0 {levels['L0']['worst_mass_matrix_relative']:.1e} "
        f"L1 {levels['L1']['worst_difference_to_pinocchio']:.1e} "
        f"L2 q {levels['L2']['max_abs_q_difference_rad']:.1e} "
        f"passed={receipt['passed']}"
    )


if __name__ == "__main__":
    main()
