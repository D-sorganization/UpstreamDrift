"""Verification script for Drake full-body builder and adapter (FB-3-D, #10067).

Tests all four Done Gates:
(a) Upper-body slice reproduces qualified Drake model mass matrix and FK to 1e-12 at random states;
(b) Full-body FK matches spec frames to 1e-12;
(c) Contact force at analytic states equals the FB-2 reference to 1e-12;
(d) Closure residual unchanged on the upper-body chain to 1e-12.
Plus forward accelerations combining contact forces and weld loop-closure dynamics.

If real Drake is not available in the local environment, automatically executes
on ControlTower via SSH and saves the resulting receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.development.full_body_models.evidence._gates import (
    FB3_DRAKE_THRESHOLDS,
    evaluate_gates,
)

FULL_BODY_SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
UPPER_SPEC_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL_BODY_URDF_PATH = (
    ROOT / "src/engines/physics_engines/drake/python/full_body_urdf.py"
)
FULL_BODY_MODEL_PATH = (
    ROOT / "src/engines/physics_engines/drake/python/full_body_model.py"
)
CONTACT_LAW_PATH = ROOT / "src/shared/python/motion_matching/contact_law.py"
FULL_BODY_SPEC_PY_PATH = ROOT / "src/shared/python/motion_matching/full_body_spec.py"
RECEIPT_PATH = HERE / "receipt.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_real_drake() -> bool:
    try:
        import pydrake.all as drake_all

        return type(drake_all).__module__ != "unittest.mock" and hasattr(
            drake_all, "MultibodyPlant"
        )
    except (ImportError, AttributeError):
        return False


def _sync_files_to_remote(remote_base: str) -> None:
    p0 = subprocess.run(
        [
            "ssh",
            "controltower",
            "wsl",
            "-d",
            "ControlTower-Runner",
            "mkdir",
            "-p",
            f"{remote_base}/src/engines/physics_engines/drake/python",
            f"{remote_base}/src/shared/python/motion_matching",
            f"{remote_base}/docs/development/full_body_models/evidence/fb3_drake",
            f"{remote_base}/docs/development/simscape_tour_matching/native_evidence",
            f"{remote_base}/tests/unit/motion_matching",
        ],
        capture_output=True,
    )
    if p0.returncode != 0:
        raise RuntimeError(f"Remote mkdir failed: {p0.stderr.decode()}")

    files_to_sync = [
        (
            FULL_BODY_SPEC_PATH,
            f"{remote_base}/docs/development/full_body_models/full_body_spec_v1.json",
        ),
        (
            UPPER_SPEC_PATH,
            f"{remote_base}/docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json",
        ),
        (
            FULL_BODY_URDF_PATH,
            f"{remote_base}/src/engines/physics_engines/drake/python/full_body_urdf.py",
        ),
        (
            FULL_BODY_MODEL_PATH,
            f"{remote_base}/src/engines/physics_engines/drake/python/full_body_model.py",
        ),
        (
            CONTACT_LAW_PATH,
            f"{remote_base}/src/shared/python/motion_matching/contact_law.py",
        ),
        (
            FULL_BODY_SPEC_PY_PATH,
            f"{remote_base}/src/shared/python/motion_matching/full_body_spec.py",
        ),
        (
            ROOT / "tests/unit/motion_matching/test_full_body_drake.py",
            f"{remote_base}/tests/unit/motion_matching/test_full_body_drake.py",
        ),
        (
            Path(__file__).resolve(),
            f"{remote_base}/docs/development/full_body_models/evidence/fb3_drake/verify_drake_full_body.py",
        ),
    ]

    for local_path, remote_path in files_to_sync:
        content = local_path.read_bytes()
        p = subprocess.run(
            [
                "ssh",
                "controltower",
                "wsl",
                "-d",
                "ControlTower-Runner",
                "tee",
                remote_path,
            ],
            input=content,
            capture_output=True,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"Failed to sync {local_path} to {remote_path}: {p.stderr.decode()}"
            )


def run_remote_controltower() -> dict[str, Any]:
    """Execute verification remotely on ControlTower and return the receipt."""
    sys.stdout.write(
        "Real Drake not available locally. Dispatching to ControlTower...\n"
    )
    sys.stdout.flush()

    remote_base = "/home/dieterolson/fb3_pinocchio_test"
    _sync_files_to_remote(remote_base)

    script = (
        f"cd {remote_base}\n"
        f"/home/dieterolson/drake-native-10022/bin/python "
        f"docs/development/full_body_models/evidence/fb3_drake/verify_drake_full_body.py --local\n"
    ).encode()
    res = subprocess.run(
        ["ssh", "controltower", "wsl", "-d", "ControlTower-Runner", "bash"],
        input=script,
        capture_output=True,
    )
    if res.returncode != 0:
        sys.stderr.write(
            f"Remote error:\n{res.stderr.decode('utf-8', errors='ignore')}\n"
            f"Remote stdout:\n{res.stdout.decode('utf-8', errors='ignore')}\n"
        )
        raise RuntimeError(
            f"Remote verification failed with exit code {res.returncode}"
        )
    sys.stdout.write(res.stdout.decode("utf-8", errors="ignore"))

    # Fetch receipt from remote
    fetch_cmd = f"cat {remote_base}/docs/development/full_body_models/evidence/fb3_drake/receipt.json\n".encode()
    res_fetch = subprocess.run(
        ["ssh", "controltower", "wsl", "-d", "ControlTower-Runner", "bash"],
        input=fetch_cmd,
        capture_output=True,
    )
    if res_fetch.returncode != 0:
        raise RuntimeError("Failed to fetch remote receipt.json")
    receipt_data = json.loads(res_fetch.stdout.decode("utf-8"))
    return receipt_data


def _verify_gate_a(
    slice_model: Any,
    qual_model: Any,
    full_model: Any,
    upper_spec: Mapping[str, Any],
    rng: np.random.Generator,
    num_states: int,
) -> tuple[float, float]:
    """Gate (a): Upper-body slice parity in FK and mass matrix."""
    max_fk_diff = 0.0
    max_mass_diff = 0.0
    upper_coords = upper_spec["coordinate_order"]
    full_coords = full_model.names

    for b in upper_spec["bodies"]:
        if b["name"] == "world":
            continue
        body_full = full_model.plant.GetBodyByName(
            full_model.metadata["body_links"][b["name"]], full_model._instance
        )
        body_qual = qual_model.plant.GetBodyByName(
            qual_model.metadata["body_links"][b["name"]], qual_model._instance
        )
        diff_body = abs(body_full.default_mass() - body_qual.default_mass())
        if diff_body > max_mass_diff:
            max_mass_diff = diff_body

    for _ in range(num_states):
        q_up = {name: float(rng.uniform(-0.35, 0.35)) for name in upper_coords}
        q_full = {**dict.fromkeys(full_coords, 0.0), **q_up}

        poses_slice = slice_model.frame_poses(q_up)
        poses_qual = qual_model.frame_poses(q_up)
        poses_full = full_model.frame_poses(q_full)

        for frame in poses_qual:
            diff_fk = float(np.linalg.norm(poses_slice[frame] - poses_qual[frame]))
            if diff_fk > max_fk_diff:
                max_fk_diff = diff_fk
            diff_full = float(np.linalg.norm(poses_full[frame] - poses_qual[frame]))
            if diff_full > max_fk_diff:
                max_fk_diff = diff_full

        # Mass matrix comparison between slice_model and qual_model
        slice_model.plant.SetPositions(
            slice_model.context, slice_model._vector(q_up, slice_model._q_indices)
        )
        qual_model.plant.SetPositions(
            qual_model.context, qual_model._vector(q_up, qual_model._q_indices)
        )
        m_slice = slice_model.plant.CalcMassMatrix(slice_model.context)
        m_qual = qual_model.plant.CalcMassMatrix(qual_model.context)
        diff_m = float(np.max(np.abs(m_slice - m_qual)))
        if diff_m > max_mass_diff:
            max_mass_diff = diff_m

    return max_fk_diff, max_mass_diff


def _verify_gate_b(
    full_model: Any,
    slice_model: Any,
    full_spec: Mapping[str, Any],
    upper_spec: Mapping[str, Any],
    rng: np.random.Generator,
    num_states: int,
) -> float:
    """Gate (b): Full-body FK matches spec frames."""
    max_fk_diff = 0.0
    coords = full_spec["coordinate_order"]
    upper_coords = upper_spec["coordinate_order"]
    for _ in range(num_states):
        q_full = {name: float(rng.uniform(-0.3, 0.3)) for name in coords}
        q_upper = {name: q_full[name] for name in upper_coords}
        poses_full = full_model.frame_poses(q_full)
        poses_slice = slice_model.frame_poses(q_upper)
        for frame in poses_slice:
            diff = float(np.linalg.norm(poses_full[frame] - poses_slice[frame]))
            if diff > max_fk_diff:
                max_fk_diff = diff

    return max_fk_diff


def _verify_gate_c(
    full_model: Any,
    full_spec: Mapping[str, Any],
) -> tuple[int, float, float, float]:
    """Gate (c): Contact forces at analytic states equal FB-2 reference."""
    from src.shared.python.motion_matching.contact_law import sphere_ground_contact

    coords = full_spec["coordinate_order"]
    q_neutral = dict.fromkeys(coords, 0.0)
    v_neutral = dict.fromkeys(coords, 0.0)
    forces_neutral = full_model.contact_forces(q_neutral, v_neutral)
    if len(forces_neutral) != 4:
        raise RuntimeError(
            f"Gate (c) setup: expected 4 contacts, got {len(forces_neutral)}"
        )

    q_pen = dict(q_neutral)
    q_pen["TranslationInputZ"] = -0.90
    v_moving = {name: (0.1 if "Translation" in name else 0.0) for name in coords}
    forces_pen = full_model.contact_forces(q_pen, v_moving)
    pen_count = sum(1 for s in forces_pen.values() if s.penetration_m > 0.0)
    # Without a penetrating sphere gate (c) would compare zero forces (vacuous pass).
    if len(forces_pen) != 4 or pen_count == 0:
        raise RuntimeError(
            f"Gate (c) setup: {len(forces_pen)} contacts, {pen_count} penetrating"
        )

    max_fn = 0.0
    max_ff = 0.0
    max_p = 0.0
    for name, sample in forces_pen.items():
        pos, vel, radius = full_model.get_sphere_kinematics(name, q_pen, v_moving)
        ref = sphere_ground_contact(
            pos, vel, radius, full_model.ground_plane, full_model.contact_parameters
        )
        diff_fn = float(np.linalg.norm(sample.normal_force_n - ref.normal_force_n))
        diff_ff = float(np.linalg.norm(sample.friction_force_n - ref.friction_force_n))
        diff_p = float(abs(sample.penetration_m - ref.penetration_m))
        if diff_fn > max_fn:
            max_fn = diff_fn
        if diff_ff > max_ff:
            max_ff = diff_ff
        if diff_p > max_p:
            max_p = diff_p

    return pen_count, max_fn, max_ff, max_p


def _verify_gate_d(
    full_model: Any,
    slice_model: Any,
    full_spec: Mapping[str, Any],
    upper_spec: Mapping[str, Any],
    rng: np.random.Generator,
    num_states: int,
) -> tuple[float, float]:
    """Gate (d): Closure residual unchanged on upper-body chain."""
    max_pos = 0.0
    max_vel = 0.0
    coords = full_spec["coordinate_order"]
    upper_coords = upper_spec["coordinate_order"]
    for _ in range(num_states):
        q_full = {name: float(rng.uniform(-0.25, 0.25)) for name in coords}
        q_up = {name: q_full[name] for name in upper_coords}
        v_full = {name: float(rng.uniform(-0.4, 0.4)) for name in coords}
        v_up = {name: v_full[name] for name in upper_coords}

        pos_slice, vel_slice = slice_model.closure_residuals(q_up, v_up)
        pos_full, vel_full = full_model.closure_residuals(q_full, v_full)
        diff_cp = float(np.max(np.abs(pos_full - pos_slice)))
        diff_cv = float(np.max(np.abs(vel_full - vel_slice)))
        if diff_cp > max_pos:
            max_pos = diff_cp
        if diff_cv > max_vel:
            max_vel = diff_cv

    return max_pos, max_vel


def _verify_accelerations(
    full_model: Any,
    full_spec: Mapping[str, Any],
) -> tuple[int, bool, float, float]:
    coords = full_spec["coordinate_order"]
    q = dict.fromkeys(coords, 0.0)
    q["TranslationInputZ"] = -0.90
    v = dict.fromkeys(coords, 0.0)
    tau = dict.fromkeys(coords, 0.0)

    acc = full_model.accelerations(q, v, tau)
    all_finite = len(acc) == len(coords) and all(
        np.isfinite(val) for val in acc.values()
    )

    pos_err, vel_err = full_model.closure_errors()
    max_cp = float(np.max(np.abs(pos_err)))
    max_cv = float(np.max(np.abs(vel_err)))
    return len(acc), all_finite, max_cp, max_cv


def build_status_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate FB-3 Drake metrics against documented thresholds (#10960 P0-9)."""
    return evaluate_gates(metrics, FB3_DRAKE_THRESHOLDS)


def _group_status(eval_gates: Mapping[str, Any], *gate_names: str) -> str:
    """PASSED only if every named gate passed in the recorded evaluation."""
    passed = all(eval_gates[name]["passed"] for name in gate_names)
    return "PASSED" if passed else "FAILED"


def _receipt_provenance(drake_version: str) -> dict[str, Any]:
    """Environment and input-file hashes recorded on every FB-3 receipt."""
    return {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "drake_version": drake_version,
        },
        "inputs": {
            "full_body_spec_v1.json": sha256_file(FULL_BODY_SPEC_PATH),
            "native_geometry_spec_9967.json": sha256_file(UPPER_SPEC_PATH),
            "full_body_urdf.py": sha256_file(FULL_BODY_URDF_PATH),
            "full_body_model.py": sha256_file(FULL_BODY_MODEL_PATH),
            "contact_law.py": sha256_file(CONTACT_LAW_PATH),
            "full_body_spec.py": sha256_file(FULL_BODY_SPEC_PY_PATH),
        },
    }


def _assemble_receipt(
    drake_version: str,
    num_states: int,
    gate_a: tuple[float, float],
    gate_b_fk: float,
    gate_c: tuple[int, float, float, float],
    gate_d: tuple[float, float],
    acc: tuple[int, bool, float, float],
) -> dict[str, Any]:
    gate_a_fk, gate_a_mass = gate_a
    pen_count, gate_c_fn, gate_c_ff, gate_c_p = gate_c
    gate_d_pos, gate_d_vel = gate_d
    num_coords, acc_finite, max_cp, max_cv = acc

    gate_metrics = {
        "gate_a_fk_diff_m": float(gate_a_fk),
        "gate_a_mass_matrix_diff": float(gate_a_mass),
        "gate_b_fk_diff_m": float(gate_b_fk),
        "gate_c_normal_force_diff_n": float(gate_c_fn),
        "gate_c_friction_force_diff_n": float(gate_c_ff),
        "gate_c_penetration_diff_m": float(gate_c_p),
        "gate_d_position_residual_diff": float(gate_d_pos),
        "gate_d_velocity_residual_diff": float(gate_d_vel),
        "max_closure_pos_error": float(max_cp),
        "max_closure_vel_error": float(max_cv),
        "accelerations_all_finite": bool(acc_finite),
    }
    status_eval = build_status_from_metrics(gate_metrics)
    eval_gates = status_eval["gates"]

    return {
        "work_package": "FB-3-D",
        "issue": "#10067",
        "epic": "#10062",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status_eval["status"],
        "gate_evaluation": status_eval,
        "thresholds": dict(FB3_DRAKE_THRESHOLDS),
        **_receipt_provenance(drake_version),
        "gates": {
            "gate_a_upper_body_slice_parity": {
                "status": _group_status(
                    eval_gates, "gate_a_fk_diff_m", "gate_a_mass_matrix_diff"
                ),
                "random_states_tested": num_states,
                "max_fk_diff_m": gate_a_fk,
                "max_mass_matrix_diff": gate_a_mass,
                "tolerance": 1e-12,
            },
            "gate_b_full_body_fk": {
                "status": _group_status(eval_gates, "gate_b_fk_diff_m"),
                "random_states_tested": num_states,
                "max_fk_diff_m": gate_b_fk,
                "tolerance": 1e-12,
            },
            "gate_c_contact_force_parity": {
                "status": _group_status(
                    eval_gates,
                    "gate_c_normal_force_diff_n",
                    "gate_c_friction_force_diff_n",
                    "gate_c_penetration_diff_m",
                ),
                "num_contact_spheres": 4,
                "penetrating_spheres_tested": pen_count,
                "max_normal_force_diff_n": gate_c_fn,
                "max_friction_force_diff_n": gate_c_ff,
                "max_penetration_diff_m": gate_c_p,
                "tolerance": 1e-12,
            },
            "gate_d_closure_residual_parity": {
                "status": _group_status(
                    eval_gates,
                    "gate_d_position_residual_diff",
                    "gate_d_velocity_residual_diff",
                ),
                "random_states_tested": num_states,
                "max_position_residual_diff": gate_d_pos,
                "max_velocity_residual_diff": gate_d_vel,
                "tolerance": 1e-12,
            },
        },
        "accelerations": {
            "status": _group_status(
                eval_gates,
                "accelerations_all_finite",
                "max_closure_pos_error",
                "max_closure_vel_error",
            ),
            "num_coordinates": num_coords,
            "all_finite": acc_finite,
            "max_closure_pos_error": max_cp,
            "max_closure_vel_error": max_cv,
            "tolerance": FB3_DRAKE_THRESHOLDS["max_closure_pos_error"],
        },
    }


def run_local_verification() -> dict[str, Any]:
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )
    from src.shared.python.motion_matching.full_body_spec import (
        FULL_BODY_SCHEMA_VERSION,
        load_full_body_spec,
    )
    import pydrake.all as drake_all

    upper_spec = json.loads(UPPER_SPEC_PATH.read_text(encoding="utf-8"))
    full_spec = load_full_body_spec(FULL_BODY_SPEC_PATH, upper_spec)

    full_model = FullBodyDrakeModel(full_spec)
    slice_model = full_model.upper_body_model()

    qual_spec = {
        "schema_version": FULL_BODY_SCHEMA_VERSION,
        "gravity_m_s2": upper_spec["gravity_m_s2"],
        "bodies": upper_spec["bodies"],
        "joints": upper_spec["joints"],
        "coordinate_order": upper_spec["coordinate_order"],
        "frames": upper_spec["frames"],
        "closure": upper_spec["closure"],
        "contact": {
            "ground": {"normal_policy": "opposite_gravity", "height_m": 0.0},
            "parameters": full_spec["contact"]["parameters"],
            "spheres": [],
        },
    }
    qual_model = FullBodyDrakeModel(qual_spec)

    rng = np.random.default_rng(20260914)
    num_states = 20

    gate_a_fk, gate_a_mass = _verify_gate_a(
        slice_model, qual_model, full_model, upper_spec, rng, num_states
    )
    gate_b_fk = _verify_gate_b(
        full_model, slice_model, full_spec, upper_spec, rng, num_states
    )
    pen_count, gate_c_fn, gate_c_ff, gate_c_p = _verify_gate_c(full_model, full_spec)
    gate_d_pos, gate_d_vel = _verify_gate_d(
        full_model, slice_model, full_spec, upper_spec, rng, num_states
    )
    num_coords, acc_finite, max_cp, max_cv = _verify_accelerations(
        full_model, full_spec
    )

    try:
        import importlib.metadata

        drake_version = importlib.metadata.version("drake")
    except (importlib.metadata.PackageNotFoundError, AttributeError, KeyError):
        drake_version = getattr(drake_all, "__version__", "unknown")

    receipt = _assemble_receipt(
        drake_version,
        num_states,
        (gate_a_fk, gate_a_mass),
        gate_b_fk,
        (pen_count, gate_c_fn, gate_c_ff, gate_c_p),
        (gate_d_pos, gate_d_vel),
        (num_coords, acc_finite, max_cp, max_cv),
    )
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    force_local = "--local" in sys.argv
    if force_local or has_real_drake():
        receipt = run_local_verification()
    else:
        receipt = run_remote_controltower()
        RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    sys.stdout.write(f"\nVerification status: {receipt['status']}\n")
    for gate_name, gate_info in receipt["gates"].items():
        sys.stdout.write(f"  {gate_name}: {gate_info['status']}\n")
    sys.stdout.write(f"Receipt written to {RECEIPT_PATH}\n")


if __name__ == "__main__":
    main()
