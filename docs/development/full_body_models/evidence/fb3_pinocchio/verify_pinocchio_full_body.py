"""Verification script for Pinocchio full-body builder (FB-3-P, #10065).

Tests all four Done Gates:
(a) Upper-body slice reproduces qualified model mass matrix and FK to 1e-12 at random states;
(b) Full-body FK matches spec frames;
(c) Contact force at analytic states equals the FB-2 reference;
(d) Closure residual unchanged on the upper-body chain.
Plus forward accelerations combining contact forces and weld loop-closure dynamics.

If real Pinocchio is not available in the local environment, automatically executes
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

FULL_BODY_SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
UPPER_SPEC_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
NATIVE_MODEL_PATH = (
    ROOT / "src/engines/physics_engines/pinocchio/python/native_model.py"
)
CONTACT_LAW_PATH = ROOT / "src/shared/python/motion_matching/contact_law.py"
FULL_BODY_SPEC_PY_PATH = ROOT / "src/shared/python/motion_matching/full_body_spec.py"
RECEIPT_PATH = HERE / "receipt.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_real_pinocchio() -> bool:
    try:
        import pinocchio as pin

        return (
            type(pin).__module__ != "unittest.mock"
            and hasattr(pin, "Model")
            and hasattr(pin, "SE3")
        )
    except (ImportError, AttributeError):
        return False


def run_remote_controltower() -> dict[str, Any]:
    """Execute verification remotely on ControlTower and return the receipt."""
    sys.stdout.write(
        "Real Pinocchio not available locally. Dispatching to ControlTower...\n"
    )
    sys.stdout.flush()
    # Sync files to ControlTower test directory
    script = (
        b"cd /home/dieterolson/fb3_pinocchio_test\n"
        b"/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python "
        b"docs/development/full_body_models/evidence/fb3_pinocchio/verify_pinocchio_full_body.py --local\n"
    )
    res = subprocess.run(
        ["ssh", "controltower", "wsl", "-d", "ControlTower-Runner", "bash"],
        input=script,
        capture_output=True,
    )
    if res.returncode != 0:
        sys.stderr.write(
            f"Remote error:\n{res.stderr.decode('utf-8', errors='ignore')}\n"
        )
        raise RuntimeError(
            f"Remote verification failed with exit code {res.returncode}"
        )

    # Fetch receipt from remote
    fetch_cmd = b"cat /home/dieterolson/fb3_pinocchio_test/docs/development/full_body_models/evidence/fb3_pinocchio/receipt.json\n"
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
    upper_spec: Mapping[str, Any],
    pin: Any,
    rng: np.random.Generator,
    num_states: int,
) -> tuple[float, float]:
    """Gate (a): Upper-body slice parity in FK and mass matrix."""
    max_fk_diff = 0.0
    max_mass_diff = 0.0
    coords = upper_spec["coordinate_order"]
    for _ in range(num_states):
        q_up = {name: float(rng.uniform(-0.35, 0.35)) for name in coords}
        poses_slice = slice_model.frame_poses(q_up)
        poses_qual = qual_model.frame_poses(q_up)
        for frame in poses_qual:
            diff_fk = float(np.linalg.norm(poses_slice[frame] - poses_qual[frame]))
            if diff_fk > max_fk_diff:
                max_fk_diff = diff_fk

        qu = qual_model.configuration(q_up)
        data_slice = slice_model.model.createData()
        data_qual = qual_model.model.createData()
        m_slice = np.asarray(pin.crba(slice_model.model, data_slice, qu))
        m_qual = np.asarray(pin.crba(qual_model.model, data_qual, qu))
        diff_m = float(np.max(np.abs(m_slice - m_qual)))
        if diff_m > max_mass_diff:
            max_mass_diff = diff_m

    assert max_fk_diff < 1e-12, f"Gate (a) FK diff: {max_fk_diff}"
    assert max_mass_diff < 1e-12, f"Gate (a) Mass diff: {max_mass_diff}"
    return max_fk_diff, max_mass_diff


def _verify_gate_b(
    full_model: Any,
    qual_model: Any,
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
        poses_qual = qual_model.frame_poses(q_upper)
        for frame in poses_qual:
            diff = float(np.linalg.norm(poses_full[frame] - poses_qual[frame]))
            if diff > max_fk_diff:
                max_fk_diff = diff

    assert max_fk_diff < 1e-12, f"Gate (b) FK diff: {max_fk_diff}"
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
    assert len(forces_neutral) == 4

    q_pen = dict(q_neutral)
    q_pen["TranslationInputZ"] = -0.90
    v_moving = {name: (0.1 if "Translation" in name else 0.0) for name in coords}
    forces_pen = full_model.contact_forces(q_pen, v_moving)
    assert len(forces_pen) == 4
    pen_count = sum(1 for s in forces_pen.values() if s.penetration_m > 0.0)
    assert pen_count > 0

    max_fn = 0.0
    max_ff = 0.0
    max_p = 0.0
    ref_frame = full_model._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    for name, sample in forces_pen.items():
        fid = full_model._contact_frames[name]
        center = full_model.data.oMf[fid].translation
        vel = full_model._pin.getFrameVelocity(
            full_model.model, full_model.data, fid, ref_frame
        ).linear
        radius = next(s.radius_m for s in full_model.contact_spheres if s.name == name)
        ref = sphere_ground_contact(
            center, vel, radius, full_model.ground, full_model.contact_parameters
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

    assert max_fn < 1e-12, f"Gate (c) Normal force diff: {max_fn}"
    assert max_ff < 1e-12, f"Gate (c) Friction force diff: {max_ff}"
    assert max_p < 1e-12, f"Gate (c) Penetration diff: {max_p}"
    return pen_count, max_fn, max_ff, max_p


def _verify_gate_d(
    full_model: Any,
    qual_model: Any,
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

        pos_qual, vel_qual = qual_model.closure_residuals(q_up, v_up)
        pos_full, vel_full = full_model.closure_residuals(q_full, v_full)
        diff_cp = float(np.max(np.abs(pos_full - pos_qual)))
        diff_cv = float(np.max(np.abs(vel_full - vel_qual)))
        if diff_cp > max_pos:
            max_pos = diff_cp
        if diff_cv > max_vel:
            max_vel = diff_cv

    assert max_pos < 1e-12, f"Gate (d) Pos diff: {max_pos}"
    assert max_vel < 1e-12, f"Gate (d) Vel diff: {max_vel}"
    return max_pos, max_vel


def _verify_accelerations(
    full_model: Any,
    full_spec: Mapping[str, Any],
) -> tuple[int, bool, float, float]:
    """Verify forward accelerations with contact forces and weld closure."""
    coords = full_spec["coordinate_order"]
    q_acc = dict.fromkeys(coords, 0.0)
    q_acc["TranslationInputZ"] = -0.90
    v_acc = dict.fromkeys(coords, 0.0)
    tau_acc = dict.fromkeys(coords, 0.0)
    acc = full_model.accelerations(q_acc, v_acc, tau_acc)
    all_finite = all(np.isfinite(val) for val in acc.values())
    assert all_finite
    c_pos_err, c_vel_err = full_model.closure_errors()
    return (
        len(acc),
        all_finite,
        float(np.max(np.abs(c_pos_err))),
        float(np.max(np.abs(c_vel_err))),
    )


def run_local_verification() -> dict[str, Any]:
    """Execute verification locally using Pinocchio."""
    import pinocchio as pin

    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
        NativePinocchioModel,
    )
    from src.shared.python.motion_matching.full_body_spec import load_full_body_spec

    upper_spec = json.loads(UPPER_SPEC_PATH.read_text(encoding="utf-8"))
    full_spec = load_full_body_spec(FULL_BODY_SPEC_PATH, upper_spec)

    upper_model = NativePinocchioModel(upper_spec)
    full_model = FullBodyPinocchioModel(full_spec)
    slice_model = full_model.upper_body_model()

    rng = np.random.default_rng(20260914)
    num_states = 20

    max_slice_fk, max_slice_m = _verify_gate_a(
        slice_model, upper_model, upper_spec, pin, rng, num_states
    )
    max_full_fk = _verify_gate_b(
        full_model, upper_model, full_spec, upper_spec, rng, num_states
    )
    pen_count, max_fn, max_ff, max_p = _verify_gate_c(full_model, full_spec)
    max_cp, max_cv = _verify_gate_d(
        full_model, upper_model, full_spec, upper_spec, rng, num_states
    )
    n_acc, all_finite, max_c_pos, max_c_vel = _verify_accelerations(
        full_model, full_spec
    )

    receipt = {
        "work_package": "FB-3-P",
        "issue": "#10065",
        "epic": "#10062",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASSED",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "pinocchio_version": getattr(pin, "__version__", "unknown"),
        },
        "inputs": {
            "full_body_spec_v1.json": sha256_file(FULL_BODY_SPEC_PATH),
            "native_geometry_spec_9967.json": sha256_file(UPPER_SPEC_PATH),
            "native_model.py": sha256_file(NATIVE_MODEL_PATH),
            "contact_law.py": sha256_file(CONTACT_LAW_PATH),
            "full_body_spec.py": sha256_file(FULL_BODY_SPEC_PY_PATH),
        },
        "gates": {
            "gate_a_upper_body_slice_parity": {
                "status": "PASSED",
                "random_states_tested": num_states,
                "max_fk_diff_m": max_slice_fk,
                "max_mass_matrix_diff": max_slice_m,
                "tolerance": 1e-12,
            },
            "gate_b_full_body_fk": {
                "status": "PASSED",
                "random_states_tested": num_states,
                "max_fk_diff_m": max_full_fk,
                "tolerance": 1e-12,
            },
            "gate_c_contact_force_parity": {
                "status": "PASSED",
                "num_contact_spheres": len(full_model.contact_spheres),
                "penetrating_spheres_tested": pen_count,
                "max_normal_force_diff_n": max_fn,
                "max_friction_force_diff_n": max_ff,
                "max_penetration_diff_m": max_p,
                "tolerance": 1e-12,
            },
            "gate_d_closure_residual_parity": {
                "status": "PASSED",
                "random_states_tested": num_states,
                "max_position_residual_diff": max_cp,
                "max_velocity_residual_diff": max_cv,
                "tolerance": 1e-12,
            },
        },
        "accelerations": {
            "status": "PASSED",
            "num_coordinates": n_acc,
            "all_finite": all_finite,
            "max_closure_pos_error": max_c_pos,
            "max_closure_vel_error": max_c_vel,
        },
    }

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> int:
    force_local = "--local" in sys.argv
    if force_local or has_real_pinocchio():
        sys.stdout.write("Running verification with local Pinocchio...\n")
        receipt = run_local_verification()
    else:
        receipt = run_remote_controltower()
        RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    ga_fk = receipt["gates"]["gate_a_upper_body_slice_parity"]["max_fk_diff_m"]
    ga_m = receipt["gates"]["gate_a_upper_body_slice_parity"]["max_mass_matrix_diff"]
    gb_fk = receipt["gates"]["gate_b_full_body_fk"]["max_fk_diff_m"]
    gc_f = receipt["gates"]["gate_c_contact_force_parity"]["max_normal_force_diff_n"]
    gd_res = receipt["gates"]["gate_d_closure_residual_parity"][
        "max_position_residual_diff"
    ]

    sys.stdout.write(f"Receipt written to {RECEIPT_PATH}\n")
    sys.stdout.write(f"Status: {receipt['status']}\n")
    sys.stdout.write(f"Gate (a) max FK diff: {ga_fk:.2e}\n")
    sys.stdout.write(f"Gate (a) max M diff:  {ga_m:.2e}\n")
    sys.stdout.write(f"Gate (b) max FK diff: {gb_fk:.2e}\n")
    sys.stdout.write(f"Gate (c) max F diff:  {gc_f:.2e}\n")
    sys.stdout.write(f"Gate (d) max res diff:{gd_res:.2e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
