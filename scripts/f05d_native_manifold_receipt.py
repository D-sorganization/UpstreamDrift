"""Freeze predeclared F05d native manifold BoxFDDP/SciPy evidence.

Generate only from the complete supported-provider native test module with
``F05D_BENCHMARK_RECEIPT=1`` and ``-o junit_family=legacy``. Missing,
skipped, failed or partial trials fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

from defusedxml import ElementTree

import crocoddyl
import mujoco
import numpy
import pinocchio
import scipy

from scripts.f05b_native_paired_receipt import _git

_STARTS = ("0.4", "0.7")
_COMPANIONS = (
    "test_state_uses_native_quaternion_tangent_and_sign_equivalence",
    "test_action_uses_native_discrete_transition_and_tangent_derivatives",
    "test_solver_rejects_stale_observation_and_bad_clock",
    "test_box_fddp_candidate_improves_exact_nonlinear_native_rollout",
    "test_native_multidof_control_exports_only_applied_torque_and_replays",
    "test_native_controller_rejects_assistance",
    "test_execution_model_actuator_order_must_match_controller",
    "test_hidden_native_callback_is_rejected",
    "test_candidate_outside_native_motor_bounds_falls_back",
    "test_loaded_native_model_mutation_is_rejected",
)
_NUMERIC = (
    "total_wall_s",
    "run_wall_s",
    "control_wall_s",
    "accepted_steps",
    "hinge_rmse_rad",
    "effort_sum_n2m2",
    "realized_native_cost",
    "call_p95_s",
)
_SOURCE_FILES = (
    "src/engines/physics_engines/mujoco/python/box_fddp_tracking.py",
    "src/engines/physics_engines/mujoco/python/native_manifold_box_fddp.py",
    "src/engines/physics_engines/mujoco/python/native_tangent_derivative.py",
    "src/engines/physics_engines/mujoco/python/native_nmpc_tracking.py",
    "src/engines/physics_engines/mujoco/python/native_torque_replay.py",
    "tests/unit/motion_matching/test_native_manifold_box_fddp.py",
    "tests/unit/motion_matching/test_f05d_native_receipt.py",
    "scripts/f05d_native_manifold_receipt.py",
)


def _properties(case: Any) -> dict[str, str]:
    return {
        str(item.get("name")): str(item.get("value"))
        for item in case.findall("./properties/property")
    }


def _cases(junit: Path) -> tuple[list[Any], str]:
    raw = junit.read_bytes()
    if len(raw) > 2_000_000 or b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("benchmark XML exceeds limit or declares entities")
    root = ElementTree.fromstring(raw)
    cases = root.findall(".//testcase")
    if any(
        case.find(tag) is not None
        for case in cases
        for tag in ("failure", "error", "skipped")
    ):
        raise ValueError("a native benchmark or companion test failed or skipped")
    names = [case.get("name") for case in cases]
    expected = [*_COMPANIONS]
    expected.extend(
        f"test_matched_native_box_and_scipy_runs_keep_frozen_replay[{start}]"
        for start in _STARTS
    )
    if sorted(names) != sorted(expected):
        raise ValueError("native benchmark test inventory is incomplete")
    return cases, hashlib.sha256(raw).hexdigest()


def _trial(case: Any, start: str) -> dict[str, object]:
    properties = _properties(case)
    required = (
        "common_bootstrap_s",
        "execution_order",
        "source_model_sha256",
        "loaded_model_sha256",
        "initial_state_sha256",
        "policy_sha256",
        "time_grid_sha256",
        "state_schema_sha256",
        "input_channel_schema_sha256",
    )
    for name in required:
        if name not in properties:
            raise ValueError(f"native trial {start} lacks {name}")
    bootstrap_s = float(properties["common_bootstrap_s"])
    row: dict[str, object] = {
        "initial_hip_rad": float(start),
        "common_bootstrap_s": bootstrap_s,
        "execution_order": json.loads(properties["execution_order"]),
        "model_sha256": properties["source_model_sha256"],
        "loaded_model_sha256": properties["loaded_model_sha256"],
        "initial_state_sha256": properties["initial_state_sha256"],
        "policy_sha256": properties["policy_sha256"],
        "time_grid_sha256": properties["time_grid_sha256"],
        "state_schema_sha256": properties["state_schema_sha256"],
        "input_channel_schema_sha256": properties["input_channel_schema_sha256"],
    }
    if row["execution_order"] not in (["box", "scipy"], ["scipy", "box"]):
        raise ValueError("native paired solver order invalid")
    if not math.isfinite(bootstrap_s) or bootstrap_s < 0:
        raise ValueError("native common bootstrap time invalid")
    for method in ("box", "scipy"):
        numeric_values: dict[str, float] = {}
        for suffix in _NUMERIC:
            key = f"{method}_{suffix}"
            if key not in properties:
                raise ValueError(f"native trial {start} lacks {key}")
            numeric_values[suffix] = float(properties[key])
            if not math.isfinite(numeric_values[suffix]) or numeric_values[suffix] < 0:
                raise ValueError(f"native trial {start} has invalid {key}")
        values: dict[str, object] = dict(numeric_values)
        statuses = json.loads(properties[f"{method}_statuses"])
        if len(statuses) != 4 or numeric_values["accepted_steps"] != sum(
            status == "optimized" for status in statuses
        ):
            raise ValueError("native accepted-step count differs from statuses")
        if (
            numeric_values["total_wall_s"] < numeric_values["run_wall_s"]
            or numeric_values["run_wall_s"] < numeric_values["control_wall_s"]
        ):
            raise ValueError("native timing components are inconsistent")
        values["statuses"] = statuses
        values["solver_id"] = properties[f"{method}_solver_id"]
        values["applied_input_sha256"] = properties[f"{method}_applied_sha256"]
        values["final_qpos"] = json.loads(properties[f"{method}_qpos_final"])
        values["time_to_fully_accepted_s"] = (
            numeric_values["total_wall_s"]
            if numeric_values["accepted_steps"] == 4
            else None
        )
        row[method] = values
    return row


def build_receipt(junit: Path, repo: Path, host_alias: str) -> dict[str, object]:
    """Bind the complete test inventory, runtime, source and exact replay IDs."""
    if not host_alias or any(char in host_alias for char in ("/", "\\", ":")):
        raise ValueError("host alias must be a public non-path label")
    cases, junit_sha256 = _cases(junit)
    trials = [
        _trial(
            next(case for case in cases if case.get("name", "").endswith(f"[{start}]")),
            start,
        )
        for start in _STARTS
    ]
    return {
        "schema_version": "upstreamdrift/f05d-manifold-boxfddp-development/1.0.0",
        "issue": 11932,
        "parent_issue": 11789,
        "selection": "retain_as_contact_free_candidate_keep_F02_TVLQR_broad_default",
        "scope": {
            "plant": "native_mujoco_floating_root_two_hinges_contact_free",
            "nq": 9,
            "nv": 8,
            "nx": 17,
            "ndx": 16,
            "nu": 2,
            "native_integrator": "Euler",
            "native_step_s": 0.01,
            "reference": (
                "native_generated_full_state_from_fixed_admissible_teacher_motor_torque"
            ),
            "executed_steps": 4,
            "horizon_steps": 4,
            "max_iterations_each": 15,
            "cooperative_wall_budget_s_each_call": 0.2,
            "prediction_and_admission_objective": (
                "same_nonlinear_native_stage_and_terminal_manifold_error_plus_effort"
            ),
            "input": "post_limit_direct_unit_motor_torque_ZOH",
            "observation": "exact_simulated_qpos_qvel_each_step",
            "replay": "independent_complete_mjSTATE_INTEGRATION_Tools_T01",
            "replay_tolerance": 1e-12,
            "timing": (
                "Common T01 preflight charged to both; each run wall includes solver "
                "setup, every accepted or fallback call, native execution, frozen "
                "input export and independent replay. Process start/pytest collection "
                "excluded. A timeout is reported after the cooperative call returns."
            ),
        },
        "runtime": {
            "host_alias": host_alias,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "mujoco": mujoco.__version__,
            "crocoddyl": crocoddyl.__version__,
            "pinocchio": getattr(pinocchio, "__version__", "unknown"),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "provenance": {
            "git_head_before_receipt": _git(repo, "rev-parse", "HEAD"),
            "tools_pin": _git(repo, "rev-parse", "HEAD:vendor/ud-tools"),
            "junit_sha256": junit_sha256,
            "source_sha256": {
                name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                for name in _SOURCE_FILES
            },
        },
        "trials": trials,
        "limits": (
            "Contact-free two-hinge software fixture only; neither hard real-time "
            "nor full-body golfer, grip/ground, muscle, capture, six-engine or "
            "full-swing qualification. F05 parent remains open."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    receipt = build_receipt(args.junit, repo, args.host_alias)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
