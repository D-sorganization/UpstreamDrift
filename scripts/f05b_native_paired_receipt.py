"""Freeze the F05b native BoxFDDP-versus-shooting paired run.

The native test emits properties only with ``F05B_BENCHMARK_RECEIPT=1`` and
``-o junit_family=legacy``. Failed, missing or partial cases are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

# The only subprocess is a fixed local Git query without a shell.
import subprocess  # nosec B404
import sys

# Only bounded local pytest XML is admitted; DTD/entity declarations reject.
from xml.etree import ElementTree  # nosec B405

import crocoddyl
import mujoco
import numpy
import pinocchio
import scipy

_STARTS = ("0.4", "0.7")
_NUMERIC = (
    "box_q_rmse_rad",
    "box_effort_l1_nm",
    "box_optimized_steps",
    "box_latency_p50_s",
    "box_latency_p95_s",
    "box_latency_worst_s",
    "box_setup_s",
    "common_contract_import_s",
    "box_controlled_export_replay_s",
    "box_iterations_total",
    "scipy_q_rmse_rad",
    "scipy_optimized_steps",
    "scipy_latency_p50_s",
    "scipy_latency_p95_s",
    "scipy_latency_worst_s",
    "scipy_controlled_export_replay_s",
    "scipy_evaluations_total",
)
_TEXT = (
    "box_statuses",
    "box_input_sha256",
    "box_initial_state_sha256",
    "box_policy_sha256",
    "box_model_sha256",
    "scipy_statuses",
    "scipy_input_sha256",
)
_SOURCE_FILES = (
    "src/engines/physics_engines/mujoco/python/box_fddp_tracking.py",
    "src/engines/physics_engines/mujoco/python/native_box_fddp_hinge.py",
    "src/engines/physics_engines/mujoco/python/native_nmpc_tracking.py",
    "src/engines/physics_engines/mujoco/python/native_torque_replay.py",
    "src/shared/python/motion_matching/bounded_nmpc.py",
    "tests/unit/motion_matching/test_box_fddp_hinge.py",
    "scripts/f05b_native_paired_receipt.py",
)


def _git(repo: Path, *args: str) -> str:
    # The checkout's .git worktree pointer is a Windows absolute path; WSL Git
    # cannot resolve it, while Windows Git accepts the same mounted cwd.
    pointer = repo / ".git"
    gitdir = pointer.read_text(encoding="utf-8") if pointer.is_file() else ""
    windows_pointer = gitdir.startswith("gitdir: ") and gitdir[9:10] == ":"
    executable = "git.exe" if windows_pointer else "git"
    # The executable is selected from two fixed local Git names; no shell.
    result = subprocess.run(  # nosec B603
        (executable, *args),
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return result.stdout.strip()


def _trials(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    if len(raw) > 2_000_000 or b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("benchmark XML exceeds limit or declares entities")
    root = ElementTree.fromstring(raw)  # nosec B314
    cases = root.findall(".//testcase")
    if any(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    ):
        raise ValueError("a benchmark or companion test failed")
    rows: list[dict[str, object]] = []
    for start in _STARTS:
        matches = [
            case
            for case in cases
            if case.get("name", "").startswith("test_box_fddp_native_tracking_replays_")
            and case.get("name", "").endswith(f"[{start}]")
        ]
        if len(matches) != 1:
            raise ValueError(f"exactly one native trial {start} is required")
        properties = {
            item.get("name"): item.get("value")
            for item in matches[0].findall("./properties/property")
        }
        if any(properties.get(name) is None for name in (*_NUMERIC, *_TEXT)):
            raise ValueError(f"native trial {start} has incomplete evidence")
        numeric = {name: float(str(properties[name])) for name in _NUMERIC}
        row: dict[str, object] = dict(numeric)
        row.update({name: str(properties[name]) for name in _TEXT})
        row["initial_q_rad"] = float(start)
        common = numeric["common_contract_import_s"]
        row["box_time_to_accepted_s"] = (
            common + numeric["box_setup_s"] + numeric["box_controlled_export_replay_s"]
        )
        row["scipy_time_to_accepted_s"] = (
            common + numeric["scipy_controlled_export_replay_s"]
        )
        rows.append(row)
    return rows


def build_receipt(junit: Path, repo: Path, host_alias: str) -> dict[str, object]:
    """Bind all predeclared trials, source, runtime and replay identities."""
    if not host_alias or any(char in host_alias for char in ("/", "\\", ":")):
        raise ValueError("host alias must be a public non-path label")
    return {
        "schema_version": "upstreamdrift/f05b-boxfddp-native-development/1.0.0",
        "issue": 11910,
        "parent_issue": 11789,
        "selection": "keep_boxfddp_as_one_hinge_candidate_retain_tvlqr_for_broad_scope",
        "scope": {
            "plant": "native_mujoco_one_hinge_contact_free_direct_unit_motor",
            "nominal_mass_kg": 0.3,
            "execution_and_second_scenario_mass_kg": 0.42,
            "native_integrator": "RK4",
            "native_step_s": 0.01,
            "executed_steps": 8,
            "horizon_steps": 4,
            "box_max_iterations": 20,
            "box_cooperative_wall_budget_s": 0.01,
            "scipy_max_evaluations": 500,
            "scipy_cooperative_wall_budget_s": 1.0,
            "prediction_objective": "box_scenario_mean_vs_scipy_scenario_max",
            "acceptance_objective": "same_max_scenario_native_linear_model_score",
            "observation": "exact_simulated_state",
            "execution_and_replay": "full_initial_state_exact_zoh_native_torque",
            "native_replay_tolerance": 1e-12,
            "timing_definition": (
                "Common T01 bootstrap charged to both; Box also charges derivative "
                "setup. Controlled/export/replay time includes every failed or "
                "fallback call. No process launch or pytest collection charged."
            ),
        },
        "runtime": {
            "host_alias": host_alias,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "mujoco": mujoco.__version__,
            "crocoddyl": getattr(crocoddyl, "__version__", "unreported"),
            "pinocchio": getattr(pinocchio, "__version__", "unreported"),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "provenance": {
            "git_head_before_receipt": _git(repo, "rev-parse", "HEAD"),
            "tools_pin": _git(repo, "rev-parse", "HEAD:vendor/ud-tools"),
            "source_sha256": {
                name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                for name in _SOURCE_FILES
            },
        },
        "trials": _trials(junit),
        "limits": (
            "The result is a local one-hinge software candidate, not a hard "
            "real-time bound, full-swing mocap fit, contact or muscle replay, "
            "manifold provider, or six-engine qualification."
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
