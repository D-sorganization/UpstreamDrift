"""Freeze the F05 native paired-test metrics with source and hardware identity.

The test emits per-case JUnit properties only with F05_BENCHMARK_RECEIPT=1 and
``-o junit_family=legacy``. This script refuses missing or failed trial rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess  # nosec B404 - fixed, local Git arguments; no shell
import sys
from xml.etree import ElementTree

import mujoco
import numpy
import scipy


_TRIAL_STARTS = ("0.4", "0.7")
_NUMERIC = (
    "nmpc_q_rmse_rad",
    "nmpc_final_q_rad",
    "nmpc_effort_l1_nm",
    "nmpc_latency_p50_s",
    "nmpc_latency_p95_s",
    "nmpc_latency_worst_s",
    "nmpc_evaluations_total",
    "nmpc_optimized_steps",
    "tvlqr_q_rmse_rad",
    "tvlqr_final_q_rad",
    "tvlqr_effort_l1_nm",
    "tvlqr_latency_p50_s",
    "tvlqr_latency_p95_s",
    "tvlqr_latency_worst_s",
)
_TEXT = (
    "nmpc_statuses",
    "nmpc_input_sha256",
    "nmpc_initial_state_sha256",
    "nmpc_policy_sha256",
    "tvlqr_input_sha256",
    "native_model_sha256",
)
_SOURCE_FILES = (
    "src/shared/python/motion_matching/bounded_nmpc.py",
    "src/engines/physics_engines/mujoco/python/native_nmpc_tracking.py",
    "tests/unit/motion_matching/test_bounded_nmpc_native.py",
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(  # nosec B603 - fixed executable, local arguments
        ("git", *args),
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return result.stdout.strip()


def _trial_rows(junit_path: Path) -> dict[str, dict[str, object]]:
    xml = ElementTree.parse(junit_path).getroot()  # nosec B314 - local test file
    rows: dict[str, dict[str, object]] = {}
    cases = xml.findall(".//testcase")
    if len(cases) != len(_TRIAL_STARTS):
        raise ValueError("receipt requires exactly two predeclared native trials")
    for start in _TRIAL_STARTS:
        matching = [
            case for case in cases if case.get("name", "").endswith(f"[{start}]")
        ]
        if len(matching) != 1 or matching[0].find("failure") is not None:
            raise ValueError(f"native trial {start} missing or failed")
        properties = {
            item.get("name"): item.get("value")
            for item in matching[0].findall("./properties/property")
        }
        if any(properties.get(key) is None for key in (*_NUMERIC, *_TEXT)):
            raise ValueError(f"native trial {start} lacks required metric/identity")
        trial: dict[str, object] = {key: float(properties[key]) for key in _NUMERIC}
        trial.update({key: str(properties[key]) for key in _TEXT})
        trial["initial_q_rad"] = float(start)
        trial["q_rmse_reduction_pct"] = (
            100.0
            * (float(trial["tvlqr_q_rmse_rad"]) - float(trial["nmpc_q_rmse_rad"]))
            / float(trial["tvlqr_q_rmse_rad"])
        )
        trial["latency_p50_ratio"] = float(trial["nmpc_latency_p50_s"]) / float(
            trial["tvlqr_latency_p50_s"]
        )
        rows[start] = trial
    return rows


def build_receipt(junit_path: Path, repo: Path, host_alias: str) -> dict[str, object]:
    """Return an anonymous, deterministic schema for the local measured run."""
    if not host_alias or any(char in host_alias for char in ("/", "\\", ":")):
        raise ValueError("host alias must be a non-path public label")
    rows = _trial_rows(junit_path)
    tools_pin = _git(repo, "rev-parse", "HEAD:vendor/ud-tools")
    source_hashes = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }
    return {
        "schema_version": "upstreamdrift/f05-native-paired-development/1.0.0",
        "issue": 11789,
        "selection": "retain_tvlqr_reject_scipy_shooting_nmpc_for_native_step_runtime",
        "scope": {
            "plant": "native_mujoco_one_hinge_contact_free_unit_motor",
            "prediction_mass_kg": 0.3,
            "execution_mass_kg": 0.42,
            "mass_perturbation_pct": 40.0,
            "integration_step_s": 0.01,
            "executed_steps": 8,
            "nmpc_horizon_steps": 4,
            "nmpc_max_evaluations_per_step": 500,
            "nmpc_cooperative_wall_budget_s": 1.0,
            "input_unit": "N*m",
            "observation_policy": "exact_simulated_state",
            "replay": "independent_frozen_zoh_native_torque",
            "replay_q_tolerance_rad": 1e-12,
            "contact_perturbation": "unsupported_in_this_fixture",
        },
        "hardware": {
            "alias": host_alias,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unreported",
            "python": sys.version.split()[0],
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "provenance": {
            "git_head_before_receipt": _git(repo, "rev-parse", "HEAD"),
            "tools_pin": tools_pin,
            "source_sha256": source_hashes,
        },
        "trials": [rows[start] for start in _TRIAL_STARTS],
        "limits": (
            "Measured controller-call latency is not a hard deadline; this "
            "cooperative SciPy solver is rejected for native 10 ms stepping. "
            "The small no-contact fixture does not qualify contact, capture, "
            "muscles, six-engine parity or full-swing tracking."
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
