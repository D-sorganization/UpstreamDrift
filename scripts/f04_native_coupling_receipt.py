"""Freeze native F04 train/holdout and paired-intervention evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, cast

import numpy as np
from defusedxml import ElementTree as ET

from scripts.f02_native_manifold_receipt import _passed_property

_TUNE = "test_native_tuning_uses_train_only_and_independent_holdout"
_INTERVENE = (
    "test_paired_native_motor_intervention_is_distinct_from_residual_covariance"
)
_IDENTITY_HASHES = (
    "initial_state_sha256",
    "policy_sha256",
    "time_grid_sha256",
)
_SOURCE = (
    "scripts/f04_native_coupling_receipt.py",
    "src/engines/physics_engines/mujoco/python/native_control_coupling.py",
    "src/engines/physics_engines/mujoco/python/native_distributed_feedback.py",
    "src/engines/physics_engines/mujoco/python/native_torque_replay.py",
    "src/shared/python/motion_matching/control_loop_tuning.py",
    "tests/unit/motion_matching/test_native_control_coupling.py",
    "tests/unit/motion_matching/test_f04_native_coupling_receipt.py",
)


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"native {label} must be SHA-256")


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"native {label} must be finite") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"native {label} must be finite with shape {shape}")
    return array


def _tuning(evidence: dict[str, object]) -> None:
    expected = {
        "source_model_sha256",
        "teacher_input_sha256",
        *_IDENTITY_HASHES,
        "initial_objective",
        "final_objective",
        "parameters",
        "holdout_phase_group_losses",
        "holdout_initial_rmse_rad",
        "holdout_final_rmse_rad",
        "cross_jacobian_norms",
        "singular_values",
        "rank_deficient",
        "uncertainty_status",
        "train_evaluations",
        "holdout_evaluations",
        "distinct_applied_inputs",
        "checkpoint_reasons",
        "max_full_state_replay_error",
        "total_wall_s",
        "total_cpu_s",
    }
    if set(evidence) != expected:
        raise ValueError("native tuning evidence fields incomplete or unknown")
    for label in ("source_model_sha256", "teacher_input_sha256", *_IDENTITY_HASHES):
        _digest(evidence[label], label)
    for label in (
        "initial_objective",
        "final_objective",
        "holdout_initial_rmse_rad",
        "holdout_final_rmse_rad",
        "max_full_state_replay_error",
        "total_wall_s",
        "total_cpu_s",
    ):
        value = evidence[label]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(f"native {label} must be finite")
    if cast(float, evidence["final_objective"]) >= cast(
        float, evidence["initial_objective"]
    ):
        raise ValueError("native tuning improvement is absent")
    if (
        cast(float, evidence["holdout_final_rmse_rad"])
        >= cast(float, evidence["holdout_initial_rmse_rad"])
        or cast(float, evidence["holdout_initial_rmse_rad"]) <= 0
    ):
        raise ValueError("native held-out tracking improvement is absent")
    if cast(float, evidence["max_full_state_replay_error"]) > 1e-12:
        raise ValueError("native tuning full-state replay failed")
    if (
        min(cast(float, evidence["total_wall_s"]), cast(float, evidence["total_cpu_s"]))
        <= 0
    ):
        raise ValueError("native tuning timing invalid")
    for label, shape in (
        ("parameters", (2,)),
        ("holdout_phase_group_losses", (2, 2)),
        ("cross_jacobian_norms", (2, 2)),
        ("singular_values", (2,)),
    ):
        values = _finite_array(evidence[label], shape, label)
        if label == "parameters":
            if np.any((values < 0) | (values > 2)):
                raise ValueError("native tuned gain scales exceed declared bounds")
        elif np.any(values < 0):
            raise ValueError(f"native {label} must be nonnegative")
    if type(evidence["rank_deficient"]) is not bool:
        raise ValueError("native local rank status must be boolean")
    if (
        evidence["uncertainty_status"]
        != "unavailable_no_resampling_or_capture_noise_model"
    ):
        raise ValueError("native uncertainty is unavailable in this fixture")
    for label in (
        "train_evaluations",
        "holdout_evaluations",
        "distinct_applied_inputs",
    ):
        if type(evidence[label]) is not int or cast(int, evidence[label]) < 1:
            raise ValueError(f"native {label} count invalid")
    if cast(int, evidence["train_evaluations"]) <= cast(
        int, evidence["holdout_evaluations"]
    ):
        raise ValueError("native training/holdout evaluation split invalid")
    reasons = evidence["checkpoint_reasons"]
    if not isinstance(reasons, list) or "accepted" not in reasons:
        raise ValueError("native tuning has no accepted block checkpoint")


def _intervention(evidence: dict[str, object], tuning: dict[str, object]) -> None:
    expected = {
        "response_rad_per_nm",
        "channel_ids",
        "response_joint_ids",
        "applied_input_sha256",
        *_IDENTITY_HASHES,
        "interpretation",
        "intervention_wall_s",
    }
    if set(evidence) != expected:
        raise ValueError("native intervention evidence fields incomplete or unknown")
    for label in _IDENTITY_HASHES:
        _digest(evidence[label], label)
        if evidence[label] != tuning[label]:
            raise ValueError(f"native intervention {label} differs from tuning policy")
    if evidence["channel_ids"] != ["hip_torque", "knee_torque"] or evidence[
        "response_joint_ids"
    ] != ["hip", "knee"]:
        raise ValueError("native intervention channel/joint order invalid")
    hashes = evidence["applied_input_sha256"]
    if not isinstance(hashes, list) or len(hashes) != 4:
        raise ValueError("native intervention needs four input identities")
    for value in hashes:
        _digest(value, "intervention input")
    if len(set(hashes)) != 4:
        raise ValueError("native intervention input histories must differ")
    response = _finite_array(evidence["response_rad_per_nm"], (2, 2), "response")
    if min(abs(response[0][1]), abs(response[1][0])) <= 1e-6:
        raise ValueError("native cross-motor response was not observed")
    if (
        evidence["interpretation"]
        != "synthetic_plant_input_intervention_not_human_control"
    ):
        raise ValueError("native intervention interpretation invalid")
    wall = evidence["intervention_wall_s"]
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(wall)
        or wall <= 0
    ):
        raise ValueError("native intervention timing invalid")


def extract_native_evidence(junit: Path) -> dict[str, dict[str, object]]:
    """Require both supported native tests and consistent policy identities."""
    raw = junit.read_bytes()
    if len(raw) > 2_000_000 or b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("native JUnit XML exceeds limit or declares entities")
    cases = ET.fromstring(raw).findall(".//testcase")
    if sorted(case.get("name") for case in cases) != sorted((_TUNE, _INTERVENE)):
        raise ValueError("native F04 test inventory incomplete")
    if any(
        case.find(outcome) is not None
        for case in cases
        for outcome in ("failure", "error", "skipped")
    ):
        raise ValueError("native F04 receipt requires passed tests")
    tuning = _passed_property(junit, _TUNE, "f04_native_tuning_evidence")
    intervention = _passed_property(
        junit, _INTERVENE, "f04_native_intervention_evidence"
    )
    _tuning(tuning)
    _intervention(intervention, tuning)
    return {"tuning": tuning, "intervention": intervention}


def build_receipt(junit: Path, *, host_alias: str) -> dict[str, Any]:
    """Bind an admitted native JUnit result to provider and source identities."""
    import mujoco
    import numpy
    import scipy

    if not host_alias or mujoco.__version__ != "3.8.0":
        raise ValueError("supported MuJoCo 3.8.0 and named host required")
    root = Path(__file__).resolve().parents[1]
    fields = (
        subprocess.run(
            ["git", "ls-tree", "HEAD", "vendor/ud-tools"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split()
    )
    if len(fields) != 4 or fields[:2] != ["160000", "commit"]:
        raise ValueError("native Tools T01 gitlink unavailable")
    return {
        "schema_version": "f04-native-coupling/1",
        "parent_issue": 11788,
        "qualification": "synthetic-contact-free-native-fixture-only",
        "coordination": "local-F04-parent-authorized-during-GitHub-quota-stop",
        "host_alias": host_alias,
        "provider_versions": {
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "tools_gitlink": fields[2],
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in _SOURCE
        },
        "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest(),
        "native_evidence": extract_native_evidence(junit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_bytes(
        (
            json.dumps(
                build_receipt(args.junit, host_alias=args.host_alias),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )


if __name__ == "__main__":
    main()
