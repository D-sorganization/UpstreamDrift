"""Freeze source-bound F02 native feedback evidence from a passed JUnit run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import cast

from defusedxml import ElementTree as ET

_TEST = "test_native_f02_feedback_replays_full_state_and_beats_frozen_nominal"
_MOVING_TEST = "test_native_model_derived_moving_reference_frozen_and_feedback_replay"
_HASH_FIELDS = (
    "source_model_sha256",
    "loaded_native_model_sha256",
    "initial_state_sha256",
    "policy_sha256",
    "time_grid_sha256",
    "state_schema_sha256",
    "input_channel_schema_sha256",
    "controlled_applied_input_sha256",
    "nominal_applied_input_sha256",
)
_NUMERIC_FIELDS = (
    "controlled_final_hip_error_rad",
    "nominal_final_hip_error_rad",
    "max_full_state_replay_error",
)
_MOVING_HASH_FIELDS = (
    "source_model_sha256",
    "teacher_applied_input_sha256",
    "frozen_applied_input_sha256",
    "controlled_applied_input_sha256",
    "heldout_initial_state_sha256",
    "policy_sha256",
    "time_grid_sha256",
)
_MOVING_NUMERIC_FIELDS = (
    "frozen_final_hip_error_rad",
    "controlled_final_hip_error_rad",
    "max_frozen_replay_error",
    "max_controlled_replay_error",
)
_SOURCE_PATHS = (
    "scripts/f02_native_manifold_receipt.py",
    "src/engines/physics_engines/mujoco/python/native_distributed_feedback.py",
    "src/engines/physics_engines/mujoco/python/native_tangent_derivative.py",
    "src/engines/physics_engines/mujoco/python/native_nmpc_tracking.py",
    "src/engines/physics_engines/mujoco/python/native_torque_replay.py",
    "src/shared/python/motion_matching/distributed_feedback.py",
    "tests/unit/motion_matching/test_native_distributed_feedback.py",
)


def _passed_property(
    junit: Path, test_name: str, property_name: str
) -> dict[str, object]:
    """Load only one named property from one passed native test."""
    root = ET.parse(junit).getroot()
    cases = [case for case in root.iter("testcase") if case.get("name") == test_name]
    if len(cases) != 1 or any(
        cases[0].find(outcome) is not None
        for outcome in ("failure", "error", "skipped")
    ):
        raise ValueError("receipt requires exactly one passed native feedback test")
    properties = cases[0].find("properties")
    values = (
        []
        if properties is None
        else [
            prop.get("value")
            for prop in properties.findall("property")
            if prop.get("name") == property_name
        ]
    )
    if len(values) != 1 or not values[0]:
        raise ValueError("native evidence property is missing or duplicated")
    evidence = json.loads(values[0])
    if not isinstance(evidence, dict):
        raise ValueError("native evidence property must be a JSON object")
    return evidence


def _validate_fields(
    evidence: dict[str, object], hashes: tuple[str, ...], numbers: tuple[str, ...]
) -> None:
    if set(evidence) != set(hashes + numbers):
        raise ValueError("native evidence property has missing or unknown fields")
    for field in hashes:
        value = evidence[field]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"native {field} must be a SHA-256 hex digest")
    for field in numbers:
        value = evidence[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"native {field} must be finite")


def extract_native_evidence(junit: Path) -> dict[str, object]:
    """Reject absent, failed, partial or contradictory native test properties."""
    evidence = _passed_property(junit, _TEST, "f02_native_evidence")
    _validate_fields(evidence, _HASH_FIELDS, _NUMERIC_FIELDS)
    if cast(float, evidence["controlled_final_hip_error_rad"]) >= cast(
        float, evidence["nominal_final_hip_error_rad"]
    ):
        raise ValueError("native feedback did not show declared hip improvement")
    if cast(float, evidence["max_full_state_replay_error"]) > 1e-12:
        raise ValueError("native full-state replay exceeded the declared tolerance")
    if (
        evidence["controlled_applied_input_sha256"]
        == evidence["nominal_applied_input_sha256"]
    ):
        raise ValueError("controlled and nominal replay inputs must differ")
    return evidence


def extract_moving_evidence(junit: Path) -> dict[str, object]:
    """Admit separate frozen-feedforward and total-feedback native replays."""
    evidence = _passed_property(junit, _MOVING_TEST, "f02_moving_evidence")
    _validate_fields(evidence, _MOVING_HASH_FIELDS, _MOVING_NUMERIC_FIELDS)
    if cast(float, evidence["controlled_final_hip_error_rad"]) >= cast(
        float, evidence["frozen_final_hip_error_rad"]
    ):
        raise ValueError("moving native reference did not improve under feedback")
    if (
        max(
            cast(float, evidence["max_frozen_replay_error"]),
            cast(float, evidence["max_controlled_replay_error"]),
        )
        > 1e-12
    ):
        raise ValueError("moving native replay exceeded full-state tolerance")
    if (
        evidence["controlled_applied_input_sha256"]
        == evidence["frozen_applied_input_sha256"]
    ):
        raise ValueError("moving native feedback must change the saved input")
    return evidence


def build_receipt(junit: Path, *, host_alias: str) -> dict[str, object]:
    """Bind the passed native property to checked-in source and runtime."""
    import mujoco
    import numpy

    if not host_alias or mujoco.__version__ != "3.8.0":
        raise ValueError("receipt requires named host and supported MuJoCo 3.8.0")
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in _SOURCE_PATHS
    }
    gitlink = (
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
    if len(gitlink) != 4 or gitlink[:2] != ["160000", "commit"]:
        raise ValueError("pinned Tools gitlink identity is unavailable")
    return {
        "schema_version": "f02-native-manifold-feedback/1",
        "issue": 11946,
        "qualification": "synthetic-contact-free-fixture-only",
        "host_alias": host_alias,
        "provider": {
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "python": sys.version.split()[0],
        },
        "tools_t01_gitlink_commit": gitlink[2],
        "fixture": {
            "nq": 9,
            "nv": 8,
            "nu": 2,
            "static_steps": 12,
            "moving_steps": 20,
            "dt_s": 0.01,
        },
        "source_sha256": source_hashes,
        "native_evidence": extract_native_evidence(junit),
        "moving_evidence": extract_moving_evidence(junit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(args.junit, host_alias=args.host_alias)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
