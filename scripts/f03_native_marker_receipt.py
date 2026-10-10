"""Freeze supported-provider F03 marker-fitting and replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import cast

from defusedxml import ElementTree as ET

from scripts.f02_native_manifold_receipt import _passed_property

_TEST = "test_native_marker_fit_improves_reachable_observations_and_replays"
_GRADIENT_TEST = "test_native_marker_cost_has_correct_tangent_gradient_and_exact_clock"
_HASHES = (
    "observation_sha256",
    "source_model_sha256",
    "loaded_native_model_sha256",
    "initial_state_sha256",
    "policy_sha256",
    "time_grid_sha256",
    "applied_input_sha256",
)
_NUMBERS = (
    "fit_marker_rmse_m",
    "zero_marker_rmse_m",
    "max_full_state_replay_error",
    "total_wall_s",
    "total_cpu_s",
)
_COUNTS = ("masked_site_samples", "observed_site_samples", "accepted_commands", "steps")
_SOURCE = (
    "scripts/f03_native_marker_receipt.py",
    "src/engines/physics_engines/mujoco/python/native_marker_fit.py",
    "src/engines/physics_engines/mujoco/python/native_manifold_box_fddp.py",
    "src/engines/physics_engines/mujoco/python/native_tangent_derivative.py",
    "src/engines/physics_engines/mujoco/python/native_torque_replay.py",
    "tests/unit/motion_matching/test_native_marker_fit.py",
    "tests/unit/motion_matching/test_f03_native_marker_receipt.py",
)


def extract_native_evidence(junit: Path) -> dict[str, object]:
    """Admit only two passed native tests and a complete, consistent receipt."""
    raw = junit.read_bytes()
    if len(raw) > 2_000_000 or b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("native XML exceeds limit or declares entities")
    cases = ET.fromstring(raw).findall(".//testcase")
    if sorted(case.get("name") for case in cases) != sorted((_TEST, _GRADIENT_TEST)):
        raise ValueError("native test inventory is incomplete")
    if any(
        case.find(outcome) is not None
        for case in cases
        for outcome in ("failure", "error", "skipped")
    ):
        raise ValueError("native receipt requires passed tests")
    evidence = _passed_property(junit, _TEST, "f03_marker_fit_evidence")
    expected = set(
        _HASHES + _NUMBERS + _COUNTS + ("statuses", "full_trajectory_accepted")
    )
    if set(evidence) != expected:
        raise ValueError("native evidence fields are incomplete or unknown")
    for field in _HASHES:
        if (
            not isinstance(evidence[field], str)
            or re.fullmatch(r"[0-9a-f]{64}", cast(str, evidence[field])) is None
        ):
            raise ValueError(f"native {field} requires SHA-256")
    for field in _NUMBERS:
        value = evidence[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"native {field} must be finite")
    for field in _COUNTS:
        if type(evidence[field]) is not int or cast(int, evidence[field]) < 0:
            raise ValueError(f"native {field} count invalid")
    if (
        evidence["masked_site_samples"] == 0
        or cast(int, evidence["observed_site_samples"]) == 0
    ):
        raise ValueError("native observation mask/availability invalid")
    steps = cast(int, evidence["steps"])
    accepted = cast(int, evidence["accepted_commands"])
    statuses = evidence["statuses"]
    if (
        steps < 1
        or not isinstance(statuses, list)
        or len(statuses) != steps
        or any(not isinstance(status, str) for status in statuses)
        or accepted < 1
        or accepted != statuses.count("optimized")
    ):
        raise ValueError("native accepted-command count conflicts with statuses")
    if type(evidence["full_trajectory_accepted"]) is not bool or evidence[
        "full_trajectory_accepted"
    ] != (accepted == steps):
        raise ValueError("native full-trajectory acceptance conflicts with commands")
    if cast(float, evidence["fit_marker_rmse_m"]) >= cast(
        float, evidence["zero_marker_rmse_m"]
    ):
        raise ValueError("native marker improvement not demonstrated")
    if cast(float, evidence["max_full_state_replay_error"]) > 1e-12:
        raise ValueError("native complete-state replay failed")
    if (
        cast(float, evidence["total_wall_s"]) <= 0
        or cast(float, evidence["total_cpu_s"]) <= 0
    ):
        raise ValueError("native total timing invalid")
    return evidence


def _tools_gitlink(root: Path, supplied: str | None) -> str:
    """Bind the pinned Tools object, allowing a Windows-owned tree in WSL."""
    if supplied is not None and re.fullmatch(r"[0-9a-f]{40}", supplied) is None:
        raise ValueError("Tools T01 gitlink must be a commit SHA")
    result = subprocess.run(
        ["git", "ls-tree", "HEAD", "vendor/ud-tools"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    fields = result.stdout.strip().split() if result.returncode == 0 else []
    discovered = (
        fields[2] if len(fields) == 4 and fields[:2] == ["160000", "commit"] else None
    )
    if supplied is not None and discovered is not None and supplied != discovered:
        raise ValueError("supplied Tools T01 gitlink differs from HEAD")
    if discovered is None and supplied is None:
        raise ValueError(
            "Tools T01 gitlink unavailable; supply verified host git ls-tree SHA"
        )
    return cast(str, discovered or supplied)


def build_receipt(
    junit: Path, *, host_alias: str, tools_gitlink: str | None = None
) -> dict[str, object]:
    """Bind JUnit evidence to checked-in source, Tools pin and native versions."""
    import crocoddyl
    import mujoco
    import numpy
    import pinocchio

    pinocchio_version = getattr(pinocchio, "__version__", None)
    if (
        not host_alias
        or mujoco.__version__ != "3.8.0"
        or crocoddyl.__version__ != "3.2.1"
        or pinocchio_version != "4.1.0"
    ):
        raise ValueError("supported MuJoCo/Crocoddyl/Pinocchio and named host required")
    root = Path(__file__).resolve().parents[1]
    gitlink = _tools_gitlink(root, tools_gitlink)
    return {
        "schema_version": "f03-native-marker-fit/1",
        "parent_issue": 11787,
        "qualification": "synthetic-contact-free-native-fixture-only",
        "coordination": "offline-parent-authorized-F03; reconcile child lease after GitHub quota recovers",
        "host_alias": host_alias,
        "provider_versions": {
            "mujoco": mujoco.__version__,
            "crocoddyl": crocoddyl.__version__,
            "pinocchio": pinocchio_version,
            "numpy": numpy.__version__,
        },
        "tools_gitlink": gitlink,
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
    parser.add_argument(
        "--tools-gitlink", help="verified Windows-host git ls-tree commit"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        args.junit, host_alias=args.host_alias, tools_gitlink=args.tools_gitlink
    )
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
