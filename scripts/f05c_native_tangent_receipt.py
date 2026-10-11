"""Freeze the supported F05c manifold-derivative/native-replay test receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

# The only subprocess is a fixed local Git query without a shell.
import subprocess  # nosec B404
import sys

# Bounded local pytest XML is accepted only without DTD/entity declarations.
from xml.etree import ElementTree  # nosec B405

import mujoco
import numpy

_SOURCE = (
    "src/engines/physics_engines/mujoco/python/native_tangent_derivative.py",
    "src/engines/physics_engines/mujoco/python/native_torque_replay.py",
    "tests/unit/motion_matching/test_native_tangent_derivative.py",
    "scripts/f05c_native_tangent_receipt.py",
)
_DERIVATIVE = (
    "derivative_p50_s",
    "derivative_p95_s",
    "derivative_worst_s",
    "derivative_calls",
    "A_sha256",
    "B_sha256",
)
_REPLAY = (
    "bundle_s",
    "replay_s",
    "model_sha256",
    "initial_state_sha256",
    "applied_input_sha256",
    "policy_sha256",
    "state_schema_sha256",
    "time_grid_sha256",
)


def _git(repo: Path, *args: str) -> str:
    pointer = repo / ".git"
    gitdir = pointer.read_text(encoding="utf-8") if pointer.is_file() else ""
    executable = (
        "git.exe" if gitdir.startswith("gitdir: ") and gitdir[9:10] == ":" else "git"
    )
    result = subprocess.run(  # nosec B603
        (executable, *args),
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return result.stdout.strip()


def _properties(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    raw = path.read_bytes()
    if len(raw) > 2_000_000 or b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("benchmark XML exceeds limit or declares entities")
    root = ElementTree.fromstring(raw)  # nosec B314
    cases = root.findall(".//testcase")
    if len(cases) != 4 or any(
        case.find("failure") is not None
        or case.find("error") is not None
        or case.find("skipped") is not None
        for case in cases
    ):
        raise ValueError(
            "all four predeclared native derivative/replay tests must pass"
        )

    def props(name: str, keys: tuple[str, ...]) -> dict[str, str]:
        matches = [case for case in cases if case.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"native receipt lacks exactly one {name}")
        values = {
            row.get("name"): row.get("value")
            for row in matches[0].findall("./properties/property")
        }
        if any(values.get(key) is None for key in keys):
            raise ValueError(f"native receipt {name} has incomplete properties")
        return {key: str(values[key]) for key in keys}

    return (
        props(
            "test_native_tangent_jacobians_match_manifold_perturbed_steps", _DERIVATIVE
        ),
        props("test_floating_two_motor_torques_replay_complete_native_state", _REPLAY),
    )


def build_receipt(junit: Path, repo: Path, host_alias: str) -> dict[str, object]:
    """Bind the exact native test and source identities without a coverage claim."""
    if not host_alias or any(char in host_alias for char in ("/", "\\", ":")):
        raise ValueError("host alias must be a public non-path label")
    derivative, replay = _properties(junit)
    return {
        "schema_version": "upstreamdrift/f05c-native-tangent-development/1.0.0",
        "issue": 11920,
        "parent_issue": 11789,
        "scope": {
            "plant": "floating_root_two_hinge_two_direct_unit_motors_contact_free",
            "nq": 9,
            "nv": 8,
            "nu": 2,
            "native_integrator": "Euler",
            "native_step_s": 0.01,
            "derivative": "mjd_transitionFD_tangent_central_checked_against_native_steps",
            "derivative_repeat_calls": 10,
            "input": "post_limit_actuator_torque_exact_ZOH",
            "initial_state": "complete_mjSTATE_INTEGRATION",
            "replay": "fresh_native_time_only_no_resets",
            "replay_tolerance": 1e-12,
            "observation_grid": "none_synthetic_native_state_only",
        },
        "runtime": {
            "host_alias": host_alias,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
        },
        "provenance": {
            "git_head_before_receipt": _git(repo, "rev-parse", "HEAD"),
            "tools_pin": _git(repo, "rev-parse", "HEAD:vendor/ud-tools"),
            "source_sha256": {
                path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
                for path in _SOURCE
            },
        },
        "derivative": {
            key: (
                float(derivative[key])
                if key.endswith("_s")
                else int(derivative[key])
                if key == "derivative_calls"
                else derivative[key]
            )
            for key in _DERIVATIVE
        },
        "replay": {
            key: float(replay[key]) if key in ("bundle_s", "replay_s") else replay[key]
            for key in _REPLAY
        },
        "limits": (
            "This is an exact-state/contact-free multi-DOF derivative and input-replay "
            "fixture, not optimized full-body feedback, hard-real-time evidence, "
            "muscle/contact/grip qualification, private capture accuracy or six-engine parity."
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
