"""One-shot bounded supervisor for the retained plain 21-interval child."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_json
from tests.opensim.native_moco_ladder_harness import (
    DEFAULT_RUNGS,
    RungResult,
    run_ladder,
    worker_command,
)
from tests.opensim.native_moco_ladder_worker import _validate_refinement_plan

MESH_INTERVALS = 21
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _persist_json(path: Path, payload: dict[str, Any]) -> None:
    validated = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    atomic_write_json(path, validated)


def _parent_hashes(parent: Path, analysis_path: Path) -> dict[str, str]:
    paths = {
        "source_osim": parent / "source.osim",
        "request": parent / "request.json",
        "solution_sto": parent / "solution.sto",
        "time_only_controls_sto": parent / "time_only_controls.sto",
    }
    paths["replay_audit"] = (
        analysis_path.parent
        / "diagnostic_retained_replay_fauldump_3e195b0e"
        / "native_audit"
        / "fresh_replay_diagnostic.json"
    )
    return {name: sha256_file(path) for name, path in paths.items()}


def _load_plan(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_refinement_plan(plan)
    if len(plan.get("normalized_mesh_points", ())) - 1 != MESH_INTERVALS or plan.get(
        "flagged_parent_cells"
    ) != [0]:
        raise ValueError("plan is not the retained plain 21-interval child")
    if plan.get("parent_request", {}).get("physical_qualification") != "unqualified":
        raise ValueError("refinement must preserve unqualified physical status")
    return plan


def _source_hashes(repo_root: Path) -> dict[str, str]:
    names = (
        "tests/opensim/native_moco_refinement_supervisor.py",
        "tests/opensim/native_moco_ladder_harness.py",
        "tests/opensim/native_moco_ladder_worker.py",
        "tests/opensim/native_moco_refinement.py",
    )
    return {name: sha256_file(repo_root / name) for name in names}


def _runtime_fingerprint(casadi_path: Path) -> dict[str, Any]:
    filenames = ("_common.pyd", "_simulation.pyd", "_moco.pyd", "osimMoco.dll")
    components = {name: sha256_file(casadi_path / name) for name in filenames}
    return {
        "coverage": "bounded Python extension and Moco DLL fingerprint; not a closed dependency-runtime identity",
        "directory": str(casadi_path.resolve()),
        "components": components,
    }


def _interpreter_fingerprint(worker_python: str) -> dict[str, str]:
    executable = Path(worker_python).resolve(strict=True)
    return {"path": str(executable), "sha256": sha256_file(executable)}


def _worker_argv(
    rung_directory: Path, plan_path: Path, worker_python: str
) -> tuple[str, ...]:
    return worker_command(
        DEFAULT_RUNGS[0],
        rung_directory,
        worker_python,
        MESH_INTERVALS,
        plan_path,
    )


def _output_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prepare_once(
    plan_path: Path,
    output_root: Path,
    worker_python: str,
    repo_root: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Validate immutable parent inputs and stage an exact plan copy."""
    if output_root.exists():
        raise FileExistsError(output_root)
    plan_path = plan_path.resolve(strict=True)
    plan = _load_plan(plan_path)
    parent = Path(plan["parent_directory"])
    analysis_path = Path(plan["analysis_path"])
    parent_before = _parent_hashes(parent, analysis_path)
    if parent_before != plan["parent_hashes"]:
        raise ValueError("parent evidence hashes differ from the bound plan")
    output_root.mkdir(parents=True, exist_ok=False)
    staged_plan = output_root / "refinement_plan.json"
    staged_plan.write_bytes(plan_path.read_bytes())
    if sha256_file(staged_plan) != sha256_file(plan_path):
        raise ValueError("staged refinement plan is not byte-identical")
    command = _worker_argv(output_root / "plain", staged_plan, worker_python)
    evidence = {
        "schema": "native-moco-single-refinement-invocation-v1",
        "head": _git_head(repo_root),
        "working_directory": str(repo_root.resolve()),
        "worker_command": list(command),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "staged_plan_sha256": sha256_file(staged_plan),
        "parent_directory": str(parent),
        "analysis_path": str(analysis_path),
        "analysis_sha256_before": sha256_file(analysis_path),
        "parent_hashes_before": parent_before,
        "attempt_id": plan["attempt_id"],
        "refinement_parent_hashes": plan["parent_hashes"],
        "source_hashes_before": _source_hashes(repo_root),
        "parent_request_contract": plan["preserved_contract"],
        "rung": "plain",
        "mesh_intervals": MESH_INTERVALS,
        "physical_qualification": "unqualified",
        "wall_timeout_seconds": 30,
        "sampled_tree_rss_threshold_bytes": 1024**3,
        "thread_variables": dict.fromkeys(THREAD_VARIABLES, "1"),
        "retry_count": 0,
    }
    return evidence, command


def _thread_limited_environment(casadi_path: Path, repo_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in THREAD_VARIABLES:
        environment[name] = "1"
    environment["CASADIPATH"] = str(casadi_path.resolve())
    environment["PYTHONPATH"] = str(repo_root)
    return environment


def _run_one(
    output_root: Path,
    worker_python: str,
) -> tuple[RungResult, tuple[str, ...]]:
    expected = _worker_argv(
        output_root / "plain", output_root / "refinement_plan.json", worker_python
    )
    observed: list[tuple[str, ...]] = []

    def command_for(rung: Any, directory: Path) -> tuple[str, ...]:
        command = _worker_argv(
            directory, output_root / "refinement_plan.json", worker_python
        )
        observed.append(command)
        return command

    result = run_ladder(
        DEFAULT_RUNGS[:1],
        command_for,
        output_root,
        timeout_seconds=30,
    )[0]
    if observed != [expected]:
        raise ValueError("actual run_ladder command differs from invocation plan")
    return result, observed[0]


def run_attempt(
    plan_path: Path,
    output_root: Path,
    worker_python: str,
    casadi_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    runtime_before = _runtime_fingerprint(casadi_path)
    evidence, expected_command = prepare_once(
        plan_path, output_root, worker_python, repo_root
    )
    evidence["runtime_fingerprint_before"] = runtime_before
    evidence["worker_interpreter_before"] = _interpreter_fingerprint(worker_python)
    evidence["source_plan_sha256_before"] = sha256_file(plan_path)
    evidence["staged_plan_sha256_before"] = sha256_file(
        output_root / "refinement_plan.json"
    )
    worker_environment = _thread_limited_environment(casadi_path, repo_root)
    os.environ.update(worker_environment)
    evidence["worker_environment"] = {
        name: worker_environment[name]
        for name in ("CASADIPATH", "PYTHONPATH", *THREAD_VARIABLES)
    }
    evidence["parent_python_version"] = sys.version
    _persist_json(output_root / "invocation.json", evidence)
    run_error = None
    try:
        result, actual_command = _run_one(output_root, worker_python)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        result = None
        actual_command = expected_command
        run_error = {"type": type(exc).__name__, "detail": str(exc)}
    actual_parent_hashes = _parent_hashes(
        Path(evidence["parent_directory"]), Path(evidence["analysis_path"])
    )
    evidence["parent_hashes_after"] = actual_parent_hashes
    evidence["parent_inputs_unchanged"] = (
        actual_parent_hashes == evidence["parent_hashes_before"]
    )
    evidence["analysis_sha256_after"] = sha256_file(Path(evidence["analysis_path"]))
    evidence["analysis_file_unchanged"] = (
        evidence["analysis_sha256_after"] == evidence["analysis_sha256_before"]
    )
    evidence["source_hashes_after"] = _source_hashes(repo_root)
    evidence["source_files_unchanged"] = (
        evidence["source_hashes_after"] == evidence["source_hashes_before"]
    )
    evidence["runtime_fingerprint_after"] = _runtime_fingerprint(casadi_path)
    evidence["runtime_files_unchanged"] = (
        evidence["runtime_fingerprint_after"] == evidence["runtime_fingerprint_before"]
    )
    evidence["worker_interpreter_after"] = _interpreter_fingerprint(worker_python)
    evidence["worker_interpreter_unchanged"] = (
        evidence["worker_interpreter_after"] == evidence["worker_interpreter_before"]
    )
    evidence["source_plan_sha256_after"] = sha256_file(plan_path)
    evidence["staged_plan_sha256_after"] = sha256_file(
        output_root / "refinement_plan.json"
    )
    evidence["plan_files_unchanged"] = (
        evidence["source_plan_sha256_after"] == evidence["plan_sha256"]
        and evidence["staged_plan_sha256_after"] == evidence["staged_plan_sha256"]
    )
    evidence["head_after"] = _git_head(repo_root)
    evidence["head_unchanged"] = evidence["head_after"] == evidence["head"]
    evidence["result"] = (
        dataclasses.asdict(result)
        if result is not None
        else {
            "status": "supervisor_error",
            "detail": run_error["detail"] if run_error else "unknown supervisor error",
            "owned_tree_cleanup_verified": False,
        }
    )
    evidence["supervisor_exception"] = run_error
    evidence["actual_worker_command"] = list(actual_command)
    evidence["expected_command_matches_result_protocol"] = list(
        expected_command
    ) == list(actual_command)
    plain_output = output_root / "plain"
    evidence["output_sha256"] = (
        _output_hashes(plain_output) if plain_output.is_dir() else {}
    )
    evidence["result_sha256"] = result.receipt_sha256 if result is not None else None
    evidence["provenance_unchanged"] = all(
        evidence[key]
        for key in (
            "parent_inputs_unchanged",
            "analysis_file_unchanged",
            "source_files_unchanged",
            "runtime_files_unchanged",
            "worker_interpreter_unchanged",
            "head_unchanged",
            "plan_files_unchanged",
            "expected_command_matches_result_protocol",
        )
    )
    evidence["final_status"] = (
        evidence["result"]["status"]
        if evidence["provenance_unchanged"]
        else "provenance_changed"
    )
    _persist_json(output_root / "supervisor_result.json", evidence)
    return evidence


def run_stub(
    plan_path: Path,
    output_root: Path,
    worker_python: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Exercise plan, output, and exact command protocol without launching SDK."""
    evidence, command = prepare_once(plan_path, output_root, worker_python, repo_root)
    expected = _worker_argv(
        output_root / "plain", output_root / "refinement_plan.json", worker_python
    )
    if command != expected:
        raise ValueError("stub command differs from the reviewed worker protocol")
    evidence["status"] = "stub_preflight_pass_no_sdk_launch"
    evidence["sdk_launched"] = False
    evidence["owned_tree_cleanup_verified"] = None
    evidence["worker_command"] = list(command)
    _persist_json(output_root / "stub_result.json", evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--casadi-path", type=Path)
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args()
    if args.stub:
        run_stub(args.plan, args.output_root, args.worker_python, args.repo_root)
        return 0
    if args.casadi_path is None:
        parser.error("--casadi-path is required outside stub mode")
    result = run_attempt(
        args.plan,
        args.output_root,
        args.worker_python,
        args.casadi_path,
        args.repo_root,
    )
    return 0 if result["final_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
