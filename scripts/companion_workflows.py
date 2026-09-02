"""Validate and execute the governed UpstreamDrift companion workflows.

The registry is data; this module is the sole execution boundary. Commands are
always argv-based Python modules, receive only declared environment keys, run
with ``shell=False``, and may change only their declared artifact files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence, Set
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import jsonschema

REGISTRY_PATH = Path("scripts/config/companion_workflows.v1.json")
DEFAULT_REPORT = Path("dist/companion-workflows/execution-report.v1.json")
REGISTRY_ID = "upstreamdrift-companion-workflows"
REGISTRY_VERSION = "1.0.0"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_ENVIRONMENT_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FORBIDDEN_ARGUMENT = re.compile(r"[;&|<>`$(){}\[\]!*?~\r\n]")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_INHERITED_ENV = frozenset(
    {
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
)
_ALLOWED_FIXED_ENV = frozenset(
    {
        "MPLBACKEND",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUTF8",
    }
)
_ALLOWED_MODULES = frozenset(
    {"scripts.companion_catalog", "scripts.companion_workflow_tasks"}
)
_FAILURE_CLASSES = frozenset(
    {"bad-input", "stale-version", "unavailable-engine", "unsupported-dependency"}
)
_ARTIFACT_TYPES = frozenset({"csv", "json", "markdown", "png", "sha256"})
_VERIFICATION_METHODS = frozenset(
    {"csv-header", "json-object", "non-empty", "png-signature", "sha256-sidecar"}
)
_SUPPORT_TIERS = frozenset({"experimental", "extended", "not_applicable", "supported"})
_OUTPUT_PREFIX = PurePosixPath("dist/companion-workflows/artifacts")
_RECORD_REQUIRED_KEYS = frozenset(
    {
        "id",
        "title",
        "kind",
        "executor",
        "argv",
        "cwd",
        "environment",
        "expected_exit_code",
        "expected_artifacts",
        "documentation_paths",
        "program_ids",
        "prerequisites",
        "support_tier",
        "availability",
        "determinism",
        "verification_method",
        "scientific_limitations",
    }
)


class WorkflowContractError(ValueError):
    """Raised when workflow registry data violates its fail-closed contract."""


class WorkflowExecutionError(RuntimeError):
    """Raised when a command or its artifact evidence violates its contract."""


def _exact_keys(
    value: Mapping[str, Any], *, required: Set[str], optional: Set[str], label: str
) -> None:
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        raise WorkflowContractError(
            f"{label} has unknown or missing keys; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowContractError(f"{label} must be a non-empty string")
    return value


def _repo_relative(value: Any, *, label: str, allow_root: bool = False) -> str:
    text = _non_empty_string(value, label=label).replace("\\", "/")
    if allow_root and text == ".":
        return text
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or ".." in candidate.parts
        or candidate.parts in ((), (".",))
    ):
        raise WorkflowContractError(f"{label} must be a contained repo-relative path")
    return candidate.as_posix()


def _safe_argument(value: Any, *, index: int) -> str:
    text = _non_empty_string(value, label=f"argv[{index}]")
    if _FORBIDDEN_ARGUMENT.search(text):
        raise WorkflowContractError(f"argv[{index}] contains a shell metacharacter")
    if text.startswith(("http://", "https://")):
        raise WorkflowContractError(
            f"argv[{index}] contains a mutable external reference"
        )
    candidate = PurePosixPath(text.replace("\\", "/"))
    if candidate.is_absolute() or PureWindowsPath(text).is_absolute():
        raise WorkflowContractError(f"argv[{index}] must not be absolute")
    if ".." in candidate.parts:
        raise WorkflowContractError(f"argv[{index}] contains path traversal")
    return text


def _string_list(value: Any, *, label: str, non_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise WorkflowContractError(f"{label} must be an array of strings")
    if non_empty and not value:
        raise WorkflowContractError(f"{label} must not be empty")
    if len(value) != len(set(value)):
        raise WorkflowContractError(f"{label} contains duplicate values")
    return list(value)


def _validate_environment(raw: Any, *, repo_root: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise WorkflowContractError("environment must be an object")
    _exact_keys(
        raw,
        required={"inherit", "fixed", "python_paths"},
        optional=set(),
        label="environment",
    )
    inherited = _string_list(raw["inherit"], label="environment.inherit")
    fixed_raw = raw["fixed"]
    if not isinstance(fixed_raw, Mapping):
        raise WorkflowContractError("environment.fixed must be an object")
    fixed: dict[str, str] = {}
    for key, value in fixed_raw.items():
        if not isinstance(key, str) or not _ENVIRONMENT_KEY.fullmatch(key):
            raise WorkflowContractError(f"invalid fixed environment key: {key!r}")
        fixed[key] = _non_empty_string(value, label=f"environment.fixed.{key}")
    invalid_inherited = set(inherited) - _ALLOWED_INHERITED_ENV
    invalid_fixed = set(fixed) - _ALLOWED_FIXED_ENV
    overlap = set(inherited) & set(fixed)
    if invalid_inherited or invalid_fixed or overlap:
        raise WorkflowContractError(
            "environment keys are not safely declared; "
            f"invalid inherited={sorted(invalid_inherited)}, "
            f"invalid fixed={sorted(invalid_fixed)}, overlap={sorted(overlap)}"
        )
    python_paths = [
        _repo_relative(path, label="environment python path", allow_root=True)
        for path in _string_list(raw["python_paths"], label="environment.python_paths")
    ]
    for path in python_paths:
        candidate = repo_root if path == "." else repo_root / path
        if not candidate.is_dir():
            raise WorkflowContractError(
                f"environment python path does not exist: {path}"
            )
    return {
        "inherit": sorted(inherited),
        "fixed": dict(sorted(fixed.items())),
        "python_paths": python_paths,
    }


def _validate_verification(raw: Any, *, artifact_type: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise WorkflowContractError("artifact verification must be an object")
    _exact_keys(
        raw,
        required={"method", "minimum_bytes"},
        optional={"required_columns"},
        label="artifact verification",
    )
    method = _non_empty_string(raw["method"], label="artifact verification method")
    if method not in _VERIFICATION_METHODS:
        raise WorkflowContractError(
            f"unsupported artifact verification method: {method}"
        )
    minimum_bytes = raw["minimum_bytes"]
    if (
        not isinstance(minimum_bytes, int)
        or isinstance(minimum_bytes, bool)
        or minimum_bytes < 1
    ):
        raise WorkflowContractError("artifact minimum_bytes must be a positive integer")
    required_columns = _string_list(
        raw.get("required_columns", []), label="artifact required_columns"
    )
    if method == "csv-header" and not required_columns:
        raise WorkflowContractError("csv-header verification requires columns")
    if method != "csv-header" and required_columns:
        raise WorkflowContractError("required_columns is only valid for csv-header")
    expected_method = {
        "csv": "csv-header",
        "json": "json-object",
        "markdown": "non-empty",
        "png": "png-signature",
        "sha256": "sha256-sidecar",
    }[artifact_type]
    if method != expected_method:
        raise WorkflowContractError(
            f"artifact type {artifact_type} requires verification method {expected_method}"
        )
    result: dict[str, Any] = {"method": method, "minimum_bytes": minimum_bytes}
    if required_columns:
        result["required_columns"] = required_columns
    return result


def _validate_artifacts(raw: Any, *, executable: bool) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise WorkflowContractError("expected_artifacts must be an array")
    artifacts: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise WorkflowContractError(
                f"expected_artifacts[{index}] must be an object"
            )
        _exact_keys(
            item,
            required={"path", "type", "required", "verification"},
            optional=set(),
            label=f"expected_artifacts[{index}]",
        )
        path = _repo_relative(item["path"], label="artifact path")
        candidate = PurePosixPath(path)
        if candidate != _OUTPUT_PREFIX and _OUTPUT_PREFIX not in candidate.parents:
            raise WorkflowContractError(
                f"artifact path must be below {_OUTPUT_PREFIX.as_posix()}"
            )
        artifact_type = _non_empty_string(item["type"], label="artifact type")
        if artifact_type not in _ARTIFACT_TYPES:
            raise WorkflowContractError(f"unsupported artifact type: {artifact_type}")
        required = item["required"]
        if not isinstance(required, bool):
            raise WorkflowContractError("artifact required must be boolean")
        if path in paths:
            raise WorkflowContractError(f"duplicate artifact path: {path}")
        paths.add(path)
        artifacts.append(
            {
                "path": path,
                "type": artifact_type,
                "required": required,
                "verification": _validate_verification(
                    item["verification"], artifact_type=artifact_type
                ),
            }
        )
    if executable and not artifacts:
        raise WorkflowContractError("executable workflows require expected artifacts")
    return sorted(artifacts, key=lambda item: item["path"])


def _validate_determinism(raw: Any, *, repo_root: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise WorkflowContractError("determinism must be an object")
    _exact_keys(
        raw,
        required={
            "fixture_kind",
            "fixture_paths",
            "absolute_tolerance",
            "relative_tolerance",
        },
        optional=set(),
        label="determinism",
    )
    fixture_kind = _non_empty_string(raw["fixture_kind"], label="fixture_kind")
    if fixture_kind not in {"inline", "repository"}:
        raise WorkflowContractError("fixture_kind must be inline or repository")
    fixture_paths = [
        _repo_relative(path, label="fixture path")
        for path in _string_list(raw["fixture_paths"], label="fixture_paths")
    ]
    if fixture_kind == "repository" and not fixture_paths:
        raise WorkflowContractError("repository fixtures require fixture_paths")
    if fixture_kind == "inline" and fixture_paths:
        raise WorkflowContractError("inline fixtures cannot declare fixture_paths")
    for path in fixture_paths:
        if not (repo_root / path).is_file():
            raise WorkflowContractError(f"fixture path does not exist: {path}")
    tolerances: dict[str, float] = {}
    for key in ("absolute_tolerance", "relative_tolerance"):
        value = raw[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise WorkflowContractError(f"{key} must be a non-negative number")
        tolerances[key] = float(value)
    return {
        "fixture_kind": fixture_kind,
        "fixture_paths": sorted(fixture_paths),
        **tolerances,
    }


def _validate_identity(raw: Mapping[str, Any]) -> tuple[str, str]:
    workflow_id = _non_empty_string(raw["id"], label="workflow id")
    if not _IDENTIFIER.fullmatch(workflow_id):
        raise WorkflowContractError(f"invalid workflow id: {workflow_id}")
    kind = _non_empty_string(raw["kind"], label="workflow kind")
    if kind not in {"failure-fixture", "workflow"}:
        raise WorkflowContractError(f"unsupported workflow kind: {kind}")
    return workflow_id, kind


def _validate_availability(
    raw: Any,
) -> tuple[str, str | None, bool]:
    if not isinstance(raw, Mapping):
        raise WorkflowContractError("availability must be an object")
    _exact_keys(raw, required={"state", "reason"}, optional=set(), label="availability")
    state = _non_empty_string(raw["state"], label="availability state")
    if state not in {"available", "unavailable"}:
        raise WorkflowContractError(
            "workflow availability must be available or unavailable"
        )
    reason = raw["reason"]
    if state == "available" and reason is not None:
        raise WorkflowContractError("available workflow reason must be null")
    if state == "unavailable":
        reason = _non_empty_string(reason, label="unavailable reason")
    return state, reason, state == "available"


def _validate_execution(
    raw: Mapping[str, Any], *, kind: str, executable: bool
) -> tuple[str | None, list[str], int | None, str | None]:
    executor = raw["executor"]
    argv_raw = raw["argv"]
    if not isinstance(argv_raw, list):
        raise WorkflowContractError("argv must be an array")
    argv = [_safe_argument(value, index=index) for index, value in enumerate(argv_raw)]
    if executable:
        if executor != "python-module":
            raise WorkflowContractError(f"unsupported executor: {executor}")
        if not argv or not _MODULE.fullmatch(argv[0]):
            raise WorkflowContractError(
                "python-module argv must begin with a module name"
            )
        if argv[0] not in _ALLOWED_MODULES:
            raise WorkflowContractError(
                f"python module is not an approved workflow boundary: {argv[0]}"
            )
    elif executor is not None or argv:
        raise WorkflowContractError(
            "unavailable workflow must not expose an executor or argv"
        )
    expected_exit = raw["expected_exit_code"]
    if executable and (
        not isinstance(expected_exit, int)
        or isinstance(expected_exit, bool)
        or not 0 <= expected_exit <= 255
    ):
        raise WorkflowContractError(
            "expected_exit_code must be an integer from 0 to 255"
        )
    if not executable and expected_exit is not None:
        raise WorkflowContractError(
            "unavailable workflow expected_exit_code must be null"
        )
    failure_class = raw.get("failure_class")
    if kind == "failure-fixture":
        if failure_class not in _FAILURE_CLASSES:
            raise WorkflowContractError(
                f"invalid failure fixture class: {failure_class}"
            )
        if not executable or expected_exit == 0:
            raise WorkflowContractError(
                "failure fixtures must execute with a nonzero exit"
            )
    elif failure_class is not None:
        raise WorkflowContractError("ordinary workflows cannot declare failure_class")
    elif executable and expected_exit != 0:
        raise WorkflowContractError("ordinary workflows must expect exit code zero")
    return executor, argv, expected_exit, failure_class


def _validate_references(
    raw: Mapping[str, Any], *, repo_root: Path, program_ids: Set[str]
) -> tuple[str, list[str], list[str], str]:
    cwd = _repo_relative(raw["cwd"], label="cwd", allow_root=True)
    cwd_path = repo_root if cwd == "." else repo_root / cwd
    if not cwd_path.is_dir():
        raise WorkflowContractError(f"workflow cwd does not exist: {cwd}")
    documentation_paths = [
        _repo_relative(path, label="documentation path")
        for path in _string_list(
            raw["documentation_paths"], label="documentation_paths", non_empty=True
        )
    ]
    for path in documentation_paths:
        if not (repo_root / path).is_file():
            raise WorkflowContractError(f"documentation path does not exist: {path}")
    referenced_programs = _string_list(
        raw["program_ids"], label="program_ids", non_empty=True
    )
    dangling = set(referenced_programs) - program_ids
    if dangling:
        raise WorkflowContractError(
            f"workflow has dangling program IDs: {sorted(dangling)}"
        )
    support_tier = _non_empty_string(raw["support_tier"], label="support_tier")
    if support_tier not in _SUPPORT_TIERS:
        raise WorkflowContractError(f"unsupported support tier: {support_tier}")
    return cwd, sorted(documentation_paths), sorted(referenced_programs), support_tier


def _validate_record(
    raw: Any, *, repo_root: Path, source_commit: str, program_ids: Set[str]
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise WorkflowContractError("workflow record must be an object")
    _exact_keys(
        raw,
        required=_RECORD_REQUIRED_KEYS,
        optional={"failure_class"},
        label="workflow record",
    )
    workflow_id, kind = _validate_identity(raw)
    state, reason, executable = _validate_availability(raw["availability"])
    executor, argv, expected_exit, failure_class = _validate_execution(
        raw, kind=kind, executable=executable
    )
    cwd, documentation_paths, referenced_programs, support_tier = _validate_references(
        raw, repo_root=repo_root, program_ids=program_ids
    )
    record: dict[str, Any] = {
        "id": workflow_id,
        "title": _non_empty_string(raw["title"], label="workflow title"),
        "kind": kind,
        "executor": executor,
        "argv": argv,
        "cwd": cwd,
        "environment": _validate_environment(raw["environment"], repo_root=repo_root),
        "expected_exit_code": expected_exit,
        "expected_artifacts": _validate_artifacts(
            raw["expected_artifacts"], executable=executable
        ),
        "documentation_paths": documentation_paths,
        "program_ids": referenced_programs,
        "prerequisites": sorted(
            _string_list(raw["prerequisites"], label="prerequisites", non_empty=True)
        ),
        "support_tier": support_tier,
        "availability": {"state": state, "reason": reason},
        "determinism": _validate_determinism(raw["determinism"], repo_root=repo_root),
        "verification_method": _non_empty_string(
            raw["verification_method"], label="verification_method"
        ),
        "scientific_limitations": sorted(
            _string_list(
                raw["scientific_limitations"],
                label="scientific_limitations",
                non_empty=True,
            )
        ),
        "source_commit": source_commit,
    }
    if failure_class is not None:
        record["failure_class"] = failure_class
    return record


def parse_registry(
    payload: bytes,
    *,
    repo_root: Path,
    source_commit: str,
    program_ids: Set[str],
) -> list[dict[str, Any]]:
    """Parse strict registry bytes into source-commit-bound manifest records.

    Postcondition: records are unique and sorted by stable workflow ID.
    """
    if not _COMMIT.fullmatch(source_commit):
        raise WorkflowContractError("source commit must be an exact lowercase commit")
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowContractError(
            f"workflow registry is not canonical JSON: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise WorkflowContractError("workflow registry must be an object")
    _exact_keys(
        raw,
        required={"registry_id", "version", "workflows"},
        optional=set(),
        label="workflow registry",
    )
    if raw["registry_id"] != REGISTRY_ID or raw["version"] != REGISTRY_VERSION:
        raise WorkflowContractError("workflow registry identity or version is stale")
    workflows_raw = raw["workflows"]
    if not isinstance(workflows_raw, list) or not workflows_raw:
        raise WorkflowContractError("workflow registry must contain records")
    records = [
        _validate_record(
            item,
            repo_root=repo_root,
            source_commit=source_commit,
            program_ids=program_ids,
        )
        for item in workflows_raw
    ]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise WorkflowContractError("workflow registry contains duplicate IDs")
    failure_classes = [
        record["failure_class"]
        for record in records
        if record["kind"] == "failure-fixture"
    ]
    if set(failure_classes) != _FAILURE_CLASSES or len(failure_classes) != 4:
        raise WorkflowContractError(
            "workflow registry must contain exactly one fixture for each failure class"
        )
    available_successes = [
        record
        for record in records
        if record["kind"] == "workflow"
        and record["availability"]["state"] == "available"
    ]
    if len(available_successes) < 10:
        raise WorkflowContractError(
            "workflow registry requires at least ten executable workflows"
        )
    if not any(record["availability"]["state"] == "unavailable" for record in records):
        raise WorkflowContractError(
            "workflow registry requires an explicit unavailable record"
        )
    return sorted(records, key=lambda record: record["id"])


def referenced_fixture_paths(records: Iterable[Mapping[str, Any]]) -> tuple[Path, ...]:
    """Return unique repository fixture inputs in canonical order."""
    paths = {
        Path(path)
        for record in records
        for path in record["determinism"]["fixture_paths"]
    }
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _snapshot_files(scan_root: Path, *, repo_root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not scan_root.exists():
        return snapshot
    for directory, names, files in os.walk(scan_root):
        names[:] = [name for name in names if name != ".git"]
        base = Path(directory)
        for name in files:
            path = base / name
            relative = path.relative_to(repo_root).as_posix()
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _execution_scope(repo_root: Path, declared_paths: Set[str]) -> Path:
    prefix = "dist/companion-workflows/"
    if declared_paths and all(path.startswith(prefix) for path in declared_paths):
        return repo_root / "dist/companion-workflows"
    return repo_root


def _artifact_path(repo_root: Path, raw: str) -> Path:
    path = (repo_root / raw).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise WorkflowExecutionError(
            f"artifact path escapes repository: {raw}"
        ) from exc
    return path


def _verify_artifact(
    repo_root: Path, artifact: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _artifact_path(repo_root, artifact["path"])
    if not path.is_file():
        if artifact["required"]:
            raise WorkflowExecutionError(
                f"required artifact is missing: {artifact['path']}"
            )
        return None
    payload = path.read_bytes()
    verification = artifact["verification"]
    if len(payload) < verification["minimum_bytes"]:
        raise WorkflowExecutionError(
            f"artifact is smaller than declared: {artifact['path']}"
        )
    method = verification["method"]
    if method == "json-object":
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkflowExecutionError(
                f"artifact is not valid JSON: {artifact['path']}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise WorkflowExecutionError(
                f"JSON artifact must be an object: {artifact['path']}"
            )
    elif method == "csv-header":
        text = payload.decode("utf-8")
        header = next(iter(csv.reader(text.splitlines())), [])
        missing = set(verification["required_columns"]) - set(header)
        if missing:
            raise WorkflowExecutionError(
                f"CSV artifact lacks declared columns {sorted(missing)}: {artifact['path']}"
            )
    elif method == "png-signature":
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise WorkflowExecutionError(
                f"artifact lacks PNG signature: {artifact['path']}"
            )
    elif method == "sha256-sidecar":
        fields = payload.decode("ascii").strip().split()
        target = path.with_suffix("")
        if len(fields) != 2 or fields[1] != target.name or not target.is_file():
            raise WorkflowExecutionError(f"invalid SHA-256 sidecar: {artifact['path']}")
        if fields[0] != hashlib.sha256(target.read_bytes()).hexdigest():
            raise WorkflowExecutionError(
                f"SHA-256 sidecar mismatch: {artifact['path']}"
            )
    return {
        "path": artifact["path"],
        "type": artifact["type"],
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "verification_method": method,
    }


def execute_record(
    repo_root: Path,
    record: Mapping[str, Any],
    *,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Execute one validated manifest record and return artifact evidence.

    Preconditions: the record is source-bound, available, and already schema
    validated. Postconditions: exit code and every changed file match the record.
    """
    root = repo_root.resolve()
    if record["availability"]["state"] != "available":
        raise WorkflowExecutionError(f"workflow is unavailable: {record['id']}")
    if (
        expected_source_commit is not None
        and record["source_commit"] != expected_source_commit
    ):
        raise WorkflowExecutionError(
            f"workflow source commit does not match executor authority: {record['id']}"
        )
    declared_paths = {artifact["path"] for artifact in record["expected_artifacts"]}
    for raw in declared_paths:
        path = _artifact_path(root, raw)
        if path.exists():
            if not path.is_file():
                raise WorkflowExecutionError(f"declared artifact is not a file: {raw}")
            path.unlink()
    scan_root = _execution_scope(root, declared_paths)
    before = _snapshot_files(scan_root, repo_root=root)
    environment = {
        key: os.environ[key]
        for key in record["environment"]["inherit"]
        if key in os.environ
    }
    environment.update(record["environment"]["fixed"])
    python_paths = record["environment"]["python_paths"]
    if python_paths:
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    cwd = root if record["cwd"] == "." else root / record["cwd"]
    command = [sys.executable, "-m", *record["argv"]]
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkflowExecutionError(
            f"workflow {record['id']} exceeded the 120-second execution limit"
        ) from exc
    if completed.returncode != record["expected_exit_code"]:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise WorkflowExecutionError(
            f"workflow {record['id']} exit code {completed.returncode} != "
            f"{record['expected_exit_code']}: {detail}"
        )
    after = _snapshot_files(scan_root, repo_root=root)
    changed = {
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    }
    undeclared = changed - declared_paths
    if undeclared:
        raise WorkflowExecutionError(
            f"workflow {record['id']} produced undeclared outputs: {sorted(undeclared)}"
        )
    evidence = [
        item
        for artifact in record["expected_artifacts"]
        if (item := _verify_artifact(root, artifact)) is not None
    ]
    return {
        "workflow_id": record["id"],
        "kind": record["kind"],
        "expected_exit_code": record["expected_exit_code"],
        "actual_exit_code": completed.returncode,
        "executor": record["executor"],
        "argv": list(record["argv"]),
        "cwd": record["cwd"],
        "artifacts": evidence,
    }


def execute_all(
    repo_root: Path,
    *,
    report_path: Path = DEFAULT_REPORT,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Execute every available record and write canonical aggregate evidence."""
    from scripts import companion_catalog

    root = repo_root.resolve()
    catalog = companion_catalog.build_catalog(root, require_clean=require_clean)
    schema = json.loads(
        (root / "docs/api/contracts/upstreamdrift-companion-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(catalog)
    records = catalog["workflows"]
    source_commit = catalog["source"]["commit"]
    development_skipped = ["companion-export"] if not require_clean else []
    executions = [
        execute_record(root, record, expected_source_commit=source_commit)
        for record in records
        if record["availability"]["state"] == "available"
        and record["id"] not in development_skipped
    ]
    registry_input = next(
        item
        for item in catalog["source"]["inputs"]
        if item["path"] == REGISTRY_PATH.as_posix()
    )
    report = {
        "report_version": "1.0.0",
        "source_commit": source_commit,
        "registry_path": REGISTRY_PATH.as_posix(),
        "registry_sha256": registry_input["sha256"],
        "executed_workflow_ids": [row["workflow_id"] for row in executions],
        "failure_fixture_ids": [
            row["workflow_id"] for row in executions if row["kind"] == "failure-fixture"
        ],
        "unavailable_workflow_ids": [
            record["id"]
            for record in records
            if record["availability"]["state"] == "unavailable"
        ],
        "development_skipped_workflow_ids": development_skipped,
        "executions": executions,
    }
    if require_clean and report_path.is_absolute():
        raise WorkflowExecutionError("authoritative report path must be repo-relative")
    destination = report_path if report_path.is_absolute() else root / report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_all_parser = subparsers.add_parser("execute-all")
    execute_all_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--workflow-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public fail-closed workflow executor."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "execute-all":
            report = execute_all(args.repo_root, report_path=args.report)
            sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
            return 0
        from scripts import companion_catalog

        root = args.repo_root.resolve()
        catalog = companion_catalog.build_catalog(root)
        record = next(
            (row for row in catalog["workflows"] if row["id"] == args.workflow_id),
            None,
        )
        if record is None:
            raise WorkflowContractError(f"unknown workflow ID: {args.workflow_id}")
        evidence = execute_record(
            root, record, expected_source_commit=catalog["source"]["commit"]
        )
        sys.stdout.write(json.dumps(evidence, sort_keys=True) + "\n")
        return 0
    except (
        WorkflowContractError,
        WorkflowExecutionError,
        jsonschema.ValidationError,
    ) as exc:
        sys.stderr.write(f"companion workflow execution refused: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
