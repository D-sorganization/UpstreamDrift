"""Atomic serial runner for the prospective shaft--ground factorial."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_CHECKPOINT_SCHEMA = "articulated-structural-factorial-checkpoint/1.1.0"
_LEGACY_LAUNCH_SCHEMA = "articulated-structural-factorial-launch/1.1.0"
_ENRICHED_LAUNCH_SCHEMA = "articulated-structural-factorial-launch/1.2.0"
_ENRICHED_EXECUTION_SCHEMAS = {
    "evidence_sidecar_schema": ("articulated-structural-factorial-evidence/1.0.0"),
    "runtime_audit_schema": ("articulated-structural-factorial-runtime-audit/1.4.0"),
    "enrichment_audit_schema": (
        "articulated-structural-factorial-enrichment-audit/1.0.0"
    ),
}


@dataclass(frozen=True, slots=True)
class StructuralCase:
    """One state, pathway cell, direction, engine, and integration step."""

    source_case_index: int
    source_sample_index: int
    source_time_s: float
    cell_id: str
    shaft_activation: str
    ground_activation: str
    velocity_factor: float
    engine: str
    time_step_s: float
    case_key: str


@dataclass(frozen=True, slots=True)
class StructuralCheckpoint:
    """One validated checkpoint in registered case order."""

    case: StructuralCase
    path: Path
    status: str
    resumed: bool


@dataclass(frozen=True, slots=True)
class StructuralEvaluation:
    """JSON summary plus compressed arrays required for engine parity."""

    result: Mapping[str, object]
    parity_arrays: Mapping[str, NDArray[Any]]


class NativeEngineUnavailable(Exception):
    """A registered native engine is unavailable in the execution runtime."""

    def __init__(self, *, engine: str, detail: str) -> None:
        if not engine or not detail:
            raise ValueError("engine and detail must be nonempty")
        self.engine = engine
        self.detail = detail
        super().__init__(f"{engine}: {detail}")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_sha256(plan: Mapping[str, object]) -> str:
    """Return the canonical plan identity used by launch and checkpoints."""

    return _canonical_sha256(plan)


def _strings(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _numbers(
    value: object, *, name: str, decreasing: bool = False
) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{name} must contain numbers")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item == 0.0 for item in result):
        raise ValueError(f"{name} must contain finite nonzero numbers")
    if decreasing and any(
        later >= earlier for earlier, later in zip(result, result[1:], strict=False)
    ):
        raise ValueError(f"{name} must be strictly decreasing")
    return result


def _states(value: object) -> tuple[tuple[int, int, float], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("design.states must be a nonempty list")
    result: list[tuple[int, int, float]] = []
    for index, raw in enumerate(value):
        state = _mapping(raw, name=f"design.states[{index}]")
        case = state.get("source_case_index")
        sample = state.get("source_sample_index")
        time_s = state.get("source_time_s")
        if (
            isinstance(case, bool)
            or not isinstance(case, int)
            or case < 0
            or isinstance(sample, bool)
            or not isinstance(sample, int)
            or sample < 0
            or isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not math.isfinite(float(time_s))
            or float(time_s) < 0.0
        ):
            raise ValueError("registered state values must be finite and nonnegative")
        result.append((case, sample, float(time_s)))
    if len(set(result)) != len(result):
        raise ValueError("design.states must be unique")
    return tuple(result)


def _cells(value: object) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("design.factorial_cells must be a nonempty list")
    result: list[tuple[str, str, str]] = []
    for index, raw in enumerate(value):
        cell = _mapping(raw, name=f"design.factorial_cells[{index}]")
        cell_id = cell.get("cell_id")
        shaft = cell.get("shaft_activation")
        ground = cell.get("ground_activation")
        if (
            not isinstance(cell_id, str)
            or not re.fullmatch(r"[01]{4}", cell_id)
            or shaft not in {"rigid", "bending", "torsion", "coupled"}
            or ground not in {"fixed", "translation", "free_moment", "coupled"}
        ):
            raise ValueError("factorial cell identity or activation is invalid")
        result.append((cell_id, str(shaft), str(ground)))
    if len(set(result)) != len(result):
        raise ValueError("design.factorial_cells must be unique")
    return tuple(result)


def build_registered_cases(plan: Mapping[str, object]) -> tuple[StructuralCase, ...]:
    """Expand the complete deterministic case matrix."""

    design = _mapping(plan.get("design"), name="design")
    states = _states(design.get("states"))
    cells = _cells(design.get("factorial_cells"))
    velocities = _numbers(
        design.get("velocity_factors"), name="design.velocity_factors"
    )
    engines = _strings(design.get("engines"), name="design.engines")
    steps = _numbers(
        design.get("time_steps_s"), name="design.time_steps_s", decreasing=True
    )
    return tuple(
        StructuralCase(
            source_case_index=case,
            source_sample_index=sample,
            source_time_s=time_s,
            cell_id=cell_id,
            shaft_activation=shaft,
            ground_activation=ground,
            velocity_factor=velocity,
            engine=engine,
            time_step_s=step,
            case_key=(
                f"state=case{case}-sample{sample}/cell={cell_id}/"
                f"v={velocity:+g}/{engine}/dt={step:g}"
            ),
        )
        for case, sample, time_s in states
        for cell_id, shaft, ground in cells
        for velocity in velocities
        for engine in engines
        for step in steps
    )


def _launch_contract(
    plan: Mapping[str, object], execution: Mapping[str, object]
) -> tuple[str, dict[str, str]]:
    plan_schema = plan.get("schema_version")
    if plan_schema == "articulated-structural-factorial-plan/1.2.0":
        return _LEGACY_LAUNCH_SCHEMA, {}
    if plan_schema != "articulated-structural-factorial-plan/1.3.0":
        raise ValueError("structural plan schema is not launchable")
    if any(
        execution.get(name) != expected
        for name, expected in _ENRICHED_EXECUTION_SCHEMAS.items()
    ):
        raise ValueError("enriched execution schema identity is invalid")
    preregistration = _mapping(plan.get("preregistration"), name="preregistration")
    amendment = _mapping(
        preregistration.get("operational_amendment"),
        name="preregistration.operational_amendment",
    )
    if (
        amendment.get("detected_before_scientific_outcome_inspection") is not True
        or amendment.get("registered_design_or_gate_change") is not False
        or amendment.get("legacy_prefix_promotable") is not False
        or amendment.get("legacy_revision_resume_permitted") is not False
        or amendment.get("requires_full_legacy_prefix_replay") is not True
        or amendment.get("requires_exact_enrichment_audit") is not True
    ):
        raise ValueError("enriched retention amendment is not fail-closed")
    return _ENRICHED_LAUNCH_SCHEMA, dict(_ENRICHED_EXECUTION_SCHEMAS)


def build_launch_manifest(
    *, plan: Mapping[str, object], execution_revision: str
) -> dict[str, object]:
    """Bind the frozen plan to one immutable runner revision."""

    if not _SHA40.fullmatch(execution_revision):
        raise ValueError("execution_revision must be a lowercase 40-character SHA")
    execution = _mapping(plan.get("execution"), name="execution")
    if execution.get("worker_count") != 1:
        raise ValueError("plan execution.worker_count must be one")
    if execution.get("maximum_python_process_count") != 1:
        raise ValueError("maximum_python_process_count must be one")
    analysis = _mapping(plan.get("analysis"), name="analysis")
    if analysis.get("outcome_matching") != "prohibited":
        raise ValueError("outcome matching must remain prohibited")
    cases = build_registered_cases(plan)
    launch_schema, enriched_identity = _launch_contract(plan, execution)
    return {
        "schema_version": launch_schema,
        "plan_sha256": plan_sha256(plan),
        "execution_revision": execution_revision,
        "registered_case_count": len(cases),
        "worker_count": 1,
        "maximum_python_process_count": 1,
        "checkpoint_policy": "one_atomic_checkpoint_per_attempt",
        "parity_sidecar_policy": "one_sha256_bound_npz_per_completed_attempt",
        "status": "ready_for_disclosed_timing_probe_then_registered_execution",
        **enriched_identity,
    }


def _validate_launch(
    *, plan: Mapping[str, object], launch: Mapping[str, object]
) -> tuple[str, str]:
    execution = _mapping(plan.get("execution"), name="execution")
    expected_schema, enriched_identity = _launch_contract(plan, execution)
    if launch.get("schema_version") != expected_schema:
        raise ValueError("launch identity schema is invalid")
    expected_plan = plan_sha256(plan)
    execution_revision = launch.get("execution_revision")
    if (
        launch.get("plan_sha256") != expected_plan
        or not isinstance(execution_revision, str)
        or not _SHA40.fullmatch(execution_revision)
        or launch.get("worker_count") != 1
        or launch.get("maximum_python_process_count") != 1
        or launch.get("checkpoint_policy") != "one_atomic_checkpoint_per_attempt"
        or launch.get("parity_sidecar_policy")
        != "one_sha256_bound_npz_per_completed_attempt"
        or launch.get("registered_case_count") != len(build_registered_cases(plan))
        or any(
            launch.get(name) != expected for name, expected in enriched_identity.items()
        )
    ):
        raise ValueError("launch identity does not match the frozen plan")
    return expected_plan, execution_revision


def _case_payload(case: StructuralCase) -> dict[str, object]:
    return {
        "source_case_index": case.source_case_index,
        "source_sample_index": case.source_sample_index,
        "source_time_s": case.source_time_s,
        "cell_id": case.cell_id,
        "shaft_activation": case.shaft_activation,
        "ground_activation": case.ground_activation,
        "velocity_factor": case.velocity_factor,
        "engine": case.engine,
        "time_step_s": case.time_step_s,
    }


def _path(directory: Path, case: StructuralCase) -> Path:
    digest = hashlib.sha256(case.case_key.encode("utf-8")).hexdigest()[:20]
    return directory / f"case-{digest}.json"


def checkpoint_path(directory: Path, case: StructuralCase) -> Path:
    """Return the deterministic JSON path for one registered case."""

    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path")
    return _path(directory, case)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".npz")


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_atomic_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> str:
    if not arrays:
        raise ValueError("completed evaluation must contain parity arrays")
    normalized: dict[str, NDArray[Any]] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parity array names must be nonempty strings")
        array = np.asarray(value)
        if array.dtype.hasobject or array.ndim < 1:
            raise ValueError("parity arrays must be non-object arrays with dimensions")
        if np.issubdtype(array.dtype, np.number) and np.any(~np.isfinite(array)):
            raise ValueError("numeric parity arrays must be finite")
        normalized[name] = array
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **normalized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(
    *,
    path: Path,
    case: StructuralCase,
    plan_hash: str,
    execution_revision: str,
) -> StructuralCheckpoint:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint is unreadable: {path.name}") from exc
    root = _mapping(payload, name="checkpoint")
    identity = _mapping(root.get("identity"), name="checkpoint.identity")
    if (
        root.get("checkpoint_schema_version") != _CHECKPOINT_SCHEMA
        or identity.get("plan_sha256") != plan_hash
        or identity.get("execution_revision") != execution_revision
        or identity.get("case_key") != case.case_key
        or root.get("case") != _case_payload(case)
    ):
        raise ValueError("checkpoint launch identity does not match the requested run")
    outcome = _mapping(root.get("outcome"), name="checkpoint.outcome")
    status = outcome.get("status")
    if status not in {"completed", "unavailable", "failed"}:
        raise ValueError("checkpoint outcome status is invalid")
    sidecar = _sidecar_path(path)
    if status == "completed":
        metadata = _mapping(outcome.get("parity_sidecar"), name="parity_sidecar")
        if (
            metadata.get("path") != sidecar.name
            or not sidecar.is_file()
            or metadata.get("sha256")
            != hashlib.sha256(sidecar.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "completed checkpoint parity sidecar is missing or corrupt"
            )
    elif sidecar.exists():
        raise ValueError("non-completed checkpoint must not have a parity sidecar")
    return StructuralCheckpoint(case=case, path=path, status=str(status), resumed=True)


def run_serial_cases(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
    evaluator: Callable[[StructuralCase], StructuralEvaluation],
    case_start: int = 0,
    case_stop: int | None = None,
) -> tuple[StructuralCheckpoint, ...]:
    """Run or resume one registered half-open slice, retaining typed absence."""

    if not isinstance(checkpoint_dir, Path):
        raise TypeError("checkpoint_dir must be a pathlib.Path")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    plan_hash, execution_revision = _validate_launch(plan=plan, launch=launch)
    registered_cases = build_registered_cases(plan)
    resolved_stop = len(registered_cases) if case_stop is None else case_stop
    if (
        isinstance(case_start, bool)
        or not isinstance(case_start, int)
        or isinstance(resolved_stop, bool)
        or not isinstance(resolved_stop, int)
        or case_start < 0
        or resolved_stop <= case_start
        or resolved_stop > len(registered_cases)
    ):
        raise ValueError("case slice must satisfy 0 <= start < stop <= case count")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[StructuralCheckpoint] = []
    for case in registered_cases[case_start:resolved_stop]:
        path = _path(checkpoint_dir, case)
        if path.exists():
            checkpoints.append(
                _load(
                    path=path,
                    case=case,
                    plan_hash=plan_hash,
                    execution_revision=execution_revision,
                )
            )
            continue
        try:
            evaluation = evaluator(case)
        except NativeEngineUnavailable as exc:
            if exc.engine != case.engine:
                raise ValueError(
                    "typed unavailable engine does not match case"
                ) from exc
            outcome: dict[str, object] = {
                "status": "unavailable",
                "failure": {
                    "code": "native_engine_unavailable",
                    "engine": exc.engine,
                    "detail": exc.detail,
                },
            }
        else:
            if not isinstance(evaluation, StructuralEvaluation):
                raise TypeError("evaluator must return a StructuralEvaluation")
            sidecar = _sidecar_path(path)
            sidecar_sha256 = _write_atomic_npz(sidecar, evaluation.parity_arrays)
            outcome = {
                "status": "completed",
                "result": dict(evaluation.result),
                "parity_sidecar": {"path": sidecar.name, "sha256": sidecar_sha256},
            }
        _write_atomic(
            path,
            {
                "checkpoint_schema_version": _CHECKPOINT_SCHEMA,
                "identity": {
                    "plan_sha256": plan_hash,
                    "execution_revision": execution_revision,
                    "case_key": case.case_key,
                },
                "case": _case_payload(case),
                "outcome": outcome,
            },
        )
        checkpoints.append(
            StructuralCheckpoint(
                case=case,
                path=path,
                status=str(outcome["status"]),
                resumed=False,
            )
        )
    return tuple(checkpoints)


def load_registered_checkpoints(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
) -> tuple[StructuralCheckpoint, ...]:
    """Load a complete run while revalidating every identity."""

    plan_hash, execution_revision = _validate_launch(plan=plan, launch=launch)
    checkpoints = []
    for case in build_registered_cases(plan):
        path = _path(checkpoint_dir, case)
        if not path.is_file():
            raise FileNotFoundError(
                f"registered checkpoint is missing: {case.case_key}"
            )
        checkpoints.append(
            _load(
                path=path,
                case=case,
                plan_hash=plan_hash,
                execution_revision=execution_revision,
            )
        )
    return tuple(checkpoints)


def load_available_checkpoints(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
) -> tuple[StructuralCheckpoint, ...]:
    """Validate every checkpoint currently visible without requiring completion."""

    plan_hash, execution_revision = _validate_launch(plan=plan, launch=launch)
    cases = build_registered_cases(plan)
    known_json = {_path(checkpoint_dir, case).name for case in cases}
    known_npz = {Path(name).with_suffix(".npz").name for name in known_json}
    visible_json = {path.name for path in checkpoint_dir.glob("case-*.json")}
    visible_npz = {path.name for path in checkpoint_dir.glob("case-*.npz")}
    unknown = (visible_json - known_json) | (visible_npz - known_npz)
    if unknown:
        raise ValueError("checkpoint directory contains unregistered case files")
    checkpoints = []
    for case in cases:
        path = _path(checkpoint_dir, case)
        if path.is_file():
            checkpoints.append(
                _load(
                    path=path,
                    case=case,
                    plan_hash=plan_hash,
                    execution_revision=execution_revision,
                )
            )
    return tuple(checkpoints)


__all__ = [
    "NativeEngineUnavailable",
    "StructuralCase",
    "StructuralCheckpoint",
    "StructuralEvaluation",
    "build_launch_manifest",
    "build_registered_cases",
    "checkpoint_path",
    "load_available_checkpoints",
    "load_registered_checkpoints",
    "plan_sha256",
    "run_serial_cases",
]
