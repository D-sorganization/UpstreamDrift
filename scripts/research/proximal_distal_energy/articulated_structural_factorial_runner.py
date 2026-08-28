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

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_CHECKPOINT_SCHEMA = "articulated-structural-factorial-checkpoint/1.0.0"
_LAUNCH_SCHEMA = "articulated-structural-factorial-launch/1.0.0"


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
    return {
        "schema_version": _LAUNCH_SCHEMA,
        "plan_sha256": plan_sha256(plan),
        "execution_revision": execution_revision,
        "registered_case_count": len(cases),
        "worker_count": 1,
        "maximum_python_process_count": 1,
        "checkpoint_policy": "one_atomic_checkpoint_per_attempt",
        "status": "ready_for_disclosed_timing_probe_then_registered_execution",
    }


def _validate_launch(
    *, plan: Mapping[str, object], launch: Mapping[str, object]
) -> tuple[str, str]:
    if launch.get("schema_version") != _LAUNCH_SCHEMA:
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
        or launch.get("registered_case_count") != len(build_registered_cases(plan))
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


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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
    return StructuralCheckpoint(case=case, path=path, status=str(status), resumed=True)


def run_serial_cases(
    *,
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    checkpoint_dir: Path,
    evaluator: Callable[[StructuralCase], Mapping[str, object]],
) -> tuple[StructuralCheckpoint, ...]:
    """Run or resume all cases, retaining only typed native absence."""

    if not isinstance(checkpoint_dir, Path):
        raise TypeError("checkpoint_dir must be a pathlib.Path")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    plan_hash, execution_revision = _validate_launch(plan=plan, launch=launch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[StructuralCheckpoint] = []
    for case in build_registered_cases(plan):
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
            result = evaluator(case)
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
            if not isinstance(result, Mapping):
                raise TypeError("evaluator must return a mapping")
            outcome = {"status": "completed", "result": dict(result)}
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


__all__ = [
    "NativeEngineUnavailable",
    "StructuralCase",
    "StructuralCheckpoint",
    "build_launch_manifest",
    "build_registered_cases",
    "load_registered_checkpoints",
    "plan_sha256",
    "run_serial_cases",
]
