"""Atomic serial checkpoint runner for the preregistered #9153 study."""

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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class StudyCase:
    """One immutable engine, variant, and integration-step combination."""

    source_case_index: int
    source_sample_index: int
    source_time_s: float
    engine: str
    variant: str
    time_step_s: float
    case_key: str


@dataclass(frozen=True, slots=True)
class CaseCheckpoint:
    """Validated retained checkpoint returned in registered case order."""

    case: StudyCase
    path: Path
    status: str
    resumed: bool


class NativeEngineUnavailable(Exception):
    """A registered native operator is absent from the qualified runtime."""

    def __init__(self, *, engine: str, detail: str) -> None:
        if not engine or not detail:
            raise ValueError("engine and detail must be nonempty")
        self.engine = engine
        self.detail = detail
        super().__init__(f"{engine}: {detail}")


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_strings(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    items = tuple(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must contain unique values")
    return items


def _require_steps(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("design.time_steps_s must be a nonempty list")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError("design.time_steps_s must contain numbers")
    steps = tuple(float(item) for item in value)
    if any(not math.isfinite(step) or not (step > 0.0) for step in steps) or any(
        later >= earlier for earlier, later in zip(steps, steps[1:], strict=False)
    ):
        raise ValueError("design.time_steps_s must be positive and decreasing")
    return steps


def _require_smoke_states(value: object) -> tuple[tuple[int, int, float], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("design.smoke_states must be a nonempty list")
    states: list[tuple[int, int, float]] = []
    for slot, item in enumerate(value):
        state = _require_mapping(item, name=f"design.smoke_states[{slot}]")
        case_index = state.get("source_case_index")
        sample_index = state.get("source_sample_index")
        source_time = state.get("source_time_s")
        if (
            isinstance(case_index, bool)
            or not isinstance(case_index, int)
            or case_index < 0
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
            or isinstance(source_time, bool)
            or not isinstance(source_time, (int, float))
            or not math.isfinite(float(source_time))
            or float(source_time) < 0.0
        ):
            raise ValueError("smoke state indices/time must be finite and nonnegative")
        states.append((case_index, sample_index, float(source_time)))
    if len(set(states)) != len(states):
        raise ValueError("design.smoke_states must be unique")
    return tuple(states)


def build_registered_cases(manifest: Mapping[str, object]) -> tuple[StudyCase, ...]:
    """Expand the manifest design in engine, variant, then step order."""

    execution = _require_mapping(manifest.get("execution"), name="execution")
    if execution.get("worker_count") != 1:
        raise ValueError("execution.worker_count must be one")
    if execution.get("case_checkpointing") != "atomic_per_case":
        raise ValueError("execution.case_checkpointing must be atomic_per_case")
    design = _require_mapping(manifest.get("design"), name="design")
    engines = _require_strings(design.get("engines"), name="design.engines")
    variants = _require_strings(design.get("variants"), name="design.variants")
    steps = _require_steps(design.get("time_steps_s"))
    states = _require_smoke_states(design.get("smoke_states"))
    return tuple(
        StudyCase(
            source_case_index=case_index,
            source_sample_index=sample_index,
            source_time_s=source_time,
            engine=engine,
            variant=variant,
            time_step_s=step,
            case_key=(
                f"state=case{case_index}-sample{sample_index}/"
                f"{engine}/{variant}/dt={step:g}"
            ),
        )
        for case_index, sample_index, source_time in states
        for engine in engines
        for variant in variants
        for step in steps
    )


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _study_identity(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    case: StudyCase,
) -> dict[str, str]:
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("schema_version must be a nonempty string")
    identity = _require_mapping(manifest.get("identity"), name="identity")
    source_revision = identity.get("source_revision")
    source_data_sha256 = identity.get("source_data_sha256")
    if not isinstance(source_revision, str) or not _SHA40.fullmatch(source_revision):
        raise ValueError(
            "identity.source_revision must be a lowercase 40-character SHA"
        )
    if not isinstance(source_data_sha256, str) or not _SHA256.fullmatch(
        source_data_sha256
    ):
        raise ValueError("identity.source_data_sha256 must be a lowercase SHA-256")
    if not _SHA40.fullmatch(execution_revision):
        raise ValueError("execution_revision must be a lowercase 40-character SHA")
    return {
        "study_schema_version": schema_version,
        "source_revision": source_revision,
        "source_data_sha256": source_data_sha256,
        "execution_revision": execution_revision,
        "plan_sha256": _canonical_sha256(manifest),
        "case_key": case.case_key,
    }


def _checkpoint_path(checkpoint_dir: Path, case: StudyCase) -> Path:
    digest = hashlib.sha256(case.case_key.encode("utf-8")).hexdigest()[:20]
    return checkpoint_dir / f"case-{digest}.json"


def _case_payload(case: StudyCase) -> dict[str, object]:
    return {
        "source_case_index": case.source_case_index,
        "source_sample_index": case.source_sample_index,
        "source_time_s": case.source_time_s,
        "engine": case.engine,
        "variant": case.variant,
        "time_step_s": case.time_step_s,
    }


def _load_resumed_checkpoint(
    *,
    path: Path,
    case: StudyCase,
    expected_identity: Mapping[str, str],
) -> CaseCheckpoint:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint is unreadable: {path.name}") from exc
    root = _require_mapping(payload, name="checkpoint")
    actual_identity = _require_mapping(root.get("identity"), name="checkpoint.identity")
    for name, expected in expected_identity.items():
        if actual_identity.get(name) != expected:
            raise ValueError(f"checkpoint {name} does not match the requested run")
    if root.get("case") != _case_payload(case):
        raise ValueError("checkpoint case payload does not match its case key")
    outcome = _require_mapping(root.get("outcome"), name="checkpoint.outcome")
    status = outcome.get("status")
    if status not in {"completed", "unavailable", "failed"}:
        raise ValueError("checkpoint outcome status is invalid")
    return CaseCheckpoint(case=case, path=path, status=status, resumed=True)


def _write_atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_serial_cases(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    checkpoint_dir: Path,
    evaluator: Callable[[StudyCase], Mapping[str, object]],
) -> tuple[CaseCheckpoint, ...]:
    """Evaluate every registered case once and retain atomic resumable records.

    Unexpected evaluator exceptions propagate without a completed checkpoint.
    Only the explicit native-unavailability condition is converted into a
    retained typed outcome.
    """

    if not isinstance(checkpoint_dir, Path):
        raise TypeError("checkpoint_dir must be a pathlib.Path")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    cases = build_registered_cases(manifest)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[CaseCheckpoint] = []
    for case in cases:
        identity = _study_identity(
            manifest=manifest,
            execution_revision=execution_revision,
            case=case,
        )
        path = _checkpoint_path(checkpoint_dir, case)
        if path.exists():
            checkpoints.append(
                _load_resumed_checkpoint(
                    path=path,
                    case=case,
                    expected_identity=identity,
                )
            )
            continue
        try:
            result = evaluator(case)
        except NativeEngineUnavailable as exc:
            if exc.engine != case.engine:
                raise ValueError(
                    "native-unavailability engine must match the registered case"
                ) from exc
            outcome: dict[str, object] = {
                "status": "unavailable",
                "failure": {
                    "code": "native_engine_unavailable",
                    "detail": exc.detail,
                    "engine": exc.engine,
                },
            }
        else:
            if not isinstance(result, Mapping):
                raise TypeError("evaluator must return a mapping")
            outcome = {"status": "completed", "result": dict(result)}
        payload = {
            "checkpoint_schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "identity": identity,
            "case": _case_payload(case),
            "outcome": outcome,
        }
        _write_atomic_json(path, payload)
        checkpoints.append(
            CaseCheckpoint(
                case=case,
                path=path,
                status=str(outcome["status"]),
                resumed=False,
            )
        )
    return tuple(checkpoints)


__all__ = [
    "CaseCheckpoint",
    "NativeEngineUnavailable",
    "StudyCase",
    "build_registered_cases",
    "run_serial_cases",
]
