"""JSON persistence for training-controller domain types.

PR2 lives entirely in-process; PR3 spawns subprocess workers that
exchange these dicts over stdin/stdout, and PR5 writes them to durable
storage. Keeping serialization in one module means new fields land in
one place and the wire format stays in lock-step with the schema
version on :class:`TrainingConfig`.

Round-trip invariant: ``from_dict(to_dict(x)) == x`` for every
supported type — locked in by tests in
``tests/unit/training/test_persistence.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import CURRENT_SCHEMA_VERSION, TrainingConfig, TrainingFramework
from .errors import TrainingConfigError
from .identifiers import JobId, RunId
from .job import RunResult, TrainingJob
from .metrics import MetricKind, TrainingMetric
from .resources import ResourceRequest
from .status import TrainingStatus

__all__ = [
    "run_result_from_dict",
    "run_result_to_dict",
    "training_config_from_dict",
    "training_config_to_dict",
    "training_job_from_dict",
    "training_job_to_dict",
    "training_metric_from_dict",
    "training_metric_to_dict",
]


def _require_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrainingConfigError(
            f"{field_name} must be an int (got {type(value).__name__})"
        )
    return value


def _require_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TrainingConfigError(
            f"{field_name} must be a string (got {type(value).__name__})"
        )
    return value


def _resources_to_dict(req: ResourceRequest) -> dict[str, Any]:
    return {
        "cpu_cores": req.cpu_cores,
        "gpu_count": req.gpu_count,
        "memory_mb": req.memory_mb,
        "gpu_memory_mb": req.gpu_memory_mb,
    }


def _resources_from_dict(data: dict[str, Any]) -> ResourceRequest:
    return ResourceRequest(
        cpu_cores=int(data.get("cpu_cores", 1)),
        gpu_count=int(data.get("gpu_count", 0)),
        memory_mb=int(data.get("memory_mb", 1024)),
        gpu_memory_mb=(
            int(data["gpu_memory_mb"])
            if data.get("gpu_memory_mb") is not None
            else None
        ),
    )


def training_config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Serialize a :class:`TrainingConfig` to a JSON-safe dict.

    The dict embeds :data:`schema_version` so the reverse path can
    detect and reject incompatible versions explicitly.
    """

    if not isinstance(config, TrainingConfig):
        raise TypeError(f"expected TrainingConfig (got {type(config).__name__})")
    return {
        "schema_version": config.schema_version,
        "framework": config.framework.value,
        "entry_point": config.entry_point,
        "output_dir": str(config.output_dir),
        "hyperparameters": dict(config.hyperparameters),
        "dataset_id": config.dataset_id,
        "resources": _resources_to_dict(config.resources),
        "max_epochs": config.max_epochs,
        "max_steps": config.max_steps,
        "seed": config.seed,
        "tags": dict(config.tags),
    }


def training_config_from_dict(data: dict[str, Any]) -> TrainingConfig:
    """Rebuild a :class:`TrainingConfig` from its dict form.

    Raises:
        TrainingConfigError: When the schema version is newer than
            this build understands, or when any field fails validation.
    """

    if not isinstance(data, dict):
        raise TrainingConfigError(
            f"training_config_from_dict expected a dict (got {type(data).__name__})"
        )
    version = _require_int(
        data.get("schema_version", CURRENT_SCHEMA_VERSION),
        field_name="schema_version",
    )
    if version > CURRENT_SCHEMA_VERSION:
        raise TrainingConfigError(
            f"training config schema_version={version} is newer than this build "
            f"supports (max={CURRENT_SCHEMA_VERSION}); upgrade the training package"
        )
    framework_raw = _require_str(data["framework"], field_name="framework")
    try:
        framework = TrainingFramework(framework_raw)
    except ValueError as exc:
        raise TrainingConfigError(f"unknown framework {framework_raw!r}") from exc
    return TrainingConfig(
        framework=framework,
        entry_point=_require_str(data["entry_point"], field_name="entry_point"),
        output_dir=Path(_require_str(data["output_dir"], field_name="output_dir")),
        schema_version=version,
        hyperparameters=dict(data.get("hyperparameters") or {}),
        dataset_id=data.get("dataset_id"),
        resources=_resources_from_dict(data.get("resources") or {}),
        max_epochs=data.get("max_epochs"),
        max_steps=data.get("max_steps"),
        seed=data.get("seed"),
        tags=dict(data.get("tags") or {}),
    )


def training_metric_to_dict(metric: TrainingMetric) -> dict[str, Any]:
    return {
        "name": metric.name,
        "value": metric.value,
        "step": metric.step,
        "timestamp": metric.timestamp,
        "kind": metric.kind.value,
        "tags": dict(metric.tags),
    }


def training_metric_from_dict(data: dict[str, Any]) -> TrainingMetric:
    if not isinstance(data, dict):
        raise TrainingConfigError(
            f"training_metric_from_dict expected a dict (got {type(data).__name__})"
        )
    kind_raw = data.get("kind", MetricKind.SCALAR.value)
    try:
        kind = MetricKind(kind_raw)
    except ValueError as exc:
        raise TrainingConfigError(f"unknown metric kind {kind_raw!r}") from exc
    return TrainingMetric(
        name=_require_str(data["name"], field_name="name"),
        value=float(data["value"]),
        step=_require_int(data["step"], field_name="step"),
        timestamp=float(data["timestamp"]),
        kind=kind,
        tags=dict(data.get("tags") or {}),
    )


def training_job_to_dict(job: TrainingJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id.value,
        "config": training_config_to_dict(job.config),
        "status": job.status.value,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message,
        "run_id": job.run_id.value if job.run_id is not None else None,
    }


def training_job_from_dict(data: dict[str, Any]) -> TrainingJob:
    if not isinstance(data, dict):
        raise TrainingConfigError(
            f"training_job_from_dict expected a dict (got {type(data).__name__})"
        )
    status_raw = _require_str(data["status"], field_name="status")
    try:
        status = TrainingStatus(status_raw)
    except ValueError as exc:
        raise TrainingConfigError(f"unknown status {status_raw!r}") from exc
    run_id_value = data.get("run_id")
    return TrainingJob(
        job_id=JobId(_require_str(data["job_id"], field_name="job_id")),
        config=training_config_from_dict(data["config"]),
        status=status,
        created_at=float(data["created_at"]),
        started_at=(
            float(data["started_at"]) if data.get("started_at") is not None else None
        ),
        completed_at=(
            float(data["completed_at"])
            if data.get("completed_at") is not None
            else None
        ),
        error_message=data.get("error_message"),
        run_id=RunId(run_id_value) if run_id_value is not None else None,
    )


def run_result_to_dict(result: RunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id.value,
        "status": result.status.value,
        "duration_s": result.duration_s,
        "final_metrics": [training_metric_to_dict(m) for m in result.final_metrics],
        "artifacts": [str(p) for p in result.artifacts],
        "error": result.error,
    }


def run_result_from_dict(data: dict[str, Any]) -> RunResult:
    if not isinstance(data, dict):
        raise TrainingConfigError(
            f"run_result_from_dict expected a dict (got {type(data).__name__})"
        )
    status_raw = _require_str(data["status"], field_name="status")
    try:
        status = TrainingStatus(status_raw)
    except ValueError as exc:
        raise TrainingConfigError(f"unknown status {status_raw!r}") from exc
    return RunResult(
        run_id=RunId(_require_str(data["run_id"], field_name="run_id")),
        status=status,
        duration_s=float(data["duration_s"]),
        final_metrics=tuple(
            training_metric_from_dict(m) for m in (data.get("final_metrics") or ())
        ),
        artifacts=tuple(Path(p) for p in (data.get("artifacts") or ())),
        error=data.get("error"),
    )
