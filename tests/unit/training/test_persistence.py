"""Round-trip tests for :mod:`training.persistence`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    CURRENT_SCHEMA_VERSION,
    MetricKind,
    ResourceRequest,
    RunResult,
    TrainingConfig,
    TrainingConfigError,
    TrainingFramework,
    TrainingJob,
    TrainingMetric,
    TrainingStatus,
    new_job_id,
    new_run_id,
    run_result_from_dict,
    run_result_to_dict,
    training_config_from_dict,
    training_config_to_dict,
    training_job_from_dict,
    training_job_to_dict,
    training_metric_from_dict,
    training_metric_to_dict,
)

pytestmark = pytest.mark.unit


class TestTrainingConfigRoundTrip:
    def _config(self) -> TrainingConfig:
        return TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="my_module:train",
            output_dir=Path("/tmp/out"),
            hyperparameters={"lr": 1e-3, "batch_size": 32},
            dataset_id="ds-001",
            resources=ResourceRequest(
                cpu_cores=4, gpu_count=1, memory_mb=8192, gpu_memory_mb=12000
            ),
            max_epochs=10,
            seed=42,
            tags={"owner": "claude", "experiment": "alpha"},
        )

    def test_round_trip_preserves_equality(self) -> None:
        original = self._config()
        rebuilt = training_config_from_dict(training_config_to_dict(original))
        assert rebuilt == original

    def test_to_dict_is_json_serializable(self) -> None:
        config = self._config()
        data = training_config_to_dict(config)
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        rebuilt = training_config_from_dict(decoded)
        assert rebuilt == config

    def test_schema_version_embedded(self) -> None:
        data = training_config_to_dict(self._config())
        assert data["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_rejects_newer_schema_version(self) -> None:
        data = training_config_to_dict(self._config())
        data["schema_version"] = CURRENT_SCHEMA_VERSION + 1
        with pytest.raises(TrainingConfigError, match="newer"):
            training_config_from_dict(data)

    def test_rejects_unknown_framework(self) -> None:
        data = training_config_to_dict(self._config())
        data["framework"] = "tensorflow"
        with pytest.raises(TrainingConfigError, match="unknown framework"):
            training_config_from_dict(data)

    def test_rejects_non_dict_input(self) -> None:
        with pytest.raises(TrainingConfigError):
            training_config_from_dict("not a dict")  # type: ignore[arg-type]


class TestTrainingMetricRoundTrip:
    def test_round_trip(self) -> None:
        metric = TrainingMetric(
            name="val_loss",
            value=0.123,
            step=5,
            timestamp=1700_000_000.0,
            kind=MetricKind.LOSS,
            tags={"split": "val"},
        )
        rebuilt = training_metric_from_dict(training_metric_to_dict(metric))
        assert rebuilt == metric

    def test_default_kind_round_trip(self) -> None:
        metric = TrainingMetric(name="x", value=1.0, step=0, timestamp=0.0)
        rebuilt = training_metric_from_dict(training_metric_to_dict(metric))
        assert rebuilt.kind is MetricKind.SCALAR

    def test_rejects_unknown_kind(self) -> None:
        data = training_metric_to_dict(
            TrainingMetric(name="x", value=1.0, step=0, timestamp=0.0)
        )
        data["kind"] = "unknown_kind"
        with pytest.raises(TrainingConfigError, match="unknown metric kind"):
            training_metric_from_dict(data)


class TestTrainingJobRoundTrip:
    def test_pending_job_round_trip(self) -> None:
        cfg = TrainingConfig(
            framework=TrainingFramework.GYMNASIUM,
            entry_point="m:train",
            output_dir=Path("/tmp/out"),
        )
        job = TrainingJob(
            job_id=new_job_id(),
            config=cfg,
            status=TrainingStatus.PENDING,
            created_at=100.0,
        )
        rebuilt = training_job_from_dict(training_job_to_dict(job))
        assert rebuilt == job

    def test_running_job_round_trip(self) -> None:
        cfg = TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="m:train",
            output_dir=Path("/tmp/out"),
        )
        job = TrainingJob(
            job_id=new_job_id(),
            config=cfg,
            status=TrainingStatus.RUNNING,
            created_at=100.0,
            started_at=110.0,
            run_id=new_run_id(),
        )
        rebuilt = training_job_from_dict(training_job_to_dict(job))
        assert rebuilt == job

    def test_failed_job_round_trip(self) -> None:
        cfg = TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="m:train",
            output_dir=Path("/tmp/out"),
        )
        job = TrainingJob(
            job_id=new_job_id(),
            config=cfg,
            status=TrainingStatus.FAILED,
            created_at=100.0,
            started_at=110.0,
            completed_at=120.0,
            error_message="CUDA OOM",
            run_id=new_run_id(),
        )
        rebuilt = training_job_from_dict(training_job_to_dict(job))
        assert rebuilt == job

    def test_rejects_unknown_status(self) -> None:
        cfg = TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="m:train",
            output_dir=Path("/tmp/out"),
        )
        job = TrainingJob(
            job_id=new_job_id(),
            config=cfg,
            status=TrainingStatus.PENDING,
            created_at=100.0,
        )
        data = training_job_to_dict(job)
        data["status"] = "imaginary"
        with pytest.raises(TrainingConfigError, match="unknown status"):
            training_job_from_dict(data)


class TestRunResultRoundTrip:
    def test_completed_round_trip(self) -> None:
        result = RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=12.5,
            final_metrics=(
                TrainingMetric(
                    name="final_loss",
                    value=0.05,
                    step=10,
                    timestamp=1.0,
                    kind=MetricKind.LOSS,
                ),
            ),
            artifacts=(Path("/tmp/checkpoint.pt"),),
        )
        rebuilt = run_result_from_dict(run_result_to_dict(result))
        assert rebuilt == result

    def test_failed_round_trip(self) -> None:
        result = RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.FAILED,
            duration_s=1.0,
            error="OOM",
        )
        rebuilt = run_result_from_dict(run_result_to_dict(result))
        assert rebuilt == result

    def test_cancelled_round_trip(self) -> None:
        result = RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.CANCELLED,
            duration_s=0.5,
        )
        rebuilt = run_result_from_dict(run_result_to_dict(result))
        assert rebuilt == result
