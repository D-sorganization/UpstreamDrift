"""Design-by-Contract tests for :mod:`training_controller.view_model`."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import (
    JobId,
    TrainingConfig,
    TrainingFramework,
    TrainingJob,
    TrainingStatus,
)
from training.metrics import MetricKind
from src.tools.training_controller.view_model import (
    DashboardModel,
    GpuSnapshot,
    JobRow,
    MetricSeries,
    ResourceSnapshot,
    job_row_from_training_job,
)


pytestmark = pytest.mark.unit


def _config() -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="m:train",
        output_dir=Path("/tmp/training-controller-tests"),
        dataset_id="dataset-1",
    )


def _job(
    *,
    status: TrainingStatus = TrainingStatus.PENDING,
    started_at: float | None = None,
    completed_at: float | None = None,
    error_message: str | None = None,
) -> TrainingJob:
    return TrainingJob(
        job_id=JobId("job-1"),
        config=_config(),
        status=status,
        created_at=100.0,
        started_at=started_at,
        completed_at=completed_at,
        error_message=error_message,
    )


# --------------------------------------------------------------------- JobRow


class TestJobRow:
    def test_valid_row(self) -> None:
        row = JobRow(
            job_id="job-1",
            framework="pytorch",
            status="running",
            dataset_id="dataset-1",
            elapsed_s=42.0,
        )
        assert row.elapsed_s == 42.0
        assert row.error_message is None

    @pytest.mark.parametrize("bad_id", ["", 7, None])
    def test_rejects_bad_job_id(self, bad_id: object) -> None:
        with pytest.raises((ValueError, TypeError)):
            JobRow(
                job_id=bad_id,  # type: ignore[arg-type]
                framework="pytorch",
                status="running",
                dataset_id=None,
                elapsed_s=0.0,
            )

    def test_rejects_negative_elapsed(self) -> None:
        with pytest.raises(ValueError):
            JobRow(
                job_id="job-1",
                framework="pytorch",
                status="running",
                dataset_id=None,
                elapsed_s=-1.0,
            )

    def test_rejects_empty_dataset_id(self) -> None:
        with pytest.raises(ValueError):
            JobRow(
                job_id="job-1",
                framework="pytorch",
                status="running",
                dataset_id="",
                elapsed_s=0.0,
            )

    def test_rejects_empty_error_message(self) -> None:
        with pytest.raises(ValueError):
            JobRow(
                job_id="job-1",
                framework="pytorch",
                status="failed",
                dataset_id=None,
                elapsed_s=0.0,
                error_message="",
            )


class TestJobRowFromTrainingJob:
    def test_pending_job_has_zero_elapsed(self) -> None:
        row = job_row_from_training_job(_job(), now=999.0)
        assert row.elapsed_s == 0.0
        assert row.status == "pending"
        assert row.dataset_id == "dataset-1"

    def test_running_job_uses_now_minus_started(self) -> None:
        job = _job(status=TrainingStatus.RUNNING, started_at=200.0)
        row = job_row_from_training_job(job, now=350.0)
        assert row.elapsed_s == pytest.approx(150.0)

    def test_terminal_job_uses_completed_minus_started(self) -> None:
        job = _job(
            status=TrainingStatus.COMPLETED,
            started_at=200.0,
            completed_at=275.0,
        )
        row = job_row_from_training_job(job, now=999.0)
        assert row.elapsed_s == pytest.approx(75.0)

    def test_failed_job_passes_error_message(self) -> None:
        job = _job(
            status=TrainingStatus.FAILED,
            started_at=200.0,
            completed_at=210.0,
            error_message="boom",
        )
        row = job_row_from_training_job(job, now=999.0)
        assert row.error_message == "boom"

    def test_negative_now_rejected(self) -> None:
        with pytest.raises(ValueError):
            job_row_from_training_job(_job(), now=-1.0)

    def test_non_job_rejected(self) -> None:
        with pytest.raises(TypeError):
            job_row_from_training_job("not a job", now=1.0)  # type: ignore[arg-type]


# ----------------------------------------------------------------- MetricSeries


class TestMetricSeries:
    def test_valid_series(self) -> None:
        series = MetricSeries(
            name="loss",
            kind=MetricKind.LOSS,
            steps=(0, 1, 2),
            values=(1.0, 0.5, 0.25),
        )
        assert series.smoothed is None
        assert series.values == (1.0, 0.5, 0.25)

    def test_smoothed_must_match_length(self) -> None:
        with pytest.raises(ValueError):
            MetricSeries(
                name="reward",
                kind=MetricKind.REWARD,
                steps=(0, 1),
                values=(1.0, 2.0),
                smoothed=(1.0, 1.5, 2.0),
            )

    def test_steps_and_values_must_match(self) -> None:
        with pytest.raises(ValueError):
            MetricSeries(
                name="loss",
                kind=MetricKind.LOSS,
                steps=(0, 1, 2),
                values=(1.0, 0.5),
            )

    def test_negative_step_rejected(self) -> None:
        with pytest.raises(ValueError):
            MetricSeries(
                name="loss",
                kind=MetricKind.LOSS,
                steps=(0, -1),
                values=(1.0, 0.5),
            )

    def test_name_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError):
            MetricSeries(
                name="",
                kind=MetricKind.LOSS,
                steps=(),
                values=(),
            )

    def test_kind_must_be_metric_kind(self) -> None:
        with pytest.raises(TypeError):
            MetricSeries(
                name="loss",
                kind="loss",  # type: ignore[arg-type]
                steps=(0,),
                values=(1.0,),
            )


# ---------------------------------------------------------------- GpuSnapshot


class TestGpuSnapshot:
    def test_valid_snapshot(self) -> None:
        snap = GpuSnapshot(
            index=0,
            name="A100",
            utilization_percent=75.0,
            memory_used_mb=1024,
            memory_total_mb=8192,
        )
        assert snap.name == "A100"

    def test_none_utilization_allowed(self) -> None:
        GpuSnapshot(
            index=0,
            name="A100",
            utilization_percent=None,
            memory_used_mb=0,
            memory_total_mb=8192,
        )

    def test_utilization_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            GpuSnapshot(
                index=0,
                name="A100",
                utilization_percent=150.0,
                memory_used_mb=0,
                memory_total_mb=8192,
            )

    def test_used_exceeds_total_rejected(self) -> None:
        with pytest.raises(ValueError):
            GpuSnapshot(
                index=0,
                name="A100",
                utilization_percent=0.0,
                memory_used_mb=9000,
                memory_total_mb=8192,
            )


# ------------------------------------------------------------- ResourceSnapshot


class TestResourceSnapshot:
    def test_available_snapshot(self) -> None:
        snap = ResourceSnapshot(cpu_percent=12.5, memory_percent=40.0)
        assert snap.available is True
        assert snap.gpus == ()

    def test_unavailable_helper(self) -> None:
        snap = ResourceSnapshot.unavailable()
        assert snap.available is False
        assert snap.cpu_percent is None
        assert snap.memory_percent is None
        assert snap.gpus == ()

    def test_unavailable_with_populated_field_rejected(self) -> None:
        with pytest.raises(ValueError):
            ResourceSnapshot(
                cpu_percent=10.0,
                memory_percent=None,
                gpus=(),
                available=False,
            )

    def test_cpu_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            ResourceSnapshot(cpu_percent=120.0, memory_percent=20.0)

    def test_non_gpu_snapshot_in_tuple_rejected(self) -> None:
        with pytest.raises(TypeError):
            ResourceSnapshot(
                cpu_percent=1.0,
                memory_percent=1.0,
                gpus=("not-a-gpu",),  # type: ignore[arg-type]
            )


# ------------------------------------------------------------- DashboardModel


class TestDashboardModel:
    def _row(self, job_id: str = "job-1") -> JobRow:
        return JobRow(
            job_id=job_id,
            framework="pytorch",
            status="running",
            dataset_id=None,
            elapsed_s=1.0,
        )

    def test_empty_model_with_no_selection(self) -> None:
        model = DashboardModel(
            jobs=(),
            selected_job_id=None,
            metric_series_for_selected=(),
            resources=ResourceSnapshot.unavailable(),
        )
        assert model.selected_row is None

    def test_selected_row_must_exist(self) -> None:
        with pytest.raises(ValueError):
            DashboardModel(
                jobs=(),
                selected_job_id=JobId("unknown"),
                metric_series_for_selected=(),
                resources=ResourceSnapshot.unavailable(),
            )

    def test_selected_row_returns_match(self) -> None:
        model = DashboardModel(
            jobs=(self._row(),),
            selected_job_id=JobId("job-1"),
            metric_series_for_selected=(),
            resources=ResourceSnapshot.unavailable(),
        )
        assert model.selected_row is not None
        assert model.selected_row.job_id == "job-1"

    def test_metric_series_only_with_selection(self) -> None:
        series = MetricSeries(
            name="loss",
            kind=MetricKind.LOSS,
            steps=(0,),
            values=(1.0,),
        )
        with pytest.raises(ValueError):
            DashboardModel(
                jobs=(self._row(),),
                selected_job_id=None,
                metric_series_for_selected=(series,),
                resources=ResourceSnapshot.unavailable(),
            )

    def test_find_row_returns_match(self) -> None:
        model = DashboardModel(
            jobs=(self._row("job-1"), self._row("job-2")),
            selected_job_id=None,
            metric_series_for_selected=(),
            resources=ResourceSnapshot.unavailable(),
        )
        assert model.find_row(JobId("job-1")) is not None
        assert model.find_row(JobId("missing")) is None

    def test_find_row_type_check(self) -> None:
        model = DashboardModel(
            jobs=(),
            selected_job_id=None,
            metric_series_for_selected=(),
            resources=ResourceSnapshot.unavailable(),
        )
        with pytest.raises(TypeError):
            model.find_row("not-a-job-id")  # type: ignore[arg-type]
