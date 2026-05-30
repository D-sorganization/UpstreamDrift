"""Tests for :mod:`training_controller.controller`.

Exercises the headless MVC controller against a real
:class:`training.Scheduler` + :class:`training.runtime.InProcessDriver`
+ a stub :class:`TrainingJobRunner`. No PyQt; no realtime; no display.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from src.shared.python.training import (
    CompatibilityChecker,
    CompatibilityError,
    Dataset,
    DatasetRegistry,
    JobRegistry,
    RunResult,
    Scheduler,
    TrainingConfig,
    TrainingFramework,
    TrainingStatus,
    new_run_id,
)
from src.shared.python.training.contracts import CancelToken, ProgressSink
from src.shared.python.training.metrics import MetricKind, TrainingMetric
from src.shared.python.training.resource_monitor import GpuSample, ResourceSample
from src.shared.python.training.runtime import InProcessDriver, RunnerRegistry
from src.tools.training_controller.controller import (
    DEFAULT_ROLLING_WINDOW,
    TrainingDashboardController,
)
from src.tools.training_controller.view_model import (
    DashboardModel,
    JobRow,
    ResourceSnapshot,
)


pytestmark = pytest.mark.unit


# ------------------------------------------------------------------- fixtures


class _SuccessRunner:
    framework = TrainingFramework.PYTORCH

    def can_run(self, config: TrainingConfig) -> bool:
        del config
        return True

    def prepare(self, config: TrainingConfig) -> None:
        del config

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        del config, progress, cancel
        return RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=0.0,
        )


class _BlockingRunner:
    framework = TrainingFramework.PYTORCH

    def __init__(self) -> None:
        self.started = threading.Event()

    def can_run(self, config: TrainingConfig) -> bool:
        del config
        return True

    def prepare(self, config: TrainingConfig) -> None:
        del config

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        del config, progress
        self.started.set()
        while not cancel.is_cancelled:
            time.sleep(0.01)
        return RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.CANCELLED,
            duration_s=0.0,
        )


def _config(
    *, framework: TrainingFramework = TrainingFramework.PYTORCH
) -> TrainingConfig:
    return TrainingConfig(
        framework=framework,
        entry_point="m:train",
        output_dir=Path("/tmp/training-controller-tests"),
        dataset_id="dataset-1",
    )


def _make_controller(
    *,
    runner: object | None = None,
    resource_provider=None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> tuple[TrainingDashboardController, Scheduler]:
    runners = RunnerRegistry()
    runners.register(runner if runner is not None else _SuccessRunner())
    driver = InProcessDriver(runners, max_workers=1)
    scheduler = Scheduler(
        registry=JobRegistry(),
        runners=runners,
        driver=driver,
    )
    datasets = DatasetRegistry(
        initial=(
            Dataset(
                dataset_id="dataset-1",
                name="Dataset 1",
                path=Path("/tmp/dataset-1"),
                format="custom",
            ),
        )
    )
    controller = TrainingDashboardController(
        scheduler,
        datasets,
        CompatibilityChecker(),
        resource_provider=resource_provider,
        rolling_window=rolling_window,
    )
    return controller, scheduler


# ------------------------------------------------------------------- construction


class TestConstruction:
    def test_rejects_non_scheduler(self) -> None:
        with pytest.raises(TypeError):
            TrainingDashboardController(
                "not-a-scheduler",  # type: ignore[arg-type]
                DatasetRegistry(),
                CompatibilityChecker(),
            )

    def test_rejects_non_dataset_registry(self) -> None:
        controller, scheduler = _make_controller()
        controller.close()
        with pytest.raises(TypeError):
            TrainingDashboardController(
                scheduler,
                "not-a-registry",  # type: ignore[arg-type]
                CompatibilityChecker(),
            )
        scheduler.shutdown()

    def test_rejects_non_compat_checker(self) -> None:
        controller, scheduler = _make_controller()
        controller.close()
        with pytest.raises(TypeError):
            TrainingDashboardController(
                scheduler,
                DatasetRegistry(),
                "not-a-checker",  # type: ignore[arg-type]
            )
        scheduler.shutdown()

    def test_rejects_bad_rolling_window(self) -> None:
        controller, scheduler = _make_controller()
        controller.close()
        with pytest.raises(ValueError):
            TrainingDashboardController(
                scheduler,
                DatasetRegistry(),
                CompatibilityChecker(),
                rolling_window=0,
            )
        scheduler.shutdown()

    def test_exposes_backend_references(self) -> None:
        controller, scheduler = _make_controller()
        try:
            assert controller.scheduler is scheduler
            assert isinstance(controller.dataset_registry, DatasetRegistry)
            assert isinstance(controller.compatibility_checker, CompatibilityChecker)
        finally:
            controller.close()
            scheduler.shutdown()


# ----------------------------------------------------------------- submit gate


class TestSubmitGate:
    def test_submit_passes_through_to_scheduler(self) -> None:
        controller, scheduler = _make_controller()
        try:
            job = controller.submit_job(_config())
            assert job.status is TrainingStatus.QUEUED
            assert scheduler.get(job.job_id) is job
        finally:
            controller.close()
            scheduler.shutdown()

    def test_submit_runs_compat_check_first(self) -> None:
        controller, scheduler = _make_controller()
        try:
            # PyTorch + drake is compatible per the default map.
            controller.submit_job(_config(), target_engine="drake")
        finally:
            controller.close()
            scheduler.shutdown()

    def test_compat_failure_blocks_submit(self) -> None:
        # Gymnasium framework but no gymnasium runner registered — so
        # we route the (config, engine) pair through the controller's
        # compat check (engine unknown -> error).
        controller, scheduler = _make_controller()
        try:
            with pytest.raises(CompatibilityError):
                controller.submit_job(_config(), target_engine="bogus-engine")
            # No job was admitted.
            assert scheduler.registry.list() == ()
        finally:
            controller.close()
            scheduler.shutdown()

    def test_rejects_non_config(self) -> None:
        controller, scheduler = _make_controller()
        try:
            with pytest.raises(TypeError):
                controller.submit_job("not-a-config")  # type: ignore[arg-type]
        finally:
            controller.close()
            scheduler.shutdown()


# ----------------------------------------------------------------- lifecycle


class TestLifecycle:
    def test_cancel_pause_resume_pass_through(self) -> None:
        runner = _BlockingRunner()
        controller, scheduler = _make_controller(runner=runner)
        try:
            job = controller.submit_job(_config())
            scheduler.start(job.job_id)
            assert runner.started.wait(timeout=2.0)
            paused = controller.pause_job(job.job_id)
            assert paused.status is TrainingStatus.PAUSED
            resumed = controller.resume_job(job.job_id)
            assert resumed.status is TrainingStatus.RUNNING
            cancelled = controller.cancel_job(job.job_id)
            assert cancelled.status is TrainingStatus.CANCELLED
        finally:
            controller.close()
            scheduler.shutdown()

    def test_lifecycle_methods_reject_non_jobid(self) -> None:
        controller, scheduler = _make_controller()
        try:
            for fn in (
                controller.cancel_job,
                controller.pause_job,
                controller.resume_job,
            ):
                with pytest.raises(TypeError):
                    fn("not-a-jobid")  # type: ignore[arg-type]
        finally:
            controller.close()
            scheduler.shutdown()


# ----------------------------------------------------------------- model build


class TestCurrentModel:
    def test_empty_model_when_no_jobs(self) -> None:
        controller, scheduler = _make_controller()
        try:
            model = controller.current_model()
            assert isinstance(model, DashboardModel)
            assert model.jobs == ()
            assert model.selected_job_id is None
            assert model.metric_series_for_selected == ()
            assert model.resources.available is False
        finally:
            controller.close()
            scheduler.shutdown()

    def test_model_lists_submitted_jobs(self) -> None:
        controller, scheduler = _make_controller()
        try:
            controller.submit_job(_config())
            controller.submit_job(_config())
            model = controller.current_model()
            assert len(model.jobs) == 2
            assert all(isinstance(row, JobRow) for row in model.jobs)
        finally:
            controller.close()
            scheduler.shutdown()

    def test_resource_snapshot_populated_when_provider_returns_sample(
        self,
    ) -> None:
        sample = ResourceSample(
            timestamp=1.0,
            cpu_percent=12.5,
            memory_used_mb=512,
            memory_total_mb=8192,
            gpus=(
                GpuSample(
                    index=0,
                    name="A100",
                    utilization_percent=80.0,
                    memory_used_mb=1024,
                    memory_total_mb=8192,
                ),
            ),
        )
        controller, scheduler = _make_controller(resource_provider=lambda: sample)
        try:
            snap = controller.current_model().resources
            assert isinstance(snap, ResourceSnapshot)
            assert snap.available is True
            assert snap.cpu_percent == pytest.approx(12.5)
            assert snap.memory_percent == pytest.approx(100.0 * 512 / 8192)
            assert len(snap.gpus) == 1
            assert snap.gpus[0].name == "A100"
        finally:
            controller.close()
            scheduler.shutdown()


# ----------------------------------------------------------------- selection


class TestSelection:
    def test_select_unknown_job_raises(self) -> None:
        controller, scheduler = _make_controller()
        try:
            from src.shared.python.training import JobId  # local import for readability

            with pytest.raises(KeyError):
                controller.select_job(JobId("unknown"))
        finally:
            controller.close()
            scheduler.shutdown()

    def test_select_and_clear(self) -> None:
        controller, scheduler = _make_controller()
        try:
            job = controller.submit_job(_config())
            controller.select_job(job.job_id)
            assert controller.selected_job_id == job.job_id
            controller.select_job(None)
            assert controller.selected_job_id is None
        finally:
            controller.close()
            scheduler.shutdown()


# ----------------------------------------------------------------- observers


class TestObservers:
    def test_observer_fires_on_submit(self) -> None:
        controller, scheduler = _make_controller()
        try:
            calls = []
            controller.on_model_change(lambda: calls.append(True))
            controller.submit_job(_config())
            # submit causes a PENDING -> QUEUED transition which routes
            # through scheduler.on_status_change -> controller -> notify.
            assert calls, "observer was not fired on submit"
        finally:
            controller.close()
            scheduler.shutdown()

    def test_observer_fires_on_selection_change(self) -> None:
        controller, scheduler = _make_controller()
        try:
            job = controller.submit_job(_config())
            calls = []
            controller.on_model_change(lambda: calls.append(True))
            controller.select_job(job.job_id)
            assert calls
        finally:
            controller.close()
            scheduler.shutdown()

    def test_unsubscribe_stops_notifications(self) -> None:
        controller, scheduler = _make_controller()
        try:
            calls = []
            unsubscribe = controller.on_model_change(lambda: calls.append(True))
            unsubscribe()
            controller.submit_job(_config())
            assert calls == []
        finally:
            controller.close()
            scheduler.shutdown()

    def test_unsubscribe_is_idempotent(self) -> None:
        controller, scheduler = _make_controller()
        try:
            unsubscribe = controller.on_model_change(lambda: None)
            unsubscribe()
            unsubscribe()  # second call must not raise
        finally:
            controller.close()
            scheduler.shutdown()

    def test_observer_exception_does_not_propagate(self) -> None:
        controller, scheduler = _make_controller()
        try:

            def boom() -> None:
                raise RuntimeError("kaboom")

            controller.on_model_change(boom)
            # Should NOT raise — observer fan-out logs and continues.
            controller.submit_job(_config())
        finally:
            controller.close()
            scheduler.shutdown()


# ----------------------------------------------------------------- metrics


class TestMetricIngest:
    def test_ingest_buffers_metric(self) -> None:
        controller, scheduler = _make_controller()
        try:
            job = controller.submit_job(_config())
            controller.select_job(job.job_id)
            metric = TrainingMetric(
                name="loss",
                value=0.5,
                step=0,
                timestamp=1.0,
                kind=MetricKind.LOSS,
            )
            controller.ingest_metric(job.job_id, metric)
            buffered = controller.metrics_for(job.job_id)
            assert buffered == (metric,)
        finally:
            controller.close()
            scheduler.shutdown()

    def test_ingest_for_selected_job_triggers_notify(self) -> None:
        controller, scheduler = _make_controller()
        try:
            job = controller.submit_job(_config())
            controller.select_job(job.job_id)
            calls = []
            controller.on_model_change(lambda: calls.append(True))
            controller.ingest_metric(
                job.job_id,
                TrainingMetric(name="loss", value=1.0, step=0, timestamp=1.0),
            )
            assert calls, "observer should fire when selected job receives metric"
        finally:
            controller.close()
            scheduler.shutdown()

    def test_ingest_for_unselected_job_does_not_notify(self) -> None:
        controller, scheduler = _make_controller()
        try:
            job_a = controller.submit_job(_config())
            job_b = controller.submit_job(_config())
            controller.select_job(job_a.job_id)
            calls = []
            controller.on_model_change(lambda: calls.append(True))
            controller.ingest_metric(
                job_b.job_id,
                TrainingMetric(name="loss", value=1.0, step=0, timestamp=1.0),
            )
            assert calls == []
        finally:
            controller.close()
            scheduler.shutdown()

    def test_metric_series_appears_in_model_for_selected_job(self) -> None:
        controller, scheduler = _make_controller()
        try:
            job = controller.submit_job(_config())
            controller.select_job(job.job_id)
            for step, value in enumerate([1.0, 0.7, 0.4]):
                controller.ingest_metric(
                    job.job_id,
                    TrainingMetric(
                        name="loss",
                        value=value,
                        step=step,
                        timestamp=float(step),
                        kind=MetricKind.LOSS,
                    ),
                )
            model = controller.current_model()
            assert len(model.metric_series_for_selected) == 1
            series = model.metric_series_for_selected[0]
            assert series.name == "loss"
            assert series.steps == (0, 1, 2)
            assert series.values == (1.0, 0.7, 0.4)
            assert series.smoothed is None  # LOSS gets no smoothing
        finally:
            controller.close()
            scheduler.shutdown()

    def test_reward_series_is_smoothed(self) -> None:
        controller, scheduler = _make_controller(rolling_window=2)
        try:
            job = controller.submit_job(_config())
            controller.select_job(job.job_id)
            for step, value in enumerate([1.0, 3.0, 5.0]):
                controller.ingest_metric(
                    job.job_id,
                    TrainingMetric(
                        name="episode_reward",
                        value=value,
                        step=step,
                        timestamp=float(step),
                        kind=MetricKind.REWARD,
                    ),
                )
            model = controller.current_model()
            assert len(model.metric_series_for_selected) == 1
            series = model.metric_series_for_selected[0]
            assert series.kind is MetricKind.REWARD
            assert series.smoothed is not None
            # Window=2, so smoothed[0]=1.0 (only one value), smoothed[1]=2.0,
            # smoothed[2]=4.0.
            assert series.smoothed == pytest.approx((1.0, 2.0, 4.0))
        finally:
            controller.close()
            scheduler.shutdown()

    def test_clear_metrics_drops_buffer(self) -> None:
        controller, scheduler = _make_controller()
        try:
            job = controller.submit_job(_config())
            controller.ingest_metric(
                job.job_id,
                TrainingMetric(name="loss", value=1.0, step=0, timestamp=0.0),
            )
            controller.clear_metrics(job.job_id)
            assert controller.metrics_for(job.job_id) == ()
        finally:
            controller.close()
            scheduler.shutdown()

    def test_clear_all_metrics(self) -> None:
        controller, scheduler = _make_controller()
        try:
            job = controller.submit_job(_config())
            controller.ingest_metric(
                job.job_id,
                TrainingMetric(name="loss", value=1.0, step=0, timestamp=0.0),
            )
            controller.clear_metrics()
            assert controller.metrics_for(job.job_id) == ()
        finally:
            controller.close()
            scheduler.shutdown()

    def test_metrics_for_unknown_returns_empty(self) -> None:
        controller, scheduler = _make_controller()
        try:
            from src.shared.python.training import JobId

            assert controller.metrics_for(JobId("never-seen")) == ()
        finally:
            controller.close()
            scheduler.shutdown()


# ----------------------------------------------------------------- close


class TestClose:
    def test_close_unhooks_scheduler(self) -> None:
        controller, scheduler = _make_controller()
        try:
            calls: list[bool] = []
            controller.on_model_change(lambda: calls.append(True))
            controller.close()
            # After close, scheduler events no longer reach the controller,
            # so observers do not fire.
            controller.submit_job(_config())
            assert calls == []
        finally:
            scheduler.shutdown()

    def test_close_is_idempotent(self) -> None:
        controller, scheduler = _make_controller()
        controller.close()
        controller.close()  # must not raise
        scheduler.shutdown()
