"""PyQt smoke tests for :mod:`src.tools.training_controller.gui`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _load_qt():
    try:
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtTest import QTest
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    return QTest, Qt, QTimer


pytestmark = pytest.mark.unit


def _make_controller():
    from src.shared.python.training import (
        CompatibilityChecker,
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
    from src.shared.python.training.runtime import InProcessDriver, RunnerRegistry
    from src.tools.training_controller import TrainingDashboardController

    class _Runner:
        framework = TrainingFramework.PYTORCH

        def can_run(self, config: TrainingConfig) -> bool:
            return config.framework is self.framework

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

    runners = RunnerRegistry()
    runners.register(_Runner())
    scheduler = Scheduler(
        registry=JobRegistry(),
        runners=runners,
        driver=InProcessDriver(runners, max_workers=1),
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
    )
    return controller, scheduler


def test_main_window_renders_empty_controller(qapp) -> None:
    _load_qt()
    from src.tools.training_controller.gui import MainWindow

    del qapp
    controller, scheduler = _make_controller()
    window = MainWindow(controller)
    try:
        assert window.windowTitle() == "Training Controller"
        assert window.main_widget.job_table.rowCount() == 0
        assert window.main_widget.dataset_list.count() == 1
        assert "monitoring unavailable" in window.main_widget.resource_label.text()
    finally:
        window.close()
        scheduler.shutdown()


def test_submit_button_opens_dialog_and_submits_job(qapp) -> None:
    QTest, Qt, QTimer = _load_qt()
    from src.tools.training_controller.gui import MainWindow

    del qapp
    controller, scheduler = _make_controller()
    window = MainWindow(controller)
    try:
        window.show()

        def _accept_dialog() -> None:
            dialog = window.findChild(object, "training-submit-dialog")
            assert dialog is not None
            dialog.entry_edit.setText("module:train")
            dialog.output_edit.setText(str(Path("/tmp/training-controller-gui")))
            QTest.mouseClick(dialog.submit_button, Qt.MouseButton.LeftButton)

        QTimer.singleShot(0, _accept_dialog)
        QTest.mouseClick(window.main_widget.submit_button, Qt.MouseButton.LeftButton)

        assert window.main_widget.job_table.rowCount() == 1
    finally:
        window.close()
        scheduler.shutdown()


def test_lifecycle_buttons_call_controller_methods(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QTest, Qt, _ = _load_qt()
    from src.shared.python.training import (
        JobId,
        TrainingConfig,
        TrainingFramework,
    )
    from src.tools.training_controller import TrainingDashboardController
    from src.tools.training_controller.gui import MainWindow

    del qapp
    controller, scheduler = _make_controller()
    job = controller.submit_job(
        TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="module:train",
            output_dir=Path("/tmp/training-controller-gui"),
            dataset_id="dataset-1",
        )
    )
    window = MainWindow(controller)
    calls: list[tuple[str, JobId]] = []
    monkeypatch.setattr(
        TrainingDashboardController,
        "cancel_job",
        lambda _self, job_id: calls.append(("cancel", job_id)),
    )
    monkeypatch.setattr(
        TrainingDashboardController,
        "pause_job",
        lambda _self, job_id: calls.append(("pause", job_id)),
    )
    monkeypatch.setattr(
        TrainingDashboardController,
        "resume_job",
        lambda _self, job_id: calls.append(("resume", job_id)),
    )
    try:
        window.main_widget.job_table.selectRow(0)
        QTest.mouseClick(window.main_widget.cancel_button, Qt.MouseButton.LeftButton)
        QTest.mouseClick(window.main_widget.pause_button, Qt.MouseButton.LeftButton)
        QTest.mouseClick(window.main_widget.resume_button, Qt.MouseButton.LeftButton)
        assert calls == [
            ("cancel", job.job_id),
            ("pause", job.job_id),
            ("resume", job.job_id),
        ]
    finally:
        window.close()
        scheduler.shutdown()
