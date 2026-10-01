"""Tests for matcher process lifecycle, RunContext binding, and error handling (R08, #11148)."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from PyQt6.QtCore import QCoreApplication, QObject, QProcess
from PyQt6.QtWidgets import QApplication
import pytest

from src.tools.motion_matching.gui import MotionMatchingWidget, RunWorker
from src.tools.motion_matching.process_lifecycle import (
    RunContext,
    RunResult,
    RunStatus,
    format_failure_diagnostic,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """Session-scoped offscreen QApplication."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv[:1])
    return app


@pytest.fixture
def widget() -> MotionMatchingWidget:
    return MotionMatchingWidget()


def _drain_events(iterations: int = 15, delay_ms: int = 10) -> None:
    """Process pending Qt events to service asynchronous signals."""
    for _ in range(iterations):
        QCoreApplication.processEvents()
        if delay_ms > 0:
            import time

            time.sleep(delay_ms / 1000.0)
            QCoreApplication.processEvents()


class TestProcessLifecycle:
    """Test suite covering R08 acceptance criteria."""

    def test_missing_executable_restores_controls_and_reports_recovery(
        self, widget: MotionMatchingWidget
    ) -> None:
        """Starting a missing executable emits exactly one terminal result and restores UI."""
        results: list[RunResult] = []
        finish_codes: list[int] = []

        widget._worker.result_ready.connect(results.append)
        widget._worker.finished.connect(finish_codes.append)

        # Set user inputs to confirm they are retained on failure
        widget.stature.setValue(1.85)
        widget.mass.setValue(82.0)

        # Launch a nonexistent command directly on the widget
        bad_cmd = ["nonexistent_exec_ud_11148_xyz", "--some-arg"]
        ctx = RunContext(
            run_id="test-missing-1",
            owning_panel="matching",
            stage_name="build",
        )
        widget._set_active_buttons(widget.run_button, widget.stop_button, running=True)
        widget._worker.start([bad_cmd], context=ctx, stage_names=["build"])

        _drain_events(iterations=20, delay_ms=10)

        # Worker must not be stuck in running state
        assert not widget._worker.is_running()

        # UI controls must be restored
        assert widget.run_button.isEnabled()
        assert not widget.stop_button.isEnabled()

        # Exactly one terminal event must be emitted
        assert len(results) == 1
        assert len(finish_codes) == 1

        result = results[0]
        assert result.status == RunStatus.FAILED_TO_START
        assert result.failing_stage == "build"
        assert result.recovery_action is not None
        assert (
            "executable" in result.recovery_action.lower()
            or "verify" in result.recovery_action.lower()
        )

        # Results label must identify failing stage and recovery guidance
        label_text = widget.results.text()
        assert "build" in label_text.lower()
        assert (
            "recovery" in label_text.lower()
            or "action" in label_text.lower()
            or "executable" in label_text.lower()
        )

        # User inputs must be retained
        assert widget.stature.value() == 1.85
        assert widget.mass.value() == 82.0

    def test_denied_executable_restores_controls_and_reports_recovery(
        self, widget: MotionMatchingWidget
    ) -> None:
        """Permission-denied or failed-to-start process emits exactly one terminal event."""
        results: list[RunResult] = []
        finish_codes: list[int] = []

        widget._worker.result_ready.connect(results.append)
        widget._worker.finished.connect(finish_codes.append)

        ctx = RunContext(
            run_id="test-denied-1",
            owning_panel="matching",
            stage_name="match",
        )
        widget._set_active_buttons(widget.run_button, widget.stop_button, running=True)
        widget._worker.start(
            [[sys.executable, "-c", "import sys; sys.exit(0)"]],
            context=ctx,
            stage_names=["match"],
        )

        # Simulate QProcess FailedToStart error on active process
        proc = widget._worker._process
        assert proc is not None
        proc.errorOccurred.emit(QProcess.ProcessError.FailedToStart)

        _drain_events(iterations=10, delay_ms=5)

        assert not widget._worker.is_running()
        assert widget.run_button.isEnabled()
        assert not widget.stop_button.isEnabled()
        assert len(results) == 1
        assert len(finish_codes) == 1
        assert results[0].status == RunStatus.FAILED_TO_START
        assert results[0].failing_stage == "match"

    def test_crashed_process_emits_single_terminal_result_and_restores_controls(
        self, widget: MotionMatchingWidget
    ) -> None:
        """Process crash emitting both errorOccurred and finished emits only one terminal result."""
        results: list[RunResult] = []
        finish_codes: list[int] = []

        widget._worker.result_ready.connect(results.append)
        widget._worker.finished.connect(finish_codes.append)

        ctx = RunContext(
            run_id="test-crash-1",
            owning_panel="experiment",
            stage_name="downswing_experiment",
        )
        widget._set_active_buttons(
            widget.exp_run_btn, widget.exp_stop_btn, running=True
        )
        widget._worker.start(
            [[sys.executable, "-c", "import sys; sys.exit(0)"]],
            context=ctx,
            stage_names=["downswing_experiment"],
        )

        proc = widget._worker._process
        assert proc is not None

        # Emit both crash error and crash exit from finished
        proc.errorOccurred.emit(QProcess.ProcessError.Crashed)
        proc.finished.emit(-11, QProcess.ExitStatus.CrashExit)

        _drain_events(iterations=10, delay_ms=5)

        assert not widget._worker.is_running()
        assert widget.exp_run_btn.isEnabled()
        assert not widget.exp_stop_btn.isEnabled()

        # Idempotent finalizer: exactly one terminal result
        assert len(results) == 1
        assert len(finish_codes) == 1
        assert results[0].status == RunStatus.CRASHED
        assert results[0].failing_stage == "downswing_experiment"
        assert (
            "crashed" in widget.exp_results.text().lower()
            or "crash" in widget.exp_results.text().lower()
        )

    def test_cancellation_emits_single_terminal_result_and_restores_controls(
        self, widget: MotionMatchingWidget
    ) -> None:
        """Calling stop() on a running worker terminates cleanly with single cancelled result."""
        results: list[RunResult] = []
        finish_codes: list[int] = []

        widget._worker.result_ready.connect(results.append)
        widget._worker.finished.connect(finish_codes.append)

        ctx = RunContext(
            run_id="test-cancel-1",
            owning_panel="matching",
            stage_name="match",
        )
        widget._set_active_buttons(widget.run_button, widget.stop_button, running=True)
        # Start a sleeping command
        cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
        widget._worker.start([cmd], context=ctx, stage_names=["match"])

        assert widget._worker.is_running()

        # Cancel the run
        widget.stop()

        _drain_events(iterations=15, delay_ms=10)

        assert not widget._worker.is_running()
        assert widget.run_button.isEnabled()
        assert not widget.stop_button.isEnabled()

        assert len(results) == 1
        assert len(finish_codes) == 1
        assert results[0].status == RunStatus.CANCELLED
        assert results[0].exit_code == -1

    def test_tab_switching_preserves_output_and_result_routing(
        self, widget: MotionMatchingWidget
    ) -> None:
        """Switching tabs while a run is active routes output and status strictly to the initiating panel."""
        ctx = RunContext(
            run_id="test-tab-routing-1",
            owning_panel="matching",
            stage_name="match",
        )
        widget.log.clear()
        widget.exp_log.clear()
        widget.mjx_log.clear()

        # Start on Matching tab
        widget.tabs.setCurrentIndex(0)
        cmd = [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('MATCH_STEP_OUTPUT_TEST\\n'); sys.stdout.flush()",
        ]
        widget._worker.start([cmd], context=ctx, stage_names=["match"])

        # Immediately switch to other tabs (Downswing experiment, MJX, Club-Only, Tour Baselines)
        for tab_idx in [1, 2, 3, 4]:
            widget.tabs.setCurrentIndex(tab_idx)
            _drain_events(iterations=5, delay_ms=5)

        _drain_events(iterations=20, delay_ms=10)

        # Output must be routed strictly to Matching tab log
        assert "MATCH_STEP_OUTPUT_TEST" in widget.log.toPlainText()
        assert "MATCH_STEP_OUTPUT_TEST" not in widget.exp_log.toPlainText()
        assert "MATCH_STEP_OUTPUT_TEST" not in widget.mjx_log.toPlainText()

    def test_experiment_tab_switching_preserves_output_routing(
        self, widget: MotionMatchingWidget
    ) -> None:
        """Switching tabs while downswing experiment runs routes output strictly to exp_log."""
        ctx = RunContext(
            run_id="test-exp-tab-routing",
            owning_panel="experiment",
            stage_name="downswing_experiment",
        )
        widget.log.clear()
        widget.exp_log.clear()

        widget.tabs.setCurrentIndex(1)
        cmd = [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('EXP_STEP_OUTPUT_TEST\\n'); sys.stdout.flush()",
        ]
        widget._worker.start([cmd], context=ctx, stage_names=["downswing_experiment"])

        # Switch to Matching tab (0) and Tour Baselines tab (4)
        widget.tabs.setCurrentIndex(0)
        _drain_events(iterations=5, delay_ms=5)
        widget.tabs.setCurrentIndex(4)

        _drain_events(iterations=20, delay_ms=10)

        assert "EXP_STEP_OUTPUT_TEST" in widget.exp_log.toPlainText()
        assert "EXP_STEP_OUTPUT_TEST" not in widget.log.toPlainText()

    def test_repeated_start_stop_disposes_process_objects_without_leak(
        self, widget: MotionMatchingWidget
    ) -> None:
        """Repeated start and stop cycles dispose QProcess instances and do not duplicate callbacks."""
        results: list[RunResult] = []
        widget._worker.result_ready.connect(results.append)

        for i in range(5):
            ctx = RunContext(
                run_id=f"test-repeat-{i}",
                owning_panel="matching",
                stage_name="match",
            )
            cmd = [sys.executable, "-c", "import time; time.sleep(2)"]
            widget._worker.start([cmd], context=ctx, stage_names=["match"])
            assert widget._worker.is_running()
            widget.stop()
            _drain_events(iterations=10, delay_ms=5)
            assert not widget._worker.is_running()

        # Exactly 5 results for 5 cycles (1 per cycle)
        assert len(results) == 5

        # All child QProcess instances of worker should have been deleted/disposed
        child_procs = [c for c in widget._worker.children() if isinstance(c, QProcess)]
        assert len(child_procs) == 0

    def test_failing_stage_and_recovery_action_identification(
        self, widget: MotionMatchingWidget
    ) -> None:
        """Multi-stage run failure distinguishes stage 1 failure from stage 2 failure."""
        results: list[RunResult] = []
        widget._worker.result_ready.connect(results.append)

        # Stage 1 succeeds, Stage 2 fails with exit code 42
        ctx = RunContext(
            run_id="test-multi-stage",
            owning_panel="matching",
            stage_name="build",
        )
        cmd1 = [sys.executable, "-c", "import sys; sys.exit(0)"]
        cmd2 = [sys.executable, "-c", "import sys; sys.exit(42)"]

        widget._worker.start([cmd1, cmd2], context=ctx, stage_names=["build", "match"])
        _drain_events(iterations=25, delay_ms=10)

        assert not widget._worker.is_running()
        assert len(results) == 1
        res = results[0]
        assert res.exit_code == 42
        assert res.failing_stage == "match"
        assert res.status == RunStatus.FAILED
        assert res.recovery_action is not None
        assert "Step failed with exit code 42" in widget.results.text()
        assert "match" in widget.results.text().lower()
