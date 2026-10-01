"""Process lifecycle, immutable RunContext, and idempotent worker execution for Motion Matching (R08, #11148)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from typing import Any
import uuid

from PyQt6.QtCore import QObject, QProcess, pyqtSignal

from src.tools.motion_matching import pipeline

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    """Execution status for a motion matching process run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED_TO_START = "failed_to_start"
    CRASHED = "crashed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RunContext:
    """Immutable context identifying the initiating run, owning panel, and workflow stage."""

    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    owning_panel: str = "matching"  # "matching" | "experiment" | "mjx"
    stage_name: str = "pipeline"
    request: Any = None
    output_dir: Path | str | None = None
    commands: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class RunResult:
    """Terminal outcome of a process queue execution."""

    context: RunContext
    exit_code: int
    status: RunStatus
    failing_stage: str | None = None
    error_message: str | None = None
    recovery_action: str | None = None
    output: str = ""

    @property
    def is_success(self) -> bool:
        """Return True if the run completed normally with code 0."""
        return self.exit_code == 0 and self.status == RunStatus.COMPLETED


def format_failure_diagnostic(result: RunResult) -> str:
    """Format user-facing failure diagnostics identifying failing stage and recovery action."""
    stage = result.failing_stage or result.context.stage_name or "unknown stage"
    action = (
        result.recovery_action
        or "Review log output above for error traceback or missing input files."
    )

    if result.status == RunStatus.FAILED_TO_START:
        return (
            f"Stage '{stage}' failed to start.\n"
            f"Error: {result.error_message or 'Executable not found or execution denied.'}\n"
            f"Recovery: {action}\n"
            f"(Step failed with exit code {result.exit_code}; see log for details.)"
        )

    if result.status == RunStatus.CRASHED:
        return (
            f"Stage '{stage}' crashed unexpectedly.\n"
            f"Error: {result.error_message or 'Process terminated by signal/crash.'}\n"
            f"Recovery: {action}\n"
            f"(Step failed with exit code {result.exit_code}; see log for details.)"
        )

    if result.status == RunStatus.CANCELLED:
        return (
            f"Stage '{stage}' was cancelled by user.\n"
            f"Action: {action}\n"
            f"(Step failed with exit code {result.exit_code}; see log for details.)"
        )

    return (
        f"Stage '{stage}' failed with exit code {result.exit_code}.\n"
        f"Recovery: {action}\n"
        f"Step failed with exit code {result.exit_code}; see log for details."
    )


class RunWorker(QObject):
    """Executes a queue of external CLI commands sequentially with robust lifecycle management."""

    started = pyqtSignal()
    output_received = pyqtSignal(str)
    finished = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    context_output_received = pyqtSignal(str, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._queue: list[tuple[str, list[str]]] = []
        self._cwd: Path = pipeline.REPO_ROOT
        self._context: RunContext | None = None
        self._last_result: RunResult | None = None
        self._current_stage: str = ""
        self._is_finalizing: bool = False
        self._was_cancelled: bool = False
        self._accumulated_output: list[str] = []

    @property
    def current_context(self) -> RunContext | None:
        """Return the active RunContext, if a run is ongoing or recently completed."""
        return self._context

    @property
    def last_result(self) -> RunResult | None:
        """Return the last terminal RunResult."""
        return self._last_result

    def is_running(self) -> bool:
        """Return True if an external process step is actively executing."""
        return self._process is not None

    def start(
        self,
        commands: list[list[str]],
        cwd: Path | None = None,
        context: RunContext | None = None,
        stage_names: list[str] | None = None,
    ) -> None:
        """Initiate execution of command queue bound to an immutable RunContext."""
        if self._process is not None or self._is_finalizing:
            return

        self._was_cancelled = False
        self._accumulated_output.clear()
        self._cwd = cwd or pipeline.REPO_ROOT

        resolved_stages = list(stage_names) if stage_names else []
        self._queue = []
        for i, cmd in enumerate(commands):
            st_name = (
                resolved_stages[i]
                if i < len(resolved_stages)
                else (Path(cmd[0]).stem if cmd else f"step_{i + 1}")
            )
            self._queue.append((st_name, cmd))

        if context is None:
            first_stage = self._queue[0][0] if self._queue else "pipeline"
            self._context = RunContext(
                run_id=f"run-{uuid.uuid4().hex[:8]}",
                owning_panel="matching",
                stage_name=first_stage,
                commands=tuple(tuple(cmd) for cmd in commands),
            )
        else:
            self._context = context

        if not self._queue:
            success_result = RunResult(
                context=self._context,
                exit_code=0,
                status=RunStatus.COMPLETED,
                output="",
            )
            self._last_result = success_result
            self.result_ready.emit(success_result)
            self.finished.emit(0)
            return

        self.started.emit()
        self._next()

    def _next(self) -> None:
        """Advance to the next command in the queue or finalize on success."""
        if not self._queue:
            self._finalize(
                status=RunStatus.COMPLETED,
                exit_code=0,
                error_msg=None,
                recovery=None,
            )
            return

        self._current_stage, command = self._queue.pop(0)
        banner = f"$ {' '.join(command)}\n"
        self._accumulated_output.append(banner)
        self.output_received.emit(banner)
        if self._context is not None:
            self.context_output_received.emit(banner, self._context)

        process = QProcess(self)
        process.setWorkingDirectory(str(self._cwd))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        process.readyReadStandardOutput.connect(self._on_process_output)
        process.finished.connect(self._on_step_finished)
        process.errorOccurred.connect(self._on_step_error)

        self._process = process
        process.start(command[0], command[1:])

    def _on_process_output(self) -> None:
        """Read standard output from the active process and emit signals."""
        if self._process is None:
            return
        chunk = self._process.readAllStandardOutput().data().decode(errors="replace")
        if chunk:
            self._accumulated_output.append(chunk)
            self.output_received.emit(chunk)
            if self._context is not None:
                self.context_output_received.emit(chunk, self._context)

    def _on_step_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        """Handle normal or abnormal exit of an individual step."""
        if self._is_finalizing:
            return

        if self._was_cancelled:
            self._finalize_cancellation()
            return

        if status == QProcess.ExitStatus.CrashExit:
            self._finalize(
                status=RunStatus.CRASHED,
                exit_code=code if code != 0 else -11,
                error_msg=f"Process crashed with exit code {code}.",
                recovery="Check system logs, native dependencies, and memory limits.",
            )
            return

        if code != 0:
            self._finalize(
                status=RunStatus.FAILED,
                exit_code=code,
                error_msg=f"Step failed with exit code {code}.",
                recovery="Review log output above for error traceback or missing input files.",
            )
            return

        # Step succeeded; clean up completed process object
        self._dispose_process()
        self._next()

    def _on_step_error(self, error: QProcess.ProcessError) -> None:
        """Handle QProcess startup or runtime failures idempotently."""
        if self._is_finalizing:
            return

        if self._was_cancelled:
            return

        if error == QProcess.ProcessError.FailedToStart:
            self._finalize(
                status=RunStatus.FAILED_TO_START,
                exit_code=127,
                error_msg="Executable failed to start (file not found or access denied).",
                recovery="Verify the executable exists in PATH or current environment and has execute permissions.",
            )
        elif error == QProcess.ProcessError.Crashed:
            self._finalize(
                status=RunStatus.CRASHED,
                exit_code=-11,
                error_msg="Process crashed or was terminated abruptly.",
                recovery="Check system logs, native dependencies, and memory limits.",
            )
        else:
            self._finalize(
                status=RunStatus.FAILED,
                exit_code=1,
                error_msg=f"Process error: {error.name}",
                recovery="Review process arguments, system permissions, and environment variables.",
            )

    def _finalize_cancellation(self) -> None:
        """Finalize the active run as cancelled by the user."""
        self._finalize(
            status=RunStatus.CANCELLED,
            exit_code=-1,
            error_msg="Process execution was cancelled by user.",
            recovery="Re-run the workflow when ready.",
        )

    def stop(self) -> None:
        """Cancel the ongoing run cleanly and idempotently."""
        self._was_cancelled = True
        self._queue.clear()
        if self._process is not None:
            proc = self._process
            try:
                proc.kill()
            except RuntimeError:
                pass
            self._finalize_cancellation()

    def _dispose_process(self) -> None:
        """Disconnect, unparent, and schedule deletion of current QProcess instance."""
        if self._process is not None:
            proc = self._process
            self._process = None
            try:
                proc.readyReadStandardOutput.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                proc.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                proc.errorOccurred.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                proc.setParent(None)
            except RuntimeError:
                pass
            proc.deleteLater()

    def _finalize(
        self,
        status: RunStatus,
        exit_code: int,
        error_msg: str | None,
        recovery: str | None,
    ) -> None:
        """Idempotent terminal finalizer emitting exactly one result."""
        if self._is_finalizing:
            return
        self._is_finalizing = True
        self._queue.clear()

        # Read any remaining stdout before disposal
        if self._process is not None:
            try:
                remaining = (
                    self._process.readAllStandardOutput()
                    .data()
                    .decode(errors="replace")
                )
                if remaining:
                    self._accumulated_output.append(remaining)
                    self.output_received.emit(remaining)
                    if self._context is not None:
                        self.context_output_received.emit(remaining, self._context)
            except RuntimeError:
                pass

        self._dispose_process()

        assert self._context is not None
        failing_stage = self._current_stage if status != RunStatus.COMPLETED else None
        result = RunResult(
            context=self._context,
            exit_code=exit_code,
            status=status,
            failing_stage=failing_stage,
            error_message=error_msg,
            recovery_action=recovery,
            output="".join(self._accumulated_output),
        )
        self._last_result = result

        try:
            self.result_ready.emit(result)
            self.finished.emit(exit_code)
        finally:
            self._is_finalizing = False
