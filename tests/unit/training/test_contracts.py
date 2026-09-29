"""Tests for :mod:`training.contracts` — Protocols and ThreadingCancelToken."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from training import (
    CancelToken,
    ProgressSink,
    RunResult,
    ThreadingCancelToken,
    TrainingConfig,
    TrainingFramework,
    TrainingJobRunner,
    TrainingMetric,
    TrainingStatus,
    new_run_id,
)

pytestmark = pytest.mark.unit


class TestThreadingCancelToken:
    def test_initially_not_cancelled(self) -> None:
        token = ThreadingCancelToken()
        assert token.is_cancelled is False

    def test_request_cancel_flips_flag(self) -> None:
        token = ThreadingCancelToken()
        token.request_cancel()
        assert token.is_cancelled is True

    def test_idempotent_cancel(self) -> None:
        token = ThreadingCancelToken()
        token.request_cancel()
        token.request_cancel()  # second call must not raise
        assert token.is_cancelled is True

    def test_satisfies_cancel_token_protocol(self) -> None:
        token = ThreadingCancelToken()
        assert isinstance(token, CancelToken)

    def test_cross_thread_visibility(self) -> None:
        token = ThreadingCancelToken()
        observed: list[bool] = []
        done = threading.Event()

        def waiter() -> None:
            while not token.is_cancelled:
                pass
            observed.append(True)
            done.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        token.request_cancel()
        assert done.wait(timeout=2.0), "cancel signal not seen across threads"
        thread.join(timeout=1.0)
        assert observed == [True]


class _FullSink:
    """Minimal :class:`ProgressSink` implementation for protocol checks."""

    def __init__(self) -> None:
        self.metrics: list[TrainingMetric] = []
        self.status_calls: list[tuple[TrainingStatus, str | None]] = []

    def emit_metric(self, metric: TrainingMetric) -> None:
        self.metrics.append(metric)

    def emit_status(
        self, status: TrainingStatus, *, message: str | None = None
    ) -> None:
        self.status_calls.append((status, message))


class _MissingMethodSink:
    """Sink with one method missing — must NOT satisfy the protocol."""

    def emit_metric(self, metric: TrainingMetric) -> None:
        pass


class TestProgressSinkProtocol:
    def test_full_sink_satisfies_protocol(self) -> None:
        assert isinstance(_FullSink(), ProgressSink)

    def test_missing_method_fails_protocol(self) -> None:
        assert not isinstance(_MissingMethodSink(), ProgressSink)


class _FullRunner:
    framework = TrainingFramework.PYTORCH

    def can_run(self, config: TrainingConfig) -> bool:
        return True

    def prepare(self, config: TrainingConfig) -> None:
        pass

    def run(
        self,
        config: TrainingConfig,
        *,
        progress: ProgressSink,
        cancel: CancelToken,
    ) -> RunResult:
        return RunResult(
            run_id=new_run_id(),
            status=TrainingStatus.COMPLETED,
            duration_s=0.0,
        )


class _MissingMethodRunner:
    framework = TrainingFramework.PYTORCH

    def can_run(self, config: TrainingConfig) -> bool:
        return True


class TestTrainingJobRunnerProtocol:
    def test_full_runner_satisfies_protocol(self) -> None:
        assert isinstance(_FullRunner(), TrainingJobRunner)

    def test_missing_method_fails_protocol(self) -> None:
        assert not isinstance(_MissingMethodRunner(), TrainingJobRunner)

    def test_runner_can_be_invoked_end_to_end(self) -> None:
        """Sanity check: a Protocol-conformant runner plugs into a sink + token."""
        runner = _FullRunner()
        sink = _FullSink()
        token = ThreadingCancelToken()
        config = TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="m:train",
            output_dir=Path("/tmp/out"),
        )
        runner.prepare(config)
        result = runner.run(config, progress=sink, cancel=token)
        assert result.status is TrainingStatus.COMPLETED
