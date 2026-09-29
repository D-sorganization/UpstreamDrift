"""Tests for :mod:`training.runtime.runner_registry`."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import (
    RunResult,
    TrainingConfig,
    TrainingFramework,
    TrainingStatus,
    new_run_id,
)
from training.contracts import CancelToken, ProgressSink
from training.runtime import RunnerRegistry
from training.runtime.runner_registry import NoRunnerAvailableError

pytestmark = pytest.mark.unit


def _pytorch_config() -> TrainingConfig:
    return TrainingConfig(
        framework=TrainingFramework.PYTORCH,
        entry_point="m:train",
        output_dir=Path("/tmp/out"),
    )


class _FakeRunner:
    def __init__(
        self,
        framework: TrainingFramework,
        *,
        accepts: bool = True,
    ) -> None:
        self.framework = framework
        self._accepts = accepts

    def can_run(self, config: TrainingConfig) -> bool:
        return self._accepts

    def prepare(self, config: TrainingConfig) -> None:
        return None

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


class TestRunnerRegistryRegister:
    def test_register_and_get(self) -> None:
        registry = RunnerRegistry()
        runner = _FakeRunner(TrainingFramework.PYTORCH)
        registry.register(runner)
        assert registry.get(TrainingFramework.PYTORCH) is runner

    def test_register_replaces_existing(self) -> None:
        registry = RunnerRegistry()
        a = _FakeRunner(TrainingFramework.PYTORCH)
        b = _FakeRunner(TrainingFramework.PYTORCH)
        registry.register(a)
        registry.register(b)
        assert registry.get(TrainingFramework.PYTORCH) is b

    def test_register_rejects_non_runner(self) -> None:
        registry = RunnerRegistry()
        with pytest.raises(TypeError):
            registry.register("not a runner")  # type: ignore[arg-type]

    def test_register_rejects_runner_without_framework_attr(self) -> None:
        class _Bad:
            framework = "pytorch"  # wrong type

            def can_run(self, c: TrainingConfig) -> bool:
                return True

            def prepare(self, c: TrainingConfig) -> None:
                return None

            def run(self, c, *, progress, cancel):  # type: ignore[no-untyped-def]
                return RunResult(
                    run_id=new_run_id(),
                    status=TrainingStatus.COMPLETED,
                    duration_s=0.0,
                )

        with pytest.raises(TypeError):
            RunnerRegistry().register(_Bad())  # type: ignore[arg-type]

    def test_unregister(self) -> None:
        registry = RunnerRegistry()
        registry.register(_FakeRunner(TrainingFramework.PYTORCH))
        registry.unregister(TrainingFramework.PYTORCH)
        with pytest.raises(NoRunnerAvailableError):
            registry.get(TrainingFramework.PYTORCH)


class TestRunnerRegistryResolve:
    def test_resolve_uses_can_run(self) -> None:
        registry = RunnerRegistry()
        accepting = _FakeRunner(TrainingFramework.PYTORCH, accepts=True)
        registry.register(accepting)
        assert registry.resolve(_pytorch_config()) is accepting

    def test_resolve_rejects_when_can_run_false(self) -> None:
        registry = RunnerRegistry()
        registry.register(_FakeRunner(TrainingFramework.PYTORCH, accepts=False))
        with pytest.raises(NoRunnerAvailableError, match="declined"):
            registry.resolve(_pytorch_config())

    def test_resolve_missing_framework_raises(self) -> None:
        registry = RunnerRegistry()
        with pytest.raises(NoRunnerAvailableError, match="no runner registered"):
            registry.resolve(_pytorch_config())


class TestRunnerRegistryIntrospection:
    def test_frameworks_snapshot(self) -> None:
        registry = RunnerRegistry()
        registry.register(_FakeRunner(TrainingFramework.PYTORCH))
        registry.register(_FakeRunner(TrainingFramework.GYMNASIUM))
        assert registry.frameworks() == {
            TrainingFramework.PYTORCH,
            TrainingFramework.GYMNASIUM,
        }

    def test_len(self) -> None:
        registry = RunnerRegistry()
        assert len(registry) == 0
        registry.register(_FakeRunner(TrainingFramework.PYTORCH))
        assert len(registry) == 1
