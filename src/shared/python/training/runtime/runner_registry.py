"""Registry mapping :class:`TrainingFramework` → :class:`TrainingJobRunner`.

The scheduler does not import framework adapters directly — it asks
the registry for a runner that handles a given :class:`TrainingConfig`.
This decouples scheduler-level code from PyTorch / gymnasium /
TensorFlow imports, so headless CI (and the contracts test suite) can
exercise scheduling without paying the framework import cost.

Registering is explicit and idempotent (re-registering an adapter for
the same framework replaces it). Lookup raises :class:`KeyError` when
no adapter matches — the scheduler converts that into a clear
``FAILED`` status with a domain error message.

NM-05 (#10620) classical / small-MLP dynamics pilots admit via
:func:`~src.shared.python.training.scheduler.neural_dynamics_baseline_budget`
and run through ``DynamicsBaselineTrainer`` — they are intentionally
outside this framework-adapter registry (reuse anchor retained here).
"""

from __future__ import annotations

import threading

from ..config import TrainingConfig, TrainingFramework
from ..contracts import TrainingJobRunner
from ..errors import TrainingError

__all__ = [
    "NM05_DYNAMICS_BASELINE_SCHEMA",
    "NoRunnerAvailableError",
    "RunnerRegistry",
]

# Wire id for NM-05 pilot receipts; mirrors neural_motion.baselines.BASELINE_SCHEMA
# without importing the neural package at registry import time.
NM05_DYNAMICS_BASELINE_SCHEMA = "neural-dynamics-baselines/1.0.0"


class NoRunnerAvailableError(TrainingError, LookupError):
    """Raised when no registered runner can handle a config."""


class RunnerRegistry:
    """Thread-safe lookup table of :class:`TrainingJobRunner` adapters."""

    __slots__ = ("_all_runners", "_lock", "_runners")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runners: dict[TrainingFramework, TrainingJobRunner] = {}
        self._all_runners: dict[TrainingFramework, list[TrainingJobRunner]] = {}

    def register(self, runner: TrainingJobRunner) -> None:
        """Register a runner adapter.

        Raises:
            TypeError: When ``runner`` does not satisfy the
                :class:`TrainingJobRunner` Protocol or its
                ``framework`` attribute is not a
                :class:`TrainingFramework`.
        """

        if not isinstance(runner, TrainingJobRunner):
            raise TypeError("runner does not satisfy the TrainingJobRunner Protocol")
        framework = getattr(runner, "framework", None)
        if not isinstance(framework, TrainingFramework):
            raise TypeError(
                f"runner.framework must be a TrainingFramework (got {framework!r})"
            )
        with self._lock:
            self._runners[framework] = runner
            if framework not in self._all_runners:
                self._all_runners[framework] = []
            if runner not in self._all_runners[framework]:
                self._all_runners[framework].append(runner)

    def unregister(self, framework: TrainingFramework) -> None:
        """Remove the runner for ``framework`` if present."""

        with self._lock:
            self._runners.pop(framework, None)
            self._all_runners.pop(framework, None)

    def get(self, framework: TrainingFramework) -> TrainingJobRunner:
        """Return the registered runner for ``framework``.

        Raises:
            NoRunnerAvailableError: When no runner is registered for
                the framework.
        """

        with self._lock:
            try:
                return self._runners[framework]
            except KeyError as exc:
                raise NoRunnerAvailableError(
                    f"no runner registered for framework {framework.value!r}"
                ) from exc

    def resolve(self, config: TrainingConfig) -> TrainingJobRunner:
        """Return the runner that can handle ``config``.

        Goes through two filters: framework match first, then candidate
        runner :meth:`can_run` predicates.

        Raises:
            NoRunnerAvailableError: When no registered runner matches
                or all candidates decline via ``can_run``.
        """

        with self._lock:
            candidates = list(reversed(self._all_runners.get(config.framework, [])))
            if not candidates and config.framework in self._runners:
                candidates = [self._runners[config.framework]]

        if not candidates:
            raise NoRunnerAvailableError(
                f"no runner registered for framework {config.framework.value!r}"
            )

        for runner in candidates:
            if runner.can_run(config):
                return runner

        raise NoRunnerAvailableError(
            f"all runners for {config.framework.value!r} declined the job "
            f"(can_run returned False)"
        )

    def frameworks(self) -> frozenset[TrainingFramework]:
        """Snapshot of registered frameworks."""

        with self._lock:
            return frozenset(self._runners.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._runners)
