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
"""

from __future__ import annotations

import threading

from ..config import TrainingConfig, TrainingFramework
from ..contracts import TrainingJobRunner
from ..errors import TrainingError

__all__ = ["NoRunnerAvailableError", "RunnerRegistry"]


class NoRunnerAvailableError(TrainingError, LookupError):
    """Raised when no registered runner can handle a config."""


class RunnerRegistry:
    """Thread-safe lookup table of :class:`TrainingJobRunner` adapters."""

    __slots__ = ("_lock", "_runners")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runners: dict[TrainingFramework, TrainingJobRunner] = {}

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

    def unregister(self, framework: TrainingFramework) -> None:
        """Remove the runner for ``framework`` if present."""

        with self._lock:
            self._runners.pop(framework, None)

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

        Goes through two filters: framework match first, then the
        runner's own :meth:`can_run` predicate (which may decline a
        config that demands a resource the runner cannot supply).

        Raises:
            NoRunnerAvailableError: When no registered runner matches
                or all candidates decline via ``can_run``.
        """

        runner = self.get(config.framework)
        if not runner.can_run(config):
            raise NoRunnerAvailableError(
                f"runner for {config.framework.value!r} declined the job "
                f"(can_run returned False)"
            )
        return runner

    def frameworks(self) -> frozenset[TrainingFramework]:
        """Snapshot of registered frameworks."""

        with self._lock:
            return frozenset(self._runners.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._runners)
