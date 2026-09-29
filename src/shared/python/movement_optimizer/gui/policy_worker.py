# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Background workers for Movement Optimizer policy searches."""

from __future__ import annotations

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from movement_optimizer.models.swingset import (
    DEFAULT_POLICY_DT_S,
    CyclicPolicyBounds,
    CyclicPolicySearchSpace,
    SwingSetConfig,
    optimize_cyclic_policy,
    optimize_cyclic_policy_iterative,
)


class PolicyOptimizationWorker(QThread):
    """Run swingset policy search off the GUI thread."""

    progress = pyqtSignal(int, int, float, object)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        config: SwingSetConfig,
        iterative: bool,
        steps: int,
        cycles: float,
        bounds: CyclicPolicyBounds,
        budget: int,
        seed: int,
        search_space: CyclicPolicySearchSpace,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._iterative = iterative
        self._steps = steps
        self._cycles = cycles
        self._bounds = bounds
        self._budget = budget
        self._seed = seed
        self._search_space = search_space

    def run(self) -> None:
        """Execute the selected policy search and signal the GUI thread."""

        def _progress(
            completed: int,
            total: int,
            best_score: float,
            params: object,
        ) -> None:
            self.progress.emit(completed, total, best_score, params)

        try:
            if self._iterative:
                result = optimize_cyclic_policy_iterative(
                    self._config,
                    steps=self._steps,
                    dt_s=DEFAULT_POLICY_DT_S,
                    cycles=self._cycles,
                    bounds=self._bounds,
                    budget=self._budget,
                    seed=self._seed,
                    progress_callback=_progress,
                )
            else:
                result = optimize_cyclic_policy(
                    self._config,
                    steps=self._steps,
                    dt_s=DEFAULT_POLICY_DT_S,
                    cycles=self._cycles,
                    search_space=self._search_space,
                    progress_callback=_progress,
                )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.succeeded.emit(result)
