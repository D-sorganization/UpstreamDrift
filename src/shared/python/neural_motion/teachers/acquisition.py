"""Active-learning candidate selection that never consumes test labels."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from src.shared.python.neural_motion.episodes.splits import FamilySplitPlan
from src.shared.python.neural_motion.episodes.store import EpisodeStore

from .types import (
    ACQUISITION_SCHEMA,
    AcquisitionEntry,
    AcquisitionLog,
    AcquisitionStrategy,
)

__all__ = ["ActiveLearningAcquirer"]


class ActiveLearningAcquirer:
    """Select acquisition candidates from non-test splits only.

    Strategies:
    - ``UNCERTAINTY`` — highest provided uncertainty score
    - ``COVERAGE`` — highest provided coverage gap score
    - ``RANDOM_CONTROL`` — seeded random among eligible trials

    Requesting a pool that is entirely in the test split fails closed.
    """

    def __init__(
        self,
        *,
        store: EpisodeStore,
        split_plan: FamilySplitPlan,
        rng_seed: int,
    ) -> None:
        if not isinstance(store, EpisodeStore):
            raise TypeError("store must be an EpisodeStore")
        if not isinstance(split_plan, FamilySplitPlan):
            raise TypeError("split_plan must be a FamilySplitPlan")
        self._store = store
        self._split_plan = split_plan
        self._rng = np.random.default_rng(rng_seed)

    def select(
        self,
        *,
        candidate_trial_ids: Sequence[str],
        n: int,
        strategies: Sequence[AcquisitionStrategy],
        scores: Mapping[str, float],
        coverage: Mapping[str, float],
    ) -> AcquisitionLog:
        if n < 1:
            raise ValueError("n must be >= 1")
        if not strategies:
            raise ValueError("strategies must be non-empty")
        if not candidate_trial_ids:
            raise ValueError("candidate_trial_ids must be non-empty")

        eligible: list[str] = []
        test_hits = 0
        for trial_id in candidate_trial_ids:
            split = self._split_plan.split_of(trial_id)
            if split == "test":
                test_hits += 1
                continue
            eligible.append(trial_id)

        if not eligible:
            raise ValueError(
                "active acquisition cannot consume test labels; "
                "candidate pool contains only test-split trials"
            )

        selected: list[AcquisitionEntry] = []
        remaining = list(dict.fromkeys(eligible))
        strategy_cycle = list(strategies)
        step = 0
        while remaining and len(selected) < n:
            strategy = strategy_cycle[step % len(strategy_cycle)]
            step += 1
            pick = self._pick(strategy, remaining, scores, coverage)
            remaining.remove(pick)
            selected.append(
                AcquisitionEntry(
                    trial_id=pick,
                    strategy=strategy,
                    score=self._score_for(strategy, pick, scores, coverage),
                )
            )

        digest = _digest(
            {
                "selected": [entry.as_dict() for entry in selected],
                "test_hits_ignored": test_hits,
            }
        )
        return AcquisitionLog(
            schema=ACQUISITION_SCHEMA,
            selected=tuple(selected),
            test_labels_consumed=0,
            content_digest=digest,
        )

    def _pick(
        self,
        strategy: AcquisitionStrategy,
        remaining: Sequence[str],
        scores: Mapping[str, float],
        coverage: Mapping[str, float],
    ) -> str:
        if strategy is AcquisitionStrategy.RANDOM_CONTROL:
            return str(self._rng.choice(list(remaining)))
        if strategy is AcquisitionStrategy.UNCERTAINTY:
            return max(remaining, key=lambda tid: float(scores.get(tid, 0.0)))
        if strategy is AcquisitionStrategy.COVERAGE:
            return max(remaining, key=lambda tid: float(coverage.get(tid, 0.0)))
        raise ValueError(f"unsupported strategy {strategy!r}")

    @staticmethod
    def _score_for(
        strategy: AcquisitionStrategy,
        trial_id: str,
        scores: Mapping[str, float],
        coverage: Mapping[str, float],
    ) -> float:
        if strategy is AcquisitionStrategy.COVERAGE:
            return float(coverage.get(trial_id, 0.0))
        if strategy is AcquisitionStrategy.UNCERTAINTY:
            return float(scores.get(trial_id, 0.0))
        return 0.0


def _digest(payload: Mapping[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
