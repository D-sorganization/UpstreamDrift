# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Thread-safe cache for optimisation solutions."""

from __future__ import annotations

import hashlib
import json
import logging
import threading

from ..observability import metrics
from .result import OptimizationResult

logger = logging.getLogger(__name__)


class SolutionCache:
    """Thread-safe cache of optimisation results keyed by config hash."""

    def __init__(self) -> None:
        self._store: dict[str, OptimizationResult] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _config_key(
        exercise_type: str,
        body_mass: float,
        height: float,
        seg_mults: dict[str, float],
        bar_mass: float,
        duration: float,
        smoothness: float,
        bar_depth: float = 0.0,
        bar_height: float = 0.0,
    ) -> str:
        blob = json.dumps(
            {
                "ex": exercise_type,
                "bm": round(body_mass, 2),
                "h": round(height, 3),
                "sm": {k: round(v, 3) for k, v in sorted(seg_mults.items())},
                "bar": round(bar_mass, 1),
                "dur": round(duration, 2),
                "smooth": round(smoothness, 2),
                "b_dep": round(bar_depth, 3),
                "b_ht": round(bar_height, 3),
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def get(
        self,
        exercise_type: str,
        body_mass: float,
        height: float,
        seg_mults: dict[str, float],
        bar_mass: float,
        duration: float,
        smoothness: float,
        bar_depth: float = 0.0,
        bar_height: float = 0.0,
    ) -> OptimizationResult | None:
        key = self._config_key(
            exercise_type,
            body_mass,
            height,
            seg_mults,
            bar_mass,
            duration,
            smoothness,
            bar_depth,
            bar_height,
        )
        with self._lock:
            result = self._store.get(key)
        metrics.increment(
            "solution_cache_lookup_total",
            exercise_type=exercise_type,
            outcome="hit" if result is not None else "miss",
        )
        return result

    def put(
        self,
        exercise_type: str,
        body_mass: float,
        height: float,
        seg_mults: dict[str, float],
        bar_mass: float,
        duration: float,
        smoothness: float,
        result: OptimizationResult,
        bar_depth: float = 0.0,
        bar_height: float = 0.0,
    ) -> None:
        key = self._config_key(
            exercise_type,
            body_mass,
            height,
            seg_mults,
            bar_mass,
            duration,
            smoothness,
            bar_depth,
            bar_height,
        )
        with self._lock:
            self._store[key] = result
            size = len(self._store)
        metrics.increment("solution_cache_put_total", exercise_type=exercise_type)
        metrics.observe("solution_cache_entries", size)
        logger.debug("Cached solution for key=%s", key)

    def clear(self) -> None:
        with self._lock:
            size = len(self._store)
            self._store.clear()
        metrics.increment("solution_cache_clear_total")
        metrics.observe("solution_cache_entries", 0)
        logger.debug("Cleared solution cache entries=%d", size)
