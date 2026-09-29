"""Helpers for performance regression detection.

This module provides a tiny dependency-free harness for measuring the
median wall-clock time of a callable across N iterations and comparing it
against a checked-in baseline (``tests/benchmarks/baseline.json``).

Used by ``test_regression_benchmarks.py`` (issue #3510) to wire critical
hot paths into CI without introducing a new heavyweight dependency.
The intent is *regression detection* (catch a 5x slowdown), not micro-
benchmarking; for fine-grained measurement use pytest-benchmark.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

BASELINE_PATH = Path(__file__).with_name("baseline.json")

# Generous regression multiplier — fail only on egregious slowdowns until
# baselines are stabilized across runners. See issue #3510.
DEFAULT_REGRESSION_MULTIPLIER = 5.0
DEFAULT_MIN_REGRESSION_THRESHOLD_SECONDS = 10e-6


def measure_median_seconds(
    func: Callable[..., Any],
    *args: Any,
    iterations: int = 200,
    warmup: int = 10,
    **kwargs: Any,
) -> float:
    """Measure median per-call wall-clock time of ``func``.

    Preconditions:
        - ``iterations`` must be > 0
        - ``warmup`` must be >= 0

    Args:
        func: Callable to benchmark.
        *args: Positional arguments forwarded to ``func``.
        iterations: Number of timed invocations. Must be > 0.
        warmup: Number of untimed warmup invocations.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        Median per-invocation time in seconds.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    for _ in range(warmup):
        func(*args, **kwargs)

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def load_baseline() -> dict[str, float]:
    """Load the checked-in baseline file. Returns empty dict if missing."""
    if not BASELINE_PATH.exists():
        return {}
    try:
        data = json.loads(BASELINE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    measurements = data.get("measurements", {}) if isinstance(data, dict) else {}
    return {k: float(v) for k, v in measurements.items() if isinstance(v, int | float)}


def assert_within_regression_threshold(
    name: str,
    measured_seconds: float,
    multiplier: float = DEFAULT_REGRESSION_MULTIPLIER,
    minimum_threshold_seconds: float = DEFAULT_MIN_REGRESSION_THRESHOLD_SECONDS,
) -> None:
    """Assert ``measured_seconds`` is within ``multiplier`` x baseline for ``name``.

    Preconditions:
        - ``measured_seconds`` must be >= 0
        - ``multiplier`` must be > 1.0
        - ``minimum_threshold_seconds`` must be >= 0

    If the baseline file does not contain ``name``, this is a no-op so a fresh
    benchmark on a new metric can record a baseline without failing CI.
    """
    if measured_seconds < 0:
        raise ValueError(
            f"measured_seconds must be non-negative, got {measured_seconds}"
        )
    if multiplier <= 1.0:
        raise ValueError(f"multiplier must be > 1.0, got {multiplier}")
    if minimum_threshold_seconds < 0:
        raise ValueError(
            "minimum_threshold_seconds must be non-negative, "
            f"got {minimum_threshold_seconds}"
        )

    baseline = load_baseline().get(name)
    if baseline is None:
        return  # no baseline yet — record this run, do not fail

    threshold = max(baseline * multiplier, minimum_threshold_seconds)
    assert measured_seconds <= threshold, (
        f"Performance regression for {name!r}: "
        f"measured {measured_seconds * 1e6:.1f}us, "
        f"baseline {baseline * 1e6:.1f}us, "
        f"threshold {threshold * 1e6:.1f}us ({multiplier}x). "
        f"Investigate or update tests/benchmarks/baseline.json if intentional."
    )
