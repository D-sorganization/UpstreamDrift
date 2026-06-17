# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for lightweight runtime metrics instrumentation."""

from __future__ import annotations

import pytest
from conftest import make_test_result

from movement_optimizer.observability import InMemoryMetrics, MetricSample, metrics
from movement_optimizer.trajectory import SolutionCache


@pytest.fixture(autouse=True)
def clear_global_metrics() -> None:
    metrics.clear()


def test_in_memory_metrics_records_snapshot_and_clear() -> None:
    recorder = InMemoryMetrics()

    recorder.increment("optimizer_runs_total", exercise_type="squat", outcome="success")
    recorder.observe("optimizer_elapsed_seconds", 1.25, exercise_type="squat")

    assert recorder.snapshot() == [
        MetricSample(
            name="optimizer_runs_total",
            value=1.0,
            labels={"exercise_type": "squat", "outcome": "success"},
        ),
        MetricSample(
            name="optimizer_elapsed_seconds",
            value=1.25,
            labels={"exercise_type": "squat"},
        ),
    ]

    recorder.clear()
    assert recorder.snapshot() == []


def test_in_memory_metrics_rejects_empty_metric_name() -> None:
    recorder = InMemoryMetrics()

    with pytest.raises(ValueError, match="metric name"):
        recorder.increment("")


def test_solution_cache_records_hit_miss_put_and_clear() -> None:
    cache = SolutionCache()
    seg_mults = {"torso": 1.0}

    assert cache.get("squat", 75.0, 1.75, seg_mults, 60.0, 2.0, 1.0) is None

    result = make_test_result()
    cache.put("squat", 75.0, 1.75, seg_mults, 60.0, 2.0, 1.0, result)
    assert cache.get("squat", 75.0, 1.75, seg_mults, 60.0, 2.0, 1.0) is result

    cache.clear()

    samples = metrics.snapshot()
    assert [sample.name for sample in samples] == [
        "solution_cache_lookup_total",
        "solution_cache_put_total",
        "solution_cache_entries",
        "solution_cache_lookup_total",
        "solution_cache_clear_total",
        "solution_cache_entries",
    ]
    assert samples[0].labels == {"exercise_type": "squat", "outcome": "miss"}
    assert samples[3].labels == {"exercise_type": "squat", "outcome": "hit"}
    assert samples[2].value == 1.0
    assert samples[-1].value == 0.0


def test_trajectory_optimizer_records_completion_metrics(squat_optimizer) -> None:
    opt, _, _, _, _ = squat_optimizer

    result = opt.optimize()

    samples = metrics.snapshot()
    by_name = {sample.name: sample for sample in samples}
    assert by_name["trajectory_optimization_started_total"].labels == {
        "exercise_type": "squat",
        "mode": "single",
    }
    assert by_name["trajectory_optimization_completed_total"].labels == {
        "exercise_type": "squat",
        "mode": "single",
        "outcome": "success" if result.success else "failed",
    }
    assert by_name["trajectory_optimization_elapsed_seconds"].value == result.elapsed_s
    assert by_name["trajectory_optimization_cost"].value == result.cost
    assert by_name["trajectory_optimization_evaluations"].value == float(result.n_evals)
