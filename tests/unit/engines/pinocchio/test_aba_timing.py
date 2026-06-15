from __future__ import annotations

import pytest

from src.engines.pinocchio.benchmarks.aba_timing import run_aba_timing_benchmark

pytestmark = pytest.mark.unit


def test_aba_timing_benchmark_reports_unavailable_when_pinocchio_missing() -> None:
    result = run_aba_timing_benchmark(iterations=2, pinocchio_module=None)

    assert result.available is False
    assert result.iterations == 0
    assert result.mean_seconds is None
    assert "Pinocchio" in result.reason


def test_aba_timing_benchmark_validates_iterations() -> None:
    result = run_aba_timing_benchmark(iterations=0, pinocchio_module=None)

    assert result.available is False
    assert "positive" in result.reason
