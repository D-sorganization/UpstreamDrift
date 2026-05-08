"""Unit tests for performance regression testing and CI gates.

Validates:
    - PerformanceBaseline canonical metrics and regression detection.
    - PerformanceGate CI-gated merge blocking on >10% regression.
    - BenchmarkResult throughput computation.
    - Performance measurement recording and trend analysis.
"""

from __future__ import annotations

import pytest
import numpy as np
from src.shared.python.motion_matching.performance_baseline import (
    REGRESSION_TOLERANCE,
    BenchmarkResult,
    PerformanceBaseline,
    PerformanceGate,
    PerformanceMetric,
    create_performance_baseline,
    measure_execution_time,
)


class TestBenchmarkResult:
    """Test cases for BenchmarkResult."""

    def test_benchmark_result_valid_creation(self) -> None:
        """Test creating valid benchmark result."""
        result = BenchmarkResult(
            name="jacobian_test",
            engine="drake",
            execution_time_s=0.05,
            memory_peak_mb=150.0,
            iterations=10,
        )
        assert result.name == "jacobian_test"
        assert result.engine == "drake"
        assert result.execution_time_s == 0.05

    def test_benchmark_result_throughput_property(self) -> None:
        """Test throughput computation."""
        result = BenchmarkResult(
            name="test",
            engine="drake",
            execution_time_s=1.0,
            memory_peak_mb=100.0,
            iterations=10,
        )
        assert result.throughput_hz == pytest.approx(10.0)

    def test_benchmark_result_zero_execution_time_raises(self) -> None:
        """Test zero execution time raises."""
        with pytest.raises(ValueError, match="execution_time_s must be positive"):
            BenchmarkResult(
                name="test",
                engine="drake",
                execution_time_s=0.0,
                memory_peak_mb=100.0,
                iterations=10,
            )

    def test_benchmark_result_negative_memory_raises(self) -> None:
        """Test negative memory raises."""
        with pytest.raises(ValueError, match="memory_peak_mb must be non-negative"):
            BenchmarkResult(
                name="test",
                engine="drake",
                execution_time_s=0.1,
                memory_peak_mb=-50.0,
                iterations=10,
            )

    def test_benchmark_result_zero_iterations_raises(self) -> None:
        """Test zero iterations raises."""
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            BenchmarkResult(
                name="test",
                engine="drake",
                execution_time_s=0.1,
                memory_peak_mb=100.0,
                iterations=0,
            )

    def test_benchmark_result_frozen(self) -> None:
        """Test BenchmarkResult is immutable."""
        result = BenchmarkResult(
            name="test",
            engine="drake",
            execution_time_s=0.1,
            memory_peak_mb=100.0,
            iterations=10,
        )
        with pytest.raises(AttributeError):
            result.execution_time_s = 0.2  # type: ignore


class TestPerformanceBaseline:
    """Test cases for PerformanceBaseline."""

    def test_baseline_initialization(self) -> None:
        """Test baseline initializes with all engines."""
        baseline = PerformanceBaseline()
        assert len(baseline.baselines) == 4
        assert "drake" in baseline.baselines
        assert "mujoco" in baseline.baselines

    def test_get_baseline_valid_engine(self) -> None:
        """Test getting baseline for valid engine."""
        baseline = PerformanceBaseline()
        metrics = baseline.get_baseline("drake")
        assert isinstance(metrics, dict)
        assert "jacobian_computation_s" in metrics

    def test_get_baseline_invalid_engine(self) -> None:
        """Test getting baseline for invalid engine."""
        baseline = PerformanceBaseline()
        with pytest.raises(ValueError):
            baseline.get_baseline("invalid_engine")

    def test_get_metric_baseline(self) -> None:
        """Test getting specific metric baseline."""
        baseline = PerformanceBaseline()
        value = baseline.get_metric_baseline("drake", "jacobian_computation_s")
        assert value == 0.05

    def test_get_metric_baseline_missing_metric(self) -> None:
        """Test getting missing metric raises."""
        baseline = PerformanceBaseline()
        with pytest.raises(KeyError):
            baseline.get_metric_baseline("drake", "nonexistent_metric")

    def test_check_regression_no_regression(self) -> None:
        """Test regression check detects no regression."""
        baseline = PerformanceBaseline()
        result = baseline.check_regression(
            engine="drake",
            metric_name="jacobian_computation_s",
            current_value=0.055,  # 10% higher, within tolerance
        )
        assert result["regression_detected"] is False
        assert result["percent_change"] == pytest.approx(10.0, abs=1.0)

    def test_check_regression_detects_regression(self) -> None:
        """Test regression check detects regression."""
        baseline = PerformanceBaseline()
        result = baseline.check_regression(
            engine="drake",
            metric_name="jacobian_computation_s",
            current_value=0.07,  # 40% higher, exceeds tolerance
        )
        assert result["regression_detected"] is True
        assert result["percent_change"] > 30.0

    def test_check_regression_improvement(self) -> None:
        """Test regression check for improvements."""
        baseline = PerformanceBaseline()
        result = baseline.check_regression(
            engine="drake",
            metric_name="jacobian_computation_s",
            current_value=0.03,  # 40% faster
        )
        assert result["regression_detected"] is False

    def test_record_measurement(self) -> None:
        """Test recording measurements."""
        baseline = PerformanceBaseline()
        baseline.record_measurement(
            engine="drake",
            metric_name="jacobian_computation_s",
            measurement=0.048,
        )
        key = ("drake", "jacobian_computation_s")
        assert key in baseline.measurements
        assert len(baseline.measurements[key]) == 1
        assert baseline.measurements[key][0] == 0.048

    def test_record_multiple_measurements(self) -> None:
        """Test recording multiple measurements for trend."""
        baseline = PerformanceBaseline()
        for value in [0.048, 0.052, 0.050]:
            baseline.record_measurement(
                engine="drake",
                metric_name="jacobian_computation_s",
                measurement=value,
            )
        key = ("drake", "jacobian_computation_s")
        assert len(baseline.measurements[key]) == 3

    def test_get_regression_report(self) -> None:
        """Test getting regression report."""
        baseline = PerformanceBaseline()
        baseline.record_measurement(
            engine="drake",
            metric_name="jacobian_computation_s",
            measurement=0.048,
        )
        baseline.record_measurement(
            engine="mujoco",
            metric_name="jacobian_computation_s",
            measurement=0.15,  # Regression
        )
        report = baseline.get_regression_report()
        assert report["total_measurements"] == 2
        assert report["regressions_detected"] == 1


class TestPerformanceGate:
    """Test cases for PerformanceGate CI gating."""

    def test_gate_creation(self) -> None:
        """Test creating performance gate."""
        baseline = PerformanceBaseline()
        gate = PerformanceGate(baseline)
        assert gate.baseline is baseline
        assert len(gate.blocked_metrics) == 0

    def test_gate_pass_no_regression(self) -> None:
        """Test gate passes when no regression."""
        baseline = PerformanceBaseline()
        gate = PerformanceGate(baseline)
        result = gate.check_gate(
            engine="drake",
            metric_name="jacobian_computation_s",
            current_value=0.048,
        )
        assert result is True
        assert len(gate.blocked_metrics) == 0

    def test_gate_block_on_regression(self) -> None:
        """Test gate blocks on regression."""
        baseline = PerformanceBaseline()
        gate = PerformanceGate(baseline)
        result = gate.check_gate(
            engine="drake",
            metric_name="jacobian_computation_s",
            current_value=0.08,  # 60% regression
        )
        assert result is False
        assert len(gate.blocked_metrics) == 1

    def test_gate_multiple_regressions(self) -> None:
        """Test gate tracks multiple regressions."""
        baseline = PerformanceBaseline()
        gate = PerformanceGate(baseline)
        gate.check_gate(
            engine="drake",
            metric_name="jacobian_computation_s",
            current_value=0.08,
        )
        gate.check_gate(
            engine="mujoco",
            metric_name="simulation_step_s",
            current_value=2.5,
        )
        assert len(gate.blocked_metrics) == 2

    def test_gate_status(self) -> None:
        """Test getting gate status."""
        baseline = PerformanceBaseline()
        gate = PerformanceGate(baseline)
        gate.check_gate(
            engine="drake",
            metric_name="jacobian_computation_s",
            current_value=0.08,
        )
        status = gate.get_gate_status()
        assert status["gate_open"] is False
        assert status["blocked_count"] == 1


class TestPerformanceMetric:
    """Test cases for PerformanceMetric."""

    def test_metric_creation(self) -> None:
        """Test creating performance metric."""
        metric = PerformanceMetric(
            name="jacobian_time_ms",
            value=50.0,
            unit="ms",
            timestamp="2026-05-06T12:00:00Z",
        )
        assert metric.name == "jacobian_time_ms"
        assert metric.value == 50.0
        assert metric.unit == "ms"

    def test_metric_is_namedtuple(self) -> None:
        """Test PerformanceMetric is immutable namedtuple."""
        metric = PerformanceMetric(
            name="test",
            value=1.0,
            unit="s",
            timestamp="2026-05-06T12:00:00Z",
        )
        with pytest.raises(AttributeError):
            metric.value = 2.0  # type: ignore


class TestMeasureExecutionTime:
    """Test cases for measure_execution_time function."""

    def test_measure_single_iteration(self) -> None:
        """Test measuring single function call."""
        def dummy_func() -> int:
            return 42

        duration, result = measure_execution_time(dummy_func, iterations=1)
        assert duration >= 0.0
        assert result == 42

    def test_measure_multiple_iterations(self) -> None:
        """Test measuring multiple iterations."""
        import time

        def slow_func() -> int:
            time.sleep(0.0005)  # 0.5 ms
            return 42

        duration, result = measure_execution_time(slow_func, iterations=3)
        assert duration > 0.0  # At least some time recorded
        assert result == 42

    def test_measure_zero_iterations_raises(self) -> None:
        """Test zero iterations raises."""
        def dummy_func() -> int:
            return 42

        with pytest.raises(ValueError):
            measure_execution_time(dummy_func, iterations=0)


class TestCreatePerformanceBaseline:
    """Test cases for create_performance_baseline factory."""

    def test_create_baseline(self) -> None:
        """Test factory creates valid baseline."""
        baseline = create_performance_baseline()
        assert isinstance(baseline, PerformanceBaseline)
        assert len(baseline.baselines) == 4

    def test_created_baseline_is_functional(self) -> None:
        """Test created baseline is immediately functional."""
        baseline = create_performance_baseline()
        metrics = baseline.get_baseline("drake")
        assert len(metrics) > 0
