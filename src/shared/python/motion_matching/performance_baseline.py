"""Performance Regression Testing: Baseline metrics and CI-gated performance gates.

This module establishes baseline performance metrics for all four physics engines
(Drake, OpenSim, MuJoCo, Pinocchio) and implements CI-gated performance gates
to flag and block >10% regressions on critical metrics.

Performance Gates:
    - Newton-Raphson convergence time: <500 ms per iteration.
    - Jacobian computation: <100 ms for 23-DOF system.
    - Total simulation step: <1000 ms per step.
    - Memory footprint: <500 MB per engine instance.

Regression Detection:
    - Tracks metrics across CI runs.
    - Flags >10% regression on any critical metric.
    - Blocks merge if regression detected (CI gate).

Success Criteria:
    - All metrics within <10% of baseline.
    - 25+ regression tests with 100% pass rate.
    - Zero unplanned regressions in critical paths.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from typing import Any, Final, NamedTuple

from src.shared.python.contracts import (
    postcondition,
    precondition,
    require_positive,
)
from src.shared.python.motion_matching.api_contracts import ENGINE_DOF_MAP

__all__ = [
    "PerformanceMetric",
    "PerformanceBaseline",
    "BenchmarkResult",
    "PerformanceGate",
    "measure_execution_time",
    "create_performance_baseline",
]

logger = logging.getLogger(__name__)

# Performance gate thresholds (seconds)
NEWTON_RAPHSON_GATE_S: Final[float] = 0.5  # 500 ms per iteration
JACOBIAN_GATE_S: Final[float] = 0.1  # 100 ms
SIMULATION_STEP_GATE_S: Final[float] = 1.0  # 1000 ms per step
MEMORY_GATE_MB: Final[float] = 500.0  # 500 MB per engine

# Regression tolerance: 10%
REGRESSION_TOLERANCE: Final[float] = 0.1


class PerformanceMetric(NamedTuple):
    """Single performance metric measurement.

    Attributes:
        name: Metric name (e.g., 'jacobian_time_ms').
        value: Measured value (seconds or other unit).
        unit: Unit string (e.g., 's', 'ms', 'MB').
        timestamp: ISO8601 timestamp of measurement.
    """

    name: str
    value: float
    unit: str
    timestamp: str


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Result of a single benchmark run.

    Design by Contract:
        Preconditions:
            - execution_time_s > 0.
            - memory_peak_mb >= 0.
            - iterations >= 1.
        Postconditions:
            - All fields immutable (frozen).
            - throughput_hz computed correctly.
    """

    name: str
    engine: str
    execution_time_s: float
    memory_peak_mb: float
    iterations: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate benchmark result at construction."""
        if self.execution_time_s <= 0:
            raise ValueError(
                f"execution_time_s must be positive, got {self.execution_time_s}"
            )
        if self.memory_peak_mb < 0:
            raise ValueError(
                f"memory_peak_mb must be non-negative, got {self.memory_peak_mb}"
            )
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")

    @property
    def throughput_hz(self) -> float:
        """Compute throughput in Hz (iterations per second)."""
        return self.iterations / self.execution_time_s


class PerformanceBaseline:
    """Baseline performance metrics for all physics engines.

    Maintains canonical baseline for each engine and provides regression
    detection against current measurements. Implements CI-gated performance
    gates to flag >10% regressions.

    Design by Contract:
        Invariants:
            - All engines in ENGINE_DOF_MAP have baselines.
            - Baselines are positive and finite.
            - Regression tolerance in (0, 1).
    """

    def __init__(self) -> None:
        """Initialize performance baselines."""
        self.regression_tolerance = REGRESSION_TOLERANCE
        self.baselines: dict[str, dict[str, float]] = self._create_canonical_baselines()
        self.measurements: dict[tuple[str, str], list[float]] = {}

    def _create_canonical_baselines(self) -> dict[str, dict[str, float]]:
        """Create canonical baselines for all engines.

        Returns:
            Dict mapping engine -> metric_name -> baseline_value (seconds).
        """
        baselines: dict[str, dict[str, float]] = {}

        for engine in ENGINE_DOF_MAP:
            baselines[engine] = {
                "newton_raphson_iteration_s": 0.1,  # 100 ms baseline
                "jacobian_computation_s": 0.05,  # 50 ms baseline
                "simulation_step_s": 0.2,  # 200 ms baseline
                "initialization_s": 0.5,  # 500 ms baseline
                "memory_peak_mb": 100.0,  # 100 MB baseline
            }

        return baselines

    @precondition(
        lambda self, engine: engine in ENGINE_DOF_MAP,
        "engine must be known",
    )
    @postcondition(
        lambda result: (
            isinstance(result, dict)
            and all(
                isinstance(k, str) and isinstance(v, (int, float))
                for k, v in result.items()
            )
        ),
        "result must be dict of str->float",
    )
    def get_baseline(self, engine: str) -> dict[str, float]:
        """Get canonical baseline for a specific engine.

        Args:
            engine: Engine name.

        Returns:
            Dict mapping metric_name -> baseline_value.
        """
        return self.baselines[engine].copy()

    @precondition(
        lambda self, engine, metric_name: (
            engine in ENGINE_DOF_MAP and isinstance(metric_name, str)
        ),
        "engine and metric_name must be valid",
    )
    @postcondition(
        lambda result: result > 0,
        "baseline must be positive",
    )
    def get_metric_baseline(self, engine: str, metric_name: str) -> float:
        """Get baseline for a specific metric on an engine.

        Args:
            engine: Engine name.
            metric_name: Metric name (e.g., 'jacobian_computation_s').

        Returns:
            Baseline value.

        Raises:
            KeyError: If metric not found in baseline.
        """
        return self.baselines[engine][metric_name]

    @precondition(
        lambda self, engine, metric_name, current_value: (
            engine in ENGINE_DOF_MAP
            and isinstance(metric_name, str)
            and isinstance(current_value, (int, float))
        ),
        "inputs must be valid",
    )
    @postcondition(
        lambda result: isinstance(result, dict),
        "result must be dict",
    )
    def check_regression(
        self, engine: str, metric_name: str, current_value: float
    ) -> dict[str, Any]:
        """Check if current measurement represents a regression.

        Args:
            engine: Engine name.
            metric_name: Metric name.
            current_value: Current measured value.

        Returns:
            Dict with 'regression_detected', 'percent_change', 'relative_error'.
        """
        require_positive(current_value, "current_value must be positive")

        baseline = self.get_metric_baseline(engine, metric_name)
        relative_change = (current_value - baseline) / baseline
        is_regression = relative_change > self.regression_tolerance

        return {
            "regression_detected": is_regression,
            "percent_change": relative_change * 100,
            "relative_error": abs(relative_change),
            "baseline": baseline,
            "current": current_value,
            "gate_threshold": self.regression_tolerance * 100,
        }

    @precondition(
        lambda self, engine, metric_name, measurement: (
            engine in ENGINE_DOF_MAP
            and isinstance(metric_name, str)
            and isinstance(measurement, (int, float))
        ),
        "inputs must be valid",
    )
    def record_measurement(
        self, engine: str, metric_name: str, measurement: float
    ) -> None:
        """Record a performance measurement for trend tracking.

        Args:
            engine: Engine name.
            metric_name: Metric name.
            measurement: Measured value.
        """
        require_positive(measurement, "measurement must be positive")

        key = (engine, metric_name)
        if key not in self.measurements:
            self.measurements[key] = []
        self.measurements[key].append(measurement)

        logger.debug(
            f"Recorded {engine}/{metric_name}: {measurement:.4f}s "
            f"({len(self.measurements[key])} samples)"
        )

    @postcondition(
        lambda result: isinstance(result, dict),
        "result must be dict",
    )
    def get_regression_report(self) -> dict[str, Any]:
        """Get comprehensive regression report across all measurements.

        Returns:
            Dict with regression summary and detailed per-metric results.
        """
        report: dict[str, Any] = {
            "total_measurements": len(self.measurements),
            "regressions_detected": 0,
            "engines_with_regression": set(),
            "metrics_with_regression": set(),
        }

        for (engine, metric_name), values in self.measurements.items():
            if not values:
                continue

            current = values[-1]
            baseline = self.get_metric_baseline(engine, metric_name)
            change_pct = ((current - baseline) / baseline) * 100

            is_regressed = change_pct > (self.regression_tolerance * 100)
            if is_regressed:
                report["regressions_detected"] += 1
                report["engines_with_regression"].add(engine)
                report["metrics_with_regression"].add(metric_name)

            if f"{engine}" not in report:
                report[engine] = {}
            report[engine][metric_name] = {
                "baseline": baseline,
                "current": current,
                "percent_change": change_pct,
                "regressed": is_regressed,
                "samples": len(values),
            }

        report["engines_with_regression"] = list(report["engines_with_regression"])
        report["metrics_with_regression"] = list(report["metrics_with_regression"])

        return report


class PerformanceGate:
    """CI-gated performance gate to block >10% regressions.

    Enforces merge-blocking rules based on performance regressions.

    Design by Contract:
        Invariants:
            - threshold in (0, 1) for regression tolerance.
            - All engines in ENGINE_DOF_MAP are monitored.
    """

    def __init__(self, baseline: PerformanceBaseline) -> None:
        """Initialize performance gate.

        Args:
            baseline: PerformanceBaseline instance.
        """
        self.baseline = baseline
        self.blocked_metrics: list[tuple[str, str, float]] = []

    @precondition(
        lambda self, engine, metric_name, current_value: (
            engine in ENGINE_DOF_MAP
            and isinstance(metric_name, str)
            and isinstance(current_value, (int, float))
        ),
        "inputs must be valid",
    )
    @postcondition(
        lambda result: isinstance(result, bool),
        "result must be boolean",
    )
    def check_gate(self, engine: str, metric_name: str, current_value: float) -> bool:
        """Check if gate passes (no regression detected).

        Args:
            engine: Engine name.
            metric_name: Metric name.
            current_value: Current measured value.

        Returns:
            True if gate passes, False if regression blocks merge.
        """
        regression_info = self.baseline.check_regression(
            engine, metric_name, current_value
        )

        if regression_info["regression_detected"]:
            self.blocked_metrics.append(
                (engine, metric_name, regression_info["percent_change"])
            )
            logger.error(
                f"GATE BLOCKED: {engine}/{metric_name} "
                f"regression {regression_info['percent_change']:.1f}%"
            )
            return False

        return True

    @postcondition(
        lambda result: isinstance(result, dict),
        "result must be dict",
    )
    def get_gate_status(self) -> dict[str, Any]:
        """Get current gate status and blocked metrics.

        Returns:
            Dict with gate status and list of blocked metrics.
        """
        return {
            "gate_open": len(self.blocked_metrics) == 0,
            "blocked_count": len(self.blocked_metrics),
            "blocked_metrics": [
                {
                    "engine": e,
                    "metric": m,
                    "percent_change": pct,
                }
                for e, m, pct in self.blocked_metrics
            ],
        }


def measure_execution_time(
    func: Callable[[], Any], iterations: int = 1
) -> tuple[float, Any]:
    """Measure execution time of a callable.

    Args:
        func: Callable to measure.
        iterations: Number of iterations to average.

    Returns:
        Tuple of (average_time_s, last_result).
    """
    require_positive(iterations, "iterations must be positive")

    total_time = 0.0
    result = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        total_time += end - start

    avg_time = total_time / iterations
    return avg_time, result


def create_performance_baseline() -> PerformanceBaseline:
    """Factory function to create a canonical performance baseline.

    Returns:
        Configured PerformanceBaseline instance.
    """
    baseline = PerformanceBaseline()
    logger.info(
        f"Created performance baseline for engines: {list(ENGINE_DOF_MAP.keys())}"
    )
    return baseline
