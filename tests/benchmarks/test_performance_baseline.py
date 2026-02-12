"""Performance benchmark tests for establishing baseline metrics.

These tests use pytest-benchmark to track performance over time.
Run with: pytest tests/benchmarks/ -m benchmark --benchmark-autosave
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark
class TestPhysicsBaseline:
    """Baseline benchmarks for core physics calculations."""

    def test_vector_math_baseline(self, benchmark: pytest.fixture) -> None:
        """Benchmark basic vector math operations."""

        def vector_operations() -> float:
            """Simulate a batch of 3D vector calculations."""
            total = 0.0
            for i in range(1000):
                x, y, z = float(i), float(i + 1), float(i + 2)
                magnitude = math.sqrt(x * x + y * y + z * z)
                total += magnitude
            return total

        result = benchmark(vector_operations)
        assert result > 0

    def test_trigonometry_baseline(self, benchmark: pytest.fixture) -> None:
        """Benchmark trigonometric calculations typical in physics engines."""

        def trig_batch() -> float:
            """Simulate angle-based physics calculations."""
            total = 0.0
            for i in range(1000):
                angle = float(i) * 0.01
                total += math.sin(angle) + math.cos(angle)
            return total

        result = benchmark(trig_batch)
        assert isinstance(result, float)

    def test_matrix_flatten_baseline(self, benchmark: pytest.fixture) -> None:
        """Benchmark list-based matrix operations."""

        def matrix_ops() -> list[float]:
            """Simulate a 4x4 identity matrix creation and flatten."""
            matrix = [[0.0] * 4 for _ in range(4)]
            for i in range(4):
                matrix[i][i] = 1.0
            return [v for row in matrix for v in row]

        result = benchmark(matrix_ops)
        assert len(result) == 16
