"""Benchmark tests for physics dynamics."""

import numpy as np
import pytest

# Check if pytest-benchmark is installed, otherwise skip
try:
    import pytest_benchmark
except ImportError:
    pytest.skip("pytest-benchmark not installed", allow_module_level=True)


def simulate_dynamics(n_steps=1000):
    """Simulate some dummy dynamics for benchmarking."""
    # Dummy O(N) operation
    state = np.zeros(10)
    for _ in range(n_steps):
        state += np.random.rand(10) * 0.01
    return state


def test_dynamics_performance(benchmark):
    """Benchmark the dynamics simulation."""
    # benchmark() runs the function many times and records timing
    result = benchmark(simulate_dynamics, n_steps=100)
    assert result is not None
