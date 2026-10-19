"""Wall-clock benchmark: MOSAIC analytic derivatives beat finite differences.

This is the genuine performance claim behind ``compare_runtime``. It measures
wall-clock time, so it lives in the benchmark lane (``benchmark`` + ``slow``),
which the merge-queue ``unit-test-gate`` does not select: a loaded shared runner
can invert a ~1.3-1.5x ratio and eject a whole merge-queue batch (#13390).

The unit lane keeps deterministic contracts for the same claim in
``tests/unit/estimation/mosaic/test_analytic_derivatives.py`` (an injected
clock for the report, and regressor-evaluation counts as the work proxy).

Run with: ``python3 -m pytest tests/benchmarks/test_mosaic_analytic_derivatives_benchmark.py -m benchmark -n 0``
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.analytic_derivatives import compare_runtime
from src.shared.python.estimation.mosaic.inertial import PlanarParameterization
from src.shared.python.estimation.mosaic.planar_chain import PlanarChain

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


def test_analytic_derivatives_faster_than_finite_differences() -> None:
    chain = PlanarChain(
        link_lengths=np.array([0.9, 0.6, 0.4]),
        actuated=np.array([False, True, True]),
    )
    # Unconstrained chart coordinates, four per link (same triple pendulum as
    # the unit tests).
    theta = np.array(
        [np.log(1.3), 0.35, 0.0, np.log(0.1)]
        + [np.log(0.7), 0.2, 0.0, np.log(0.05)]
        + [np.log(0.4), 0.1, 0.0, np.log(0.02)]
    )
    pi = PlanarParameterization().from_unconstrained(theta)

    rng = np.random.default_rng(2026)
    q = rng.uniform(-1.0, 1.0, size=(10, 3))
    v = rng.uniform(-1.0, 1.0, size=(10, 3))
    a = rng.uniform(-2.0, 2.0, size=(10, 3))

    report = compare_runtime(chain, q, v, a, pi, n_repeats=50)

    assert report.max_relative_difference < 1e-4
    assert report.speedup > 1.0, report
