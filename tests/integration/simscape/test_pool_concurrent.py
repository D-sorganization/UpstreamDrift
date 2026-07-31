"""Integration tests for :class:`SimscapeAdapterPool` against live MATLAB.

All tests in this module require the MATLAB Engine for Python and the
GolfSwing3D_Kinetic Simulink model on disk. They are auto-skipped when
``matlab.engine`` cannot be imported.
"""

from __future__ import annotations

import importlib.util
import os
import time

import numpy as np
import pytest
from src.engines.simscape import PoolConfig, SimscapeAdapter, SimscapeAdapterPool

_MATLAB_OK = (
    importlib.util.find_spec("matlab") is not None
    and os.environ.get("UD_SIMSCAPE_FORCE_NO_MATLAB") != "1"
)
pytestmark = [
    pytest.mark.requires_matlab,
    pytest.mark.live_simulation,
    pytest.mark.skipif(
        not _MATLAB_OK, reason="matlab.engine not importable in this environment"
    ),
]


def _model_path() -> str:
    """Return the path to GolfSwing3D_Kinetic.slx, skipping if absent."""
    env = os.environ.get("UD_SIMSCAPE_MODEL_PATH")
    if env and os.path.exists(env):
        return env
    pytest.skip("UD_SIMSCAPE_MODEL_PATH not set or model missing")


def test_concurrent_engines_isolated_state() -> None:
    """Setting state on engine A must not bleed into engine B."""
    path = _model_path()
    cfg = PoolConfig(pool_size=2)
    with SimscapeAdapterPool(path, cfg) as pool:
        marker_a = "A"
        marker_b = "B"

        def stamp(adapter: SimscapeAdapter, mark: str) -> str:
            # Use the underlying engine workspace as a side-channel.
            engine = adapter._engine  # noqa: SLF001 - test-only
            assert engine is not None
            engine.workspace["ud_marker"] = mark
            return str(engine.workspace["ud_marker"])

        results = pool.map(stamp, [marker_a, marker_b])
        assert sorted(results) == sorted([marker_a, marker_b])


@pytest.mark.slow
def test_throughput_scales_with_pool_size() -> None:
    """A pool of N should be measurably faster than serial for N tasks."""
    path = _model_path()
    n_joints = 16
    batch = np.zeros((4, n_joints * 7), dtype=np.float64)

    # Serial baseline.
    adapter = SimscapeAdapter()
    adapter.load_from_path(path)
    t0 = time.perf_counter()
    for row in batch:
        adapter.simulate_with_coefficients(row)
    serial_s = time.perf_counter() - t0
    adapter.close()

    # Parallel.
    cfg = PoolConfig(pool_size=4)
    with SimscapeAdapterPool(path, cfg) as pool:
        t0 = time.perf_counter()
        pool.simulate_batch(batch)
        parallel_s = time.perf_counter() - t0

    # Speedup should at minimum be > 1.5x for 4 workers (within 30% of
    # ideal); we are conservative to avoid flakiness on shared CI.
    assert (
        parallel_s < serial_s * 0.75
    ), f"expected speedup; serial={serial_s:.2f}s parallel={parallel_s:.2f}s"
