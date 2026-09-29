"""Unit tests for :class:`SimscapeAdapterPool`.

These tests substitute a stub adapter for :class:`SimscapeAdapter` so
they run on hosts without a MATLAB Engine. Live tests live in
``tests/integration/simscape/test_pool_concurrent.py`` and are gated by
the ``requires_matlab`` marker.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from src.engines.simscape import PoolConfig, SimscapeAdapterPool
from src.engines.simscape._output import SimscapeOutput


def _fake_output(n_joints: int = 1) -> SimscapeOutput:
    """Return a minimal valid :class:`SimscapeOutput`."""
    n = 2
    time_arr = np.array([0.0, 0.01], dtype=np.float64)
    zeros_j = np.zeros((n, n_joints), dtype=np.float64)
    zeros_3 = np.zeros((n, 3), dtype=np.float64)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return SimscapeOutput(
        time=time_arr,
        q=zeros_j,
        qd=zeros_j.copy(),
        qdd=zeros_j.copy(),
        tau=zeros_j.copy(),
        omega=zeros_j.copy(),
        r_butt=zeros_3.copy(),
        r_clubhead=zeros_3.copy(),
        q_club=quat,
        v_clubhead=zeros_3.copy(),
    )


class _StubAdapter:
    """Drop-in replacement for :class:`SimscapeAdapter` in the pool tests.

    Tracks construction, load, simulate calls, and close. The stub
    intentionally does NOT inherit from SimscapeAdapter (the pool only
    needs duck-typed methods).
    """

    instances: list[_StubAdapter] = []

    def __init__(
        self,
        cache_max_entries: int = 16,
        startup_timeout_s: float = 90.0,
    ) -> None:
        self.cache_max_entries = cache_max_entries
        self.startup_timeout_s = startup_timeout_s
        self.loaded_path: str | None = None
        self.simulate_calls: list[np.ndarray] = []
        self.closed = False
        self.thread_ids: list[int] = []
        _StubAdapter.instances.append(self)

    def load_from_path(self, path: str) -> None:
        self.loaded_path = path

    def simulate_with_coefficients(
        self, coeffs: np.ndarray, *, opts: dict[str, Any] | None = None
    ) -> SimscapeOutput:
        self.simulate_calls.append(coeffs.copy())
        self.thread_ids.append(threading.get_ident())
        # Sleep briefly so concurrency is observable.
        time.sleep(0.005)
        # Return an output whose first-row q encodes the input's first
        # value so we can assert ordering.
        out = _fake_output()
        # We cannot mutate the frozen dataclass; fall back to attaching
        # the coeffs via a side-channel for assertions.
        object.__setattr__(out, "_marker", float(coeffs[0]))  # type: ignore[misc]
        return out

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_stub_instances() -> None:
    _StubAdapter.instances.clear()


@pytest.fixture
def slx_path(tmp_path: Path) -> Path:
    p = tmp_path / "Model.slx"
    p.write_bytes(b"FAKE")
    return p


def test_pool_init_with_invalid_size_raises(slx_path: Path) -> None:
    with pytest.raises(ValueError, match="pool_size"):
        SimscapeAdapterPool(slx_path, PoolConfig(pool_size=0))
    with pytest.raises(ValueError, match="pool_size"):
        SimscapeAdapterPool(slx_path, PoolConfig(pool_size=-1))


def test_pool_init_rejects_non_slx_path(tmp_path: Path) -> None:
    bad = tmp_path / "model.mdl"
    bad.write_bytes(b"FAKE")
    with pytest.raises(ValueError, match=".slx"):
        SimscapeAdapterPool(bad)


def test_pool_init_rejects_negative_cache_capacity(slx_path: Path) -> None:
    with pytest.raises(ValueError, match="cache_capacity"):
        SimscapeAdapterPool(
            slx_path,
            PoolConfig(pool_size=2, cache_capacity_per_worker=-1),
        )


def test_simulate_batch_preserves_order(slx_path: Path) -> None:
    cfg = PoolConfig(pool_size=4)
    pool = SimscapeAdapterPool(slx_path, cfg, adapter_factory=_StubAdapter)
    try:
        batch = np.arange(8 * 7, dtype=np.float64).reshape(8, 7)
        # Force distinct first-column values so we can verify ordering.
        for i in range(8):
            batch[i, 0] = float(i)
        outputs = pool.simulate_batch(batch)
        assert len(outputs) == 8
        markers = [o._marker for o in outputs]  # type: ignore[attr-defined]
        assert markers == [float(i) for i in range(8)]
    finally:
        pool.close()


def test_pool_size_one_equivalent_to_single_adapter(slx_path: Path) -> None:
    cfg = PoolConfig(pool_size=1)
    pool = SimscapeAdapterPool(slx_path, cfg, adapter_factory=_StubAdapter)
    try:
        batch = np.zeros((3, 7), dtype=np.float64)
        for i in range(3):
            batch[i, 0] = float(i)
        outputs = pool.simulate_batch(batch)
        assert len(outputs) == 3
        # Only one adapter instance should ever be constructed.
        assert len(_StubAdapter.instances) == 1
        adapter = _StubAdapter.instances[0]
        assert len(adapter.simulate_calls) == 3
    finally:
        pool.close()


def test_close_quits_all_workers(slx_path: Path) -> None:
    cfg = PoolConfig(pool_size=3)
    pool = SimscapeAdapterPool(slx_path, cfg, adapter_factory=_StubAdapter)
    # Force every worker to instantiate by dispatching pool_size items.
    pool.simulate_batch(np.zeros((6, 7), dtype=np.float64))
    constructed = list(_StubAdapter.instances)
    assert len(constructed) >= 1  # at least one; up to pool_size
    pool.close()
    for adapter in constructed:
        assert adapter.closed is True


def test_close_is_idempotent(slx_path: Path) -> None:
    pool = SimscapeAdapterPool(
        slx_path, PoolConfig(pool_size=2), adapter_factory=_StubAdapter
    )
    pool.close()
    pool.close()  # must not raise


def test_pool_as_context_manager_closes_on_exit(slx_path: Path) -> None:
    with SimscapeAdapterPool(
        slx_path, PoolConfig(pool_size=2), adapter_factory=_StubAdapter
    ) as pool:
        pool.simulate_batch(np.zeros((2, 7), dtype=np.float64))
        constructed = list(_StubAdapter.instances)
    for adapter in constructed:
        assert adapter.closed is True


def test_simulate_batch_after_close_raises(slx_path: Path) -> None:
    pool = SimscapeAdapterPool(
        slx_path, PoolConfig(pool_size=2), adapter_factory=_StubAdapter
    )
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.simulate_batch(np.zeros((1, 7), dtype=np.float64))


def test_simulate_batch_rejects_wrong_shape(slx_path: Path) -> None:
    pool = SimscapeAdapterPool(
        slx_path, PoolConfig(pool_size=1), adapter_factory=_StubAdapter
    )
    try:
        with pytest.raises(ValueError, match="multiple of 7"):
            pool.simulate_batch(np.zeros((2, 5), dtype=np.float64))
        with pytest.raises(ValueError):
            pool.simulate_batch(np.zeros((7,), dtype=np.float64))
        with pytest.raises(TypeError):
            pool.simulate_batch([1, 2, 3])  # type: ignore[arg-type]
    finally:
        pool.close()


def test_simulate_batch_handles_worker_failure_gracefully(
    slx_path: Path,
) -> None:
    class _FlakyAdapter(_StubAdapter):
        def simulate_with_coefficients(
            self,
            coeffs: np.ndarray,
            *,
            opts: dict[str, Any] | None = None,
        ) -> SimscapeOutput:
            if float(coeffs[0]) == 1.0:
                raise RuntimeError("simulated worker failure")
            return super().simulate_with_coefficients(coeffs, opts=opts)

    pool = SimscapeAdapterPool(
        slx_path, PoolConfig(pool_size=2), adapter_factory=_FlakyAdapter
    )
    try:
        batch = np.zeros((3, 7), dtype=np.float64)
        batch[1, 0] = 1.0  # poison the middle row
        with pytest.raises(RuntimeError, match="simulated worker failure"):
            pool.simulate_batch(batch)
        # Pool remains usable after a worker exception (executor stays
        # up; worker can still service other requests).
        good = np.zeros((2, 7), dtype=np.float64)
        good[0, 0] = 2.0
        good[1, 0] = 3.0
        outputs = pool.simulate_batch(good)
        assert len(outputs) == 2
    finally:
        pool.close()


def test_map_dispatches_in_order(slx_path: Path) -> None:
    pool = SimscapeAdapterPool(
        slx_path, PoolConfig(pool_size=3), adapter_factory=_StubAdapter
    )
    try:
        items = list(range(10))
        results = pool.map(lambda _adapter, x: x * x, items)
        assert results == [x * x for x in items]
    finally:
        pool.close()


def test_map_empty_returns_empty(slx_path: Path) -> None:
    pool = SimscapeAdapterPool(
        slx_path, PoolConfig(pool_size=2), adapter_factory=_StubAdapter
    )
    try:
        assert pool.map(lambda _a, x: x, []) == []
    finally:
        pool.close()


def test_pool_size_property(slx_path: Path) -> None:
    pool = SimscapeAdapterPool(
        slx_path, PoolConfig(pool_size=5), adapter_factory=_StubAdapter
    )
    try:
        assert pool.pool_size == 5
    finally:
        pool.close()
