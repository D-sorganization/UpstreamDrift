"""Tests for src.engines.simscape.pool and _pool_worker (offline, mocked adapter)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.simscape._output import SimscapeOutput
from src.engines.simscape._pool_worker import _PoolWorker
from src.engines.simscape.pool import PoolConfig, SimscapeAdapterPool


def _fake_output(n: int = 2, n_joints: int = 2) -> SimscapeOutput:
    return SimscapeOutput(
        time=np.linspace(0, 0.1, n),
        q=np.zeros((n, n_joints)),
        qd=np.zeros((n, n_joints)),
        qdd=np.zeros((n, n_joints)),
        tau=np.zeros((n, n_joints)),
        omega=np.zeros((n, n_joints)),
        r_butt=np.zeros((n, 3)),
        r_clubhead=np.zeros((n, 3)),
        q_club=np.tile([1.0, 0, 0, 0], (n, 1)),
        v_clubhead=np.zeros((n, 3)),
    )


class _FakeAdapter:
    """Minimal stand-in for SimscapeAdapter inside the pool worker."""

    instances: list[_FakeAdapter] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.loaded: str | None = None
        self.closed = False
        _FakeAdapter.instances.append(self)

    def load_from_path(self, path: str) -> None:
        self.loaded = path

    def simulate_with_coefficients(self, coeffs: np.ndarray) -> SimscapeOutput:
        return _fake_output(n_joints=coeffs.size // 7)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_instances() -> None:
    _FakeAdapter.instances.clear()
    yield
    _FakeAdapter.instances.clear()


@pytest.fixture
def slx(tmp_path: Path) -> Path:
    p = tmp_path / "m.slx"
    p.write_bytes(b"x")
    return p


def test_pool_worker_lazy_start_and_close(slx: Path) -> None:
    w = _PoolWorker(
        model_path=str(slx),
        cache_capacity=2,
        startup_timeout_s=5.0,
        adapter_factory=_FakeAdapter,  # type: ignore[arg-type]
    )
    assert _FakeAdapter.instances == []
    a = w.adapter
    assert a.loaded == str(slx)  # type: ignore[attr-defined]
    # subsequent calls reuse
    assert w.adapter is a
    w.close()
    assert a.closed is True  # type: ignore[attr-defined]
    w.close()  # idempotent


def test_pool_worker_close_before_use(slx: Path) -> None:
    w = _PoolWorker(
        model_path=str(slx),
        cache_capacity=0,
        startup_timeout_s=1.0,
        adapter_factory=_FakeAdapter,  # type: ignore[arg-type]
    )
    w.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = w.adapter


def test_pool_worker_close_swallows_adapter_error(slx: Path) -> None:
    class Boom(_FakeAdapter):
        def close(self) -> None:  # type: ignore[override]
            raise RuntimeError("close boom")

    w = _PoolWorker(
        model_path=str(slx),
        cache_capacity=0,
        startup_timeout_s=1.0,
        adapter_factory=Boom,  # type: ignore[arg-type]
    )
    _ = w.adapter
    w.close()  # must not raise


def test_pool_config_defaults() -> None:
    cfg = PoolConfig()
    assert cfg.pool_size == 4
    assert cfg.cache_capacity_per_worker == 16


def test_pool_rejects_bad_path() -> None:
    with pytest.raises(ValueError, match=".slx"):
        SimscapeAdapterPool("nope.mdl")


def test_pool_rejects_bad_pool_size(slx: Path) -> None:
    with pytest.raises(ValueError, match="pool_size"):
        SimscapeAdapterPool(slx, PoolConfig(pool_size=0))


def test_pool_rejects_bad_cache_capacity(slx: Path) -> None:
    with pytest.raises(ValueError, match="cache_capacity"):
        SimscapeAdapterPool(slx, PoolConfig(cache_capacity_per_worker=-1))


def test_pool_simulate_batch(slx: Path) -> None:
    pool = SimscapeAdapterPool(
        slx,
        PoolConfig(pool_size=2, startup_timeout_s=1.0),
        adapter_factory=_FakeAdapter,  # type: ignore[arg-type]
    )
    coeffs = np.zeros((3, 14))  # n_joints=2 * 7
    results = pool.simulate_batch(coeffs)
    assert len(results) == 3
    assert all(isinstance(r, SimscapeOutput) for r in results)
    pool.close()


def test_pool_simulate_batch_validation(slx: Path) -> None:
    pool = SimscapeAdapterPool(
        slx,
        PoolConfig(pool_size=1),
        adapter_factory=_FakeAdapter,  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError):
        pool.simulate_batch([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pool.simulate_batch(np.zeros(5))  # 1-D
    with pytest.raises(ValueError, match="multiple of 7"):
        pool.simulate_batch(np.zeros((2, 5)))
    pool.close()


def test_pool_map_preserves_order(slx: Path) -> None:
    pool = SimscapeAdapterPool(
        slx,
        PoolConfig(pool_size=2),
        adapter_factory=_FakeAdapter,  # type: ignore[arg-type]
    )
    items = [1, 2, 3, 4, 5]
    results = pool.map(lambda _adapter, x: x * 2, items)
    assert results == [2, 4, 6, 8, 10]
    pool.close()


def test_pool_map_empty_items(slx: Path) -> None:
    pool = SimscapeAdapterPool(
        slx,
        PoolConfig(pool_size=1),
        adapter_factory=_FakeAdapter,  # type: ignore[arg-type]
    )
    assert pool.map(lambda _a, x: x, []) == []
    pool.close()


def test_pool_close_idempotent_and_blocks_further_use(slx: Path) -> None:
    pool = SimscapeAdapterPool(
        slx,
        PoolConfig(pool_size=1),
        adapter_factory=_FakeAdapter,  # type: ignore[arg-type]
    )
    pool.close()
    pool.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        pool.simulate_batch(np.zeros((1, 7)))
    with pytest.raises(RuntimeError, match="closed"):
        pool.map(lambda _a, x: x, [1])


def test_pool_context_manager(slx: Path) -> None:
    with SimscapeAdapterPool(
        slx,
        PoolConfig(pool_size=1),
        adapter_factory=_FakeAdapter,  # type: ignore[arg-type]
    ) as pool:
        assert pool.pool_size == 1


def test_pool_accepts_path_object(tmp_path: Path) -> None:
    p = tmp_path / "m.slx"
    p.write_bytes(b"x")
    with SimscapeAdapterPool(
        p,
        PoolConfig(pool_size=1),
        adapter_factory=_FakeAdapter,  # type: ignore[arg-type]
    ) as pool:
        assert pool.pool_size == 1
