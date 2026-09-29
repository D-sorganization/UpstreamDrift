"""Unit tests for versioned HDF5 (de)serialisation of traces (M3.3).

Covers the shared on-disk schema round-tripped through
:func:`simulation_backends.trace_io.write_trace` /
:func:`~simulation_backends.trace_io.read_trace`:

* single :class:`Trace` with and without a control history;
* batched :class:`BatchTrace`;
* the schema-version compatibility guard.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.shared.python.simulation_backends.protocol import (
    SCHEMA_VERSION,
    BatchTrace,
    Trace,
)
from src.shared.python.simulation_backends.trace_io import read_trace, write_trace

pytestmark = pytest.mark.unit


def _make_single_trace(*, with_controls: bool) -> Trace:
    """Build a deterministic single-rollout trace (horizon=10 -> T=11)."""
    rng = np.random.default_rng(0)
    horizon = 10
    t = np.arange(horizon + 1, dtype=float) * 0.01
    q = rng.standard_normal((horizon + 1, 2))
    v = rng.standard_normal((horizon + 1, 2))
    u = rng.standard_normal((horizon + 1, 2)) if with_controls else None
    return Trace(
        t=t,
        q=q,
        v=v,
        u=u,
        dt=0.01,
        backend="ode",
        meta={"note": "round-trip", "envs": 1, "tol": 1e-6, "ok": True},
    )


def _make_batch_trace() -> BatchTrace:
    """Build a deterministic batched trace of shape (N=3, T=11, dim=2)."""
    rng = np.random.default_rng(0)
    num_envs, horizon = 3, 10
    t = np.arange(horizon + 1, dtype=float) * 0.02
    q = rng.standard_normal((num_envs, horizon + 1, 2))
    v = rng.standard_normal((num_envs, horizon + 1, 2))
    u = rng.standard_normal((num_envs, horizon + 1, 2))
    return BatchTrace(
        t=t,
        q=q,
        v=v,
        u=u,
        dt=0.02,
        backend="mjwarp",
        meta={"kind": "batch", "n": 3},
    )


def _assert_meta_equal(expected: dict[str, object], actual: dict[str, object]) -> None:
    """Assert two scalar-metadata mappings are equal key-for-key."""
    assert dict(actual) == dict(expected)


@pytest.mark.parametrize("with_controls", [True, False])
def test_round_trip_single_trace(tmp_path, with_controls: bool) -> None:
    """A single Trace round-trips losslessly, with controls preserved or None."""
    trace = _make_single_trace(with_controls=with_controls)
    path = tmp_path / "single.h5"

    write_trace(trace, path)
    loaded = read_trace(path)

    assert isinstance(loaded, Trace)
    np.testing.assert_allclose(loaded.t, trace.t)
    np.testing.assert_allclose(loaded.q, trace.q)
    np.testing.assert_allclose(loaded.v, trace.v)

    if with_controls:
        assert loaded.u is not None
        np.testing.assert_allclose(loaded.u, trace.u)
    else:
        assert loaded.u is None

    assert loaded.schema_version == trace.schema_version
    assert loaded.backend == trace.backend
    assert loaded.dt == pytest.approx(trace.dt)
    _assert_meta_equal(dict(trace.meta), dict(loaded.meta))


def test_round_trip_batch_trace(tmp_path) -> None:
    """A BatchTrace round-trips with shapes, values, and k--batch kind intact."""
    trace = _make_batch_trace()
    path = tmp_path / "batch.h5"

    write_trace(trace, path)

    # The on-disk discriminator must record the batched kind.
    with h5py.File(path, "r") as handle:
        kind_attr = handle.attrs["kind"]
        kind = kind_attr.decode() if isinstance(kind_attr, bytes) else kind_attr
    assert kind == "batch"

    loaded = read_trace(path)
    assert isinstance(loaded, BatchTrace)
    assert loaded.q.shape == (3, 11, 2)
    assert loaded.v.shape == (3, 11, 2)
    assert loaded.num_envs == 3
    assert loaded.num_steps == 11

    np.testing.assert_allclose(loaded.t, trace.t)
    np.testing.assert_allclose(loaded.q, trace.q)
    np.testing.assert_allclose(loaded.v, trace.v)
    assert loaded.u is not None
    np.testing.assert_allclose(loaded.u, trace.u)

    assert loaded.backend == trace.backend
    assert loaded.dt == pytest.approx(trace.dt)
    _assert_meta_equal(dict(trace.meta), dict(loaded.meta))


def test_read_rejects_incompatible_major_schema(tmp_path) -> None:
    """A file whose schema major differs from SCHEMA_VERSION is rejected."""
    trace = _make_single_trace(with_controls=True)
    path = tmp_path / "bumped.h5"
    write_trace(trace, path)

    # Tamper with the persisted schema to a future, incompatible major.
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema_version"] = "99.0.0"

    with pytest.raises(ValueError, match="schema"):
        read_trace(path)


def test_current_schema_version_round_trips(tmp_path) -> None:
    """The default schema_version is the running SCHEMA_VERSION and survives I/O."""
    trace = _make_single_trace(with_controls=False)
    assert trace.schema_version == SCHEMA_VERSION
    path = tmp_path / "version.h5"
    write_trace(trace, path)
    assert read_trace(path).schema_version == SCHEMA_VERSION


def test_empty_path_is_rejected() -> None:
    """An empty path raises ValueError on both read and write."""
    trace = _make_single_trace(with_controls=False)
    with pytest.raises(ValueError, match="non-empty"):
        write_trace(trace, "")
    with pytest.raises(ValueError, match="non-empty"):
        read_trace("")


def test_non_scalar_meta_is_skipped(tmp_path) -> None:
    """Non-scalar meta values are dropped; scalar siblings are preserved."""
    trace = Trace(
        t=np.array([0.0, 0.01]),
        q=np.zeros((2, 2)),
        v=np.zeros((2, 2)),
        dt=0.01,
        backend="ode",
        meta={"keep": 5, "drop": [1, 2, 3], "label": "x"},
    )
    path = tmp_path / "meta.h5"
    write_trace(trace, path)
    loaded = read_trace(path)

    assert loaded.meta["keep"] == 5
    assert loaded.meta["label"] == "x"
    assert "drop" not in loaded.meta
