"""Tests for src.engines.simscape._cache."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.simscape._cache import _ResultCache, make_cache_key


def test_make_cache_key_deterministic() -> None:
    coeffs = np.arange(7, dtype=np.float64)
    k1 = make_cache_key(coeffs, model_params=b"abc", matlab_version="R2024a")
    k2 = make_cache_key(coeffs, model_params=b"abc", matlab_version="R2024a")
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


def test_make_cache_key_changes_with_inputs() -> None:
    base = make_cache_key(
        np.arange(7, dtype=np.float64), model_params=b"a", matlab_version="v"
    )
    diff_coeffs = make_cache_key(
        np.arange(7, dtype=np.float64) + 1, model_params=b"a", matlab_version="v"
    )
    diff_params = make_cache_key(
        np.arange(7, dtype=np.float64), model_params=b"b", matlab_version="v"
    )
    diff_ver = make_cache_key(
        np.arange(7, dtype=np.float64), model_params=b"a", matlab_version="w"
    )
    assert len({base, diff_coeffs, diff_params, diff_ver}) == 4


def test_make_cache_key_rejects_non_ndarray() -> None:
    with pytest.raises(TypeError):
        make_cache_key([1.0, 2.0], model_params=b"a", matlab_version="v")  # type: ignore[arg-type]


def test_make_cache_key_rejects_non_1d() -> None:
    with pytest.raises(ValueError):
        make_cache_key(np.zeros((2, 2)), model_params=b"a", matlab_version="v")


def test_make_cache_key_rejects_non_bytes_params() -> None:
    with pytest.raises(TypeError):
        make_cache_key(
            np.zeros(7),
            model_params="abc",
            matlab_version="v",  # type: ignore[arg-type]
        )


def test_cache_capacity_negative_raises() -> None:
    with pytest.raises(ValueError):
        _ResultCache(capacity=-1)


def test_cache_disabled_when_capacity_zero() -> None:
    c: _ResultCache[int] = _ResultCache(capacity=0)
    c.put("k", 1)
    assert c.get("k") is None
    assert c.misses == 1
    assert c.hits == 0
    assert len(c) == 0


def test_cache_hit_miss_counters() -> None:
    c: _ResultCache[int] = _ResultCache(capacity=4)
    assert c.get("absent") is None
    c.put("a", 1)
    assert c.get("a") == 1
    assert c.hits == 1
    assert c.misses == 1


def test_cache_lru_eviction() -> None:
    c: _ResultCache[int] = _ResultCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_cache_lru_recency_on_get() -> None:
    c: _ResultCache[int] = _ResultCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1  # bumps "a" to MRU
    c.put("c", 3)  # should evict "b"
    assert c.get("b") is None
    assert c.get("a") == 1


def test_cache_put_existing_key_updates_and_bumps() -> None:
    c: _ResultCache[int] = _ResultCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 99)  # update, also bumps to MRU
    c.put("c", 3)  # should evict "b" (LRU)
    assert c.get("b") is None
    assert c.get("a") == 99


def test_cache_clear_resets_counters() -> None:
    c: _ResultCache[int] = _ResultCache(capacity=4)
    c.put("a", 1)
    c.get("a")
    c.get("missing")
    c.clear()
    assert len(c) == 0
    assert c.hits == 0
    assert c.misses == 0
