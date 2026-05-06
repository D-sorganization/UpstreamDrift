"""LRU result cache for :class:`SimscapeAdapter.simulate_with_coefficients`.

The cache is keyed on a SHA-256 digest of:

1. The polynomial coefficient vector (``coeffs``).
2. The serialised model parameters (link masses, joint damping, …).
3. The MATLAB version string (so caches do not bleed across MATLAB
   releases that may produce subtly different numerics).

Concurrency: the cache is *not* thread-safe by itself. ``SimscapeAdapter``
serialises all MATLAB Engine calls behind its own lock and is the sole
owner of an instance. The adapter pool (#039 / #4008) wraps each
adapter in its own cache; nothing is shared across processes.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

__all__ = [
    "_ResultCache",
    "make_cache_key",
]


T = TypeVar("T")


def make_cache_key(
    coeffs: np.ndarray,
    *,
    model_params: bytes,
    matlab_version: str,
) -> str:
    """Return a stable SHA-256 hex digest for ``(coeffs, model_params, matlab_version)``.

    Args:
        coeffs: 1-D float64 polynomial coefficient vector.
        model_params: Already-serialised model parameters (e.g. the
            output of ``json.dumps(...).encode("utf-8")``).
        matlab_version: MATLAB release string from ``eng.version()``.

    Returns:
        64-character lowercase hex digest.

    Raises:
        TypeError: If ``coeffs`` is not an ``np.ndarray`` or
            ``model_params`` is not ``bytes``.
        ValueError: If ``coeffs`` is not 1-D.
    """
    if not isinstance(coeffs, np.ndarray):
        raise TypeError(f"coeffs must be np.ndarray, got {type(coeffs).__name__}")
    if coeffs.ndim != 1:
        raise ValueError(f"coeffs must be 1-D, got ndim={coeffs.ndim}")
    if not isinstance(model_params, bytes):
        raise TypeError(
            f"model_params must be bytes, got {type(model_params).__name__}"
        )

    coeffs_bytes = np.ascontiguousarray(coeffs, dtype=np.float64).tobytes()
    h = hashlib.sha256()
    h.update(b"coeffs:")
    h.update(coeffs_bytes)
    h.update(b"|model_params:")
    h.update(model_params)
    h.update(b"|matlab_version:")
    h.update(matlab_version.encode("utf-8"))
    return h.hexdigest()


@dataclass
class _ResultCache(Generic[T]):
    """Bounded LRU cache for simulation results.

    Implementation uses :class:`collections.OrderedDict` so insertion
    order tracks recency; on access we ``move_to_end`` the hit. Eviction
    pops the oldest entry when ``len(self) > capacity``.

    Args:
        capacity: Maximum number of entries. ``0`` disables caching
            entirely (every ``put`` is a no-op and every ``get`` returns
            ``None``).

    Raises:
        ValueError: If ``capacity`` is negative.
    """

    capacity: int = 64
    _store: OrderedDict[str, T] = field(default_factory=OrderedDict, repr=False)
    _hits: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {self.capacity}")

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hits(self) -> int:
        """Number of successful :meth:`get` lookups since construction."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of failed :meth:`get` lookups since construction."""
        return self._misses

    def get(self, key: str) -> T | None:
        """Return the cached value for ``key`` or ``None`` on miss.

        On a hit the entry is moved to the end (most-recently-used)
        position.
        """
        if self.capacity == 0:
            self._misses += 1
            return None
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(self, key: str, value: T) -> None:
        """Insert ``value`` under ``key``, evicting the oldest if needed."""
        if self.capacity == 0:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        self._store[key] = value
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry; reset hit/miss counters."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
