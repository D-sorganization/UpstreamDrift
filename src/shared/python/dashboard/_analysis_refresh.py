"""Refresh throttling and result memoization for the advanced analysis tabs.

Issue #8932: the spectrogram and wavelet tabs recomputed their transforms and
blocked on ``canvas.draw()`` for every spinbox tick. This module holds the two
shared building blocks those tabs compose instead of each re-implementing them:

- :class:`DebouncedRefresh` coalesces a burst of change signals into a single
  call once the burst goes quiet.
- :class:`BoundedResultCache` memoizes an expensive transform on a key built by
  :func:`analysis_cache_key`, which fingerprints the signal itself so a cached
  result can never outlive the data it was computed from.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np
from PyQt6 import QtCore

DEFAULT_DEBOUNCE_MS = 150
DEFAULT_CACHE_SIZE = 8


class DebouncedRefresh:
    """Coalesce rapid change notifications into one delayed callback.

    Each :meth:`trigger` restarts a single-shot timer, so the callback runs
    exactly once, ``interval_ms`` after the *last* trigger of a burst.
    """

    def __init__(
        self,
        callback: Callable[[], Any],
        interval_ms: int = DEFAULT_DEBOUNCE_MS,
        parent: QtCore.QObject | None = None,
    ) -> None:
        """Create the debouncer.

        Raises:
            TypeError: If ``callback`` is not callable.
            ValueError: If ``interval_ms`` is negative.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        if interval_ms < 0:
            raise ValueError(f"interval_ms must be >= 0, got {interval_ms}")
        self._callback = callback
        self._timer = QtCore.QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._fire)

    @property
    def interval_ms(self) -> int:
        """Quiet period, in milliseconds, before the callback runs."""
        return self._timer.interval()

    @property
    def pending(self) -> bool:
        """True while a callback is scheduled but has not yet run."""
        return self._timer.isActive()

    def trigger(self, *_signal_args: object) -> None:
        """(Re)start the quiet period; signal payloads are ignored."""
        self._timer.start()

    def _fire(self) -> None:
        self._callback()


def analysis_cache_key(
    metric_key: str,
    dim: int,
    fs: float,
    signal: np.ndarray,
    **params: float,
) -> tuple[Hashable, ...]:
    """Build a memoization key for a transform of one signal column.

    The key is ``(metric_key, dim, fs, n_samples, <sorted params>, digest)``
    where ``digest`` hashes the signal bytes, so any change to the underlying
    data -- including an in-place overwrite that keeps the length -- produces
    a different key.

    Raises:
        ValueError: If ``dim`` is negative, ``fs`` is not a positive finite
            number, or ``signal`` is not a non-empty 1-D array.
    """
    if dim < 0:
        raise ValueError(f"dim must be >= 0, got {dim}")
    if not (math.isfinite(fs) and fs > 0):
        raise ValueError(f"fs must be a positive finite number, got {fs}")
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"signal must be a non-empty 1-D array, got {signal.shape}")
    contiguous = np.ascontiguousarray(signal)
    digest = hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()
    extra = tuple(sorted(params.items()))
    return (metric_key, int(dim), float(fs), int(signal.size), extra, digest)


class BoundedResultCache:
    """Least-recently-used memo of transform results with a hard size bound."""

    def __init__(self, maxsize: int = DEFAULT_CACHE_SIZE) -> None:
        """Create an empty cache holding at most ``maxsize`` entries.

        Raises:
            ValueError: If ``maxsize`` is less than 1.
        """
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = maxsize
        self._entries: OrderedDict[Hashable, Any] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, computing and storing it on a miss.

        Postconditions:
            ``key`` is the most recently used entry and ``len(self) <= maxsize``.
        """
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        value = compute()
        self._entries[key] = value
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop every cached entry."""
        self._entries.clear()
