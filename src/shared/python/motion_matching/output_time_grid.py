"""Canonical output time grid shared across physics engines.

Every engine emits its trajectory on an output sample grid of
``N = round(simulation_time_s * sample_rate_hz) + 1`` points spanning
``[0, simulation_time_s]``. Historically each engine built this grid with a
slightly different expression:

* Drake used ``np.arange(N) * (1 / rate)``.
* MuJoCo used ``np.linspace(0, T, N)``.
* Pinocchio used ``np.arange(N) * dt`` as its *integration clock* (a
  distinct contract — see :mod:`pinocchio...simulate`).

For integer-divisible ``(T, rate)`` (the common, spec'd case) ``arange * dt``
and inclusive ``linspace`` are **bit-identical**, so consolidating Drake and
MuJoCo onto one builder does not change any existing trajectory. For
non-divisible ``(T, rate)`` the two diverged — ``arange`` overshot ``T`` while
``linspace`` pinned the endpoint — yielding mismatched endpoints across
engines. This builder removes that divergence by always pinning the endpoint
to ``simulation_time_s`` exactly (inclusive ``linspace`` semantics).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["canonical_sample_count", "build_output_grid"]


def canonical_sample_count(simulation_time_s: float, sample_rate_hz: float) -> int:
    """Return ``N = round(T * rate) + 1``, the canonical output sample count.

    Args:
        simulation_time_s: Simulation horizon in seconds (> 0).
        sample_rate_hz: Output sample rate in Hz (> 0).

    Returns:
        Number of inclusive grid points.

    Raises:
        ValueError: if either argument is not strictly positive.
    """
    if simulation_time_s <= 0:
        raise ValueError(f"simulation_time_s must be > 0; got {simulation_time_s}")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0; got {sample_rate_hz}")
    return int(round(simulation_time_s * sample_rate_hz)) + 1


def build_output_grid(
    simulation_time_s: float, sample_rate_hz: float
) -> NDArray[np.float64]:
    """Build the canonical output time grid ``0 <= t <= simulation_time_s``.

    The grid has :func:`canonical_sample_count` points, starts at ``0.0``, is
    strictly increasing, and ends at exactly ``simulation_time_s`` (the
    endpoint is pinned, so it is aligned across every engine regardless of
    whether ``(T, rate)`` divides evenly).

    For integer-divisible ``(T, rate)`` this is bit-identical to the legacy
    ``np.arange(N) * (1 / rate)`` construction.

    Args:
        simulation_time_s: Simulation horizon in seconds (> 0).
        sample_rate_hz: Output sample rate in Hz (> 0).

    Returns:
        ``(N,)`` float64 array of sample times.
    """
    n = canonical_sample_count(simulation_time_s, sample_rate_hz)
    return np.linspace(0.0, float(simulation_time_s), n, dtype=np.float64)
