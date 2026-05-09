"""Performance budget tests for the ``DataDrivenColor`` resolver (#4814).

Two fleet-realistic workloads are exercised against a 38-marker x
654-frame channel — the canonical D-sorganization C3D fixture size:

* ``resolve_array`` — single bulk call must clear ``< 1.0 s`` (i.e.
  >= ~25 000 marker-updates per second).
* Per-frame ``resolve(...)`` loop — 654 frames must fit within the
  ``1/60 s * 654`` ≈ 10.9 s budget so live playback can hold 60 fps.

The budgets are loose by design — any reasonably modern machine
clears them by an order of magnitude — but they will catch a
regression that drops resolver throughput below interactive rates.
Tests are tagged ``perf`` so the default CI run skips them; opt in
with ``-m perf``.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from src.shared.python.plot_style import (
    ColormapId,
    DataChannel,
    DataDrivenColor,
)
from src.shared.python.plot_style.resolvers import RESOLVER_REGISTRY
from src.shared.python.plot_style.resolvers.data_driven import (
    DataDrivenColor as DataDrivenResolver,
)

pytestmark = pytest.mark.perf

N_MARKERS = 38
N_FRAMES = 654
BULK_BUDGET_S = 1.0
PER_FRAME_BUDGET_S = (1.0 / 60.0) * N_FRAMES  # ~10.9 s @ 60 fps
SCALE = float(os.environ.get("PLOT_STYLE_PERF_BUDGET_SCALE", "1.0"))


def _build_workload() -> tuple[DataDrivenColor, DataDrivenResolver]:
    """Build a 38 x 654 per-(frame, marker) workload + bound resolver."""
    rng = np.random.default_rng(42)
    values = rng.uniform(0.0, 1.0, size=(N_FRAMES, N_MARKERS)).astype(np.float64)
    channel = DataChannel(name="speed", values=values, unit="m/s")
    scale = DataDrivenColor(
        channel=channel,
        colormap=ColormapId.VIRIDIS,
        vmin=0.0,
        vmax=1.0,
    )
    resolver = DataDrivenResolver(scale)
    return scale, resolver


def test_resolve_array_bulk_under_budget() -> None:
    """Single ``resolve_array`` call must clear ``BULK_BUDGET_S`` seconds."""
    scale, resolver = _build_workload()

    # Warm-up so first-touch JIT / allocation costs do not pollute the
    # measurement.
    _ = resolver.resolve_array(scale, 8, N_MARKERS)

    start = time.perf_counter()
    rgba = resolver.resolve_array(scale, N_FRAMES, N_MARKERS)
    elapsed = time.perf_counter() - start

    assert rgba.shape == (N_FRAMES, N_MARKERS, 4)
    assert rgba.dtype == np.float64
    budget = BULK_BUDGET_S * SCALE
    assert elapsed < budget, (
        f"resolve_array bulk {elapsed:.3f} s exceeds {budget:.3f} s budget "
        f"({N_FRAMES * N_MARKERS} updates)"
    )


def test_resolve_per_frame_loop_under_60fps_budget() -> None:
    """``N_FRAMES`` per-frame ``resolve(...)`` calls must fit a 60-fps budget."""
    scale, resolver = _build_workload()

    # Warm-up.
    for _ in range(4):
        for m in range(N_MARKERS):
            resolver.resolve(scale, 0, m)

    start = time.perf_counter()
    for f in range(N_FRAMES):
        for m in range(N_MARKERS):
            resolver.resolve(scale, f, m)
    elapsed = time.perf_counter() - start

    budget = PER_FRAME_BUDGET_S * SCALE
    assert elapsed < budget, (
        f"per-frame resolve loop {elapsed:.3f} s exceeds {budget:.3f} s budget "
        f"({N_FRAMES} frames x {N_MARKERS} markers)"
    )


def test_registry_resolver_matches_direct_construction() -> None:
    """Sanity: the public registry entry resolves the same workload."""
    scale, _ = _build_workload()
    cls = RESOLVER_REGISTRY[type(scale)]
    resolver = cls(scale)
    rgba = resolver.resolve_array(scale, 4, N_MARKERS)
    assert rgba.shape == (4, N_MARKERS, 4)
    assert np.all(np.isfinite(rgba))
