"""Behavioral tests for the DTW (Dynamic Time Warping) kernel.

Targets ``src/shared/python/signal_toolkit/_dtw.py`` which previously had
no direct test coverage. Exercises:

* Distance is zero for identical sequences.
* Distance is symmetric.
* Triangle-inequality-like sanity properties.
* Time-shift / time-warp invariance (DTW's defining feature).
* Window (Sakoe-Chiba band) correctness.
* Path correctness: monotone, starts at (0,0), ends at (n-1, m-1).
* Numerical correctness against a hand-computed example.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit._dtw import (
    compute_dtw_distance,
    compute_dtw_path,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


class TestDtwDistance:
    def test_identical_sequences_zero_distance(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_dtw_distance(a, a) == 0.0

    def test_distance_is_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal(20)
        b = rng.standard_normal(25)
        assert compute_dtw_distance(a, b) >= 0.0

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.standard_normal(15)
        b = rng.standard_normal(18)
        d_ab = compute_dtw_distance(a, b)
        d_ba = compute_dtw_distance(b, a)
        assert d_ab == pytest.approx(d_ba, abs=1e-9)

    def test_constant_offset_is_root_n_times_offset(self) -> None:
        # For two equal-length constant-offset signals: each cell along the
        # diagonal contributes (offset)^2; sqrt of the sum is sqrt(n)*|offset|.
        a = np.zeros(10)
        b = np.full(10, 3.0)
        d = compute_dtw_distance(a, b)
        assert d == pytest.approx(np.sqrt(10) * 3.0, rel=1e-9)

    def test_handles_different_lengths(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        # Same shape, just dilated in time → DTW should find a near-zero match.
        d = compute_dtw_distance(a, b)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_time_warp_invariance(self) -> None:
        # DTW's reason-for-being: time-warped signals have small distance.
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        b = np.repeat(a, 2)  # time-stretched version
        d_warp = compute_dtw_distance(a, b)
        # Compare with completely unrelated signal of the same length as b.
        unrelated = np.array([5.0] * len(b))
        d_unrelated = compute_dtw_distance(a, unrelated)
        assert d_warp < d_unrelated

    def test_known_simple_distance(self) -> None:
        # Hand-computed: a=[1], b=[3], cost = (1-3)^2=4, sqrt = 2.
        d = compute_dtw_distance(np.array([1.0]), np.array([3.0]))
        assert d == pytest.approx(2.0, abs=1e-9)

    def test_window_constraint_respected(self) -> None:
        # When window is 0 only the diagonal is allowed; for equal-length
        # sequences this should match exactly the Euclidean distance.
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 4.0, 6.0, 8.0])
        d_window0 = compute_dtw_distance(a, b, window=0)
        euclid = np.sqrt(np.sum((a - b) ** 2))
        assert d_window0 == pytest.approx(euclid, abs=1e-9)

    def test_wider_window_is_no_worse_than_narrower(self) -> None:
        rng = np.random.default_rng(123)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        d_wide = compute_dtw_distance(a, b, window=20)
        d_narrow = compute_dtw_distance(a, b, window=2)
        # Larger search space cannot produce a worse minimum.
        assert d_wide <= d_narrow + 1e-9


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------


class TestDtwPath:
    def test_path_endpoints(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, path = compute_dtw_path(a, b)
        assert path[0] == (0, 0)
        assert path[-1] == (len(a) - 1, len(b) - 1)

    def test_path_is_monotone_non_decreasing(self) -> None:
        rng = np.random.default_rng(2024)
        a = rng.standard_normal(12)
        b = rng.standard_normal(15)
        _, path = compute_dtw_path(a, b)
        for (i0, j0), (i1, j1) in zip(path[:-1], path[1:], strict=True):
            assert i1 >= i0 and j1 >= j0
            # at least one index must advance per step
            assert (i1 + j1) > (i0 + j0)

    def test_distance_matches_compute_dtw_distance(self) -> None:
        rng = np.random.default_rng(99)
        a = rng.standard_normal(10)
        b = rng.standard_normal(13)
        d_path, _ = compute_dtw_path(a, b)
        d_only = compute_dtw_distance(a, b)
        assert d_path == pytest.approx(d_only, rel=1e-9, abs=1e-9)

    def test_path_zero_distance_for_identical(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        d, path = compute_dtw_path(a, a)
        assert d == pytest.approx(0.0, abs=1e-9)
        # Optimal alignment should follow the diagonal.
        assert path == [(0, 0), (1, 1), (2, 2)]
