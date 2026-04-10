from __future__ import annotations

import numpy as np

from src.shared.python.engine_core.engine_availability import (
    FASTDTW_AVAILABLE,
    NUMBA_AVAILABLE,
)

if FASTDTW_AVAILABLE:
    from fastdtw import fastdtw

if NUMBA_AVAILABLE:
    from numba import jit
else:

    def jit(*args: object, **kwargs: object) -> object:  # type: ignore[misc]
        """No-op decorator when numba is not installed."""

        def decorator(func: object) -> object:  # type: ignore[misc]
            """Return the function unchanged."""
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@jit(nopython=True, cache=True)
def _dtw_core(series1: np.ndarray, series2: np.ndarray, window: int) -> float:
    """Numba-optimized DTW distance computation core.

    This inner kernel runs ~100x faster than pure Python due to JIT compilation.

    PERFORMANCE OPTIMIZATION:
    Uses O(M) space instead of O(NM) by storing only two rows.
    This significantly reduces memory allocation overhead and improves
    cache locality, especially when Numba is not available.

    Note: fastmath=True was intentionally removed as it can introduce numerical
    instability in DTW distance calculations due to non-IEEE-compliant floating
    point optimizations.

    Args:
        series1: First sequence (1D float64 array)
        series2: Second sequence (1D float64 array)
        window: Sakoe-Chiba band width

    Returns:
        DTW distance (float)
    """
    if not (series1 is not None):
        raise ValueError("series1 must be provided")
    if not (series1 is not None):
        raise ValueError("series1 must be provided")
    n = len(series1)
    m = len(series2)

    # Use large float instead of inf for numba compatibility
    INF = 1e30

    # Allocate cost rows (O(M) space)
    # prev_row stores the costs for the previous iteration (i-1)
    prev_row = np.full(m + 1, INF, dtype=np.float64)
    prev_row[0] = 0.0

    # curr_row stores costs for current iteration (i)
    curr_row = np.full(m + 1, INF, dtype=np.float64)

    for i in range(1, n + 1):
        # Reset curr_row for current iteration
        # We need to ensure cells outside the band/calculated area are INF
        # Since we reuse the array, filling with INF is safest.
        curr_row.fill(INF)

        # Sakoe-Chiba band limits
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            cost = (series1[i - 1] - series2[j - 1]) ** 2

            # Find minimum of previous cells
            # prev_row[j] corresponds to dtw_matrix[i-1, j] (Insertion)
            min_prev = prev_row[j]

            # curr_row[j-1] corresponds to dtw_matrix[i, j-1] (Deletion)
            val_del = curr_row[j - 1]
            if val_del < min_prev:
                min_prev = val_del

            # prev_row[j-1] corresponds to dtw_matrix[i-1, j-1] (Match)
            val_match = prev_row[j - 1]
            if val_match < min_prev:
                min_prev = val_match

            curr_row[j] = cost + min_prev

        # Swap rows for next iteration
        # prev_row takes values of curr_row for next step (where it will be i-1)
        # curr_row becomes the scratch buffer
        temp = prev_row
        prev_row = curr_row
        curr_row = temp

    # After loop, result is in prev_row[m] (because we swapped at end of loop)
    return float(np.sqrt(prev_row[m]))


@jit(nopython=True, cache=True)
def _dtw_path_core(
    series1: np.ndarray, series2: np.ndarray, window: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Numba-optimized DTW path computation core.

    Note: fastmath=True was intentionally removed as it can introduce numerical
    instability in DTW distance calculations due to non-IEEE-compliant floating
    point optimizations.

    Args:
        series1: First sequence (1D float64 array)
        series2: Second sequence (1D float64 array)
        window: Sakoe-Chiba band width (-1 for none)

    Returns:
        tuple: (distance, path_i, path_j)
        path_i, path_j are arrays of indices (reversed order)
    """
    if not (series1 is not None):
        raise ValueError("series1 must be provided")
    if not (series1 is not None):
        raise ValueError("series1 must be provided")
    n = len(series1)
    m = len(series2)

    # Use large float instead of inf for numba compatibility
    INF = 1e30

    # Allocate cost matrix
    dtw_matrix = np.full((n + 1, m + 1), INF, dtype=np.float64)
    dtw_matrix[0, 0] = 0.0

    w = window if window >= 0 else max(n, m)

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)

        for j in range(j_start, j_end):
            cost = (series1[i - 1] - series2[j - 1]) ** 2

            # Find minimum of previous cells
            min_prev = dtw_matrix[i - 1, j]  # Insertion

            val_del = dtw_matrix[i, j - 1]  # Deletion
            if val_del < min_prev:
                min_prev = val_del

            val_match = dtw_matrix[i - 1, j - 1]  # Match
            if val_match < min_prev:
                min_prev = val_match

            dtw_matrix[i, j] = cost + min_prev

    distance = float(np.sqrt(dtw_matrix[n, m]))

    # Backtrack
    # BUG FIX: Path length can be up to n + m in worst case (zig-zag).
    # Previous allocation of max(n, m) caused IndexError on noisy data.
    max_len = n + m
    path_i = np.empty(max_len, dtype=np.int32)
    path_j = np.empty(max_len, dtype=np.int32)

    idx = 0
    i, j = n, m
    while i > 0 and j > 0:
        path_i[idx] = i - 1
        path_j[idx] = j - 1
        idx += 1

        v_ins = dtw_matrix[i - 1, j]
        v_del = dtw_matrix[i, j - 1]
        v_match = dtw_matrix[i - 1, j - 1]

        # Preference order for backtracking: Match, then Insertion, then Deletion
        min_val = v_match
        if v_ins < min_val:
            min_val = v_ins
        if v_del < min_val:
            min_val = v_del

        if min_val == v_match:
            i -= 1
            j -= 1
        elif min_val == v_ins:
            i -= 1
        else:
            j -= 1

    return distance, path_i[:idx], path_j[:idx]


def compute_dtw_distance(
    series1: np.ndarray,
    series2: np.ndarray,
    window: int | None = None,
) -> float:
    """Compute Dynamic Time Warping (DTW) distance between two sequences.

    Uses Euclidean distance as the local cost measure.
    Implements Sakoe-Chiba band constraint if window is specified.

    PERFORMANCE: Uses Numba JIT-compiled kernel when available (~100x speedup).

    Args:
        series1: First sequence (1D array)
        series2: Second sequence (1D array)
        window: Sakoe-Chiba band width (None for no constraint)

    Returns:
        DTW distance (float)
    """
    if not (series1 is not None):
        raise ValueError("series1 must be provided")
    if not (series1 is not None):
        raise ValueError("series1 must be provided")
    n = len(series1)
    m = len(series2)

    # Sakoe-Chiba band constraint
    w = window if window is not None else max(n, m)

    # PERFORMANCE: Use Numba-optimized kernel if available
    if NUMBA_AVAILABLE:
        # Ensure arrays are float64 for numba
        s1 = np.asarray(series1, dtype=np.float64)
        s2 = np.asarray(series2, dtype=np.float64)
        return float(_dtw_core(s1, s2, w))

    # PERFORMANCE: Use fastdtw if available (approximate linear time)
    # fastdtw uses a multi-level approach to speed up DTW
    # Only use if Numba is NOT available, as Numba provides exact calculation efficiently
    if FASTDTW_AVAILABLE and not NUMBA_AVAILABLE and window is None:
        distance, _ = fastdtw(series1, series2, dist=2)  # dist=2 means euclidean
        return float(distance)

    # Fallback: Pure Python implementation
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        # Determine band limits
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)

        for j in range(j_start, j_end):
            cost = (series1[i - 1] - series2[j - 1]) ** 2
            # Take minimum of (match, insertion, deletion)
            last_min = min(
                dtw_matrix[i - 1, j],  # Insertion
                dtw_matrix[i, j - 1],  # Deletion
                dtw_matrix[i - 1, j - 1],  # Match
            )
            dtw_matrix[i, j] = cost + last_min

    return float(np.sqrt(dtw_matrix[n, m]))


def compute_dtw_path(
    series1: np.ndarray,
    series2: np.ndarray,
    window: int | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """Compute DTW distance and optimal warping path.

    Args:
        series1: First sequence
        series2: Second sequence
        window: Sakoe-Chiba band width

    Returns:
        Tuple (distance, path). Path is list of (i, j) indices.
    """
    # Ensure inputs are float64 arrays for Numba
    if not (series1 is not None):
        raise ValueError("series1 must be provided")
    if not (series1 is not None):
        raise ValueError("series1 must be provided")
    s1 = np.asarray(series1, dtype=np.float64)
    s2 = np.asarray(series2, dtype=np.float64)

    w_val = window if window is not None else -1

    # Use Numba kernel (which works as pure python too via no-op jit)
    dist, pi, pj = _dtw_path_core(s1, s2, w_val)

    # Convert structure to list of tuples
    # pi, pj are in reverse order from backtracking
    path = []
    # Loop backwards to reverse
    for k in range(len(pi) - 1, -1, -1):
        path.append((int(pi[k]), int(pj[k])))

    return dist, path
