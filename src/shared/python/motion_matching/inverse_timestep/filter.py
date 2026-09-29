"""Speed-mask helpers for the per-timestep inverse-dynamics pipeline.

The compact dataset's per-timestep clubhead-speed distribution is highly
skewed (p25=103 mph, p50=165 mph, p99=2984 mph). The model trains on the
realistic-swing slice ``||v_clubhead|| in [50, 150] mph``, which keeps
~29% of the timesteps and spreads broadly across trials (5-9 timesteps
per trial out of 31).

This module is pure-python (numpy + pandas) and does not import torch,
so it can be used in lightweight contexts.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from src.shared.python.dataset_tools.load_compact import CompactSwingDataset

logger = logging.getLogger(__name__)


def realistic_speed_mask(
    speeds_mph: np.ndarray,
    lo: float = 50.0,
    hi: float = 150.0,
) -> np.ndarray:
    """Return a boolean mask for clubhead speeds in ``[lo, hi]`` mph.

    Preconditions:
        * ``0 <= lo < hi``.
        * ``speeds_mph`` is a 1-D finite numpy array of floats.

    Postconditions:
        * Output is a 1-D ``np.bool_`` array of the same length as
          ``speeds_mph``; ``out[i] == (lo <= speeds_mph[i] <= hi)``.

    Args:
        speeds_mph: 1-D array of clubhead speeds in mph.
        lo: Lower bound of the realistic speed window (inclusive).
        hi: Upper bound of the realistic speed window (inclusive).

    Returns:
        Boolean mask array.

    Raises:
        TypeError: If ``speeds_mph`` is not a numpy array.
        ValueError: If preconditions are violated.
    """
    if not isinstance(speeds_mph, np.ndarray):
        raise TypeError(
            f"speeds_mph must be np.ndarray, got {type(speeds_mph).__name__}"
        )
    if speeds_mph.ndim != 1:
        raise ValueError(f"speeds_mph must be 1-D, got shape {tuple(speeds_mph.shape)}")
    if not np.isfinite(speeds_mph).all():
        raise ValueError("speeds_mph contains non-finite values (NaN/Inf)")
    _validate_bounds(lo, hi)
    return (speeds_mph >= lo) & (speeds_mph <= hi)


def filter_timesteps_by_speed(
    dataset: CompactSwingDataset,
    *,
    lo: float = 50.0,
    hi: float = 150.0,
) -> CompactSwingDataset:
    """Return ``dataset`` with timesteps filtered to ``[lo, hi]`` mph.

    Operates on an eagerly materialised pandas DataFrame, replacing the
    ``timesteps`` field of the dataclass via :func:`dataclasses.replace`.
    The ``trials`` field is unchanged so callers can still cross-reference.

    Preconditions:
        * ``dataset.timesteps`` is a pandas DataFrame with a finite
          ``clubhead_speed_mph`` column.
        * ``0 <= lo < hi``.

    Postconditions:
        * Resulting timesteps DataFrame has length <= original length.
        * Every retained row has ``lo <= clubhead_speed_mph <= hi``.

    Args:
        dataset: Compact dataset handle (see :class:`CompactSwingDataset`).
        lo: Lower bound on clubhead speed in mph (inclusive).
        hi: Upper bound on clubhead speed in mph (inclusive).

    Returns:
        New :class:`CompactSwingDataset` with the timesteps frame filtered.

    Raises:
        TypeError: If the timesteps backend is not a pandas DataFrame.
        ValueError: If preconditions are violated.
    """
    timesteps_obj = dataset.timesteps
    timesteps_df = _require_pandas(timesteps_obj)
    _validate_bounds(lo, hi)
    if "clubhead_speed_mph" not in timesteps_df.columns:
        raise ValueError("timesteps DataFrame missing 'clubhead_speed_mph' column")
    speeds = timesteps_df["clubhead_speed_mph"].to_numpy(dtype=np.float64)
    mask = realistic_speed_mask(speeds, lo=lo, hi=hi)
    filtered = timesteps_df.loc[mask].reset_index(drop=True)
    n_before = len(timesteps_df)
    n_after = len(filtered)
    logger.info(
        "filter_timesteps_by_speed: %d -> %d rows (%.1f%%) at lo=%.1f hi=%.1f mph",
        n_before,
        n_after,
        100.0 * n_after / max(1, n_before),
        lo,
        hi,
    )
    return replace(dataset, timesteps=filtered)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_bounds(lo: float, hi: float) -> None:
    if not isinstance(lo, int | float):
        raise TypeError(f"lo must be a number, got {type(lo).__name__}")
    if not isinstance(hi, int | float):
        raise TypeError(f"hi must be a number, got {type(hi).__name__}")
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f):
        raise ValueError(f"lo/hi must be finite; got lo={lo_f}, hi={hi_f}")
    if lo_f < 0:
        raise ValueError(f"lo must be >= 0, got {lo_f}")
    if not lo_f < hi_f:
        raise ValueError(f"must have lo < hi; got lo={lo_f}, hi={hi_f}")


def _require_pandas(obj: Any) -> pd.DataFrame:
    import pandas as pd

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            "filter_timesteps_by_speed requires an eagerly materialised "
            "pandas DataFrame for `timesteps` (load with lazy=False); "
            f"got {type(obj).__name__}"
        )
    return obj
