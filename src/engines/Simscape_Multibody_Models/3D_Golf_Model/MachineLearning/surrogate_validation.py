"""Validation diagnostics for the dynamics surrogate.

This module isolates split selection and per-phase residual reporting so the
training loop can stay focused on the model. Time and phase metadata are used
*only* for evaluation grouping; they are never returned as model inputs.

Splits provided
---------------
- ``phase_stratified_split``: boolean mask per swing phase based on a
  normalized time column (one mask per phase covering all rows).
- ``holdout_trajectory_split``: held-out trajectory IDs (no row from a
  validation trial appears in the training mask).

Reports provided
----------------
- ``evaluate_per_phase``: RMSE per phase mask (overall, across targets).
- ``evaluate_holdout_trajectory``: RMSE on the held-out trajectory rows,
  with the count of held-out trials and rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_PHASE_BREAKPOINTS: dict[str, float] = {
    "address": 0.0,
    "top": 0.4,
    "impact": 0.7,
    "finish": 1.0,
}


def _column(df: Any, name: str) -> np.ndarray:
    """Return ``df[name]`` as a 1-D numpy array.

    Accepts pandas DataFrames or any mapping of column-name to array-like.
    """
    if hasattr(df, "columns") and hasattr(df, "__getitem__"):
        # pandas DataFrame path
        if name not in df.columns:
            raise KeyError(f"column '{name}' not present in dataframe")
        return np.asarray(df[name])
    if isinstance(df, Mapping):
        if name not in df:
            raise KeyError(f"column '{name}' not present in mapping")
        return np.asarray(df[name])
    raise TypeError(
        "df must be a pandas DataFrame or a Mapping[str, array-like]; "
        f"got {type(df).__name__}"
    )


def _normalize_breakpoints(breakpoints: Mapping[str, float]) -> list[tuple[str, float]]:
    if not breakpoints:
        raise ValueError("phase_breakpoints must contain at least one entry")
    items = sorted(breakpoints.items(), key=lambda kv: kv[1])
    values = [v for _, v in items]
    lo, hi = values[0], values[-1]
    if hi <= lo:
        raise ValueError("phase breakpoints must span a positive interval")
    # Normalize to [0, 1] across the swing window.
    return [(name, (val - lo) / (hi - lo)) for name, val in items]


def phase_stratified_split(
    df: Any,
    time_col: str = "t",
    phase_breakpoints: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Return a ``{phase_name: boolean mask}`` partition of ``df`` by time.

    The breakpoints define the *start* of each phase along a normalized
    [0, 1] swing window. ``df[time_col]`` is rescaled to that window using
    its own min/max, then assigned to the phase whose start it falls in.

    Phases partition the rows: every row belongs to exactly one phase, and
    the union of masks equals ``np.ones(len(df), dtype=bool)``.
    """
    if phase_breakpoints is None:
        phase_breakpoints = DEFAULT_PHASE_BREAKPOINTS
    phases = _normalize_breakpoints(phase_breakpoints)

    times = _column(df, time_col).astype(np.float64)
    if times.size == 0:
        return {name: np.zeros(0, dtype=bool) for name, _ in phases}
    t_min = float(np.min(times))
    t_max = float(np.max(times))
    if t_max <= t_min:
        raise ValueError(
            f"time column '{time_col}' has no positive range "
            f"(min={t_min}, max={t_max}); cannot stratify by phase"
        )
    normalized = (times - t_min) / (t_max - t_min)

    # Phase i covers [start_i, start_{i+1}); last phase is closed on the right.
    starts = [s for _, s in phases]
    masks: dict[str, np.ndarray] = {}
    n_phases = len(phases)
    for i, (name, start) in enumerate(phases):
        if i + 1 < n_phases:
            end = starts[i + 1]
            mask = (normalized >= start) & (normalized < end)
        else:
            mask = normalized >= start
        masks[name] = mask
    # Postcondition: phases partition all rows.
    coverage = np.zeros(len(times), dtype=bool)
    for mask in masks.values():
        coverage |= mask
    if not coverage.all():
        # Should not happen given normalization, but assert for DbC.
        raise AssertionError("phase masks do not cover all rows")
    return masks


def holdout_trajectory_split(
    df: Any,
    trial_id_col: str = "trial_id",
    frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_mask, val_mask)`` with disjoint trial IDs.

    A fraction ``frac`` of unique trial IDs is held out for validation; all
    rows belonging to those IDs go into ``val_mask`` and never appear in
    ``train_mask``.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1); got {frac}")

    trials = _column(df, trial_id_col)
    n_rows = len(trials)
    unique = np.array(sorted(set(trials.tolist())))
    if unique.size < 2:
        raise ValueError(
            f"trial_id_col '{trial_id_col}' must have at least 2 unique IDs; "
            f"got {unique.size}"
        )

    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(unique.size * frac)))
    n_val = min(n_val, unique.size - 1)
    perm = rng.permutation(unique.size)
    val_ids = set(unique[perm[:n_val]].tolist())

    val_mask = np.array([t in val_ids for t in trials.tolist()], dtype=bool)
    train_mask = ~val_mask

    if val_mask.sum() == 0 or train_mask.sum() == 0:
        raise AssertionError("holdout split produced an empty side")
    if (train_mask & val_mask).any():
        raise AssertionError("holdout split produced overlapping masks")
    if int(train_mask.sum()) + int(val_mask.sum()) != n_rows:
        raise AssertionError("holdout masks do not cover all rows")
    return train_mask, val_mask


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred - target
    return float(np.sqrt(np.mean(diff * diff)))


def evaluate_per_phase(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Compute RMSE per phase mask.

    Returns one float per phase. Phases with zero matching rows yield
    ``float('nan')`` so callers can distinguish "no data" from "perfect fit".
    """
    pred = np.asarray(predictions)
    tgt = np.asarray(targets)
    if pred.shape != tgt.shape:
        raise ValueError(
            f"predictions/targets shape mismatch: {pred.shape} vs {tgt.shape}"
        )
    n_rows = pred.shape[0]
    out: dict[str, float] = {}
    for name, mask in masks.items():
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape[0] != n_rows:
            raise ValueError(
                f"mask '{name}' length {mask_arr.shape[0]} != n_rows {n_rows}"
            )
        if not mask_arr.any():
            out[name] = float("nan")
            continue
        out[name] = _rmse(pred[mask_arr], tgt[mask_arr])
    return out


def evaluate_holdout_trajectory(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE on held-out rows plus simple counts."""
    pred = np.asarray(predictions)
    tgt = np.asarray(targets)
    if pred.shape != tgt.shape:
        raise ValueError(
            f"predictions/targets shape mismatch: {pred.shape} vs {tgt.shape}"
        )
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape[0] != pred.shape[0]:
        raise ValueError(f"mask length {mask_arr.shape[0]} != n_rows {pred.shape[0]}")
    if not mask_arr.any():
        return {"rmse": float("nan"), "n_rows": 0.0}
    return {
        "rmse": _rmse(pred[mask_arr], tgt[mask_arr]),
        "n_rows": float(int(mask_arr.sum())),
    }


__all__: Sequence[str] = (
    "DEFAULT_PHASE_BREAKPOINTS",
    "phase_stratified_split",
    "holdout_trajectory_split",
    "evaluate_per_phase",
    "evaluate_holdout_trajectory",
)
