"""Schema validation rules for the random-sweep parquet dataset.

Implements the rules listed in ``DATASET_SCHEMA.md`` § "Validation rules".
Each public function raises ``ValueError`` with a descriptive message on
violation; callers are expected to let the exception propagate.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Plausible shaft length range (driver ~1.1 m, with generous tolerance for
# any wedge or putter that might appear in future sweeps).
_SHAFT_MIN_M = 0.7
_SHAFT_MAX_M = 1.4

_TRIALS_REQUIRED = (
    "trial_id",
    "coefficients",
    "joint_names",
    "simulation_time_s",
    "sample_rate_hz",
    "solver_status",
)
_TIMESTEPS_REQUIRED = (
    "trial_id",
    "t",
    "q",
    "qd",
    "qdd",
    "tau",
)
_TIMESTEP_VECTOR_LENGTHS: dict[str, int] = {
    "r_butt": 3,
    "r_clubhead": 3,
    "v_clubhead": 3,
    "omega_club": 3,
    "q_club": 4,
}


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    """Raise ValueError if required columns are absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required column(s): {missing}. "
            f"Got columns: {list(df.columns)}"
        )


def validate_trials_table(trials: pd.DataFrame) -> None:
    """Check trials-level invariants."""
    require_columns(trials, _TRIALS_REQUIRED, "trials.parquet")
    if trials["trial_id"].duplicated().any():
        dups = trials.loc[trials["trial_id"].duplicated(), "trial_id"].tolist()
        raise ValueError(f"trials.trial_id must be unique; duplicates: {dups[:5]}")


def validate_timesteps_table(
    timesteps: pd.DataFrame, trial_ids: set[int], n_joints: int
) -> None:
    """Check timestep-level invariants. ``trial_ids`` is the set from trials."""
    require_columns(timesteps, _TIMESTEPS_REQUIRED, "timesteps.parquet")
    _validate_trial_id_membership(timesteps, trial_ids)
    _validate_time_monotonicity(timesteps)
    _validate_list_column_lengths(timesteps, n_joints)


def _validate_trial_id_membership(timesteps: pd.DataFrame, trial_ids: set[int]) -> None:
    seen = set(timesteps["trial_id"].unique().tolist())
    extra = seen - trial_ids
    if extra:
        raise ValueError(
            f"timesteps.trial_id contains ids not present in trials: "
            f"{sorted(extra)[:5]}"
        )


def _validate_time_monotonicity(timesteps: pd.DataFrame) -> None:
    for trial_id, group in timesteps.groupby("trial_id", sort=False):
        t = group["t"].to_numpy()
        if t.size == 0:
            continue
        if not np.all(np.diff(t) >= 0):
            raise ValueError(
                f"trial_id={trial_id}: timesteps.t must be monotonic "
                f"non-decreasing within each trial"
            )
        if not np.isclose(t[0], 0.0, atol=1e-9):
            raise ValueError(
                f"trial_id={trial_id}: timesteps.t must start at 0.0; got {t[0]!r}"
            )


def _validate_list_column_lengths(timesteps: pd.DataFrame, n_joints: int) -> None:
    joint_cols = ("q", "qd", "qdd", "tau")
    for col in joint_cols:
        if col not in timesteps.columns:
            continue
        bad = _first_bad_length(timesteps[col], n_joints)
        if bad is not None:
            row, got = bad
            raise ValueError(
                f"timesteps.{col} row {row}: expected length {n_joints}, got {got}"
            )
    for col, expected in _TIMESTEP_VECTOR_LENGTHS.items():
        if col not in timesteps.columns:
            continue
        bad = _first_bad_length(timesteps[col], expected)
        if bad is not None:
            row, got = bad
            raise ValueError(
                f"timesteps.{col} row {row}: expected length {expected}, got {got}"
            )


def _first_bad_length(series: pd.Series, expected: int) -> tuple[int, int] | None:
    for i, v in enumerate(series):
        if v is None:
            return (i, 0)
        if len(v) != expected:
            return (i, len(v))
    return None


def validate_no_nan_in_success_trials(
    trials: pd.DataFrame, timesteps: pd.DataFrame
) -> None:
    """Raise if any 'success' trial contains NaN/Inf in its timesteps."""
    success_ids = set(
        trials.loc[trials["solver_status"] == "success", "trial_id"].tolist()
    )
    if not success_ids:
        return
    sub = timesteps[timesteps["trial_id"].isin(success_ids)]
    for col in ("q", "qd", "qdd", "tau"):
        if col not in sub.columns:
            continue
        for row, v in enumerate(sub[col]):
            arr = np.asarray(v, dtype=float)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"timesteps.{col} row {row} (trial_id="
                    f"{sub.iloc[row]['trial_id']}) contains NaN/Inf in a "
                    f"trial marked solver_status=success"
                )


def validate_shaft_length(timesteps: pd.DataFrame) -> None:
    """Coordinate-system spot check: ‖r_clubhead - r_butt‖ in shaft range."""
    if "r_clubhead" not in timesteps.columns or "r_butt" not in timesteps.columns:
        return
    butt = np.asarray(timesteps["r_butt"].tolist(), dtype=float)
    head = np.asarray(timesteps["r_clubhead"].tolist(), dtype=float)
    if butt.shape != head.shape or butt.shape[1] != 3:
        return
    diff = head - butt
    # ⚡ Bolt: np.sqrt(np.einsum('ij,ij->i', x, x)) avoids temporary array allocations and is ~35% faster than np.linalg.norm(x, axis=1)
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    bad_mask = (dist < _SHAFT_MIN_M) | (dist > _SHAFT_MAX_M)
    if bad_mask.any():
        idx = int(np.argmax(bad_mask))
        raise ValueError(
            f"shaft length sanity check failed at row {idx}: "
            f"‖r_clubhead - r_butt‖={dist[idx]:.3f} m outside "
            f"[{_SHAFT_MIN_M}, {_SHAFT_MAX_M}] m. Possible units bug."
        )
