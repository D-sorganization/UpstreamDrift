"""Loader for the compact swing dataset (``COMPACT_DATASET_SCHEMA.md``).

The compactor produces ``trials.parquet`` and ``timesteps.parquet`` in a
single output directory. ``load_compact_swing_dataset`` reads both,
cross-validates the schema and the inter-table integrity rules listed
in §"Validation rules" of the schema doc, and returns a frozen
``CompactSwingDataset`` dataclass.

DbC: the function declares preconditions (path exists, both files
present) and postconditions (non-zero rows, validated schema). Failures
raise ``FileNotFoundError`` or ``ValueError`` with descriptive messages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_COEFFS,
    SCHEMA_VERSION,
    TIMESTEP_LIST_LENGTHS,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_LOGGER = logging.getLogger(__name__)

_TRIALS_FILE = "trials.parquet"
_TIMESTEPS_FILE = "timesteps.parquet"

_REQUIRED_TRIAL_COLS: tuple[str, ...] = (
    "trial_id",
    "coefficients",
    "joint_names",
    "coefficient_letters",
    "simulation_time_s",
    "sample_rate_hz",
    "clubhead_speed_max_mph",
    "total_work_J",
    "solver_status",
)

_REQUIRED_TIMESTEP_COLS: tuple[str, ...] = (
    "trial_id",
    "t",
    "q",
    "qd",
    "qdd",
    "tau",
    "r_clubhead",
    "v_clubhead",
    "r_buttend",
    "r_lhand",
    "r_rhand",
    "r_grip",
    "clubhead_speed_mph",
)


@dataclass(frozen=True)
class CompactSwingDataset:
    """In-memory handle for the compact swing dataset.

    Attributes:
        trials: One row per simulation. Either a ``polars.LazyFrame``
            (when ``lazy=True``) or a ``pandas.DataFrame``.
        timesteps: One row per simulation timestep, same backend as
            ``trials``.
        joint_names: Canonical 27-joint ordering used in every list
            column.
        coefficient_letters: ``["A", ..., "G"]``.
        schema_version: Equal to ``"compact-1.0"`` for files produced
            by this codebase.
    """

    trials: Any
    timesteps: Any
    joint_names: list[str] = field(default_factory=lambda: list(CANONICAL_JOINTS))
    coefficient_letters: list[str] = field(
        default_factory=lambda: list(COEFFICIENT_LETTERS)
    )
    schema_version: str = SCHEMA_VERSION


def load_compact_swing_dataset(
    path: str | Path,
    *,
    lazy: bool = True,
) -> CompactSwingDataset:
    """Load the compact swing parquet dataset from ``path``.

    Preconditions:
        * ``path`` exists and is a directory.
        * It contains ``trials.parquet`` and ``timesteps.parquet``.

    Postconditions:
        * ``len(result.trials) > 0`` and ``len(result.timesteps) > 0``.
        * Every list column has the documented fixed length.
        * Every ``trial_id`` referenced by ``timesteps`` exists in
          ``trials`` (FK integrity).
        * No NaN/Inf in any numeric column.

    Args:
        path: Folder containing ``trials.parquet`` and
            ``timesteps.parquet``.
        lazy: If ``True``, return ``polars.LazyFrame`` views. If
            ``False``, return eagerly materialized ``pandas.DataFrame``
            objects.

    Returns:
        A frozen :class:`CompactSwingDataset`.

    Raises:
        TypeError: If ``path`` cannot be coerced to a ``Path``.
        FileNotFoundError: If the directory or expected parquet files
            are missing.
        ValueError: If schema or value-range validation fails.
    """
    folder = _coerce_path(path)
    trials_path, timesteps_path = _check_files_exist(folder)

    # Read eagerly with pandas to validate, then optionally re-emit as
    # polars LazyFrames for downstream use.  Validation MUST run before
    # the lazy handle is constructed; otherwise schema-violation files
    # would slip through unnoticed.
    import pandas as pd

    trials_df = pd.read_parquet(trials_path)
    timesteps_df = pd.read_parquet(timesteps_path)

    _validate_required_columns(trials_df, _REQUIRED_TRIAL_COLS, "trials")
    _validate_required_columns(timesteps_df, _REQUIRED_TIMESTEP_COLS, "timesteps")
    _validate_non_empty(trials_df, timesteps_df)
    _validate_trial_uniqueness(trials_df)
    _validate_fk_integrity(trials_df, timesteps_df)
    _validate_list_lengths(trials_df, timesteps_df)
    _validate_no_nan(trials_df, timesteps_df)
    _validate_time_monotonicity(trials_df, timesteps_df)

    if lazy:
        import polars as pl

        trials_view: Any = pl.from_pandas(trials_df).lazy()
        timesteps_view: Any = pl.from_pandas(timesteps_df).lazy()
    else:
        trials_view = trials_df
        timesteps_view = timesteps_df

    return CompactSwingDataset(
        trials=trials_view,
        timesteps=timesteps_view,
        joint_names=list(CANONICAL_JOINTS),
        coefficient_letters=list(COEFFICIENT_LETTERS),
        schema_version=SCHEMA_VERSION,
    )


# ----- internal helpers --------------------------------------------------


def _coerce_path(path: str | Path) -> Path:
    if not isinstance(path, str | Path):
        raise TypeError(f"path must be str or pathlib.Path, got {type(path).__name__}")
    folder = Path(path)
    if not folder.exists():
        raise FileNotFoundError(f"compact dataset folder does not exist: {folder}")
    if not folder.is_dir():
        raise FileNotFoundError(f"compact dataset path is not a directory: {folder}")
    return folder


def _check_files_exist(folder: Path) -> tuple[Path, Path]:
    trials_path = folder / _TRIALS_FILE
    timesteps_path = folder / _TIMESTEPS_FILE
    missing = [p.name for p in (trials_path, timesteps_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"compact dataset folder {folder} missing files: {missing}"
        )
    return trials_path, timesteps_path


def _validate_required_columns(
    df: pd.DataFrame, required: tuple[str, ...], label: str
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}.parquet is missing required columns: {missing}")


def _validate_non_empty(trials_df: pd.DataFrame, timesteps_df: pd.DataFrame) -> None:
    if len(trials_df) == 0:
        raise ValueError("trials.parquet has zero rows")
    if len(timesteps_df) == 0:
        raise ValueError("timesteps.parquet has zero rows")


def _validate_trial_uniqueness(trials_df: pd.DataFrame) -> None:
    duplicate_count = int(trials_df["trial_id"].duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"trials.trial_id contains {duplicate_count} duplicate values")


def _validate_fk_integrity(trials_df: pd.DataFrame, timesteps_df: pd.DataFrame) -> None:
    trial_ids = set(trials_df["trial_id"].tolist())
    orphan_mask = ~timesteps_df["trial_id"].isin(trial_ids)
    if bool(orphan_mask.any()):
        n_orphan = int(orphan_mask.sum())
        raise ValueError(
            f"timesteps.parquet has {n_orphan} rows with trial_id not in trials.parquet"
        )


def _validate_list_lengths(trials_df: pd.DataFrame, timesteps_df: pd.DataFrame) -> None:
    """Confirm every list column matches its documented fixed length."""
    for column, expected_len in TIMESTEP_LIST_LENGTHS.items():
        bad = _first_bad_list_length(timesteps_df[column], expected_len)
        if bad is not None:
            raise ValueError(
                f"timesteps.{column} length mismatch: row {bad[0]} has length "
                f"{bad[1]}, expected {expected_len}"
            )

    bad_coef = _first_bad_list_length(trials_df["coefficients"], N_COEFFS)
    if bad_coef is not None:
        raise ValueError(
            f"trials.coefficients length mismatch: row {bad_coef[0]} has "
            f"length {bad_coef[1]}, expected {N_COEFFS}"
        )

    bad_jn = _first_bad_list_length(trials_df["joint_names"], len(CANONICAL_JOINTS))
    if bad_jn is not None:
        raise ValueError(
            f"trials.joint_names length mismatch: row {bad_jn[0]} has length "
            f"{bad_jn[1]}, expected {len(CANONICAL_JOINTS)}"
        )

    bad_cl = _first_bad_list_length(
        trials_df["coefficient_letters"], len(COEFFICIENT_LETTERS)
    )
    if bad_cl is not None:
        raise ValueError(
            f"trials.coefficient_letters length mismatch: row {bad_cl[0]} "
            f"has length {bad_cl[1]}, expected {len(COEFFICIENT_LETTERS)}"
        )


def _first_bad_list_length(
    series: pd.Series, expected_len: int
) -> tuple[int, int] | None:
    """Return ``(row_index, actual_len)`` for the first row whose list
    length differs from ``expected_len``, else ``None``."""
    for idx, value in enumerate(series.tolist()):
        if value is None:
            return idx, 0
        actual = len(value)
        if actual != expected_len:
            return idx, actual
    return None


def _validate_no_nan(trials_df: pd.DataFrame, timesteps_df: pd.DataFrame) -> None:
    """Reject NaN/Inf in any numeric or list-of-numeric column."""
    scalar_numeric_trial = (
        "simulation_time_s",
        "sample_rate_hz",
        "clubhead_speed_max_mph",
        "total_work_J",
    )
    for col in scalar_numeric_trial:
        for idx, val in enumerate(trials_df[col].tolist()):
            if val is None or _is_bad_float(val):
                raise ValueError(f"trials.{col} contains NaN/Inf at row {idx}")

    scalar_numeric_timestep = ("t", "clubhead_speed_mph")
    for col in scalar_numeric_timestep:
        for idx, val in enumerate(timesteps_df[col].tolist()):
            if val is None or _is_bad_float(val):
                raise ValueError(f"timesteps.{col} contains NaN/Inf at row {idx}")

    # The four joint-vector columns ``q/qd/qdd/tau`` may legitimately
    # carry NaN at indices where the raw dump has no source column
    # (documented in compact_swing_dataset.py — see TAU_NULL_JOINTS and
    # KNOWN_MISSING_KINEMATIC_COLUMNS). Inf is still rejected.  The
    # geometric position columns (r_*, v_*) MUST be fully finite — a
    # NaN there is evidence of a unit-conversion bug.
    nan_tolerant_cols = {"tau", "q", "qd", "qdd"}
    for col in TIMESTEP_LIST_LENGTHS:
        bad = _first_bad_list_value(
            timesteps_df[col], allow_nan=col in nan_tolerant_cols
        )
        if bad is not None:
            raise ValueError(f"timesteps.{col} contains NaN/Inf at row {bad}")

    bad_coef = _first_bad_list_value(trials_df["coefficients"])
    if bad_coef is not None:
        raise ValueError(f"trials.coefficients contains NaN/Inf at row {bad_coef}")


def _is_bad_float(val: float) -> bool:
    import math

    return not math.isfinite(float(val))


def _first_bad_list_value(series: pd.Series, *, allow_nan: bool = False) -> int | None:
    """First row index whose list contains a non-finite (or None) entry.

    With ``allow_nan=True``, NaN values are tolerated but Inf and ``None``
    still fail — matching the schema-doc exception for ``tau``.
    """
    import math

    for idx, value in enumerate(series.tolist()):
        if value is None:
            return idx
        for item in value:
            if item is None:
                return idx
            f = float(item)
            if math.isnan(f):
                if allow_nan:
                    continue
                return idx
            if math.isinf(f):
                return idx
    return None


def _validate_time_monotonicity(
    trials_df: pd.DataFrame, timesteps_df: pd.DataFrame
) -> None:
    """Each trial's ``t`` must start at 0 and be monotonic non-decreasing."""
    grouped = timesteps_df.groupby("trial_id", sort=False)
    for trial_id, group in grouped:
        times = group["t"].to_numpy()
        if len(times) == 0:
            continue
        if abs(float(times[0])) > 1e-9:
            raise ValueError(
                f"timesteps.t for trial_id={trial_id} does not start at 0 "
                f"(first value={float(times[0])!r})"
            )
        diffs = times[1:] - times[:-1]
        if (diffs < -1e-12).any():
            raise ValueError(f"timesteps.t for trial_id={trial_id} is not monotonic")
