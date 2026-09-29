"""Random-sweep parquet dataset loader.

The on-disk layout is documented in ``DATASET_SCHEMA.md``: a folder
containing ``trials.parquet`` (one row per simulation) and
``timesteps.parquet`` (one row per timestep, joinable on ``trial_id``).

The user has framed training as **per-timestep sample of (kinematics,
torques)**; ``SweepDataset.per_timestep_iter`` exposes that view.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger

from . import _validate

logger = get_logger(__name__)

# Schema version. Bump on breaking changes (column rename, dtype change,
# semantic shift in units). Minor additive changes (new optional column)
# do not require a bump.
SCHEMA_VERSION = "0.1.0"

_TRIALS_FILENAME = "trials.parquet"
_TIMESTEPS_FILENAME = "timesteps.parquet"


@dataclass(frozen=True)
class SweepDataset:
    """Loaded random-sweep dataset.

    Attributes:
        trials: One row per simulation.
        timesteps: One row per timestep; may be a polars LazyFrame when
            the loader was called with ``lazy=True``.
        joint_names: Joint ordering for the ``q``/``qd``/``qdd``/``tau``
            list-columns in ``timesteps``.
        schema_version: Schema version this dataset was loaded against.
    """

    trials: pd.DataFrame
    timesteps: Any  # pd.DataFrame or polars.LazyFrame
    joint_names: list[str]
    schema_version: str

    def n_trials(self) -> int:
        """Return the number of trials in the dataset."""
        return len(self.trials)

    def n_joints(self) -> int:
        """Return the number of joints (length of ``joint_names``)."""
        return len(self.joint_names)

    def per_timestep_iter(self) -> Iterator[tuple[int, dict]]:
        """Yield ``(trial_id, sample_dict)`` for the per-timestep framing.

        Each ``sample_dict`` contains scalar ``t`` and the per-joint
        list/array entries ``q``, ``qd``, ``qdd``, ``tau`` plus any
        club-kinematic vectors that are present.

        This intentionally materialises the ``timesteps`` frame as
        pandas so the iteration is uniform whether the loader was lazy
        or eager; callers wanting a streaming polars iterator should
        consume the ``timesteps`` LazyFrame directly.
        """
        ts = _as_pandas(self.timesteps)
        for row in ts.itertuples(index=False):
            sample = row._asdict()  # type: ignore[operator]
            trial_id = int(sample.pop("trial_id"))
            yield trial_id, sample


@precondition(
    lambda path, **_: Path(path).parent.exists(),
    "parent directory of dataset path must exist",
)
@postcondition(
    lambda result: result.n_trials() > 0,
    "loaded dataset must contain at least one trial",
)
def load_sweep_dataset(path: str | Path, *, lazy: bool = True) -> SweepDataset:
    """Load the random-sweep parquet dataset from a folder.

    Args:
        path: Folder containing ``trials.parquet`` and ``timesteps.parquet``.
        lazy: If True, return a polars LazyFrame for ``timesteps`` (large
            file). Falls back to pandas if polars is not installed.

    Returns:
        A validated :class:`SweepDataset`.

    Raises:
        FileNotFoundError: If the folder or either parquet file is missing.
            The error message includes the absolute path searched and a
            hint about ``make_synthetic_sweep`` for tests.
        ValueError: If the dataset fails any schema validation rule.
    """
    folder = Path(path)
    trials_path, timesteps_path = _resolve_files(folder)

    logger.info("Loading sweep dataset from %s", folder)
    trials = pd.read_parquet(trials_path)
    timesteps_pd = pd.read_parquet(timesteps_path)

    joint_names = _extract_joint_names(trials)
    _run_validation(trials, timesteps_pd, joint_names)

    timesteps_out = _maybe_lazy(timesteps_path, timesteps_pd, lazy=lazy)
    return SweepDataset(
        trials=trials,
        timesteps=timesteps_out,
        joint_names=joint_names,
        schema_version=SCHEMA_VERSION,
    )


def _resolve_files(folder: Path) -> tuple[Path, Path]:
    """Locate ``trials.parquet`` and ``timesteps.parquet`` under ``folder``."""
    if not folder.exists():
        raise FileNotFoundError(
            f"sweep dataset folder not found: {folder.resolve()}. "
            f"The real dataset has not been copied into the repo yet — "
            f"use motion_matching.dataset.make_synthetic_sweep(path) to "
            f"generate a small fake dataset for tests."
        )
    trials_path = folder / _TRIALS_FILENAME
    timesteps_path = folder / _TIMESTEPS_FILENAME
    if not trials_path.exists():
        raise FileNotFoundError(f"missing {_TRIALS_FILENAME} in {folder.resolve()}")
    if not timesteps_path.exists():
        raise FileNotFoundError(f"missing {_TIMESTEPS_FILENAME} in {folder.resolve()}")
    return trials_path, timesteps_path


def _extract_joint_names(trials: pd.DataFrame) -> list[str]:
    """Pull joint_names from the first row; assume uniform across trials."""
    if "joint_names" not in trials.columns or len(trials) == 0:
        raise ValueError("trials.parquet must contain a non-empty joint_names column")
    first = trials["joint_names"].iloc[0]
    return [str(n) for n in first]


def _run_validation(
    trials: pd.DataFrame, timesteps: pd.DataFrame, joint_names: list[str]
) -> None:
    _validate.validate_trials_table(trials)
    trial_ids = set(trials["trial_id"].tolist())
    _validate.validate_timesteps_table(timesteps, trial_ids, len(joint_names))
    _validate.validate_no_nan_in_success_trials(trials, timesteps)
    _validate.validate_shaft_length(timesteps)


def _maybe_lazy(timesteps_path: Path, eager: pd.DataFrame, *, lazy: bool) -> Any:
    """Return a polars LazyFrame if requested + available, else pandas."""
    if not lazy:
        return eager
    try:
        import polars as pl
    except ImportError:
        logger.info("polars not installed; falling back to pandas timesteps")
        return eager
    return pl.scan_parquet(str(timesteps_path))


def _as_pandas(timesteps: Any) -> pd.DataFrame:
    """Materialise either a pandas or polars LazyFrame to pandas."""
    if isinstance(timesteps, pd.DataFrame):
        return timesteps
    # Duck-type polars LazyFrame / DataFrame
    if hasattr(timesteps, "collect"):
        return timesteps.collect().to_pandas()
    if hasattr(timesteps, "to_pandas"):
        return timesteps.to_pandas()
    raise TypeError(f"unsupported timesteps type: {type(timesteps)!r}")
