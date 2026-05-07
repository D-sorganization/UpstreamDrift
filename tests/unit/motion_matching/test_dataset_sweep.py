"""Tests for ``load_sweep_dataset`` against the synthetic generator."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.shared.python.motion_matching.dataset import (
    SCHEMA_VERSION,
    SweepDataset,
    load_sweep_dataset,
    make_synthetic_sweep,
)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Write a small synthetic dataset and return its folder path."""
    return make_synthetic_sweep(
        tmp_path / "sweep", n_trials=4, n_joints=6, n_timesteps=20, seed=7
    )


@pytest.mark.unit
def test_load_sweep_dataset_round_trips_synthetic(tiny_dataset: Path) -> None:
    ds = load_sweep_dataset(tiny_dataset, lazy=False)

    assert isinstance(ds, SweepDataset)
    assert ds.n_trials() == 4
    assert ds.n_joints() == 6
    assert ds.schema_version == SCHEMA_VERSION
    assert isinstance(ds.timesteps, pd.DataFrame)
    assert len(ds.timesteps) == 4 * 20


@pytest.mark.unit
def test_load_sweep_dataset_rejects_missing_file_with_clear_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError) as exc:
        load_sweep_dataset(missing)
    msg = str(exc.value)
    assert "make_synthetic_sweep" in msg
    assert str(missing.resolve()) in msg


@pytest.mark.unit
def test_load_sweep_dataset_rejects_missing_columns(
    tiny_dataset: Path,
) -> None:
    trials = pd.read_parquet(tiny_dataset / "trials.parquet")
    trials = trials.drop(columns=["solver_status"])
    trials.to_parquet(tiny_dataset / "trials.parquet", index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_sweep_dataset(tiny_dataset, lazy=False)


@pytest.mark.unit
def test_load_sweep_dataset_rejects_non_monotonic_time_within_trial(
    tiny_dataset: Path,
) -> None:
    ts = pd.read_parquet(tiny_dataset / "timesteps.parquet")
    mask = ts["trial_id"] == 0
    ts.loc[mask, "t"] = ts.loc[mask, "t"].iloc[::-1].to_numpy()
    ts.to_parquet(tiny_dataset / "timesteps.parquet", index=False)
    with pytest.raises(ValueError, match="monotonic"):
        load_sweep_dataset(tiny_dataset, lazy=False)


@pytest.mark.unit
def test_load_sweep_dataset_rejects_mismatched_trial_ids(
    tiny_dataset: Path,
) -> None:
    ts = pd.read_parquet(tiny_dataset / "timesteps.parquet")
    ts.loc[ts.index[0], "trial_id"] = np.uint32(999)
    ts.to_parquet(tiny_dataset / "timesteps.parquet", index=False)
    with pytest.raises(ValueError, match="not present in trials"):
        load_sweep_dataset(tiny_dataset, lazy=False)


@pytest.mark.unit
def test_load_sweep_dataset_lazy_mode_returns_polars(tiny_dataset: Path) -> None:
    polars = pytest.importorskip("polars")
    ds = load_sweep_dataset(tiny_dataset, lazy=True)
    assert isinstance(ds.timesteps, polars.LazyFrame)


@pytest.mark.unit
def test_per_timestep_iter_yields_user_framing(tiny_dataset: Path) -> None:
    ds = load_sweep_dataset(tiny_dataset, lazy=False)
    samples = list(ds.per_timestep_iter())
    assert len(samples) == ds.n_trials() * 20
    trial_id, sample = samples[0]
    assert isinstance(trial_id, int)
    for key in ("t", "q", "qd", "qdd", "tau"):
        assert key in sample
    assert len(sample["q"]) == ds.n_joints()


@pytest.mark.unit
def test_real_dataset_path_skipped_when_absent() -> None:
    """If the real parquet dataset is checked in, smoke-load it.

    The user has stated the dataset will be copied in soon. Until then
    this test skips. When it lands, this test exercises the loader on
    real data without modification.
    """
    # Search common locations under the repo.
    here = Path(__file__).resolve()
    candidates = []
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            candidates.extend(
                [
                    parent / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/"
                    "matlab/motion_matching/data",
                    parent / "motion_matching/data",
                    parent / "data/motion_matching",
                ]
            )
            break

    real = next(
        (p for p in candidates if p.exists() and (p / "trials.parquet").exists()),
        None,
    )
    if real is None:
        pytest.skip(
            "real sweep dataset not present yet; using synthetic only. "
            "Drop trials.parquet/timesteps.parquet under "
            "motion_matching/data/ to run this test."
        )
    ds = load_sweep_dataset(real, lazy=False)
    assert ds.n_trials() > 0


@pytest.mark.unit
def test_real_10k_dataset_matches_schema_contract() -> None:
    """Issue #4074 acceptance guard for the real random-sweep dataset.

    This stays skipped until the 10k dataset is copied into a repo or
    env-configured location. Once present, it enforces the full contract
    needed before downstream Option-2/3/leaderboard training issues can run.
    """
    real = _find_real_sweep_dataset()
    if real is None:
        pytest.skip(
            "real 10k sweep dataset not present. Set "
            "UPSTREAMDRIFT_SWEEP_DATASET_PATH or place trials.parquet and "
            "timesteps.parquet under a documented motion_matching data folder."
        )

    ds = load_sweep_dataset(real, lazy=False)
    timesteps = ds.timesteps

    assert ds.schema_version == SCHEMA_VERSION
    assert ds.n_trials() == 10_000
    assert isinstance(timesteps, pd.DataFrame)
    assert set(timesteps["trial_id"]).issubset(set(ds.trials["trial_id"]))

    required_trials = {
        "trial_id",
        "coefficients",
        "joint_names",
        "simulation_time_s",
        "sample_rate_hz",
        "solver_status",
    }
    required_timesteps = {"trial_id", "t", "q", "qd", "qdd", "tau"}
    assert required_trials.issubset(ds.trials.columns)
    assert required_timesteps.issubset(timesteps.columns)

    successful_ids = set(
        ds.trials.loc[ds.trials["solver_status"] == "success", "trial_id"]
    )
    successful_timesteps = timesteps[timesteps["trial_id"].isin(successful_ids)]
    for column in ("q", "qd", "qdd", "tau"):
        values = np.asarray(successful_timesteps[column].tolist(), dtype=float)
        assert values.shape[1] == ds.n_joints()
        assert np.isfinite(values).all()


def _find_real_sweep_dataset() -> Path | None:
    configured = os.environ.get("UPSTREAMDRIFT_SWEEP_DATASET_PATH")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))

    repo_root = _repo_root()
    candidates.extend(
        [
            repo_root / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/"
            "motion_matching/data",
            repo_root / "motion_matching/data",
            repo_root / "data/motion_matching",
        ]
    )
    return next((path for path in candidates if _is_sweep_dataset_dir(path)), None)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repository root")


def _is_sweep_dataset_dir(path: Path) -> bool:
    return (path / "trials.parquet").exists() and (path / "timesteps.parquet").exists()
