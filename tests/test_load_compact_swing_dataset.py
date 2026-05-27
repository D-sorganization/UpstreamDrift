"""Unit tests for ``src.shared.python.dataset_tools.load_compact``."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip(
    "pyarrow", reason="pyarrow not installed; skipping compact dataset tests"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compact_swing_dataset import compact_swing_dataset  # noqa: E402
from src.shared.python.dataset_tools import (  # noqa: E402
    CANONICAL_JOINTS,
    SCHEMA_VERSION,
    CompactSwingDataset,
    load_compact_swing_dataset,
)

from tests.test_compact_swing_dataset_compactor import (  # noqa: E402
    _build_synthetic_raw,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def compact_dir(tmp_path: Path) -> Path:
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=3)
    compact_swing_dataset(src, out)
    return out


def test_load_returns_dataclass_with_expected_joint_names(compact_dir: Path) -> None:
    ds = load_compact_swing_dataset(compact_dir, lazy=False)
    assert isinstance(ds, CompactSwingDataset)
    assert ds.joint_names == list(CANONICAL_JOINTS)
    assert len(ds.joint_names) == 27
    assert ds.coefficient_letters == ["A", "B", "C", "D", "E", "F", "G"]
    assert ds.schema_version == SCHEMA_VERSION


def test_load_eager_returns_pandas(compact_dir: Path) -> None:
    ds = load_compact_swing_dataset(compact_dir, lazy=False)
    assert isinstance(ds.trials, pd.DataFrame)
    assert isinstance(ds.timesteps, pd.DataFrame)
    assert len(ds.trials) == 3
    assert len(ds.timesteps) == 3 * 31


def test_load_lazy_returns_polars_lazyframe(compact_dir: Path) -> None:
    polars = pytest.importorskip("polars")
    ds = load_compact_swing_dataset(compact_dir, lazy=True)
    assert isinstance(ds.trials, polars.LazyFrame)
    assert isinstance(ds.timesteps, polars.LazyFrame)
    assert ds.trials.collect().height == 3
    assert ds.timesteps.collect().height == 3 * 31


def test_load_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_compact_swing_dataset(tmp_path / "does_not_exist", lazy=False)


def test_load_rejects_directory_missing_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        load_compact_swing_dataset(empty, lazy=False)


def test_load_rejects_path_pointing_to_a_file(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("hello")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        load_compact_swing_dataset(not_a_dir, lazy=False)


def test_load_rejects_wrong_q_length(compact_dir: Path) -> None:
    timesteps = pd.read_parquet(compact_dir / "timesteps.parquet")
    bad = list(timesteps.at[0, "q"])
    timesteps.at[0, "q"] = bad[:-2]
    timesteps.to_parquet(compact_dir / "timesteps.parquet")

    with pytest.raises(ValueError, match="length mismatch"):
        load_compact_swing_dataset(compact_dir, lazy=False)


def test_load_rejects_orphan_timestep_trial_id(compact_dir: Path) -> None:
    timesteps = pd.read_parquet(compact_dir / "timesteps.parquet")
    timesteps.loc[0, "trial_id"] = 999_999
    timesteps.to_parquet(compact_dir / "timesteps.parquet")

    with pytest.raises(ValueError, match="trial_id not in"):
        load_compact_swing_dataset(compact_dir, lazy=False)


def test_load_rejects_missing_required_column(compact_dir: Path) -> None:
    timesteps = pd.read_parquet(compact_dir / "timesteps.parquet")
    timesteps.drop(columns=["clubhead_speed_mph"]).to_parquet(
        compact_dir / "timesteps.parquet"
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_compact_swing_dataset(compact_dir, lazy=False)


def test_load_rejects_non_monotonic_time(compact_dir: Path) -> None:
    timesteps = pd.read_parquet(compact_dir / "timesteps.parquet")
    # Reverse the order of t-values for the first trial only.
    first = timesteps[timesteps["trial_id"] == timesteps["trial_id"].iloc[0]]
    if len(first) >= 2:
        # Swap rows 0 and 1
        timesteps.loc[first.index[0], "t"] = 1.0
        timesteps.loc[first.index[1], "t"] = 0.5
        timesteps.to_parquet(compact_dir / "timesteps.parquet")

        with pytest.raises(ValueError, match="(start at 0|monotonic)"):
            load_compact_swing_dataset(compact_dir, lazy=False)
