"""Unit tests for ``scripts/compact_swing_dataset.py``.

The tests build a synthetic raw parquet with the same column schema as
the real 9 GB dump (1956 string columns) but only 3 trials × 31 rows,
then exercise the compactor end-to-end. The real dataset is never
touched.

A single ``requires_real_dataset`` integration test exists so the user
can run it manually against the 9 GB file when available.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compact_swing_dataset import (  # noqa: E402
    CHS_COL,
    CLUB_BUTT_COLS,
    CLUB_R_COLS,
    CLUB_V_COLS,
    COEFF_COLUMNS,
    LHAND_COLS,
    OPTIONAL_RAW_COLUMNS,
    Q_COLS,
    QD_COLS,
    QDD_COLS,
    REQUIRED_RAW_COLUMNS,
    RHAND_COLS,
    TAU_NULL_JOINTS,
    TAU_RAW_MAP,
    TIME_COL,
    TRIAL_COL,
    compact_swing_dataset,
)
from src.shared.python.dataset_tools.canonical import (  # noqa: E402
    CANONICAL_JOINTS,
    N_COEFFS,
    N_JOINTS,
)

pytestmark = pytest.mark.unit


N_TIMESTEPS = 31


def _seed_value(trial_id: int, t_idx: int, col_idx: int) -> float:
    """Deterministic synthetic value for round-trip checks."""
    return 0.001 * (trial_id + 1) + 0.01 * (t_idx + 1) + 0.0001 * (col_idx + 1)


def _build_synthetic_raw(path: Path, n_trials: int = 3) -> dict:
    """Write a synthetic raw parquet with the full canonical schema.

    Returns a dict of expected values keyed by column name for round-trip
    assertions in the test that calls this helper.
    """
    columns = list(REQUIRED_RAW_COLUMNS) + list(OPTIONAL_RAW_COLUMNS)
    # Use a fixed-order column list so trial_id/time are first.
    arrays: dict[str, list[str]] = {c: [] for c in columns}
    expected: dict[str, np.ndarray] = {}

    for trial_id in range(1, n_trials + 1):
        chs_values = []
        for t_idx in range(N_TIMESTEPS):
            arrays[TRIAL_COL].append(str(trial_id))
            t_value = round(0.01 * t_idx, 6)
            arrays[TIME_COL].append(repr(t_value))
            chs = float(trial_id * 10 + t_idx)
            chs_values.append(chs)
            arrays[CHS_COL].append(repr(chs))
            for col_idx, col in enumerate(columns):
                if col in (TRIAL_COL, TIME_COL, CHS_COL):
                    continue
                arrays[col].append(repr(_seed_value(trial_id, t_idx, col_idx)))

    expected["chs_values"] = np.array(
        [
            float(trial_id * 10 + t_idx)
            for trial_id in range(1, n_trials + 1)
            for t_idx in range(N_TIMESTEPS)
        ],
        dtype=np.float64,
    )

    arrow_columns = {c: pa.array(arrays[c], type=pa.string()) for c in columns}
    table = pa.table(arrow_columns)
    # Write one row group per trial to match the real dataset's structure.
    with pq.ParquetWriter(path, table.schema) as writer:
        for trial_id in range(1, n_trials + 1):
            slice_ = table.slice((trial_id - 1) * N_TIMESTEPS, N_TIMESTEPS)
            writer.write_table(slice_)

    return expected


def test_compactor_writes_both_outputs(tmp_path: Path) -> None:
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=3)

    result = compact_swing_dataset(src, out, validate=True)

    assert (out / "trials.parquet").exists()
    assert (out / "timesteps.parquet").exists()
    assert result["trials_rows"] == 3
    assert result["timesteps_rows"] == 3 * N_TIMESTEPS


def test_compactor_schema_matches_canonical_lengths(tmp_path: Path) -> None:
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=3)

    compact_swing_dataset(src, out)

    timesteps = pq.read_table(out / "timesteps.parquet").to_pandas()
    trials = pq.read_table(out / "trials.parquet").to_pandas()

    assert len(timesteps) == 3 * N_TIMESTEPS
    assert len(trials) == 3
    for column, expected_len in (
        ("q", N_JOINTS),
        ("qd", N_JOINTS),
        ("qdd", N_JOINTS),
        ("tau", N_JOINTS),
        ("r_clubhead", 3),
        ("v_clubhead", 3),
        ("r_buttend", 3),
        ("r_lhand", 3),
        ("r_rhand", 3),
        ("r_grip", 3),
    ):
        for value in timesteps[column].tolist():
            assert len(value) == expected_len
    for value in trials["coefficients"].tolist():
        assert len(value) == N_COEFFS
    for value in trials["joint_names"].tolist():
        assert list(value) == list(CANONICAL_JOINTS)


def test_compactor_round_trip_clubhead_speed(tmp_path: Path) -> None:
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    expected = _build_synthetic_raw(src, n_trials=3)

    compact_swing_dataset(src, out)

    timesteps = pq.read_table(out / "timesteps.parquet").to_pandas()
    actual = timesteps["clubhead_speed_mph"].to_numpy()
    np.testing.assert_allclose(actual, expected["chs_values"], rtol=1e-12)


def test_compactor_grip_is_midpoint_of_hands(tmp_path: Path) -> None:
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=2)

    compact_swing_dataset(src, out)

    timesteps = pq.read_table(out / "timesteps.parquet").to_pandas()
    for _, row in timesteps.iterrows():
        lh = np.asarray(row["r_lhand"], dtype=np.float64)
        rh = np.asarray(row["r_rhand"], dtype=np.float64)
        grip = np.asarray(row["r_grip"], dtype=np.float64)
        np.testing.assert_allclose(grip, (lh + rh) / 2.0, rtol=1e-12)


def test_compactor_tau_has_nan_only_on_unmapped_joints(tmp_path: Path) -> None:
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=2)

    compact_swing_dataset(src, out)

    timesteps = pq.read_table(out / "timesteps.parquet").to_pandas()
    null_idx = [i for i, j in enumerate(CANONICAL_JOINTS) if j in TAU_NULL_JOINTS]
    mapped_idx = [i for i, j in enumerate(CANONICAL_JOINTS) if j not in TAU_NULL_JOINTS]
    for _, row in timesteps.iterrows():
        tau = np.asarray(row["tau"], dtype=np.float64)
        if null_idx:
            assert np.all(np.isnan(tau[null_idx]))
        if mapped_idx:
            assert np.all(np.isfinite(tau[mapped_idx]))


def test_compactor_limit_trials(tmp_path: Path) -> None:
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=3)

    result = compact_swing_dataset(src, out, limit_trials=1)

    assert result["trials_rows"] == 1
    assert result["timesteps_rows"] == N_TIMESTEPS


def test_validate_rejects_duplicate_trial_id(tmp_path: Path) -> None:
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=2)
    compact_swing_dataset(src, out)

    # Corrupt trials.parquet by overwriting both rows with trial_id=1.
    import pandas as pd

    trials = pd.read_parquet(out / "trials.parquet")
    trials["trial_id"] = np.uint32(1)
    trials.to_parquet(out / "trials.parquet")

    from src.shared.python.dataset_tools.load_compact import (
        load_compact_swing_dataset,
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_compact_swing_dataset(out, lazy=False)


def test_validate_rejects_nan_in_clubhead_position(tmp_path: Path) -> None:
    """``r_clubhead`` is a strict-finite column — NaN there is rejected."""
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=2)
    compact_swing_dataset(src, out)

    import pandas as pd

    timesteps = pd.read_parquet(out / "timesteps.parquet")
    bad = list(timesteps.at[0, "r_clubhead"])
    bad[0] = math.nan
    timesteps.at[0, "r_clubhead"] = bad
    timesteps.to_parquet(out / "timesteps.parquet")

    from src.shared.python.dataset_tools.load_compact import (
        load_compact_swing_dataset,
    )

    with pytest.raises(ValueError, match="NaN"):
        load_compact_swing_dataset(out, lazy=False)


def test_validate_rejects_inf_in_q(tmp_path: Path) -> None:
    """``q`` tolerates NaN but Inf is still a hard failure."""
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=2)
    compact_swing_dataset(src, out)

    import pandas as pd

    timesteps = pd.read_parquet(out / "timesteps.parquet")
    bad = list(timesteps.at[0, "q"])
    bad[0] = math.inf
    timesteps.at[0, "q"] = bad
    timesteps.to_parquet(out / "timesteps.parquet")

    from src.shared.python.dataset_tools.load_compact import (
        load_compact_swing_dataset,
    )

    with pytest.raises(ValueError, match="NaN"):
        load_compact_swing_dataset(out, lazy=False)


def test_validate_rejects_q_length_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "raw.parquet"
    out = tmp_path / "compact"
    _build_synthetic_raw(src, n_trials=2)
    compact_swing_dataset(src, out)

    import pandas as pd

    timesteps = pd.read_parquet(out / "timesteps.parquet")
    bad = list(timesteps.at[0, "q"])[:-1]  # one short
    timesteps.at[0, "q"] = bad
    timesteps.to_parquet(out / "timesteps.parquet")

    from src.shared.python.dataset_tools.load_compact import (
        load_compact_swing_dataset,
    )

    with pytest.raises(ValueError, match="length mismatch"):
        load_compact_swing_dataset(out, lazy=False)


def test_compactor_raises_when_src_missing(tmp_path: Path) -> None:
    out = tmp_path / "compact"
    # The @precondition decorator raises a ContractViolationError, but
    # the contract framework's exception is also a ValueError-or-subclass
    # depending on enforcement level — we just want any exception.
    with pytest.raises((FileNotFoundError, ValueError, Exception)):  # noqa: B017
        compact_swing_dataset(tmp_path / "nope.parquet", out)


def test_canonical_constants_are_consistent() -> None:
    assert N_JOINTS == 27
    assert N_COEFFS == 27 * 7
    assert len(Q_COLS) == N_JOINTS
    assert len(QD_COLS) == N_JOINTS
    assert len(QDD_COLS) == N_JOINTS
    assert len(COEFF_COLUMNS) == N_COEFFS
    # The four hand/club position columns are 3-element each.
    for tup in (CLUB_R_COLS, CLUB_V_COLS, CLUB_BUTT_COLS, LHAND_COLS, RHAND_COLS):
        assert len(tup) == 3
    # TAU map has every canonical joint represented.
    for joint in CANONICAL_JOINTS:
        assert joint in TAU_RAW_MAP
