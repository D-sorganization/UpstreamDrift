"""Unit tests for surrogate validation splits and per-phase residual reports.

Covers issue #3972: phase-stratified splits and held-out trajectory splits
for the dynamics surrogate trainer.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# The module lives under src/engines/.../MachineLearning/, which is not on
# sys.path. Load it by file path so tests do not require packaging changes.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "MachineLearning"
    / "surrogate_validation.py"
)
_spec = importlib.util.spec_from_file_location("surrogate_validation", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
sv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sv)


def _make_synthetic_swings(
    n_trials: int = 6, n_steps: int = 50, seed: int = 0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        t = np.linspace(0.0, 1.0, n_steps)
        for ti in t:
            rows.append((trial, ti, float(rng.standard_normal())))
    arr = np.array(rows, dtype=np.float64)
    return {
        "trial_id": arr[:, 0].astype(int),
        "t": arr[:, 1],
        "value": arr[:, 2],
    }


def test_phase_split_partitions_data() -> None:
    df = _make_synthetic_swings()
    masks = sv.phase_stratified_split(df)
    n_rows = len(df["t"])

    # Disjoint and exhaustive.
    coverage = np.zeros(n_rows, dtype=bool)
    for name, mask in masks.items():
        assert mask.dtype == bool
        assert mask.shape == (n_rows,)
        assert not (coverage & mask).any(), f"phase '{name}' overlaps another"
        coverage |= mask
    assert coverage.all()
    # Default breakpoints have 4 phases.
    assert set(masks) == {"address", "top", "impact", "finish"}


def test_phase_breakpoints_normalize_to_swing_window() -> None:
    # A non-[0,1] time column must still produce the same partition
    # as the canonical [0,1] one.
    df_unit = {"t": np.linspace(0.0, 1.0, 100)}
    df_shifted = {"t": np.linspace(10.0, 30.0, 100)}
    masks_unit = sv.phase_stratified_split(df_unit)
    masks_shifted = sv.phase_stratified_split(df_shifted)
    for name in masks_unit:
        assert np.array_equal(masks_unit[name], masks_shifted[name])

    # Custom breakpoints expressed in raw seconds should also normalize.
    custom = {"address": 0.0, "impact": 0.5, "finish": 2.0}
    masks_custom = sv.phase_stratified_split(
        {"t": np.linspace(0.0, 2.0, 200)}, phase_breakpoints=custom
    )
    assert set(masks_custom) == {"address", "impact", "finish"}
    union = np.zeros(200, dtype=bool)
    for m in masks_custom.values():
        union |= m
    assert union.all()


def test_holdout_split_disjoint_trial_ids() -> None:
    df = _make_synthetic_swings(n_trials=10, n_steps=20)
    train, val = sv.holdout_trajectory_split(df, frac=0.3, seed=123)

    # Disjoint, complementary, non-empty.
    assert not (train & val).any()
    assert (train | val).all()
    assert train.sum() > 0
    assert val.sum() > 0

    # Disjoint trial IDs.
    train_trials = set(np.asarray(df["trial_id"])[train].tolist())
    val_trials = set(np.asarray(df["trial_id"])[val].tolist())
    assert train_trials.isdisjoint(val_trials)
    # Roughly 30% of 10 = 3 trials held out.
    assert len(val_trials) == 3


def test_evaluate_per_phase_returns_one_metric_per_phase() -> None:
    df = _make_synthetic_swings()
    masks = sv.phase_stratified_split(df)
    n = len(df["t"])
    pred = np.zeros((n, 3), dtype=np.float64)
    target = np.ones((n, 3), dtype=np.float64)  # constant residual of 1.0

    metrics = sv.evaluate_per_phase(pred, target, masks)
    assert set(metrics) == set(masks)
    for name, value in metrics.items():
        assert value == pytest.approx(1.0), f"phase {name} RMSE != 1.0"


def test_evaluate_holdout_against_synthetic_overfit_baseline() -> None:
    df = _make_synthetic_swings(n_trials=8, n_steps=30, seed=7)
    train_mask, val_mask = sv.holdout_trajectory_split(df, frac=0.25, seed=7)

    target = np.asarray(df["value"]).reshape(-1, 1)
    # Overfit baseline: predict perfectly on training rows but emit large
    # residuals everywhere else. The held-out evaluation must surface that.
    pred = target.copy()
    pred[val_mask] = target[val_mask] + 10.0

    holdout = sv.evaluate_holdout_trajectory(pred, target, val_mask)
    assert holdout["rmse"] == pytest.approx(10.0)
    assert holdout["n_rows"] == float(int(val_mask.sum()))

    # The training-side residual is zero, confirming the split actually
    # excluded those rows from "validation".
    train_only = sv.evaluate_holdout_trajectory(pred, target, train_mask)
    assert train_only["rmse"] == pytest.approx(0.0)


def test_phase_split_rejects_zero_range_time() -> None:
    df = {"t": np.zeros(10)}
    with pytest.raises(ValueError, match="positive range"):
        sv.phase_stratified_split(df)


def test_holdout_split_rejects_invalid_frac() -> None:
    df = _make_synthetic_swings(n_trials=4, n_steps=5)
    with pytest.raises(ValueError, match="frac"):
        sv.holdout_trajectory_split(df, frac=0.0)
    with pytest.raises(ValueError, match="frac"):
        sv.holdout_trajectory_split(df, frac=1.0)
