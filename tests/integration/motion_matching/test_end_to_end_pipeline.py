"""End-to-end smoke test for the motion-matching pipeline.

This test deliberately exercises every plug-point in the Option-2
NN-surrogate pipeline so that integration regressions (shim bugs, schema
mismatches between PRs, signature drift) surface here even when each
component's unit tests are individually green.

The test is a SMOKE test, not a scientific validation:

* tolerances are deliberately loose
* the surrogate is intentionally tiny and overfit on a 10-trial dataset
* components that are not yet importable are ``pytest.skip``-ed rather
  than failed -- the goal is to define the smoke-test surface that
  follow-up PRs can land green

Run with::

    python3 -m pytest tests/integration/motion_matching -m integration

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_target_from_trial(loaded_dataset, trial_id: int, fake_provenance):
    """Construct a :class:`ClubTarget` from one trial of the loaded dataset."""
    club_target_mod = pytest.importorskip(
        "src.shared.python.motion_matching.club_target"
    )
    timesteps = loaded_dataset.timesteps
    rows = timesteps[timesteps["trial_id"] == trial_id].sort_values("t")
    if len(rows) == 0:
        pytest.skip(f"no timesteps for trial_id={trial_id}")

    time = rows["t"].to_numpy(dtype=np.float64)
    butt = np.stack([np.asarray(v, dtype=np.float64) for v in rows["r_butt"]])
    head = np.stack([np.asarray(v, dtype=np.float64) for v in rows["r_clubhead"]])
    quat = np.stack([np.asarray(v, dtype=np.float64) for v in rows["q_club"]])
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1.0e-12)

    impact_idx = int(len(time) // 2)
    return club_target_mod.ClubTarget(
        time=time,
        butt=butt,
        clubhead=head,
        club_quat=quat,
        impact_idx=impact_idx,
        source=fake_provenance("smoke_trial.bin"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_synthetic_dataset_round_trip(loaded_dataset):
    """The dataset loads, has the expected shape, and contains successful trials."""
    assert len(loaded_dataset.trials) > 0
    assert (loaded_dataset.trials["solver_status"] == "success").any()
    # Per-trial timesteps present.
    assert len(loaded_dataset.timesteps) > 0


def test_surrogate_trains_on_synthetic_dataset(trained_surrogate):
    """Training returns a fully-formed bundle with finite curves."""
    pytest.importorskip("torch")
    assert trained_surrogate.model is not None
    assert trained_surrogate.norm_stats is not None
    # At least one epoch of training loss recorded.
    assert len(trained_surrogate.curves.train_loss) >= 1
    # All recorded losses are finite real numbers.
    for v in trained_surrogate.curves.train_loss:
        assert np.isfinite(v), f"non-finite training loss: {v!r}"


def test_full_pipeline_synthetic_target(
    loaded_dataset, trained_surrogate, fake_provenance
):
    """Synthetic target -> surrogate inversion produces a finite, in-bounds fit."""
    pytest.importorskip("torch")
    invert_mod = pytest.importorskip(
        "src.shared.python.motion_matching.surrogate.invert"
    )

    target_trial_id = int(loaded_dataset.trials.iloc[0]["trial_id"])
    target = _build_target_from_trial(loaded_dataset, target_trial_id, fake_provenance)

    opts = invert_mod.InvertOptions(
        n_starts=2,
        n_iters_per_start=20,
        lr=5.0e-2,
        seed=0,
        bound_strategy="clamp",
    )
    result = invert_mod.fit_swing_via_surrogate(
        target,
        trained_surrogate.model,
        opts,
        norm_stats=trained_surrogate.norm_stats,
    )

    # Shape contract: flat coefficient vector matching surrogate dim.
    expected_dim = trained_surrogate.config.coeff_dim
    assert result.theta_optimal.shape == (expected_dim,)
    assert np.all(np.isfinite(result.theta_optimal))
    # Loss is finite, non-negative, and below a deliberately loose smoke threshold.
    assert np.isfinite(result.final_loss)
    assert result.final_loss >= 0.0
    assert (
        result.final_loss < 100.0
    ), f"smoke threshold breached: final_loss={result.final_loss}"
    # History matches the configured restart/iteration grid.
    assert result.history["loss"].shape == (
        opts.n_starts,
        opts.n_iters_per_start,
    )
    # Surrogate prediction shape contract.
    assert result.surrogate_pred.butt.shape[-1] == 3
    assert result.surrogate_pred.clubhead.shape[-1] == 3
    assert result.surrogate_pred.club_quat.shape[-1] == 4
