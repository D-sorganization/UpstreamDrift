"""Training-loop tests for the compact-schema swing surrogate (#4075)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pyarrow = pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

from src.shared.python.motion_matching.surrogate.compact import (  # noqa: E402
    SurrogateConfig,
    SwingSurrogate,
    train_surrogate,
)

# --------------------------------------------------------------------------- #
# Synthetic 8-trial fixture                                                   #
# --------------------------------------------------------------------------- #


def _build_synthetic_compact_dataset(out_dir: Path, n_trials: int = 8) -> Path:
    """Write a tiny but contract-compliant compact parquet pair to ``out_dir``.

    The targets are a smooth deterministic function of the coefficients so
    that even a tiny model can learn to reduce val loss in 3 epochs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_joints = 27
    coeffs_per_joint = 7
    coeff_dim = n_joints * coeffs_per_joint
    seq_len = 31
    bounds = np.tile([1000, 1000, 500, 500, 100, 100, 25.0], n_joints).astype(
        np.float64
    )
    trials_rows: list[dict[str, object]] = []
    timesteps_rows: list[dict[str, object]] = []
    for trial_id in range(n_trials):
        coeffs = rng.uniform(-1.0, 1.0, size=coeff_dim) * bounds
        # Deterministic smooth targets that depend on the coeffs.
        weights = np.linspace(-0.1, 0.1, coeff_dim)
        scalar = float(np.dot(coeffs / bounds, weights))
        for t_idx in range(seq_len):
            t = t_idx * 0.01
            r_grip = np.array(
                [0.5 + scalar * 0.05 + 0.1 * t, 0.0, 1.0 + 0.05 * np.sin(t)],
                dtype=np.float64,
            )
            r_clubhead = r_grip + np.array(
                [0.0, 0.0, -1.1 + 0.01 * scalar], dtype=np.float64
            )
            v_clubhead = np.array(
                [10.0 + scalar, 0.0, 1.0 * np.cos(t)], dtype=np.float64
            )
            chs = float(np.linalg.norm(v_clubhead) * 2.236936)
            timesteps_rows.append(
                {
                    "trial_id": np.uint32(trial_id),
                    "t": t,
                    "q": [0.0] * 27,
                    "qd": [0.0] * 27,
                    "qdd": [0.0] * 27,
                    "tau": [0.0] * 27,
                    "r_clubhead": r_clubhead.tolist(),
                    "v_clubhead": v_clubhead.tolist(),
                    "r_buttend": r_grip.tolist(),
                    "r_lhand": r_grip.tolist(),
                    "r_rhand": r_grip.tolist(),
                    "r_grip": r_grip.tolist(),
                    "clubhead_speed_mph": chs,
                }
            )
        trials_rows.append(
            {
                "trial_id": np.uint32(trial_id),
                "coefficients": coeffs.tolist(),
                "joint_names": [f"j{i}" for i in range(n_joints)],
                "coefficient_letters": ["A", "B", "C", "D", "E", "F", "G"],
                "simulation_time_s": 0.30,
                "sample_rate_hz": 100.0,
                "clubhead_speed_max_mph": 100.0,
                "total_work_J": 250.0,
                "solver_status": "success",
            }
        )
    pd.DataFrame(trials_rows).to_parquet(out_dir / "trials.parquet")
    pd.DataFrame(timesteps_rows).to_parquet(out_dir / "timesteps.parquet")
    return out_dir


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Build the 8-trial synthetic compact dataset once per test."""
    return _build_synthetic_compact_dataset(tmp_path / "compact", n_trials=8)


@pytest.fixture
def small_config() -> SurrogateConfig:
    """A trimmed config so 3 epochs run in well under 30 s on CPU."""
    return SurrogateConfig(
        n_joints=27,
        coeffs_per_joint=7,
        seq_len=31,
        hidden_dim=32,
        n_residual_blocks=2,
    )


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.requires_torch
@pytest.mark.slow
def test_train_surrogate_reduces_val_loss(
    synthetic_dataset: Path, small_config: SurrogateConfig, tmp_path: Path
) -> None:
    """3 epochs on the synthetic fixture must reduce val loss vs epoch 1."""
    out_dir = tmp_path / "training_out"
    result = train_surrogate(
        synthetic_dataset,
        epochs=3,
        batch_size=4,
        lr=1e-3,
        seed=0,
        config=small_config,
        output_dir=out_dir,
        early_stopping_patience=10,
    )
    assert result.output_dir == out_dir
    val_losses = result.history["val_loss"]
    assert len(val_losses) == 3, f"expected 3 epochs, got {len(val_losses)}"
    assert val_losses[-1] < val_losses[0], f"val loss did not decrease: {val_losses}"
    assert result.param_count > 0


@pytest.mark.unit
@pytest.mark.requires_torch
@pytest.mark.slow
def test_train_surrogate_writes_checkpoints_and_metrics(
    synthetic_dataset: Path, small_config: SurrogateConfig, tmp_path: Path
) -> None:
    """Each epoch writes a checkpoint; ``metrics.json`` is valid JSON."""
    out_dir = tmp_path / "training_out_ckpt"
    result = train_surrogate(
        synthetic_dataset,
        epochs=3,
        batch_size=4,
        lr=1e-3,
        seed=0,
        config=small_config,
        output_dir=out_dir,
    )
    epoch_files = sorted(out_dir.glob("checkpoint_epoch_*.pt"))
    assert len(epoch_files) == 3
    assert (out_dir / "checkpoint_best.pt").exists()
    assert result.last_checkpoint.exists()
    assert result.best_checkpoint.exists()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    for key in (
        "epoch",
        "train_loss",
        "val_loss",
        "val_grip_rmse_mm",
        "val_clubhead_speed_mae_mph",
    ):
        assert key in metrics
        assert len(metrics[key]) == 3


@pytest.mark.unit
@pytest.mark.requires_torch
@pytest.mark.slow
def test_train_surrogate_resumes_from_checkpoint(
    synthetic_dataset: Path, small_config: SurrogateConfig, tmp_path: Path
) -> None:
    """A second ``train_surrogate`` call resumes from the prior checkpoint."""
    first_dir = tmp_path / "first"
    train_surrogate(
        synthetic_dataset,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        seed=0,
        config=small_config,
        output_dir=first_dir,
    )
    last_ckpt = first_dir / "checkpoint_epoch_002.pt"
    assert last_ckpt.exists()
    second_dir = tmp_path / "second"
    train_surrogate(
        synthetic_dataset,
        epochs=3,  # 1 more on top of the 2-epoch checkpoint
        batch_size=4,
        lr=1e-3,
        seed=0,
        config=small_config,
        output_dir=second_dir,
        resume_from=last_ckpt,
    )
    # Resuming from epoch 2 with epochs=3 yields one more epoch.
    epoch_files = sorted(second_dir.glob("checkpoint_epoch_*.pt"))
    assert len(epoch_files) == 1


@pytest.mark.unit
@pytest.mark.requires_torch
def test_train_surrogate_validates_args(tmp_path: Path) -> None:
    """Bad scalar hyperparams must raise ``ValueError`` before training."""
    with pytest.raises(ValueError, match="epochs"):
        train_surrogate(tmp_path, epochs=0, batch_size=4, lr=1e-3)
    with pytest.raises(ValueError, match="batch_size"):
        train_surrogate(tmp_path, epochs=1, batch_size=0, lr=1e-3)
    with pytest.raises(ValueError, match="lr"):
        train_surrogate(tmp_path, epochs=1, batch_size=4, lr=0.0)
    with pytest.raises(ValueError, match="val_fraction"):
        train_surrogate(tmp_path, epochs=1, batch_size=4, lr=1e-3, val_fraction=1.0)


@pytest.mark.unit
@pytest.mark.requires_torch
def test_train_surrogate_missing_dataset_raises(tmp_path: Path) -> None:
    """Pointing at a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        train_surrogate(
            tmp_path / "does-not-exist",
            epochs=1,
            batch_size=4,
            lr=1e-3,
        )


@pytest.mark.unit
@pytest.mark.requires_torch
def test_swing_surrogate_modules_register_parameters() -> None:
    """A freshly-built model has named parameters (sanity check for nn graph)."""
    model = SwingSurrogate()
    names = [n for n, _ in model.named_parameters()]
    assert any("input_proj" in n for n in names)
    assert any("blocks" in n for n in names)
    assert any("decoder" in n for n in names)


@pytest.mark.unit
@pytest.mark.requires_torch
@pytest.mark.slow
def test_train_surrogate_best_checkpoint_tracks_val_loss(
    synthetic_dataset: Path, small_config: SurrogateConfig, tmp_path: Path
) -> None:
    """Best checkpoint is the lowest-val-loss epoch (not lowest grip-RMSE).

    Regression lock for the audit-pass fix. Earlier revisions watched
    ``val_grip_rmse_mm`` only and could pick a "best" epoch that was
    pre-clubhead-speed convergence — saving the surrogate before the
    multi-channel objective had finished decreasing.
    """
    out_dir = tmp_path / "best_tracks_val_loss"
    result = train_surrogate(
        synthetic_dataset,
        epochs=3,
        batch_size=4,
        lr=1e-3,
        seed=0,
        config=small_config,
        output_dir=out_dir,
        early_stopping_patience=10,
    )
    val_losses = result.history["val_loss"]
    best_loss = min(val_losses)
    assert result.best_val_loss == pytest.approx(best_loss, rel=1e-6)
    # Best epoch index is 1-based and should match the argmin of val_loss.
    assert result.best_epoch == int(np.argmin(val_losses)) + 1
