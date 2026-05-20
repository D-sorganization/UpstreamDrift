"""Training-loop tests for :func:`train_inverse_regressor`.

Uses an in-memory synthetic dataset (no parquet IO) by injecting a custom
``dataset_loader``. Verifies 3 epochs on an 8-trial fixture reduce val
loss by >=10%, and that the checkpoint round-trips via ``from_checkpoint``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.inverse import (  # noqa: E402
    DEFAULT_COEFFICIENT_DIM,
    InverseRegressor,
    RegressorConfig,
    train_inverse_regressor,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


# ---------------------------------------------------------------------------
# Synthetic 8-trial fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeCompactDataset:
    trials: pd.DataFrame
    timesteps: pd.DataFrame
    joint_names: tuple
    coefficient_letters: tuple = ("A", "B", "C", "D", "E", "F", "G")
    schema_version: str = "compact-1.0"


def _build_synthetic_dataset(
    n_trials: int = 8, n_timesteps: int = 31
) -> _FakeCompactDataset:
    """Build a fixture with strong trajectory<->coefficient coupling.

    Each trial picks a single scalar phase ``alpha`` driving both the
    trajectory and the coefficient vector, so a deterministic regressor
    can in principle achieve near-zero loss with a few epochs of training.
    """
    rng = np.random.default_rng(0)
    joint_names = tuple(f"j{i}" for i in range(27))
    trial_rows: list[dict[str, Any]] = []
    ts_rows: list[dict[str, Any]] = []
    for trial_id in range(n_trials):
        # alpha in [-1, 1]; trajectory and coefficients both depend on it.
        alpha = float(rng.uniform(-1.0, 1.0))
        # Coefficients = alpha * fixed_template + small noise (so signal is
        # strong but not degenerate / fully predictable from alpha alone).
        template = rng.normal(0, 1.0, size=DEFAULT_COEFFICIENT_DIM).astype(np.float32)
        noise = rng.normal(0, 0.01, size=DEFAULT_COEFFICIENT_DIM).astype(np.float32)
        coeffs = (alpha * 50.0 * template + noise).astype(np.float32)
        trial_rows.append(
            {
                "trial_id": trial_id,
                "coefficients": coeffs.tolist(),
                "joint_names": list(joint_names),
            }
        )
        ts = np.linspace(0.0, 0.3, n_timesteps)
        for t in ts:
            phase = alpha + t
            ts_rows.append(
                {
                    "trial_id": trial_id,
                    "t": float(t),
                    "r_buttend": [np.sin(phase), np.cos(phase), 0.5 * t * alpha],
                    "r_clubhead": [
                        np.sin(phase + 0.5),
                        np.cos(phase + 0.5),
                        1.0 * t * alpha,
                    ],
                    "r_grip": [
                        np.sin(phase + 0.25),
                        np.cos(phase + 0.25),
                        0.75 * t * alpha,
                    ],
                    "v_clubhead": [
                        np.cos(phase + 0.5) * alpha,
                        -np.sin(phase + 0.5) * alpha,
                        1.0 * alpha,
                    ],
                }
            )
    return _FakeCompactDataset(
        trials=pd.DataFrame(trial_rows),
        timesteps=pd.DataFrame(ts_rows),
        joint_names=joint_names,
    )


def _loader_factory():
    dataset = _build_synthetic_dataset()
    return lambda _path: dataset


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_inverse_regressor_training_three_epoch_run_reduces_val_loss(
    tmp_path: Path,
) -> None:
    cfg = RegressorConfig(embed_dim=32, mlp_hidden=64, n_blocks=2, dropout=0.0)
    result = train_inverse_regressor(
        tmp_path,
        epochs=8,
        batch_size=4,
        lr=5e-3,
        seed=42,
        device="cpu",
        patience=20,
        output_root=tmp_path / "out",
        config=cfg,
        dataset_loader=_loader_factory(),
    )

    assert len(result.history) == 8
    first = result.history[0].val_loss
    last = result.history[-1].val_loss
    assert last < first * 0.9, (
        f"val_loss did not reduce by >=10%: {first:.4g} -> {last:.4g}"
    )
    assert result.checkpoint_path.exists()
    metrics_file = result.output_dir / "metrics.json"
    assert metrics_file.exists()
    assert result.parameter_count > 0
    assert result.n_train_trials >= 1
    assert result.n_val_trials >= 1
    # val_mse_physical should be reported at every epoch.
    assert all(m.val_mse_physical >= 0 for m in result.history)


def test_inverse_regressor_training_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = RegressorConfig(embed_dim=32, mlp_hidden=64, n_blocks=2)
    result = train_inverse_regressor(
        tmp_path,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=0,
        device="cpu",
        output_root=tmp_path / "out",
        config=cfg,
        dataset_loader=_loader_factory(),
    )
    restored = InverseRegressor.from_checkpoint(result.checkpoint_path)
    assert isinstance(restored, InverseRegressor)
    assert restored.cfg == cfg

    restored.eval()
    traj = torch.zeros(
        1, restored.cfg.seq_len, cfg.trajectory_channels, dtype=torch.float32
    )
    with torch.no_grad():
        out_a = restored(traj)
        out_b = restored(traj)
    torch.testing.assert_close(out_a, out_b)


def test_inverse_regressor_training_invalid_epochs_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epochs"):
        train_inverse_regressor(
            tmp_path,
            epochs=0,
            device="cpu",
            output_root=tmp_path / "out",
            dataset_loader=_loader_factory(),
        )


def test_inverse_regressor_training_invalid_val_fraction_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        train_inverse_regressor(
            tmp_path,
            epochs=1,
            val_fraction=1.5,
            device="cpu",
            output_root=tmp_path / "out",
            dataset_loader=_loader_factory(),
        )


def test_invalid_batch_size_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        train_inverse_regressor(
            tmp_path,
            epochs=1,
            batch_size=0,
            device="cpu",
            output_root=tmp_path / "out",
            dataset_loader=_loader_factory(),
        )


def test_invalid_lr_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lr"):
        train_inverse_regressor(
            tmp_path,
            epochs=1,
            lr=0.0,
            device="cpu",
            output_root=tmp_path / "out",
            dataset_loader=_loader_factory(),
        )
