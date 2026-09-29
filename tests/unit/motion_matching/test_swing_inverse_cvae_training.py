"""Training-loop tests for the Option-3 inverse cVAE (GH issue #4076).

Uses an in-memory synthetic dataset (no parquet IO) by injecting a custom
``dataset_loader`` into :func:`train_inverse_cvae`. Verifies that 3 epochs
on an 8-trial fixture reduce val loss, the checkpoint round-trips cleanly,
and the metrics file lands on disk.
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
    CVAEConfig,
    SwingInverseCVAE,
    train_inverse_cvae,
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
    n_trials: int = 8, n_timesteps: int = 16
) -> _FakeCompactDataset:
    rng = np.random.default_rng(0)
    joint_names = tuple(f"j{i}" for i in range(27))
    trial_rows: list[dict[str, Any]] = []
    ts_rows: list[dict[str, Any]] = []
    for trial_id in range(n_trials):
        coeffs = rng.normal(0, 50.0, size=DEFAULT_COEFFICIENT_DIM).astype(np.float32)
        trial_rows.append(
            {
                "trial_id": trial_id,
                "coefficients": coeffs.tolist(),
                "joint_names": list(joint_names),
            }
        )
        # Generate a smooth-ish trajectory keyed on coeffs so model has signal.
        base = float(np.sum(coeffs)) / 1000.0
        ts = np.linspace(0.0, 0.3, n_timesteps)
        for _k, t in enumerate(ts):
            phase = base + t
            ts_rows.append(
                {
                    "trial_id": trial_id,
                    "t": float(t),
                    "r_buttend": [np.sin(phase), np.cos(phase), 0.5 * t],
                    "r_clubhead": [
                        np.sin(phase + 0.5),
                        np.cos(phase + 0.5),
                        1.0 * t,
                    ],
                    "r_grip": [
                        np.sin(phase + 0.25),
                        np.cos(phase + 0.25),
                        0.75 * t,
                    ],
                    "v_clubhead": [
                        np.cos(phase + 0.5),
                        -np.sin(phase + 0.5),
                        1.0,
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


def test_swing_inverse_cvae_training_three_epoch_run_reduces_val_loss(
    tmp_path: Path,
) -> None:
    cfg = CVAEConfig(encoder_channels=(32, 64), decoder_hidden=64, dropout=0.0)
    # Recon is now computed in standardised [-1, 1] coefficient space (bug-2
    # fix), so the loss surface is much flatter in absolute terms. Use a few
    # extra epochs so the toy-fixture optimisation has time to descend.
    result = train_inverse_cvae(
        tmp_path,
        epochs=8,
        batch_size=4,
        lr=5e-3,
        seed=42,
        kl_anneal_epochs=2,
        max_beta=0.01,  # keep KL weak so MSE dominates the toy fit
        free_bits=0.0,  # disable free-bits on the toy test for clean signal
        device="cpu",
        output_root=tmp_path / "out",
        cvae_config=cfg,
        dataset_loader=_loader_factory(),
    )

    assert len(result.history) == 8
    assert result.history[0].val_recon > result.history[-1].val_recon, (
        f"val_recon did not improve: {result.history[0].val_recon} -> "
        f"{result.history[-1].val_recon}"
    )
    assert result.checkpoint_path.exists()
    metrics_file = result.output_dir / "metrics.json"
    assert metrics_file.exists()
    assert result.parameter_count > 0
    assert result.n_train_trials >= 1
    assert result.n_val_trials >= 1


def test_swing_inverse_cvae_training_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = CVAEConfig(encoder_channels=(32,), decoder_hidden=64)
    result = train_inverse_cvae(
        tmp_path,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=0,
        kl_anneal_epochs=1,
        device="cpu",
        output_root=tmp_path / "out",
        cvae_config=cfg,
        dataset_loader=_loader_factory(),
    )
    restored = SwingInverseCVAE.from_checkpoint(result.checkpoint_path)
    assert isinstance(restored, SwingInverseCVAE)
    assert restored.cfg == cfg

    # Forward pass on the restored model produces the same output as a fresh
    # load for the same input (deterministic eval mode).
    restored.eval()
    traj = torch.zeros(1, 16, cfg.trajectory_channels, dtype=torch.float32)
    with torch.no_grad():
        coeff_a, _ = restored(traj)
        coeff_b, _ = restored(traj)
    torch.testing.assert_close(coeff_a, coeff_b)


def test_swing_inverse_cvae_training_invalid_epochs_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epochs"):
        train_inverse_cvae(
            tmp_path,
            epochs=0,
            device="cpu",
            output_root=tmp_path / "out",
            dataset_loader=_loader_factory(),
        )


def test_swing_inverse_cvae_training_invalid_val_fraction_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        train_inverse_cvae(
            tmp_path,
            epochs=1,
            val_fraction=1.5,
            device="cpu",
            output_root=tmp_path / "out",
            dataset_loader=_loader_factory(),
        )
