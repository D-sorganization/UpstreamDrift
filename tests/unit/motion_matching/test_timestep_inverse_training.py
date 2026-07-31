"""Training-loop tests for :func:`train_timestep_inverse`.

Uses an in-memory synthetic 8-trial fixture (no parquet IO) by injecting a
custom ``dataset_loader``. Verifies a few epochs reduce val loss by >=10%
and that the checkpoint round-trips via ``from_checkpoint``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.shared.python.motion_matching.inverse_timestep import (  # noqa: E402
    TimestepInverseConfig,
    TimestepInverseDynamics,
    train_timestep_inverse,
)
from src.shared.python.motion_matching.inverse_timestep.training import (  # noqa: E402
    _make_output_dir,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


# ---------------------------------------------------------------------------
# Synthetic 8-trial fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeCompactDataset:
    trials: pd.DataFrame
    timesteps: pd.DataFrame
    joint_names: tuple = ()
    coefficient_letters: tuple = ("A", "B", "C", "D", "E", "F", "G")
    schema_version: str = "compact-1.0"


def _build_synthetic_dataset(
    n_trials: int = 8, n_timesteps: int = 31
) -> _FakeCompactDataset:
    """Build a fixture with strong (q, qd, qdd) -> tau coupling.

    tau is a fixed linear function of the state plus a small noise term,
    so a 2-block MLP can fit it in a few epochs and val_loss should fall
    well below the standardised baseline of 1.0.
    """
    rng = np.random.default_rng(0)
    n_joints = 27
    state_dim = 3 * n_joints
    weight = rng.normal(0, 0.1, size=(state_dim, n_joints)).astype(np.float32)
    bias = rng.normal(0, 0.1, size=(n_joints,)).astype(np.float32)
    speed_uniform = rng.uniform(60.0, 140.0, size=(n_trials, n_timesteps)).astype(
        np.float32
    )

    trial_rows: list[dict[str, Any]] = []
    ts_rows: list[dict[str, Any]] = []
    for trial_id in range(n_trials):
        trial_rows.append({"trial_id": trial_id})
        ts = np.linspace(0.0, 0.3, n_timesteps)
        for j, t in enumerate(ts):
            q = rng.normal(0, 0.5, size=(n_joints,)).astype(np.float32)
            qd = rng.normal(0, 0.5, size=(n_joints,)).astype(np.float32)
            qdd = rng.normal(0, 0.5, size=(n_joints,)).astype(np.float32)
            state = np.concatenate([q, qd, qdd])
            tau = state @ weight + bias
            tau += rng.normal(0, 0.01, size=(n_joints,)).astype(np.float32)
            # Mark a couple of joint indices as unmapped (NaN) to exercise
            # the masked-loss path.
            tau[3] = np.nan
            tau[7] = np.nan
            ts_rows.append(
                {
                    "trial_id": trial_id,
                    "t": float(t),
                    "q": q.tolist(),
                    "qd": qd.tolist(),
                    "qdd": qdd.tolist(),
                    "tau": tau.tolist(),
                    "clubhead_speed_mph": float(speed_uniform[trial_id, j]),
                }
            )
    return _FakeCompactDataset(
        trials=pd.DataFrame(trial_rows),
        timesteps=pd.DataFrame(ts_rows),
    )


def _loader_factory():
    dataset = _build_synthetic_dataset()
    return lambda _path: dataset


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_timestep_inverse_training_three_epoch_run_reduces_val_loss(
    tmp_path: Path,
) -> None:
    cfg = TimestepInverseConfig(hidden=64, n_blocks=2, dropout=0.0)
    result = train_timestep_inverse(
        tmp_path,
        speed_lo_mph=50.0,
        speed_hi_mph=150.0,
        epochs=10,
        batch_size=32,
        lr=5e-3,
        seed=42,
        device="cpu",
        patience=20,
        output_root=tmp_path / "out",
        config=cfg,
        dataset_loader=_loader_factory(),
    )

    assert len(result.history) == 10
    first = result.history[0].val_loss
    last = result.history[-1].val_loss
    assert (
        last < first * 0.9
    ), f"val_loss did not reduce by >=10%: {first:.4g} -> {last:.4g}"
    assert result.checkpoint_path.exists()
    metrics_file = result.output_dir / "metrics.json"
    assert metrics_file.exists()
    assert result.parameter_count > 0
    assert result.n_train_trials >= 1
    assert result.n_val_trials >= 1
    assert result.n_train_timesteps > 0
    assert result.n_val_timesteps > 0
    assert all(m.val_tau_mae_nm >= 0 for m in result.history)


def test_timestep_inverse_training_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = TimestepInverseConfig(hidden=32, n_blocks=2, dropout=0.0)
    result = train_timestep_inverse(
        tmp_path,
        speed_lo_mph=50.0,
        speed_hi_mph=150.0,
        epochs=2,
        batch_size=32,
        lr=1e-3,
        seed=0,
        device="cpu",
        patience=5,
        output_root=tmp_path / "out",
        config=cfg,
        dataset_loader=_loader_factory(),
    )
    restored = TimestepInverseDynamics.from_checkpoint(result.checkpoint_path)
    assert isinstance(restored, TimestepInverseDynamics)
    assert restored.cfg == cfg

    restored.eval()
    state = torch.zeros(1, cfg.input_dim, dtype=torch.float32)
    with torch.no_grad():
        out_a = restored(state)
        out_b = restored(state)
    torch.testing.assert_close(out_a, out_b)


def test_make_output_dir_avoids_same_second_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.shared.python.motion_matching.inverse_timestep.training.time.strftime",
        lambda _fmt: "20260507_123456",
    )
    first = _make_output_dir(tmp_path)
    second = _make_output_dir(tmp_path)

    assert first.name == "20260507_123456"
    assert second.name == "20260507_123456_001"
    assert first.is_dir()
    assert second.is_dir()
    assert second != first


def test_timestep_inverse_training_invalid_epochs_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epochs"):
        train_timestep_inverse(
            tmp_path,
            epochs=0,
            device="cpu",
            output_root=tmp_path / "out",
            dataset_loader=_loader_factory(),
        )


def test_timestep_inverse_training_invalid_val_fraction_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        train_timestep_inverse(
            tmp_path,
            epochs=1,
            val_fraction=1.5,
            device="cpu",
            output_root=tmp_path / "out",
            dataset_loader=_loader_factory(),
        )


def test_invalid_speed_window_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="speed"):
        train_timestep_inverse(
            tmp_path,
            speed_lo_mph=200.0,
            speed_hi_mph=100.0,
            epochs=1,
            device="cpu",
            output_root=tmp_path / "out",
            dataset_loader=_loader_factory(),
        )
