"""Unit tests for :func:`train_inverse_cvae` (issue #033 / GH #4002)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from src.shared.python.motion_matching.dataset import load_sweep_dataset
from src.shared.python.motion_matching.dataset.synthetic import make_synthetic_sweep
from src.shared.python.motion_matching.inverse import (
    CVAEConfig,
    TrainInverseConfig,
    train_inverse_cvae,
)

# Tiny everywhere so the suite stays under the unit-test wallclock budget.
_N_TRIALS = 12
_N_JOINTS = 4
_N_TIMESTEPS = 16


def _make_dataset(tmp_path: Path) -> object:
    folder = make_synthetic_sweep(
        tmp_path / "sweep",
        n_trials=_N_TRIALS,
        n_joints=_N_JOINTS,
        n_timesteps=_N_TIMESTEPS,
        seed=0,
    )
    return load_sweep_dataset(folder, lazy=False)


def _make_cvae_config() -> CVAEConfig:
    return CVAEConfig(
        n_joints=_N_JOINTS,
        n_timesteps=_N_TIMESTEPS,
        n_kinematic_channels=12,
        latent_dim=4,
        encoder_layers=1,
        encoder_heads=2,
        encoder_dim=16,
        decoder_hidden=16,
        dropout=0.0,
    )


def _make_train_config(**overrides: object) -> TrainInverseConfig:
    base: dict[str, object] = {
        "n_epochs": 2,
        "batch_size": 4,
        "lr": 1e-3,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "kl_warmup_epochs": 1,
        "device": "cpu",
        "seed": 7,
    }
    base.update(overrides)
    return TrainInverseConfig(**base)  # type: ignore[arg-type]


@pytest.mark.unit
def test_training_loop_runs_without_error_on_synthetic_data(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    handle = train_inverse_cvae(dataset, _make_cvae_config(), _make_train_config())
    assert len(handle.curves.train_loss) == 2
    assert all(np.isfinite(handle.curves.train_loss))
    assert "final_train_loss" in handle.train_metrics
    assert handle.model.training is False  # eval mode on return


@pytest.mark.unit
def test_train_val_test_split_by_trial_id(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    handle = train_inverse_cvae(dataset, _make_cvae_config(), _make_train_config())
    train = set(handle.train_indices.tolist())
    val = set(handle.val_indices.tolist())
    test = set(handle.test_indices.tolist())
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(_N_TRIALS))


@pytest.mark.unit
def test_kl_does_not_collapse_after_full_warmup(tmp_path: Path) -> None:
    """Final-epoch beta is at the plateau; latent ``z`` must still vary
    across a held-out batch so the encoder is doing real work.
    """
    dataset = _make_dataset(tmp_path)
    handle = train_inverse_cvae(
        dataset,
        _make_cvae_config(),
        _make_train_config(n_epochs=3, kl_warmup_epochs=1),
    )
    # Final beta hits max_beta.
    assert handle.curves.beta[-1] == pytest.approx(handle.train_config.max_beta)
    # Sample latent on a held-out batch via the encoder.
    val_idx = handle.val_indices[: min(4, len(handle.val_indices))]
    if len(val_idx) == 0:
        pytest.skip("synthetic dataset too small for val split")
    # Recreate kinematics tensor for the val rows.
    from src.shared.python.motion_matching.inverse.train import _materialize_tensors

    _, kinematics, _ = _materialize_tensors(dataset, handle.config)
    kin = torch.from_numpy(kinematics[val_idx]).float()
    with torch.no_grad():
        enc = handle.model.encode(kin, sample=True)
    z_std = enc.z.std(dim=0)
    assert torch.all(z_std > 1e-4), f"latent collapsed; z_std={z_std}"


@pytest.mark.unit
def test_train_inverse_config_validation() -> None:
    with pytest.raises(ValueError):
        TrainInverseConfig(n_epochs=0)
    with pytest.raises(ValueError):
        TrainInverseConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainInverseConfig(lr=-1e-3)
    with pytest.raises(ValueError):
        TrainInverseConfig(lambda_work=-1.0)
    with pytest.raises(ValueError):
        TrainInverseConfig(max_beta=-0.5)
    with pytest.raises(ValueError):
        TrainInverseConfig(val_fraction=0.6, test_fraction=0.6)
    with pytest.raises(ValueError):
        TrainInverseConfig(duration_s=0.0)


@pytest.mark.unit
def test_checkpoint_written_when_dir_provided(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    ckpt_dir = tmp_path / "ckpts"
    handle = train_inverse_cvae(
        dataset,
        _make_cvae_config(),
        _make_train_config(checkpoint_dir=ckpt_dir, n_epochs=2),
    )
    assert handle.checkpoint_path is not None
    assert handle.checkpoint_path.exists()


@pytest.mark.unit
def test_n_joints_mismatch_rejected(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    bad = CVAEConfig(
        n_joints=_N_JOINTS + 1,
        n_timesteps=_N_TIMESTEPS,
        n_kinematic_channels=12,
        latent_dim=4,
        encoder_layers=1,
        encoder_heads=2,
        encoder_dim=16,
        decoder_hidden=16,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="n_joints"):
        train_inverse_cvae(dataset, bad, _make_train_config())


@pytest.mark.slow
@pytest.mark.unit
def test_held_out_round_trip_accepts_lossy_recovery(tmp_path: Path) -> None:
    """Cheap end-to-end smoke: predicted coefficients on the val split
    should at least be finite and have the expected shape; the work term
    should be non-negative. Tighter recovery thresholds are validated in
    #034's rejection-sampling tests once Simscape is hooked in.
    """
    dataset = _make_dataset(tmp_path)
    handle = train_inverse_cvae(
        dataset,
        _make_cvae_config(),
        _make_train_config(n_epochs=2),
    )
    from src.shared.python.motion_matching.inverse._work_estimator import (
        analytical_total_work,
    )
    from src.shared.python.motion_matching.inverse.train import _materialize_tensors

    _, kinematics, _ = _materialize_tensors(dataset, handle.config)
    val_idx = handle.val_indices
    if len(val_idx) == 0:
        pytest.skip("synthetic dataset too small for val split")
    kin = torch.from_numpy(kinematics[val_idx]).float()
    with torch.no_grad():
        coeffs_pred, _ = handle.model(kin, sample=False)
    arr = coeffs_pred.cpu().numpy()
    assert arr.shape == (len(val_idx), _N_JOINTS * 7)
    assert np.isfinite(arr).all()
    work = analytical_total_work(arr[0], duration_s=1.0)
    assert work >= 0.0
