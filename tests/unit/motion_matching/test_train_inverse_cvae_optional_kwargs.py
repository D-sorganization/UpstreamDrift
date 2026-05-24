"""Regression tests for the optional kwargs added to ``train_inverse_cvae``.

Issue #6014 added two optional callbacks — ``on_epoch_end`` and
``should_stop`` — to support the PyTorch CVAE training-controller
adapter. These tests pin the contract so future refactors don't quietly
drop the hooks.

Guarded by ``pytest.importorskip("torch")`` because the training loop
needs torch + numpy + pandas to run end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")  # noqa: F841
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.shared.python.motion_matching.inverse import (  # noqa: E402
    DEFAULT_COEFFICIENT_DIM,
    CVAEConfig,
    train_inverse_cvae,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


@dataclass(frozen=True)
class _FakeCompactDataset:
    trials: pd.DataFrame
    timesteps: pd.DataFrame
    joint_names: tuple
    coefficient_letters: tuple = ("A", "B", "C", "D", "E", "F", "G")
    schema_version: str = "compact-1.0"


def _build_synthetic_dataset(
    n_trials: int = 6, n_timesteps: int = 12
) -> _FakeCompactDataset:
    rng = np.random.default_rng(7)
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
        base = float(np.sum(coeffs)) / 1000.0
        ts = np.linspace(0.0, 0.3, n_timesteps)
        for t in ts:
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


def _loader():
    dataset = _build_synthetic_dataset()
    return lambda _path: dataset


def test_on_epoch_end_fires_once_per_epoch(tmp_path: Path) -> None:
    seen: list[int] = []

    def _cb(metrics) -> None:
        seen.append(int(metrics.epoch))

    result = train_inverse_cvae(
        tmp_path,
        epochs=3,
        batch_size=2,
        lr=1e-3,
        seed=0,
        kl_anneal_epochs=1,
        device="cpu",
        output_root=tmp_path / "out",
        cvae_config=CVAEConfig(encoder_channels=(16,), decoder_hidden=32),
        dataset_loader=_loader(),
        on_epoch_end=_cb,
    )
    assert seen == [0, 1, 2]
    assert len(result.history) == 3


def test_should_stop_terminates_loop_early(tmp_path: Path) -> None:
    seen_epochs: list[int] = []

    def _on_epoch(metrics) -> None:
        seen_epochs.append(int(metrics.epoch))

    flag = {"stop": False}

    def _stop() -> bool:
        return flag["stop"]

    # Trip the stop signal after the first epoch.
    def _on_epoch_with_stop(metrics) -> None:
        _on_epoch(metrics)
        if int(metrics.epoch) == 0:
            flag["stop"] = True

    result = train_inverse_cvae(
        tmp_path,
        epochs=5,
        batch_size=2,
        lr=1e-3,
        seed=0,
        kl_anneal_epochs=1,
        device="cpu",
        output_root=tmp_path / "out",
        cvae_config=CVAEConfig(encoder_channels=(16,), decoder_hidden=32),
        dataset_loader=_loader(),
        on_epoch_end=_on_epoch_with_stop,
        should_stop=_stop,
    )
    # The loop ran a single epoch then exited cooperatively.
    assert seen_epochs == [0]
    assert len(result.history) == 1
    assert result.checkpoint_path.exists()


def test_default_kwargs_are_noops(tmp_path: Path) -> None:
    # Sanity: omitting both kwargs preserves the original behaviour.
    result = train_inverse_cvae(
        tmp_path,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        seed=0,
        kl_anneal_epochs=1,
        device="cpu",
        output_root=tmp_path / "out",
        cvae_config=CVAEConfig(encoder_channels=(16,), decoder_hidden=32),
        dataset_loader=_loader(),
    )
    assert len(result.history) == 1
