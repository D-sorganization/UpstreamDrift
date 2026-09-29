"""Bug-2 fix tests for the inverse cVAE.

Covers:
* Coefficient standardisation by ``COEFFICIENT_LETTER_BOUNDS`` round-trips.
* Free-bits clamping prevents the per-dim KL from collapsing to zero.
* End-to-end training with the new defaults produces non-zero ``val_kl``,
  the canonical sanity check that posterior collapse has been fixed.
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
    EncoderOutput,
    build_coefficient_bound_vector,
    kl_divergence,
    kl_divergence_per_dim,
    train_inverse_cvae,
)
from src.shared.python.motion_matching.inverse.training import (  # noqa: E402
    _kl_for_loss,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_torch]


# ---------------------------------------------------------------------------
# Coefficient standardisation
# ---------------------------------------------------------------------------


def test_coefficient_bounds_roundtrip_to_unit_cube() -> None:
    """Dividing physical-unit coeffs by the bound vector lands them in [-1, 1].

    Multiplying back recovers the original values. This is the
    standardisation step the recon MSE now operates in.
    """
    bounds = build_coefficient_bound_vector(n_joints=27)
    assert bounds.shape == (DEFAULT_COEFFICIENT_DIM,)
    # Sample physical coeffs uniformly inside the per-letter bounds.
    rng = np.random.default_rng(0)
    physical = (
        torch.from_numpy(
            rng.uniform(-1.0, 1.0, size=(8, DEFAULT_COEFFICIENT_DIM))
        ).float()
        * bounds
    )
    standardised = physical / bounds
    assert torch.all(standardised.abs() <= 1.0 + 1e-5)
    recovered = standardised * bounds
    torch.testing.assert_close(recovered, physical, rtol=1e-6, atol=1e-3)


# ---------------------------------------------------------------------------
# Free-bits behaviour
# ---------------------------------------------------------------------------


def test_kl_per_dim_matches_summed_kl() -> None:
    """``kl_divergence_per_dim`` summed over latent dim equals ``kl_divergence``."""
    rng = np.random.default_rng(1)
    mu_q = torch.from_numpy(rng.normal(size=(4, 8))).float()
    logvar_q = torch.from_numpy(rng.normal(size=(4, 8)) * 0.1).float()
    mu_p = torch.zeros(4, 8)
    logvar_p = torch.zeros(4, 8)
    kl_sum = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
    kl_dim = kl_divergence_per_dim(mu_q, logvar_q, mu_p, logvar_p).sum(dim=-1)
    torch.testing.assert_close(kl_sum, kl_dim)


def test_freebits_floor_clamps_per_dim_kl() -> None:
    """Per-dim KLs below ``free_bits`` are floored before being summed."""
    # Posterior identical to prior -> per-dim KL ~ 0.
    mu = torch.zeros(2, 4)
    logvar = torch.zeros(2, 4)
    enc = EncoderOutput(
        mu_q=mu, logvar_q=logvar, mu_p=mu, logvar_p=logvar, z=mu.clone()
    )
    free_bits = 0.5
    kl_loss, kl_uncapped = _kl_for_loss(enc, free_bits=free_bits)
    # Uncapped KL is zero (identical Gaussians).
    assert kl_uncapped.item() == pytest.approx(0.0, abs=1e-6)
    # Floor in the loss term: 4 latent dims * free_bits = 2.0 nats.
    assert kl_loss.item() == pytest.approx(4 * free_bits, abs=1e-6)


def test_freebits_zero_matches_raw_kl() -> None:
    """``free_bits=0`` reproduces the unclamped KL (smoke fixture parity)."""
    rng = np.random.default_rng(2)
    mu_q = torch.from_numpy(rng.normal(size=(3, 5))).float()
    logvar_q = torch.from_numpy(rng.normal(size=(3, 5)) * 0.1).float()
    mu_p = torch.zeros(3, 5)
    logvar_p = torch.zeros(3, 5)
    enc = EncoderOutput(
        mu_q=mu_q, logvar_q=logvar_q, mu_p=mu_p, logvar_p=logvar_p, z=mu_q.clone()
    )
    kl_loss, kl_uncapped = _kl_for_loss(enc, free_bits=0.0)
    torch.testing.assert_close(kl_loss, kl_uncapped)


# ---------------------------------------------------------------------------
# End-to-end: val_kl > 0 (posterior collapse sanity check)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeCompactDataset:
    trials: pd.DataFrame
    timesteps: pd.DataFrame
    joint_names: tuple
    coefficient_letters: tuple = ("A", "B", "C", "D", "E", "F", "G")
    schema_version: str = "compact-1.0"


def _build_synthetic_dataset(
    n_trials: int = 12, n_timesteps: int = 16
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


@pytest.mark.slow
def test_training_keeps_nonzero_val_kl(tmp_path: Path) -> None:
    """Training with the new free-bits defaults must not collapse the posterior.

    Without the bug-2 fix (free-bits + standardised recon + reduced max β)
    ``val_kl`` plummets to ~0 within a few epochs as the encoder learns
    nothing useful. With the fix it stays >= ``free_bits * latent_dim`` by
    construction (free-bits clamps the per-dim KL inside the loss but the
    reported ``val_kl`` is the *uncapped* sum, so we assert it is at least
    a small positive value).
    """
    cfg = CVAEConfig(
        latent_dim=8,
        encoder_channels=(32, 64),
        decoder_hidden=64,
        dropout=0.0,
    )
    dataset = _build_synthetic_dataset()
    result = train_inverse_cvae(
        tmp_path,
        epochs=6,
        batch_size=4,
        lr=1e-3,
        seed=0,
        kl_anneal_epochs=3,
        max_beta=0.1,
        free_bits=0.5,
        device="cpu",
        output_root=tmp_path / "out",
        cvae_config=cfg,
        dataset_loader=lambda _p: dataset,
    )
    last = result.history[-1]
    # The headline sanity check: posterior didn't collapse to the prior.
    assert last.val_kl > 0.05, (
        f"val_kl looks collapsed: {last.val_kl} (history={result.history})"
    )


def test_recon_now_o1_under_standardisation(tmp_path: Path) -> None:
    """Recon MSE must be O(1), not O(1e5+), after the standardisation fix.

    Pre-fix the recon was ``MSE(physical 189-vec)`` with bounds up to 1000,
    yielding ~692k. Post-fix it's ``MSE(physical / bounds)`` so the worst
    possible MSE is ~1.0. Asserting an upper bound here would catch any
    regression that re-introduced unstandardised recon.
    """
    cfg = CVAEConfig(encoder_channels=(32,), decoder_hidden=64, dropout=0.0)
    dataset = _build_synthetic_dataset()
    result = train_inverse_cvae(
        tmp_path,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        seed=0,
        kl_anneal_epochs=2,
        max_beta=0.05,
        free_bits=0.0,
        device="cpu",
        output_root=tmp_path / "out",
        cvae_config=cfg,
        dataset_loader=lambda _p: dataset,
    )
    # Standardised recon is bounded above by ~4.0 (square of [-1,1] swing).
    assert result.history[0].train_recon < 5.0
    assert result.history[-1].train_recon < 5.0
