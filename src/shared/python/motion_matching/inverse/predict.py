"""Inference surface for :class:`SwingInverseCVAE` (Option 3, GH issue #4076).

Exposes :func:`predict_coefficients`, which takes a measured/target hand-path
trajectory and returns N plausible 189-dim coefficient samples in physical
units, plus the prior mean (the "best single guess" if the caller does not
want to enumerate a posterior).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from .cvae import (
    DEFAULT_TRAJECTORY_CHANNELS,
    SwingInverseCVAE,
    to_numpy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoefficientPredictions:
    """N coefficient samples + the prior mean for one target trajectory.

    Attributes
    ----------
    samples
        ``(n_samples, 189)`` float32 array in physical units.
    mean
        ``(189,)`` float32 array — the prior mean decoded deterministically.
        Use this when you want a single best guess; sample-level variation
        comes from ``samples``.
    latent_mu
        ``(latent_dim,)`` prior mean. Useful for diagnostics / latent UMAP.
    latent_logvar
        ``(latent_dim,)`` prior log-variance.
    """

    samples: NDArray[np.float32]
    mean: NDArray[np.float32]
    latent_mu: NDArray[np.float32]
    latent_logvar: NDArray[np.float32]

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def coefficient_dim(self) -> int:
        return int(self.samples.shape[-1])


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_inverse_cvae(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> SwingInverseCVAE:
    """Convenience wrapper around :meth:`SwingInverseCVAE.from_checkpoint`.

    Equivalent to calling the classmethod directly; provided so callers
    can write ``from ...inverse import load_inverse_cvae`` without having
    to import the model class.
    """
    return SwingInverseCVAE.from_checkpoint(checkpoint_path, map_location=map_location)


def predict_coefficients(
    model: SwingInverseCVAE,
    target_trajectory: ArrayLike,
    *,
    n_samples: int = 8,
    seed: int | None = None,
) -> CoefficientPredictions:
    """Sample N plausible coefficient vectors from ``p(theta | trajectory)``.

    Parameters
    ----------
    model
        Trained :class:`SwingInverseCVAE`.
    target_trajectory
        Either a ``(T, 12)`` array for a single target or a ``(1, T, 12)``
        batched array. Numpy or torch input is accepted; conversion is
        explicit (no silent dtype changes).
    n_samples
        Number of samples to draw. Must be >= 1.
    seed
        Optional torch seed for reproducible sampling.

    Returns
    -------
    CoefficientPredictions
        Samples, prior-mean point estimate and the latent moments.

    Raises
    ------
    TypeError, ValueError
        For shape/dtype contract violations.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    traj = _to_traj_tensor(target_trajectory, model)

    if seed is not None:
        torch.manual_seed(int(seed))

    model.eval()
    with torch.no_grad():
        samples_t = model.sample(traj, n_samples=n_samples)
        mean_t = model.sample(traj, n_samples=1, deterministic_mean=True)
        # Prior moments for downstream diagnostics.
        ctx = model._encode_context(traj)  # noqa: SLF001 (intentional)
        mu_p, logvar_p = model._prior(ctx)  # noqa: SLF001

    return CoefficientPredictions(
        samples=to_numpy(samples_t[0]).astype(np.float32),
        mean=to_numpy(mean_t[0, 0]).astype(np.float32),
        latent_mu=to_numpy(mu_p[0]).astype(np.float32),
        latent_logvar=to_numpy(logvar_p[0]).astype(np.float32),
    )


def predict_coefficients_from_checkpoint(
    checkpoint_path: str | Path,
    target_trajectory: ArrayLike,
    *,
    n_samples: int = 8,
    seed: int | None = None,
    map_location: str | torch.device | None = None,
) -> CoefficientPredictions:
    """One-call convenience: load checkpoint + sample.

    Used by the MATLAB shim through ``pyrunfile``.
    """
    model = load_inverse_cvae(checkpoint_path, map_location=map_location)
    return predict_coefficients(
        model, target_trajectory, n_samples=n_samples, seed=seed
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_traj_tensor(
    target_trajectory: ArrayLike, model: SwingInverseCVAE
) -> torch.Tensor:
    """Coerce ``target_trajectory`` to a (1, T, 12) float32 tensor on model device.

    Accepts numpy arrays or torch tensors, 2-D ``(T, C)`` or 3-D ``(B, T, C)``.
    For batched input only the first row is used (predictions are
    single-target; batching is the caller's job).
    """
    if isinstance(target_trajectory, torch.Tensor):
        traj = target_trajectory.detach()
    else:
        traj = torch.from_numpy(np.asarray(target_trajectory))

    if traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if traj.dim() != 3:
        raise ValueError(
            f"target_trajectory must be (T, C) or (B, T, C); got {tuple(traj.shape)}"
        )
    if traj.shape[-1] != model.cfg.trajectory_channels:
        raise ValueError(
            f"trajectory channels = {traj.shape[-1]}, expected "
            f"{model.cfg.trajectory_channels} (DEFAULT={DEFAULT_TRAJECTORY_CHANNELS})"
        )
    if traj.shape[0] > 1:
        traj = traj[:1]
    traj = traj.to(dtype=torch.float32)
    # Move to the same device as the first model parameter.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return traj.to(device)
