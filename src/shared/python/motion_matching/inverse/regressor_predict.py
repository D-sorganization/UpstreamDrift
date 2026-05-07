"""Inference surface for :class:`InverseRegressor`.

Deterministic single-shot prediction: trajectory in -> 189-dim coefficient
vector out, in physical units, with each entry already clamped to its
per-letter bound.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from .regressor import (
    DEFAULT_TRAJECTORY_CHANNELS,
    InverseRegressor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_inverse_regressor(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> InverseRegressor:
    """Convenience wrapper around :meth:`InverseRegressor.from_checkpoint`."""
    return InverseRegressor.from_checkpoint(checkpoint_path, map_location=map_location)


def predict_coefficients_regressor(
    model: InverseRegressor,
    target_trajectory: ArrayLike,
) -> NDArray[np.float32]:
    """Predict 189-dim coefficient vectors in physical units.

    Parameters
    ----------
    model
        Trained :class:`InverseRegressor`.
    target_trajectory
        Either a ``(T, 12)`` array for a single target or a ``(B, T, 12)``
        batched array. Numpy or torch input accepted.

    Returns
    -------
    np.ndarray
        ``(B, coefficient_dim)`` float32 array in physical units. Even for
        single-target ``(T, C)`` input, the output keeps a leading batch
        dimension of 1 so callers don't have to special-case the shape.

    Raises
    ------
    TypeError, ValueError
        For shape/dtype contract violations.
    """
    traj = _to_traj_tensor(target_trajectory, model)
    model.eval()
    with torch.no_grad():
        pred = model(traj)
    return pred.detach().cpu().numpy().astype(np.float32)


def predict_coefficients_regressor_from_checkpoint(
    checkpoint_path: str | Path,
    target_trajectory: ArrayLike,
    *,
    map_location: str | torch.device | None = None,
) -> NDArray[np.float32]:
    """One-call convenience: load checkpoint + predict."""
    model = load_inverse_regressor(checkpoint_path, map_location=map_location)
    return predict_coefficients_regressor(model, target_trajectory)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_traj_tensor(
    target_trajectory: ArrayLike, model: InverseRegressor
) -> torch.Tensor:
    """Coerce input to a 3-D ``(B, T, C)`` float32 tensor on the model device.

    Accepts numpy arrays or torch tensors; 2-D ``(T, C)`` is unsqueezed to
    ``(1, T, C)``. Unlike the cVAE's helper, *all* batch rows are kept —
    the regressor is deterministic and batch-friendly.
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
    traj = traj.to(dtype=torch.float32)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return traj.to(device)
