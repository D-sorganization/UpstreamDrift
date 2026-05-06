"""Per-feature z-score normalization helpers for the SwingSurrogate.

Stats are fitted on the **training split only** per APPROACH.md and
DATA.md, then re-used on val/test/inference. Quaternions are not
normalized (they are already unit-norm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NormalizationStats:
    """Per-feature z-score statistics fitted on the training split.

    Attributes:
        coeffs_mean: ``(D,)`` mean of the input coefficient vector.
        coeffs_std: ``(D,)`` std of the input coefficient vector.
        butt_mean: ``(3,)`` mean of butt position over (trials x time).
        butt_std: ``(3,)`` std of butt position.
        clubhead_mean: ``(3,)`` mean of clubhead position.
        clubhead_std: ``(3,)`` std of clubhead position.
    """

    coeffs_mean: np.ndarray
    coeffs_std: np.ndarray
    butt_mean: np.ndarray
    butt_std: np.ndarray
    clubhead_mean: np.ndarray
    clubhead_std: np.ndarray


_EPS = 1.0e-8


def fit_stats(
    coeffs: np.ndarray,
    butt: np.ndarray,
    clubhead: np.ndarray,
) -> NormalizationStats:
    """Fit z-score statistics from training-split arrays.

    Args:
        coeffs: ``(N_trials, D)`` coefficient matrix.
        butt: ``(N_trials, T, 3)`` butt-position trajectories.
        clubhead: ``(N_trials, T, 3)`` clubhead-position trajectories.

    Returns:
        A :class:`NormalizationStats` whose ``*_std`` arrays are floored
        at ``1e-8`` to avoid division-by-zero on constant features.

    Raises:
        ValueError: If shapes are inconsistent or any array is empty.
    """
    if coeffs.ndim != 2:
        raise ValueError(f"coeffs must be 2-D, got shape {coeffs.shape}")
    if butt.ndim != 3 or butt.shape[-1] != 3:
        raise ValueError(f"butt must be (N,T,3), got {butt.shape}")
    if clubhead.shape != butt.shape:
        raise ValueError(f"clubhead shape {clubhead.shape} != butt shape {butt.shape}")
    if coeffs.shape[0] == 0:
        raise ValueError("cannot fit stats on empty arrays")
    return NormalizationStats(
        coeffs_mean=coeffs.mean(axis=0).astype(np.float32),
        coeffs_std=np.maximum(coeffs.std(axis=0), _EPS).astype(np.float32),
        butt_mean=butt.mean(axis=(0, 1)).astype(np.float32),
        butt_std=np.maximum(butt.std(axis=(0, 1)), _EPS).astype(np.float32),
        clubhead_mean=clubhead.mean(axis=(0, 1)).astype(np.float32),
        clubhead_std=np.maximum(clubhead.std(axis=(0, 1)), _EPS).astype(np.float32),
    )


def zscore_coeffs(coeffs: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    """Apply ``(x - mean) / std`` to a batch of coefficient vectors."""
    mean = torch.as_tensor(stats.coeffs_mean, dtype=coeffs.dtype, device=coeffs.device)
    std = torch.as_tensor(stats.coeffs_std, dtype=coeffs.dtype, device=coeffs.device)
    return (coeffs - mean) / std


def zscore_positions(
    positions: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """Z-score a batch of position tensors of shape ``(B, T, 3)``."""
    m = torch.as_tensor(mean, dtype=positions.dtype, device=positions.device)
    s = torch.as_tensor(std, dtype=positions.dtype, device=positions.device)
    return (positions - m) / s


def denormalize_positions(
    z: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """Inverse of :func:`zscore_positions`."""
    m = torch.as_tensor(mean, dtype=z.dtype, device=z.device)
    s = torch.as_tensor(std, dtype=z.dtype, device=z.device)
    return z * s + m
