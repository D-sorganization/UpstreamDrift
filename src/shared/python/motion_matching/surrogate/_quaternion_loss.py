"""Quaternion-aware orientation loss for the SwingSurrogate.

Implements ``L_quat = mean(1 - <q_pred, q_true>^2)`` per APPROACH.md.

The squared-inner-product form is sign-invariant (``q`` and ``-q``
represent the same rotation in the unit-quaternion double cover) and
smooth everywhere on the unit sphere, which is what the inversion
gradient path in #029 needs.
"""

from __future__ import annotations

import torch


def quaternion_loss(q_pred: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    """Compute the sign-invariant quaternion supervision loss.

    Args:
        q_pred: Predicted unit quaternions, shape ``(..., 4)``.
        q_true: Target unit quaternions, shape ``(..., 4)``.

    Returns:
        A scalar tensor equal to ``mean(1 - <q_pred, q_true>^2)``. The
        loss is zero when ``q_pred == q_true`` or ``q_pred == -q_true``,
        and is bounded above by ``1.0``.

    Raises:
        ValueError: If the two tensors have different shapes or last
            dim is not 4.
    """
    if q_pred.shape != q_true.shape:
        raise ValueError(
            f"q_pred shape {tuple(q_pred.shape)} != q_true shape {tuple(q_true.shape)}"
        )
    if q_pred.shape[-1] != 4:
        raise ValueError(f"quaternion last dim must be 4, got {q_pred.shape[-1]}")
    inner = (q_pred * q_true).sum(dim=-1)
    return (1.0 - inner.pow(2)).mean()
