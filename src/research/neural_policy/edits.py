"""Rank-1 weight interventions and activation-direction probes (#14478).

Edit recipe: :math:`W' = W - \\alpha\\, u (u^{\\mathsf T} W)` with optional row mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import require


@dataclass(frozen=True)
class LowRankEditRecipe:
    """Serializable rank-1 weight intervention (not a synergy claim)."""

    mask_rows: tuple[int, ...] | None
    u: tuple[float, ...]
    alpha: float
    u_dot_w: tuple[float, ...]
    intervention: str = "rank1_masked_subtract_u_uT_W"

    def as_dict(self) -> dict:
        """JSON-friendly run-record payload."""
        return {
            "intervention": self.intervention,
            "mask_rows": None if self.mask_rows is None else list(self.mask_rows),
            "u": list(self.u),
            "alpha": self.alpha,
            "u_dot_w": list(self.u_dot_w),
        }


def probe_activation_direction(
    activations: NDArray[np.floating],
    loss_grad_wrt_activations: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return a unit vector along the gradient in activation space.

    Preconditions:
        - ``activations`` and ``loss_grad_wrt_activations`` are 1D, same length.
    Postconditions:
        - Return has L2 norm 1 when the gradient is non-zero; else zero vector.
    """
    require(activations.ndim == 1, "activations must be 1D", activations.ndim)
    require(
        loss_grad_wrt_activations.ndim == 1,
        "loss_grad_wrt_activations must be 1D",
        loss_grad_wrt_activations.ndim,
    )
    require(
        activations.shape == loss_grad_wrt_activations.shape,
        "activation vector and gradient must match in shape",
        (activations.shape, loss_grad_wrt_activations.shape),
    )
    g = np.asarray(loss_grad_wrt_activations, dtype=float)
    norm = float(np.linalg.norm(g))
    if norm < 1e-15:
        return np.zeros_like(g)
    return g / norm


def apply_low_rank_weight_edit(
    weight: NDArray[np.floating],
    u: NDArray[np.floating],
    alpha: float,
    mask_rows: NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.floating], LowRankEditRecipe]:
    """Return ``W' = W - alpha * u (u.T @ W)`` with optional row mask.

    Rows not in ``mask_rows`` are unchanged. When ``mask_rows`` is None, all rows
    are updated.

    Preconditions:
        - ``weight`` is 2D, ``u`` is 1D with ``len(u) == weight.shape[0]``.
    Postconditions:
        - Unmasked rows match the input ``weight`` exactly.
    """
    require(weight.ndim == 2, "weight must be 2D", weight.ndim)
    u_vec = np.asarray(u, dtype=float).reshape(-1)
    require(
        u_vec.shape[0] == weight.shape[0],
        "u length must match weight rows",
        (u_vec.shape[0], weight.shape[0]),
    )
    require(np.isfinite(alpha), "alpha must be finite", alpha)

    w = np.asarray(weight, dtype=float)
    out = w.copy()
    projection = u_vec @ w
    u_dot_w = tuple(float(x) for x in projection)
    update = alpha * np.outer(u_vec, projection)
    rows_arr: np.ndarray | None
    if mask_rows is None:
        out -= update
        rows_arr = None
        rows_tuple: tuple[int, ...] | None = None
    else:
        rows_arr = np.asarray(mask_rows, dtype=int)
        for r in rows_arr:
            out[r, :] -= update[r, :]
        rows_tuple = tuple(int(x) for x in rows_arr.reshape(-1))

    recipe = LowRankEditRecipe(
        mask_rows=rows_tuple,
        u=tuple(float(x) for x in u_vec),
        alpha=float(alpha),
        u_dot_w=u_dot_w,
    )
    if mask_rows is not None:
        keep = np.ones(out.shape[0], dtype=bool)
        keep[rows_arr] = False
        require(
            np.allclose(out[keep, :], w[keep, :], rtol=0.0, atol=0.0),
            "unmasked rows must be bit-identical",
            None,
        )
    return out, recipe


def reverse_low_rank_edit(
    weight_edited: NDArray[np.floating],
    recipe: LowRankEditRecipe,
) -> NDArray[np.floating]:
    """Invert :func:`apply_low_rank_weight_edit` using stored ``u_dot_w``."""
    w = np.asarray(weight_edited, dtype=float)
    u = np.asarray(recipe.u, dtype=float)
    p = np.asarray(recipe.u_dot_w, dtype=float)
    update = recipe.alpha * np.outer(u, p)
    if recipe.mask_rows is None:
        return w + update
    rows = np.asarray(recipe.mask_rows, dtype=int)
    out = w.copy()
    for r in rows:
        out[r, :] += update[r, :]
    return out
