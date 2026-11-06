"""LoRA-style low-rank adapter for fine-tune comparison (#14479)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import require


class LoRAAdapter:
    """``y = W x + (B A) x`` with optional rank ``r`` (A, B zero when r=0)."""

    def __init__(
        self,
        weight: NDArray[np.floating],
        rank: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        require(weight.ndim == 2, "weight must be 2D", weight.ndim)
        require(rank >= 0, "rank must be non-negative", rank)
        self._w = np.asarray(weight, dtype=float)
        self._in_dim = int(self._w.shape[1])
        self._rank = int(rank)
        if self._rank == 0:
            self._a = np.zeros((0, self._in_dim), dtype=float)
            self._b = np.zeros((self._w.shape[0], 0), dtype=float)
        else:
            gen = rng if rng is not None else np.random.default_rng()
            bound = float(np.sqrt(1.0 / self._in_dim))
            self._a = gen.uniform(-bound, bound, size=(self._rank, self._in_dim))
            self._b = np.zeros((self._w.shape[0], self._rank), dtype=float)

    @property
    def weight(self) -> NDArray[np.floating]:
        return self._w

    @property
    def a(self) -> NDArray[np.floating]:
        return self._a

    @property
    def b(self) -> NDArray[np.floating]:
        return self._b

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        require(x.ndim == 1, "x must be 1D", x.ndim)
        require(
            int(x.shape[0]) == self._in_dim,
            "input dim must match weight columns",
            (x.shape[0], self._in_dim),
        )
        out = self._w @ x
        if self._rank > 0:
            out = out + self._b @ (self._a @ x)
        return out


def project_box(
    values: NDArray[np.floating],
    lower: NDArray[np.floating],
    upper: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Element-wise clip to ``[lower, upper]`` (same shape)."""
    require(
        values.shape == lower.shape == upper.shape,
        "values, lower, upper must match shape",
        (values.shape, lower.shape, upper.shape),
    )
    return np.minimum(np.maximum(values, lower), upper)
