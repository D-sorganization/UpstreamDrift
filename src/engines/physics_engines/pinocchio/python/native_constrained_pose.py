"""Native adapters for shared closed-chain marker-pose fitting.

This module exposes only marker position and weld-pose residual oracles.  It
does not smooth poses, differentiate capture data, identify effort, alter a
state, or claim a forward-dynamics match.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class _MarkerResult(Protocol):
    """Subset of the native marker derivative result used by this adapter."""

    @property
    def positions_m(self) -> NDArray[np.float64]: ...


class _NativeConstrainedPoseModel(Protocol):
    """Minimal engine contract required by the shared static pose solver."""

    def marker_derivatives(
        self,
        coordinates: dict[str, float],
        bodies: Sequence[str],
        offsets: NDArray[np.float64],
    ) -> _MarkerResult: ...

    def closure_residuals(
        self, coordinates: dict[str, float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


class NativeConstrainedPoseOracle:
    """Map ordered native pose vectors to shared marker/closure callables."""

    def __init__(
        self,
        model: _NativeConstrainedPoseModel,
        coordinate_names: Sequence[str],
        marker_bodies: Sequence[str],
        marker_offsets_m: NDArray[np.float64],
    ) -> None:
        names = tuple(coordinate_names)
        bodies = tuple(marker_bodies)
        offsets = np.array(marker_offsets_m, dtype=float, copy=True)
        if (
            not names
            or len(names) != len(set(names))
            or not bodies
            or offsets.shape != (len(bodies), 3)
            or not np.isfinite(offsets).all()
        ):
            raise ValueError("Invalid native coordinate or marker specification")
        self._model = model
        self._names = names
        self._bodies = bodies
        self._offsets = offsets
        self._offsets.setflags(write=False)

    def _coordinates(self, vector: NDArray[np.float64]) -> dict[str, float]:
        values = np.asarray(vector, dtype=float)
        if values.shape != (len(self._names),) or not np.isfinite(values).all():
            raise ValueError("Expected a finite native coordinate vector")
        return dict(zip(self._names, values.tolist(), strict=True))

    def forward(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ordered fixed-marker world positions for one native pose."""
        result = self._model.marker_derivatives(
            self._coordinates(coordinates), self._bodies, self._offsets
        )
        positions = np.asarray(result.positions_m, dtype=float)
        if (
            positions.shape != (len(self._bodies), 3)
            or not np.isfinite(positions).all()
        ):
            raise ValueError("Native model returned invalid marker positions")
        return positions.copy()

    def closure(self, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the refreshed weld pose residual for one native pose."""
        pose, _ = self._model.closure_residuals(self._coordinates(coordinates))
        residual = np.asarray(pose, dtype=float)
        if residual.ndim != 1 or not residual.size or not np.isfinite(residual).all():
            raise ValueError("Native model returned invalid closure residual")
        return residual.copy()
