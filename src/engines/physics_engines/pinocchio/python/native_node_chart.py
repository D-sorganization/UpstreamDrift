"""Native weld-closure chart for shooting nodes, shared by audit drivers.

The wrapper owns only the state/residual layout: coordinates followed by rates,
twelve pose/rate closure rows, and the explicit numerical scales that define
the local chart. Dynamics stay in the engine; the retraction stays in
node_retraction. It is a coordination boundary, never a state-reset device.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.node_retraction import (
    NodeRetraction,
    retract_node,
    scaled_tangent_basis,
)

Array = NDArray[np.float64]
_CLOSURE_ROWS = 12


class NativeClosureOracle(Protocol):
    """Engine boundary required to build a closure-preserving node chart."""

    def closure_residuals(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float] | None = None,
    ) -> tuple[Array, Array]: ...

    def closure_trajectory_linearization(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        accelerations: Mapping[str, float],
        *,
        finite_difference_step: float,
    ) -> Any: ...


class NativeNodeChart:
    """Closure residual, Jacobian, tangent basis and retraction at one layout."""

    def __init__(
        self,
        coordinate_names: Sequence[str],
        engine: NativeClosureOracle,
        *,
        state_scales: Array,
        residual_scales: Array,
        radius: float,
        tolerance: float,
        jacobian_step: float,
    ) -> None:
        names = tuple(coordinate_names)
        scales = np.array(state_scales, dtype=float, copy=True)
        residual = np.array(residual_scales, dtype=float, copy=True)
        if (
            not names
            or len(set(names)) != len(names)
            or scales.shape != (2 * len(names),)
            or residual.shape != (_CLOSURE_ROWS,)
            or not np.isfinite(scales).all()
            or not np.isfinite(residual).all()
            or np.any(scales <= 0)
            or np.any(residual <= 0)
            or any(
                not np.isfinite(value) or value <= 0
                for value in (radius, tolerance, jacobian_step)
            )
        ):
            raise ValueError("Invalid native node chart names, scales or settings")
        scales.setflags(write=False)
        residual.setflags(write=False)
        self._names, self._engine = names, engine
        self._scales, self._residual = scales, residual
        self._radius, self._tolerance, self._step = radius, tolerance, jacobian_step

    @property
    def dimension(self) -> int:
        return len(self._names)

    def _split(self, state: Array) -> tuple[dict[str, float], dict[str, float]]:
        x = np.asarray(state, dtype=float)
        if x.shape != (2 * self.dimension,) or not np.isfinite(x).all():
            raise ValueError("State must be finite native q followed by rates")
        n = self.dimension
        return (
            dict(zip(self._names, x[:n].tolist(), strict=True)),
            dict(zip(self._names, x[n:].tolist(), strict=True)),
        )

    def closure(self, state: Array) -> Array:
        """Return the twelve pose/rate weld residuals at one state."""
        q, v = self._split(state)
        value = np.concatenate(self._engine.closure_residuals(q, v))
        if value.shape != (_CLOSURE_ROWS,) or not np.isfinite(value).all():
            raise ValueError("Invalid native closure residual")
        return value

    def jacobian(self, state: Array) -> Array:
        """Return d(closure)/d(q, v) using the engine's local linearization."""
        q, v = self._split(state)
        linear = self._engine.closure_trajectory_linearization(
            q, v, dict.fromkeys(self._names, 0.0), finite_difference_step=self._step
        )
        value = np.hstack((linear.dq[:_CLOSURE_ROWS], linear.dv[:_CLOSURE_ROWS]))
        if (
            value.shape != (_CLOSURE_ROWS, 2 * self.dimension)
            or not np.isfinite(value).all()
        ):
            raise ValueError("Invalid native closure Jacobian")
        return value

    def basis(self, reference: Array) -> Array:
        """Return the scaled orthonormal tangent basis at a closure-valid node."""
        return scaled_tangent_basis(
            self.jacobian(reference) / self._residual[:, None], self._scales
        )

    def retract(
        self, reference: Array, basis: Array, coordinates: Array
    ) -> NodeRetraction:
        """Retract chart coordinates onto the closure manifold near reference."""
        return retract_node(
            reference,
            basis,
            coordinates,
            self.closure,
            self.jacobian,
            state_scales=self._scales,
            residual_scales=self._residual,
            radius=self._radius,
            tolerance=self._tolerance,
        )

    def describe(self) -> dict[str, Any]:
        """Return the explicit numerical chart settings for receipts."""
        return {
            "dimension": self.dimension,
            "closure_rows": _CLOSURE_ROWS,
            "state_scales": self._scales.tolist(),
            "residual_scales": self._residual.tolist(),
            "radius": self._radius,
            "tolerance": self._tolerance,
            "jacobian_difference_step": self._step,
        }
