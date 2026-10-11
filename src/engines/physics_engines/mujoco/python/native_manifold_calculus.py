"""Shared tangent calculus for native Crocoddyl state and action adapters.

Each caller supplies its validated physical state and native retraction;
this module neither normalizes configurations nor changes native dynamics.
"""

from __future__ import annotations

from typing import Any

import crocoddyl
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _selected_jacobians(
    state: Any, first: Any, second: Any, component: Any
) -> list[Array]:
    if component == crocoddyl.Jcomponent.first:
        return [state._jacobian(first)]
    if component == crocoddyl.Jcomponent.second:
        return [state._jacobian(second)]
    return [state._jacobian(first), state._jacobian(second)]


def difference_jacobians(
    state: Any, before: Array, after: Array, component: Any
) -> list[Array]:
    """Differentiate native difference by perturbing either tangent origin."""

    def first(direction: Array) -> Array:
        return state.diff(state.integrate(before, direction), after)

    def second(direction: Array) -> Array:
        return state.diff(before, state.integrate(after, direction))

    return _selected_jacobians(state, first, second, component)


def integration_jacobians(
    state: Any, base: Array, tangent: Array, component: Any
) -> list[Array]:
    """Differentiate native retraction in the resulting state's tangent."""
    output = state.integrate(base, tangent)

    def first(direction: Array) -> Array:
        displaced = state.integrate(base, direction)
        return state.diff(output, state.integrate(displaced, tangent))

    def second(direction: Array) -> Array:
        return state.diff(output, state.integrate(base, tangent + direction))

    return _selected_jacobians(state, first, second, component)


def set_tracking_cost_derivatives(
    data: Any,
    state: Any,
    reference: Array,
    state_weights: Array,
    command: Array,
    input_weights: Array,
) -> None:
    """Set quadratic tracking gradient and Gauss–Newton curvature blocks.

    ``data.Fx`` and ``data.Fu`` must already be the admitted native step's
    tangent derivatives. Curvature omits second derivatives of the composed
    residual, including native dynamics and manifold difference.
    """
    error = state.diff(reference, data.xnext)
    jacobian = state.Jdiff(reference, data.xnext, crocoddyl.Jcomponent.second)[0]
    gradient = 2 * jacobian.T @ (state_weights * error)
    curvature = 2 * jacobian.T @ np.diag(state_weights) @ jacobian
    data.Lx = data.Fx.T @ gradient
    data.Lu = data.Fu.T @ gradient + 2 * input_weights * command
    data.Lxx = data.Fx.T @ curvature @ data.Fx
    data.Lxu = data.Fx.T @ curvature @ data.Fu
    data.Luu = data.Fu.T @ curvature @ data.Fu + 2 * np.diag(input_weights)
