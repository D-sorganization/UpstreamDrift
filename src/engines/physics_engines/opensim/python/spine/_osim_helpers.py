"""Single-hop OpenSim SWIG adapters for spine models (SPN-7a, #12750).

Each function takes an already-loaded ``opensim.Model`` and wraps one SWIG
chain so callers stay within the two-level call-chain limit.  No OpenSim import
is needed here: the model object carries the bindings.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["body_transform_in", "set_coordinate_values", "transform_to_matrix"]


def _coordinate(model: object, name: str) -> object:
    coords = model.getCoordinateSet()  # type: ignore[attr-defined]
    if not coords.getIndex(name) >= 0:
        raise ValueError(f"model has no coordinate {name!r}")
    return coords.get(name)


def _body(model: object, name: str) -> object:
    bodies = model.getBodySet()  # type: ignore[attr-defined]
    if not bodies.getIndex(name) >= 0:
        raise ValueError(f"model has no body {name!r}")
    return bodies.get(name)


def transform_to_matrix(transform: object) -> np.ndarray:
    """Convert an ``opensim.Transform`` to a (4, 4) float array."""
    rot = transform.R()  # type: ignore[attr-defined]
    pos = transform.p()  # type: ignore[attr-defined]
    out = np.eye(4)
    for i in range(3):
        out[i, 3] = pos.get(i)
        for j in range(3):
            out[i, j] = rot.get(i, j)
    return out


def set_coordinate_values(
    model: object, state: object, values: Mapping[str, float]
) -> None:
    """Reset all coordinates to 0, set ``values`` (radians/m), realize positions.

    Raises ``ValueError`` for an unknown coordinate name.
    """
    coords = model.getCoordinateSet()  # type: ignore[attr-defined]
    for name in values:
        _coordinate(model, name)
    for i in range(coords.getSize()):
        coords.get(i).setValue(state, 0.0, False)
    for name, value in values.items():
        _coordinate(model, name).setValue(state, float(value), False)  # type: ignore[attr-defined]
    model.realizePosition(state)  # type: ignore[attr-defined]


def body_transform_in(
    model: object, state: object, body_name: str, base_name: str
) -> np.ndarray:
    """(4, 4) pose of body ``body_name`` expressed in body ``base_name``'s frame.

    ``X_base_body = inv(X_ground_base) @ X_ground_body``; the state must be
    realized to Position (``set_coordinate_values`` does so).
    """
    x_body = transform_to_matrix(_body(model, body_name).getTransformInGround(state))  # type: ignore[attr-defined]
    x_base = transform_to_matrix(_body(model, base_name).getTransformInGround(state))  # type: ignore[attr-defined]
    return np.linalg.inv(x_base) @ x_body
