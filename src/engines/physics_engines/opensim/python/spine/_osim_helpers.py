"""Single-hop OpenSim SWIG adapters for spine models (SPN-7a, #12750).

Each function takes an already-loaded ``opensim.Model`` and wraps one SWIG
chain so callers stay within the two-level call-chain limit.  No OpenSim import
is needed here: the model object carries the bindings.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = [
    "body_mass",
    "body_transform_in",
    "coordinate_value",
    "muscle_max_isometric_force",
    "muscle_moment_arm",
    "set_coordinate_values",
    "set_independent_values",
    "state_derivatives",
    "transform_to_matrix",
]


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


def body_mass(model: object, name: str) -> float:
    """Mass (kg) of body ``name``; raises ``ValueError`` for an unknown body."""
    return float(_body(model, name).getMass())  # type: ignore[attr-defined]


def coordinate_value(model: object, state: object, name: str) -> float:
    """Current value (rad or m) of coordinate ``name`` in ``state``."""
    return float(_coordinate(model, name).getValue(state))  # type: ignore[attr-defined]


def set_independent_values(
    model: object, state: object, values: Mapping[str, float]
) -> None:
    """Prescribe independent coordinates and project the coupled ones.

    Every coordinate is first reset to 0, the ``values`` are set (radians or
    metres) and ``model.assemble`` satisfies the coupler constraints with the
    prescribed coordinates locked, so the dependent coordinates take the values
    of their coupling functions.  Raises ``ValueError`` for an unknown name.
    """
    set_coordinate_values(model, state, values)
    locked = [_coordinate(model, name) for name in values]
    for coord in locked:
        coord.setLocked(state, True)  # type: ignore[attr-defined]
    try:
        model.assemble(state)  # type: ignore[attr-defined]
    finally:
        for coord in locked:
            coord.setLocked(state, False)  # type: ignore[attr-defined]
    model.realizePosition(state)  # type: ignore[attr-defined]


def muscle_max_isometric_force(model: object, name: str) -> float:
    """Maximum isometric force (N) of muscle ``name``."""
    muscles = model.getMuscles()  # type: ignore[attr-defined]
    if not muscles.getIndex(name) >= 0:
        raise ValueError(f"model has no muscle {name!r}")
    return float(muscles.get(name).getMaxIsometricForce())


def muscle_moment_arm(
    model: object, state: object, muscle: str, coordinate: str
) -> float:
    """Moment arm (m or dimensionless) of ``muscle`` about ``coordinate``."""
    muscles = model.getMuscles()  # type: ignore[attr-defined]
    if not muscles.getIndex(muscle) >= 0:
        raise ValueError(f"model has no muscle {muscle!r}")
    path = muscles.get(muscle).getGeometryPath()
    return float(path.computeMomentArm(state, _coordinate(model, coordinate)))


def state_derivatives(model: object, state: object) -> np.ndarray:
    """All state-variable derivatives of ``state`` (realized to Acceleration)."""
    model.realizeAcceleration(state)  # type: ignore[attr-defined]
    names = model.getStateVariableNames()  # type: ignore[attr-defined]
    return np.array(
        [
            model.getStateVariableDerivativeValue(state, names.get(i))  # type: ignore[attr-defined]
            for i in range(names.getSize())
        ],
        dtype=float,
    )
