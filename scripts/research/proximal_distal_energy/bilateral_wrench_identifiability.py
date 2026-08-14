"""Structural identifiability maps for bilateral hand contact wrenches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearMapAudit:
    """Rank, nullspace, and nonzero conditioning of a linear measurement map."""

    matrix_shape: tuple[int, int]
    rank: int
    nullity: int
    singular_values: FloatArray
    minimum_nonzero_singular_value: float
    nonzero_condition_number: float
    right_null_basis: FloatArray


def _finite_array(value: object, *, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def point_force_wrench_map(
    contact_positions_m: object,
    reference_position_m: object = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Map two three-axis point forces to one net wrench about a reference.

    Inputs are ordered ``[F_lead, F_trail]`` and outputs are ordered
    ``[resultant_force, resultant_moment]``.
    """

    positions = _finite_array(
        contact_positions_m, shape=(2, 3), name="contact_positions_m"
    )
    reference = _finite_array(
        reference_position_m, shape=(3,), name="reference_position_m"
    )
    identity = np.eye(3)
    return np.block(
        [
            [identity, identity],
            [_skew(positions[0] - reference), _skew(positions[1] - reference)],
        ]
    )


def full_hand_wrench_map(
    contact_positions_m: object,
    reference_position_m: object = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Map two six-axis hand wrenches to one net wrench about a reference.

    Inputs are ordered ``[F_lead, M_lead, F_trail, M_trail]``.
    """

    positions = _finite_array(
        contact_positions_m, shape=(2, 3), name="contact_positions_m"
    )
    reference = _finite_array(
        reference_position_m, shape=(3,), name="reference_position_m"
    )
    identity = np.eye(3)
    zeros = np.zeros((3, 3))
    return np.block(
        [
            [identity, zeros, identity, zeros],
            [
                _skew(positions[0] - reference),
                identity,
                _skew(positions[1] - reference),
                identity,
            ],
        ]
    )


def internal_axial_measurement(contact_positions_m: object) -> FloatArray:
    """Return one normalized row that observes the invisible axial force mode."""

    positions = _finite_array(
        contact_positions_m, shape=(2, 3), name="contact_positions_m"
    )
    separation = positions[1] - positions[0]
    span = float(np.linalg.norm(separation))
    if span <= np.finfo(float).eps:
        raise ValueError("contact positions must be distinct")
    direction = separation / span
    return np.concatenate((direction, -direction))[None, :] / np.sqrt(2.0)


def audit_linear_map(
    matrix: object, *, relative_tolerance: float = 1e-12
) -> LinearMapAudit:
    """Audit structural rank and nullity using an explicit relative tolerance."""

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must contain only finite values")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive and finite")

    _, singular_values, right_vectors_t = np.linalg.svd(array, full_matrices=True)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    threshold = relative_tolerance * max(array.shape) * scale
    rank = int(np.count_nonzero(singular_values > threshold))
    nullity = int(array.shape[1] - rank)
    nonzero = singular_values[:rank]
    if nonzero.size:
        minimum = float(nonzero[-1])
        condition = float(nonzero[0] / nonzero[-1])
    else:
        minimum = 0.0
        condition = float("inf")
    null_basis = right_vectors_t[rank:, :].T.copy()
    return LinearMapAudit(
        matrix_shape=(int(array.shape[0]), int(array.shape[1])),
        rank=rank,
        nullity=nullity,
        singular_values=singular_values,
        minimum_nonzero_singular_value=minimum,
        nonzero_condition_number=condition,
        right_null_basis=null_basis,
    )
