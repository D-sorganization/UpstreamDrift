"""Physically consistent inertial-parameter algebra for MOSAIC.

Rigid-body inverse dynamics is linear in the ten inertial parameters of each
body, ``pi = (m, h, I)`` with ``h = m c`` the first mass moment and ``I`` the
rotational inertia about the body frame origin.  The set of *physically
consistent* parameters is the convex cone of parameters whose 4x4
pseudo-inertia ``J = [[Sigma, h], [h^T, m]]`` with ``Sigma = tr(I)/2 - I`` is
positive semidefinite (Wensing, Kim and Slotine, RA-L 2018).  Two
representations are provided:

* the linear 10-vector, kept inside the convex cone by Euclidean projection in
  the pseudo-inertia (Frobenius) metric; this preserves the linearity that the
  variable-projection inner solve exploits;
* the unconstrained log-Cholesky coordinates of Rucker and Wensing (RA-L 2022),
  which map all of R^10 onto the interior of the cone and are used for the
  final manifold polish.

The planar fixture uses the 4-vector ``(m, h_x, h_y, I_o)`` whose consistency
condition ``m I_o >= |h|^2`` is a rotated second-order cone.

All functions are vectorized over a leading batch axis of bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

from src.shared.python.core.contracts import ensure, require

FloatArray: TypeAlias = npt.NDArray[np.float64]

PLANAR_PARAMETERS_PER_BODY = 4
SPATIAL_PARAMETERS_PER_BODY = 10
_PSEUDO_DIM = 4
_INERTIA_INDEX = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_DEFAULT_TOLERANCE = 1e-12


def _as_batch(parameters: FloatArray, width: int) -> FloatArray:
    array = np.asarray(parameters, dtype=np.float64)
    require(array.ndim == 2, "parameters must be a (bodies, width) array", array.shape)
    require(array.shape[1] == width, f"each body needs {width} parameters", array.shape)
    require(bool(np.all(np.isfinite(array))), "parameters must be finite")
    return array


def _inertia_tensor(flat: FloatArray) -> FloatArray:
    tensor = np.zeros((flat.shape[0], 3, 3), dtype=np.float64)
    for column, (row, col) in enumerate(_INERTIA_INDEX):
        tensor[:, row, col] = flat[:, column]
        tensor[:, col, row] = flat[:, column]
    return tensor


def _flatten_inertia(tensor: FloatArray) -> FloatArray:
    return np.stack([tensor[:, row, col] for row, col in _INERTIA_INDEX], axis=1)


def pseudo_inertia(parameters: FloatArray) -> FloatArray:
    """Return the batch of 4x4 pseudo-inertia matrices of spatial parameters.

    Postcondition: each matrix is symmetric.
    """
    batch = _as_batch(parameters, SPATIAL_PARAMETERS_PER_BODY)
    inertia = _inertia_tensor(batch[:, 4:])
    trace = np.einsum("bii->b", inertia)
    sigma = 0.5 * trace[:, None, None] * np.eye(3) - inertia
    pseudo = np.zeros((batch.shape[0], _PSEUDO_DIM, _PSEUDO_DIM), dtype=np.float64)
    pseudo[:, :3, :3] = sigma
    pseudo[:, :3, 3] = batch[:, 1:4]
    pseudo[:, 3, :3] = batch[:, 1:4]
    pseudo[:, 3, 3] = batch[:, 0]
    ensure(bool(np.allclose(pseudo, np.swapaxes(pseudo, 1, 2))), "pseudo symmetric")
    return pseudo


def parameters_from_pseudo_inertia(pseudo: FloatArray) -> FloatArray:
    """Invert :func:`pseudo_inertia` for a batch of symmetric 4x4 matrices."""
    array = np.asarray(pseudo, dtype=np.float64)
    require(
        array.ndim == 3 and array.shape[1:] == (4, 4),
        "need (bodies, 4, 4)",
        array.shape,
    )
    sigma = array[:, :3, :3]
    inertia = np.einsum("bii->b", sigma)[:, None, None] * np.eye(3) - sigma
    return np.concatenate(
        [array[:, 3, 3:4], array[:, :3, 3], _flatten_inertia(inertia)], axis=1
    )


def is_spatial_consistent(
    parameters: FloatArray, tolerance: float = _DEFAULT_TOLERANCE
) -> npt.NDArray[np.bool_]:
    """Return a boolean per body: pseudo-inertia PSD within ``tolerance``."""
    eigenvalues = np.linalg.eigvalsh(pseudo_inertia(parameters))
    return np.all(eigenvalues >= -tolerance, axis=1)


def project_spatial_consistent(
    parameters: FloatArray, floor: float = 0.0
) -> FloatArray:
    """Euclidean projection onto the physically consistent cone.

    The projection is performed in the pseudo-inertia Frobenius metric by
    clipping eigenvalues at ``floor`` (``>= 0``), which is the nearest PSD
    matrix (Higham).  Consistent bodies are returned unchanged.
    """
    require(floor >= 0.0, "floor must be non-negative", floor)
    eigenvalues, vectors = np.linalg.eigh(pseudo_inertia(parameters))
    clipped = np.maximum(eigenvalues, floor)
    projected = np.einsum("bik,bk,bjk->bij", vectors, clipped, vectors)
    return parameters_from_pseudo_inertia(
        0.5 * (projected + np.swapaxes(projected, 1, 2))
    )


def to_log_cholesky(parameters: FloatArray) -> FloatArray:
    """Map consistent spatial parameters to unconstrained log-Cholesky coordinates.

    Precondition: every body is strictly physically consistent.  Coordinates are
    ``(log diag(L), strictly-lower entries of L)`` with ``J = L L^T``.
    """
    pseudo = pseudo_inertia(parameters)
    require(
        bool(np.all(np.linalg.eigvalsh(pseudo) > 0.0)),
        "bodies must be strictly consistent",
    )
    lower = np.linalg.cholesky(pseudo)
    diag = np.log(np.einsum("bii->bi", lower))
    rows, cols = np.tril_indices(_PSEUDO_DIM, k=-1)
    return np.concatenate([diag, lower[:, rows, cols]], axis=1)


def from_log_cholesky(theta: FloatArray) -> FloatArray:
    """Inverse of :func:`to_log_cholesky`; every input maps to a consistent body."""
    batch = _as_batch(theta, SPATIAL_PARAMETERS_PER_BODY)
    lower = np.zeros((batch.shape[0], _PSEUDO_DIM, _PSEUDO_DIM), dtype=np.float64)
    index = np.arange(_PSEUDO_DIM)
    lower[:, index, index] = np.exp(batch[:, :_PSEUDO_DIM])
    rows, cols = np.tril_indices(_PSEUDO_DIM, k=-1)
    lower[:, rows, cols] = batch[:, _PSEUDO_DIM:]
    pseudo = np.einsum("bik,bjk->bij", lower, lower)
    return parameters_from_pseudo_inertia(pseudo)


def planar_to_unconstrained(parameters: FloatArray) -> FloatArray:
    """Map strictly consistent planar bodies to coordinates ``(log m, h_x, h_y, log I_c)``.

    ``I_c = I_o - |h|^2 / m`` is the inertia about the centre of mass, so the
    image of R^4 under :func:`planar_from_unconstrained` is exactly the interior
    of the consistency cone.
    """
    batch = _as_batch(parameters, PLANAR_PARAMETERS_PER_BODY)
    require(bool(np.all(batch[:, 0] > 0.0)), "masses must be positive")
    inertia_com = batch[:, 3] - (batch[:, 1] ** 2 + batch[:, 2] ** 2) / batch[:, 0]
    require(bool(np.all(inertia_com > 0.0)), "bodies must be strictly consistent")
    return np.stack(
        [np.log(batch[:, 0]), batch[:, 1], batch[:, 2], np.log(inertia_com)], axis=1
    )


def planar_from_unconstrained(theta: FloatArray) -> FloatArray:
    """Inverse of :func:`planar_to_unconstrained`; every input is consistent."""
    batch = _as_batch(theta, PLANAR_PARAMETERS_PER_BODY)
    mass = np.exp(batch[:, 0])
    inertia_origin = np.exp(batch[:, 3]) + (batch[:, 1] ** 2 + batch[:, 2] ** 2) / mass
    return np.stack([mass, batch[:, 1], batch[:, 2], inertia_origin], axis=1)


def planar_unconstrained_jacobian(theta: FloatArray) -> FloatArray:
    """Analytic ``d pi / d theta`` of :func:`planar_from_unconstrained`, shape ``(bodies, 4, 4)``."""
    batch = _as_batch(theta, PLANAR_PARAMETERS_PER_BODY)
    mass = np.exp(batch[:, 0])
    hx, hy = batch[:, 1], batch[:, 2]
    jacobian = np.zeros((batch.shape[0], 4, 4), dtype=np.float64)
    jacobian[:, 0, 0] = mass
    jacobian[:, 1, 1] = 1.0
    jacobian[:, 2, 2] = 1.0
    jacobian[:, 3, 0] = -(hx**2 + hy**2) / mass
    jacobian[:, 3, 1] = 2.0 * hx / mass
    jacobian[:, 3, 2] = 2.0 * hy / mass
    jacobian[:, 3, 3] = np.exp(batch[:, 3])
    return jacobian


class InertialParameterization(Protocol):
    """Smooth bijection between unconstrained coordinates and consistent ``pi``."""

    def to_unconstrained(self, parameters: FloatArray) -> FloatArray: ...

    def from_unconstrained(self, theta: FloatArray) -> FloatArray: ...

    def jacobian(self, theta: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class PlanarParameterization:
    """Flat-vector wrapper of the planar ``(log m, h, log I_c)`` coordinates."""

    def to_unconstrained(self, parameters: FloatArray) -> FloatArray:
        return planar_to_unconstrained(
            parameters.reshape(-1, PLANAR_PARAMETERS_PER_BODY)
        ).ravel()

    def from_unconstrained(self, theta: FloatArray) -> FloatArray:
        return planar_from_unconstrained(
            theta.reshape(-1, PLANAR_PARAMETERS_PER_BODY)
        ).ravel()

    def jacobian(self, theta: FloatArray) -> FloatArray:
        """Block-diagonal ``d pi / d theta`` of shape ``(n_pi, n_pi)``."""
        blocks = planar_unconstrained_jacobian(
            theta.reshape(-1, PLANAR_PARAMETERS_PER_BODY)
        )
        return block_diag(*blocks)


@dataclass(frozen=True)
class SpatialLogCholeskyParameterization:
    """Flat-vector wrapper of the spatial log-Cholesky coordinates (FD Jacobian)."""

    finite_difference_step: float = 1e-7

    def to_unconstrained(self, parameters: FloatArray) -> FloatArray:
        return to_log_cholesky(
            parameters.reshape(-1, SPATIAL_PARAMETERS_PER_BODY)
        ).ravel()

    def from_unconstrained(self, theta: FloatArray) -> FloatArray:
        return from_log_cholesky(theta.reshape(-1, SPATIAL_PARAMETERS_PER_BODY)).ravel()

    def jacobian(self, theta: FloatArray) -> FloatArray:
        step = self.finite_difference_step
        columns = [
            (
                self.from_unconstrained(theta + step * unit)
                - self.from_unconstrained(theta - step * unit)
            )
            / (2.0 * step)
            for unit in np.eye(theta.size)
        ]
        return np.stack(columns, axis=1)


def planar_consistency_margin(parameters: FloatArray) -> FloatArray:
    """Return ``m I_o - |h|^2`` per planar body (non-negative when consistent)."""
    batch = _as_batch(parameters, PLANAR_PARAMETERS_PER_BODY)
    return batch[:, 0] * batch[:, 3] - batch[:, 1] ** 2 - batch[:, 2] ** 2


def is_planar_consistent(
    parameters: FloatArray, tolerance: float = _DEFAULT_TOLERANCE
) -> npt.NDArray[np.bool_]:
    """Return a boolean per planar body (positive mass, inertia, and margin)."""
    batch = _as_batch(parameters, PLANAR_PARAMETERS_PER_BODY)
    positive = (batch[:, 0] >= -tolerance) & (batch[:, 3] >= -tolerance)
    return positive & (planar_consistency_margin(batch) >= -tolerance)


def project_planar_consistent(parameters: FloatArray) -> FloatArray:
    """Project planar parameters onto the cone ``{m I_o >= |h|^2, m, I_o >= 0}``.

    The cone is a rotated second-order cone; the projection is Euclidean in the
    coordinates ``(m, sqrt(2) h, I_o)`` and leaves consistent bodies unchanged.
    """
    batch = _as_batch(parameters, PLANAR_PARAMETERS_PER_BODY)
    root2 = np.sqrt(2.0)
    s = (batch[:, 0] + batch[:, 3]) / root2
    x = np.stack(
        [(batch[:, 0] - batch[:, 3]) / root2, root2 * batch[:, 1], root2 * batch[:, 2]],
        axis=1,
    )
    norm = np.linalg.norm(x, axis=1)
    inside = norm <= s
    polar = norm <= -s
    scale = np.where(
        inside, 1.0, 0.5 * (s + norm) / np.maximum(norm, np.finfo(float).tiny)
    )
    s_new = np.where(inside, s, 0.5 * (s + norm))
    x_new = x * scale[:, None]
    s_new = np.where(polar, 0.0, s_new)
    x_new = np.where(polar[:, None], 0.0, x_new)
    mass = (s_new + x_new[:, 0]) / root2
    inertia = (s_new - x_new[:, 0]) / root2
    projected = np.stack(
        [mass, x_new[:, 1] / root2, x_new[:, 2] / root2, inertia], axis=1
    )
    ensure(
        bool(np.all(is_planar_consistent(projected, tolerance=1e-9))),
        "projection consistent",
    )
    return projected
