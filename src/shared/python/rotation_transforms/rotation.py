"""Shared rotation representation conversions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from contracts import require, require_finite, require_unit_vector


def _as_finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    require(array.shape == shape, f"{name} must have shape {shape}", array.shape)
    require_finite(array, name)
    return array


class Rotation:
    """Immutable rotation stored internally as a unit quaternion.

    Public quaternions use the repository convention ``[w, x, y, z]``.
    SciPy uses ``[x, y, z, w]`` internally, so all crossings are normalized
    through small adapters here.
    """

    __slots__ = ("_rotation",)

    def __init__(self, q: Any) -> None:
        q_array = _as_finite_array(q, (4,), "quaternion")
        norm = np.linalg.norm(q_array)
        require(bool(norm > 1e-12), "quaternion must be non-zero", norm)
        q_array = q_array / norm
        if q_array[0] < 0:
            q_array = -q_array
        self._rotation = SciPyRotation.from_quat(
            [q_array[1], q_array[2], q_array[3], q_array[0]]
        )

    @classmethod
    def identity(cls) -> Rotation:
        """Create the identity rotation."""
        return cls([1.0, 0.0, 0.0, 0.0])

    @classmethod
    def from_quaternion(cls, q: Any) -> Rotation:
        """Create a rotation from a ``[w, x, y, z]`` quaternion."""
        return cls(q)

    @classmethod
    def from_rotation_matrix(cls, matrix: Any) -> Rotation:
        """Create a rotation from a 3x3 rotation matrix."""
        matrix_array = _as_finite_array(matrix, (3, 3), "rotation matrix")
        orth_err = np.max(np.abs(matrix_array @ matrix_array.T - np.eye(3)))
        require(
            bool(orth_err < 1e-6),
            f"rotation matrix must be orthogonal (max err={orth_err:.2e})",
        )
        det = np.linalg.det(matrix_array)
        require(bool(abs(det - 1.0) < 1e-6), "rotation matrix must have det=+1")
        rotation = SciPyRotation.from_matrix(matrix_array)
        return cls(_scipy_to_wxyz(rotation))

    @classmethod
    def from_euler(cls, a: float, b: float, c: float, convention: str) -> Rotation:
        """Create a rotation from Euler angles in radians."""
        _validate_convention(convention)
        rotation = SciPyRotation.from_euler(convention, [a, b, c])
        return cls(_scipy_to_wxyz(rotation))

    @classmethod
    def from_axis_angle(cls, axis: Any, angle: float) -> Rotation:
        """Create a rotation from a unit or non-zero axis and angle."""
        axis_array = _as_finite_array(axis, (3,), "axis")
        require_unit_vector(axis_array, "axis")
        rotation = SciPyRotation.from_rotvec(axis_array * float(angle))
        return cls(_scipy_to_wxyz(rotation))

    @classmethod
    def from_rodrigues(cls, vector: Any) -> Rotation:
        """Create a rotation from a Rodrigues/rotation vector."""
        vector_array = _as_finite_array(vector, (3,), "Rodrigues vector")
        rotation = SciPyRotation.from_rotvec(vector_array)
        return cls(_scipy_to_wxyz(rotation))

    def as_quaternion(self) -> np.ndarray:
        """Return a unit quaternion as ``[w, x, y, z]``."""
        return _scipy_to_wxyz(self._rotation)

    def as_rotation_matrix(self) -> np.ndarray:
        """Return a 3x3 rotation matrix."""
        return self._rotation.as_matrix()

    def as_euler(self, convention: str) -> tuple[float, float, float]:
        """Return Euler angles in radians for the given convention."""
        _validate_convention(convention)
        values = self._rotation.as_euler(convention)
        return (float(values[0]), float(values[1]), float(values[2]))

    def as_axis_angle(self) -> tuple[np.ndarray, float]:
        """Return ``(unit_axis, angle)``."""
        vector = self._rotation.as_rotvec()
        angle = float(np.linalg.norm(vector))
        if angle <= 1e-12:
            return np.array([1.0, 0.0, 0.0]), 0.0
        return vector / angle, angle

    def as_rodrigues(self) -> np.ndarray:
        """Return the Rodrigues/rotation vector."""
        return self._rotation.as_rotvec()

    def compose(self, other: Rotation) -> Rotation:
        """Compose this rotation with another: ``self * other``."""
        if not isinstance(other, Rotation):
            raise ValueError("other must be a Rotation")
        return Rotation(_scipy_to_wxyz(self._rotation * other._rotation))

    def inverse(self) -> Rotation:
        """Return the inverse rotation."""
        return Rotation(_scipy_to_wxyz(self._rotation.inv()))

    def __repr__(self) -> str:
        return f"Rotation(q={self.as_quaternion()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rotation):
            return NotImplemented
        q_self = self.as_quaternion()
        q_other = other.as_quaternion()
        return bool(
            np.allclose(q_self, q_other, atol=1e-10)
            or np.allclose(q_self, -q_other, atol=1e-10)
        )

    def __hash__(self) -> int:
        """Hash based on normalized quaternion (rounded to discrete bins).

        Two Rotation instances that compare equal must have the same hash.
        Since equality uses numerical tolerance (atol=1e-10), we round the
        quaternion to a coarse grid (1e-8 precision) for hashing.
        """
        q = self.as_quaternion()
        # Round to 8 decimal places for robust hashing across equivalent rotations
        q_rounded = tuple(np.round(q, 8).astype(float))
        return hash(q_rounded)


def _scipy_to_wxyz(rotation: SciPyRotation) -> np.ndarray:
    q_xyzw = rotation.as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    if q_wxyz[0] < 0:
        q_wxyz = -q_wxyz
    return q_wxyz


def _validate_convention(convention: str) -> None:
    if len(convention) != 3 or any(axis not in "xyzXYZ" for axis in convention):
        raise ValueError(f"Unknown Euler convention: {convention}")
