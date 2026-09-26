"""Classified errors, field origins and explicit-raise validators (IA-U2, #9703).

Every check here raises explicitly so the invariants survive ``python -O`` and
any disabled contract level. Numeric boundary values are copied into read-only
float arrays; strings and booleans are never coerced into numbers.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, TypeAlias, cast

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]

#: Absolute orthonormality/determinant tolerance for a proper rotation.
ROTATION_TOLERANCE = 1e-9
#: Absolute unit-norm tolerance for a (w, x, y, z) quaternion; never normalized.
QUATERNION_TOLERANCE = 1e-10
#: Relative tolerance for symmetry and the principal-moment triangle inequality.
SYMMETRY_RELATIVE_TOLERANCE = 1e-10
#: Absolute unit-norm tolerance for a declared contact normal.
UNIT_VECTOR_TOLERANCE = 1e-10

_SHA256 = re.compile(r"[0-9a-f]{64}")


class PreImpactBundleError(ValueError):
    """A bundle contract violation, classified by a stable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


class AbsentFieldError(PreImpactBundleError):
    """Numeric use of a field whose declared origin is ``absent``."""

    def __init__(self, message: str) -> None:
        super().__init__("absent_field", message)


class FieldOrigin(str, enum.Enum):
    """Where a field value came from; ``ABSENT`` means no value exists."""

    MEASURED = "measured"
    IDENTIFIED = "identified"
    PRESCRIBED = "prescribed"
    SYNTHETIC = "synthetic"
    ABSENT = "absent"


def fail(code: str, message: str) -> NoReturn:
    """Raise a classified contract error (never an ``assert``)."""
    raise PreImpactBundleError(code, message)


def finite_array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    """Copy strictly real numeric data of ``shape`` into a read-only array."""
    if isinstance(value, (str, bytes, Mapping)) or value is None:
        fail("type", f"{name} must be numeric, got {type(value).__name__}")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        fail("type", f"{name} must contain real numbers, not {raw.dtype}")
    if isinstance(value, bool) or (
        not isinstance(value, np.ndarray)
        and any(isinstance(item, (bool, np.bool_)) for item in raw.astype(object).flat)
    ):
        fail("type", f"{name} must not contain booleans")
    if raw.shape != shape:
        fail("shape", f"{name} must have shape {shape}, got {raw.shape}")
    array = np.array(raw, dtype=np.float64, copy=True)
    if not bool(np.all(np.isfinite(array))):
        fail("non_finite", f"{name} must be finite")
    array.setflags(write=False)
    return array


def finite_scalar(value: object, name: str, *, positive: bool = False) -> float:
    """Return a finite real scalar, optionally strictly positive."""
    result = float(finite_array(value, (), name))
    if positive and result <= 0.0:
        fail("out_of_range", f"{name} must be > 0")
    return result


def identifier(value: object, name: str) -> str:
    """Return a nonempty trimmed identifier string."""
    if not isinstance(value, str) or not value or value != value.strip():
        fail("identifier", f"{name} must be a nonempty trimmed string")
    return str(value)


def sha256_hex(value: object, name: str) -> str:
    """Return a lowercase 64-character SHA-256 hex digest."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        fail("hash", f"{name} must be a lowercase SHA-256 hex digest")
    return str(value)


def strictly_increasing(array: FloatArray, name: str) -> None:
    """Refuse repeated or decreasing samples."""
    if array.size >= 2 and not bool(np.all(np.diff(array) > 0.0)):
        fail("non_monotonic_time", f"{name} must be strictly increasing")


def proper_rotation(value: object, name: str) -> FloatArray:
    """Return a read-only proper orthonormal rotation (det = +1)."""
    rotation = finite_array(value, (3, 3), name)
    orthonormal = np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=ROTATION_TOLERANCE
    )
    determinant = float(np.linalg.det(rotation))
    if not orthonormal or abs(determinant - 1.0) > ROTATION_TOLERANCE:
        fail(
            "improper_rotation",
            f"{name} must be orthonormal with det=+1 (det={determinant:.12g})",
        )
    return rotation


def unit_quaternion(value: object, name: str) -> FloatArray:
    """Return a read-only (w, x, y, z) quaternion already of unit norm."""
    quaternion = finite_array(value, (4,), name)
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > QUATERNION_TOLERANCE:
        fail("non_unit_quaternion", f"{name} must be unit length (|q|={norm:.12g})")
    return quaternion


def unit_vector(value: object, name: str) -> FloatArray:
    """Return a read-only finite unit three-vector (never normalized)."""
    vector = finite_array(value, (3,), name)
    if abs(float(np.linalg.norm(vector)) - 1.0) > UNIT_VECTOR_TOLERANCE:
        fail("non_unit_vector", f"{name} must be a unit vector")
    return vector


def _scale(matrix: FloatArray) -> float:
    return max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)


def symmetric(value: object, size: int, name: str) -> FloatArray:
    """Return a read-only finite symmetric ``size``-square matrix."""
    matrix = finite_array(value, (size, size), name)
    tolerance = SYMMETRY_RELATIVE_TOLERANCE * _scale(matrix)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        fail("asymmetric", f"{name} must be symmetric")
    return matrix


def positive_definite(value: object, size: int, name: str) -> FloatArray:
    """Return a symmetric positive-definite matrix (Cholesky must succeed)."""
    matrix = symmetric(value, size, name)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        fail("not_positive_definite", f"{name} must be positive definite")
    return matrix


def positive_semidefinite(value: object, size: int, name: str) -> FloatArray:
    """Return a symmetric positive-semidefinite matrix."""
    matrix = symmetric(value, size, name)
    floor = -SYMMETRY_RELATIVE_TOLERANCE * _scale(matrix)
    if float(np.min(np.linalg.eigvalsh(matrix))) < floor:
        fail("not_positive_semidefinite", f"{name} must be positive semidefinite")
    return matrix


def com_inertia(value: object, name: str) -> FloatArray:
    """Return a physically realizable COM inertia: SPD plus triangle inequality."""
    inertia = positive_definite(value, 3, name)
    moments = np.linalg.eigvalsh(inertia)
    slack = SYMMETRY_RELATIVE_TOLERANCE * _scale(inertia)
    if float(moments[2] - moments[1] - moments[0]) > slack:
        fail("triangle_inequality", f"{name} principal moments violate I3<=I1+I2")
    return inertia


def reject_unknown(
    data: object, allowed: frozenset[str], name: str
) -> Mapping[str, Any]:
    """Require a string-keyed mapping without fields this version ignores."""
    if not isinstance(data, Mapping):
        fail("type", f"{name} must be an object")
    data = cast(Mapping[str, Any], data)
    unknown = set(data) - allowed
    if unknown:
        fail("unknown_field", f"{name} has unknown fields {sorted(unknown)}")
    missing = {key for key in allowed if key not in data}
    if missing:
        fail("missing_field", f"{name} is missing fields {sorted(missing)}")
    return data


@dataclass(frozen=True, eq=False)
class Quantity:
    """A value paired with its origin; ``ABSENT`` carries no value at all.

    Postconditions: present values are read-only float arrays; absent values
    raise :class:`AbsentFieldError` on ``.value``, ``float()`` or ``np.asarray``.
    Absent is never a zero placeholder.
    """

    origin: FieldOrigin
    raw: FloatArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.origin, FieldOrigin):
            fail("origin", "origin must be a FieldOrigin")
        if self.origin is FieldOrigin.ABSENT:
            if self.raw is not None:
                fail("origin", "an absent field must not carry a value")
            return
        if self.raw is None:
            fail("origin", f"a {self.origin.value} field requires a value")
        array = np.array(self.raw, dtype=np.float64, copy=True)
        if not bool(np.all(np.isfinite(array))):
            fail("non_finite", "quantity values must be finite")
        array.setflags(write=False)
        object.__setattr__(self, "raw", array)

    @classmethod
    def absent(cls) -> Quantity:
        """Explicitly absent field."""
        return cls(FieldOrigin.ABSENT, None)

    @property
    def is_absent(self) -> bool:
        return self.origin is FieldOrigin.ABSENT

    @property
    def value(self) -> FloatArray:
        """The read-only value; raises :class:`AbsentFieldError` when absent."""
        if self.raw is None:
            raise AbsentFieldError("field is absent; no numeric value exists")
        return self.raw

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> FloatArray:
        value = self.value
        return np.array(value, dtype=dtype) if dtype is not None else value

    def __float__(self) -> float:
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        value = None if self.raw is None else self.raw.tolist()
        return {"origin": self.origin.value, "value": value}


def parse_quantity(
    raw: object,
    name: str,
    shape: tuple[int, ...],
    *,
    required: bool = False,
) -> Quantity:
    """Parse ``{"origin", "value"}``; absent requires ``value`` to be null."""
    data = reject_unknown(raw, frozenset({"origin", "value"}), name)
    try:
        origin = FieldOrigin(data["origin"])
    except ValueError:
        fail("origin", f"{name}.origin {data['origin']!r} is not a declared origin")
    if origin is FieldOrigin.ABSENT:
        if data["value"] is not None:
            fail("origin", f"{name} is absent but carries a value")
        if required:
            fail("absent_required", f"{name} is required and may not be absent")
        return Quantity.absent()
    if data["value"] is None:
        fail("origin", f"{name} is {origin.value} but has no value")
    return Quantity(origin, finite_array(data["value"], shape, name))


def positive(quantity: Quantity, name: str) -> None:
    """Strict positivity check for a present scalar quantity."""
    if not quantity.is_absent and not float(quantity.value) > 0.0:
        fail("out_of_range", f"{name} must be > 0")


def nonnegative(quantity: Quantity, name: str) -> None:
    """Nonnegative check for a present quantity."""
    if not quantity.is_absent and bool(np.any(quantity.value < 0.0)):
        fail("out_of_range", f"{name} must be >= 0")


__all__ = [
    "AbsentFieldError",
    "FieldOrigin",
    "PreImpactBundleError",
    "Quantity",
]
