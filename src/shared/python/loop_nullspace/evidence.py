"""Evidence-axis and small value types for closed-loop snapshots (NSX-10, D2).

Every user-visible number is a :class:`QuantityRecord` carrying how it depends
on the allocation choice, which model and input produced it, and whether it was
validated.  ``INVARIANT`` means "identical for every admissible allocation,
given model X"; it never means "measured".  Pure numpy; no engine imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

SYMMETRY_RTOL = 1e-9
ROTATION_ATOL = 1e-9


class AllocationDependence(Enum):
    """How a quantity depends on the choice among admissible allocations."""

    INVARIANT = "INVARIANT"
    FAMILY_DEPENDENT = "FAMILY_DEPENDENT"
    UNAVAILABLE = "UNAVAILABLE"


class ValidationStatus(Enum):
    """Whether a quantity has passed its validity gate."""

    UNVALIDATED = "UNVALIDATED"
    VALIDATED = "VALIDATED"


def readonly_float(value: ArrayLike, name: str) -> Array:
    """Return a read-only float64 copy of ``value``.

    Raises:
        ValueError: If the array contains NaN or infinity.
    """
    arr = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    arr.flags.writeable = False
    return arr


def check_shape(arr: Array, shape: tuple[int, ...], name: str) -> None:
    """Raise ``ValueError`` unless ``arr.shape == shape``."""
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")


def check_rotation(rotation: Array, name: str) -> None:
    """Require a 3x3 orthonormal matrix with determinant +1."""
    check_shape(rotation, (3, 3), name)
    if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > ROTATION_ATOL:
        raise ValueError(f"{name} must be orthonormal (R^T R = I)")
    if abs(np.linalg.det(rotation) - 1.0) > ROTATION_ATOL:
        raise ValueError(f"{name} must have determinant +1 (proper rotation)")


def check_symmetric(mat: Array, name: str) -> None:
    """Require ``mat`` symmetric to a tolerance scaled by its magnitude."""
    scale = max(1.0, float(np.max(np.abs(mat))))
    if np.max(np.abs(mat - mat.T)) > SYMMETRY_RTOL * scale:
        raise ValueError(f"{name} must be symmetric")


@dataclass(frozen=True)
class ModelProvenance:
    """Which model produced a quantity (name, version, content hash)."""

    name: str
    version: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("ModelProvenance requires non-empty name and version")


@dataclass(frozen=True)
class InputProvenance:
    """Which input a quantity came from (kind, source id, free-text note)."""

    kind: str
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.kind or not self.source:
            raise ValueError("InputProvenance requires non-empty kind and source")


@dataclass(frozen=True)
class SpatialInertia:
    """Rigid-body inertia: mass, centre of mass and inertia about the COM.

    Raises:
        ValueError: If mass <= 0, the inertia is not symmetric positive
            semidefinite, or its principal moments violate the triangle
            inequality ``I1 <= I2 + I3``.
    """

    mass: float
    com: Array
    inertia: Array

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError(f"SpatialInertia mass must be positive, got {self.mass}")
        com = readonly_float(self.com, "SpatialInertia.com")
        inertia = readonly_float(self.inertia, "SpatialInertia.inertia")
        check_shape(com, (3,), "SpatialInertia.com")
        check_shape(inertia, (3, 3), "SpatialInertia.inertia")
        check_symmetric(inertia, "SpatialInertia.inertia")
        scale = max(1e-300, float(np.max(np.abs(inertia))))
        moments = np.sort(np.linalg.eigvalsh(0.5 * (inertia + inertia.T)))
        if moments[0] < -SYMMETRY_RTOL * scale:
            raise ValueError("SpatialInertia.inertia must be positive semidefinite")
        if moments[2] > moments[0] + moments[1] + SYMMETRY_RTOL * scale:
            raise ValueError(
                "SpatialInertia.inertia violates the triangle inequality "
                "(largest principal moment exceeds the sum of the other two)"
            )
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "com", com)
        object.__setattr__(self, "inertia", inertia)


@dataclass(frozen=True)
class GripFrame:
    """Grip frame: ``rotation`` is world <- grip, ``origin`` in world."""

    rotation: Array
    origin: Array

    def __post_init__(self) -> None:
        rotation = readonly_float(self.rotation, "GripFrame.rotation")
        origin = readonly_float(self.origin, "GripFrame.origin")
        check_rotation(rotation, "GripFrame.rotation")
        check_shape(origin, (3,), "GripFrame.origin")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "origin", origin)


@dataclass(frozen=True)
class FootContact:
    """A foot contact point: world position, unit normal and friction ``mu``."""

    name: str
    position: Array
    normal: Array
    mu: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FootContact requires a non-empty name")
        position = readonly_float(self.position, "FootContact.position")
        normal = readonly_float(self.normal, "FootContact.normal")
        check_shape(position, (3,), "FootContact.position")
        check_shape(normal, (3,), "FootContact.normal")
        if abs(float(np.linalg.norm(normal)) - 1.0) > ROTATION_ATOL:
            raise ValueError("FootContact.normal must be a unit vector")
        if not np.isfinite(self.mu) or self.mu <= 0.0:
            raise ValueError(f"FootContact.mu must be positive, got {self.mu}")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "mu", float(self.mu))


@dataclass(frozen=True)
class QuantityRecord:
    """A user-visible quantity with its evidence axes (ADR-0054 D2).

    ``hypotheses`` are allocation priors (the choice among admissible
    allocations).  ``conditioning`` is an orthogonal axis: declared hypotheses
    the value is conditioned on irrespective of allocation dependence, so an
    ``INVARIANT`` record may carry it.  Ids are free text here, exactly like
    ``hypotheses``; registry resolution happens at display time.

    Raises:
        TypeError: If ``conditioning`` is a bare string.
        ValueError: If units are empty, ``INVARIANT`` carries hypotheses,
            ``FAMILY_DEPENDENT`` carries none, ``UNAVAILABLE`` lacks a reason
            or has non-NaN values, or an available quantity is non-finite.
    """

    values: Array
    units: str
    allocation_dependence: AllocationDependence
    model_provenance: ModelProvenance
    input_provenance: InputProvenance
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    conventions: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    reason: str | None = None
    conditioning: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError("QuantityRecord requires units")
        values = np.array(self.values, dtype=np.float64, copy=True)
        dep = self.allocation_dependence
        if dep is AllocationDependence.UNAVAILABLE:
            if not self.reason:
                raise ValueError("UNAVAILABLE QuantityRecord requires a reason")
            if not np.all(np.isnan(values)):
                raise ValueError("UNAVAILABLE QuantityRecord values must be all NaN")
        else:
            if not np.all(np.isfinite(values)):
                raise ValueError("available QuantityRecord values must be finite")
            if dep is AllocationDependence.INVARIANT and self.hypotheses:
                raise ValueError("INVARIANT QuantityRecord must not carry hypotheses")
            if dep is AllocationDependence.FAMILY_DEPENDENT and not self.hypotheses:
                raise ValueError(
                    "FAMILY_DEPENDENT QuantityRecord requires at least one hypothesis"
                )
        if isinstance(self.conditioning, str):
            raise TypeError("QuantityRecord.conditioning must be a sequence of ids")
        conditioning = tuple(self.conditioning)
        if not all(isinstance(c, str) and c for c in conditioning):
            raise ValueError("QuantityRecord.conditioning ids must be non-empty str")
        values.flags.writeable = False
        object.__setattr__(self, "conditioning", conditioning)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "conventions", tuple(self.conventions))
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))


def family_range_record(
    values: Array,
    *,
    unavailable_reason: str | None,
    hypotheses: tuple[str, ...],
    units: str,
    model: ModelProvenance,
    source: InputProvenance,
    label: str,
) -> QuantityRecord:
    """A ``FAMILY_DEPENDENT`` range record, or ``UNAVAILABLE`` with a reason.

    ``values`` is the ``(frames, 2)`` ``[lower, upper]`` array.  When
    ``unavailable_reason`` is given the record carries all-NaN values of the
    same shape and that reason; otherwise it carries ``values`` and
    ``hypotheses``.  ``label`` is recorded as the record's convention.
    """
    common: dict[str, Any] = {
        "units": units,
        "model_provenance": model,
        "input_provenance": source,
        "conventions": (label,),
    }
    if unavailable_reason is not None:
        return QuantityRecord(
            values=np.full(values.shape, np.nan),
            allocation_dependence=AllocationDependence.UNAVAILABLE,
            reason=unavailable_reason,
            **common,
        )
    return QuantityRecord(
        values=values,
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        hypotheses=hypotheses,
        **common,
    )
