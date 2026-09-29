"""Canonical :class:`SegmentProperties` dataclass.

A :class:`SegmentProperties` instance fully describes the
anthropometric properties of a single rigid body segment in SI
units. Every instance is validated on construction against a
strict set of physical-realisability invariants (Design by
Contract). Once an instance exists, it is guaranteed to be
self-consistent and physically plausible.

References for published ratios used by downstream estimators
(public scientific literature):

* de Leva, P. (1996). *Adjustments to Zatsiorsky-Seluyanov's
  segment inertia parameters.* Journal of Biomechanics.
* Dempster, W. T. (1955). *Space requirements of the seated
  operator.* WADC Technical Report.
* Zatsiorsky, V. M. (2002). *Kinetics of Human Motion.*

This module itself contains no ratios — it only enforces the
contract every estimator must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ._types import INERTIA_NUMERIC_TOL

if TYPE_CHECKING:
    from ._types import FloatArray


# --------------------------------------------------------------------------- #
# Validation helpers (module-private, DRY).                                   #
# --------------------------------------------------------------------------- #
def _require_non_empty_str(value: object, label: str) -> None:
    """Raise ``ValueError`` if *value* is not a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")


def _require_positive(value: float, label: str) -> None:
    """Raise ``ValueError`` if *value* is not strictly positive."""
    if not (isinstance(value, (int, float)) and np.isfinite(value) and value > 0):
        raise ValueError(f"{label} must be a positive finite number, got {value!r}")


def _coerce_array(value: object, label: str, shape: tuple[int, ...]) -> FloatArray:
    """Return *value* as a finite ndarray of *shape* or raise."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


def _validate_inertia_tensor(tensor: FloatArray) -> None:
    """Enforce all physical invariants on a 3x3 inertia tensor.

    Raises
    ------
    ValueError
        If the tensor is not symmetric, not positive-definite, or
        violates the triangle inequality on its principal moments.
    """
    if not np.allclose(tensor, tensor.T, atol=INERTIA_NUMERIC_TOL):
        raise ValueError(
            "inertia_tensor must be symmetric "
            f"(max asymmetry={np.max(np.abs(tensor - tensor.T)):.3e})"
        )

    eigenvalues = np.linalg.eigvalsh(tensor)
    if np.any(eigenvalues <= 0):
        raise ValueError(
            "inertia_tensor must be positive-definite "
            f"(eigenvalues={eigenvalues.tolist()})"
        )

    ix, iy, iz = sorted(float(e) for e in eigenvalues)
    pairs = (
        ("Ix+Iy >= Iz", ix + iy, iz),
        ("Iy+Iz >= Ix", iy + iz, ix),
        ("Ix+Iz >= Iy", ix + iz, iy),
    )
    for label, lhs, rhs in pairs:
        if lhs + INERTIA_NUMERIC_TOL < rhs:
            raise ValueError(
                "inertia_tensor violates triangle inequality on "
                f"principal moments: {label} (got {lhs:.6e} < {rhs:.6e}); "
                f"sorted eigenvalues=({ix:.6e}, {iy:.6e}, {iz:.6e})"
            )


# --------------------------------------------------------------------------- #
# Public dataclass.                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SegmentProperties:
    """Anthropometric properties of a single body segment, SI units only.

    Invariants (all enforced in :meth:`__post_init__`)
    --------------------------------------------------
    * ``name``, ``body_part_id``, ``source_method`` are non-empty strings.
    * ``length_m``, ``mass_kg``, ``source_subject_height_m``, and
      ``source_subject_mass_kg`` are strictly positive finite floats.
    * ``com_xyz_m`` has shape ``(3,)`` and ``|com| <= 2 * length_m``.
    * ``inertia_tensor`` has shape ``(3, 3)``, is symmetric to
      ``1e-9``, is positive-definite, and its principal moments
      satisfy the triangle inequality.

    The dataclass is frozen — mutate by constructing a new
    instance via :func:`dataclasses.replace`.
    """

    name: str
    body_part_id: str
    length_m: float
    proximal_marker: str | None
    distal_marker: str | None
    mass_kg: float
    com_xyz_m: FloatArray = field()
    inertia_tensor: FloatArray = field()
    source_method: str
    source_subject_height_m: float
    source_subject_mass_kg: float

    def __post_init__(self) -> None:
        # 1. String identifiers.
        _require_non_empty_str(self.name, "name")
        _require_non_empty_str(self.body_part_id, "body_part_id")
        _require_non_empty_str(self.source_method, "source_method")

        # 2. Positive scalars.
        _require_positive(self.length_m, "length_m")
        _require_positive(self.mass_kg, "mass_kg")
        _require_positive(self.source_subject_height_m, "source_subject_height_m")
        _require_positive(self.source_subject_mass_kg, "source_subject_mass_kg")

        # 3. Optional marker labels — if present, must be non-empty.
        for label, value in (
            ("proximal_marker", self.proximal_marker),
            ("distal_marker", self.distal_marker),
        ):
            if value is not None:
                _require_non_empty_str(value, label)

        # 4. Center-of-mass vector, normalised to a contiguous ndarray.
        com = _coerce_array(self.com_xyz_m, "com_xyz_m", (3,))
        com_norm = float(np.linalg.norm(com))
        if com_norm > 2.0 * self.length_m:
            raise ValueError(
                "com_xyz_m magnitude exceeds 2 * length_m "
                f"(|com|={com_norm:.6e}, length={self.length_m:.6e})"
            )
        object.__setattr__(self, "com_xyz_m", com)

        # 5. Inertia tensor — full physical invariant suite.
        tensor = _coerce_array(self.inertia_tensor, "inertia_tensor", (3, 3))
        _validate_inertia_tensor(tensor)
        object.__setattr__(self, "inertia_tensor", tensor)
