"""Regression-based segment inertia estimator.

Wave 2 of the anthropometrics EPIC (#4797). Foundation #4800
already exposes the canonical :class:`SegmentProperties` data
model; this module provides three closed-form inertia
computations and a builder that composes them into a fully
validated :class:`SegmentProperties` instance.

Methods
-------
``cylinder``
    Solid cylinder of given mass, length, and radius. Body axis
    is along x; principal moments::

        I_x        = m * r**2 / 2
        I_y = I_z  = m * (3 * r**2 + L**2) / 12

``ellipsoid``
    Solid ellipsoid with semi-axes a, b, c::

        I_x = m * (b**2 + c**2) / 5
        I_y = m * (a**2 + c**2) / 5
        I_z = m * (a**2 + b**2) / 5

``gyration_radii``
    de Leva-style radii of gyration. For a segment of length L and
    dimensionless gyration ratios ``k = (k_x, k_y, k_z)``::

        I_i = m * (k_i * L)**2

DRY reuse
---------
The cylinder and ellipsoid analytical formulas already exist in
:mod:`model_generation.inertia.primitives`
(``cylinder_inertia``, ``ellipsoid_inertia``). Those primitives
return component dicts; we delegate to them and pack the result
into a 3x3 :class:`numpy.ndarray`. The gyration-radii method has
no equivalent in :mod:`model_generation.inertia.primitives`, so
it is implemented here.

The composed :class:`SegmentProperties` carries its own
positive-definiteness and triangle-inequality validation, so any
inertia tensor that is non-physical for the supplied parameters
will be rejected at construction time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from model_generation.inertia.primitives import (
    cylinder_inertia as _primitive_cylinder_inertia,
)
from model_generation.inertia.primitives import (
    ellipsoid_inertia as _primitive_ellipsoid_inertia,
)

from anthropometrics.segment_properties import SegmentProperties

if TYPE_CHECKING:
    from anthropometrics._types import FloatArray


# --------------------------------------------------------------------------- #
# Internal validation helpers (DRY).                                          #
# --------------------------------------------------------------------------- #
def _require_positive(value: float, label: str) -> None:
    """Raise ``ValueError`` if *value* is not a strictly positive finite float."""
    if not (isinstance(value, (int, float)) and np.isfinite(value) and value > 0):
        raise ValueError(f"{label} must be a positive finite number, got {value!r}")


def _diag_tensor(ix: float, iy: float, iz: float) -> FloatArray:
    """Return a 3x3 diagonal inertia tensor as a float ndarray."""
    return np.diag(np.asarray([ix, iy, iz], dtype=float))


# --------------------------------------------------------------------------- #
# Inertia computations.                                                       #
# --------------------------------------------------------------------------- #
def inertia_from_cylinder(
    mass_kg: float,
    length_m: float,
    radius_m: float,
) -> FloatArray:
    """Return the inertia tensor of a solid cylinder about its CoM.

    The cylinder's body axis is **x**. Off-diagonal moments are zero.

    Parameters
    ----------
    mass_kg
        Segment mass (kg). Must be strictly positive.
    length_m
        Cylinder length along its axis (m). Must be strictly positive.
    radius_m
        Cylinder radius (m). Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        A ``(3, 3)`` symmetric, positive-definite inertia tensor in kg*m^2.

    Raises
    ------
    ValueError
        If any input is not a strictly positive finite float.
    """
    _require_positive(mass_kg, "mass_kg")
    _require_positive(length_m, "length_m")
    _require_positive(radius_m, "radius_m")

    primitive = _primitive_cylinder_inertia(
        float(mass_kg), float(radius_m), float(length_m), axis="x"
    )
    return _diag_tensor(primitive["ixx"], primitive["iyy"], primitive["izz"])


def inertia_from_ellipsoid(
    mass_kg: float,
    a_m: float,
    b_m: float,
    c_m: float,
) -> FloatArray:
    """Return the inertia tensor of a solid ellipsoid about its CoM.

    The ellipsoid has semi-axes ``a_m``, ``b_m``, ``c_m`` aligned with
    the x, y, z body axes respectively. Off-diagonal moments are zero.

    Parameters
    ----------
    mass_kg
        Segment mass (kg). Must be strictly positive.
    a_m, b_m, c_m
        Semi-axes (m). Each must be strictly positive.

    Returns
    -------
    numpy.ndarray
        A ``(3, 3)`` symmetric, positive-definite inertia tensor in kg*m^2.

    Raises
    ------
    ValueError
        If any input is not a strictly positive finite float.
    """
    _require_positive(mass_kg, "mass_kg")
    _require_positive(a_m, "a_m")
    _require_positive(b_m, "b_m")
    _require_positive(c_m, "c_m")

    primitive = _primitive_ellipsoid_inertia(
        float(mass_kg), float(a_m), float(b_m), float(c_m)
    )
    return _diag_tensor(primitive["ixx"], primitive["iyy"], primitive["izz"])


def inertia_from_gyration_radii(
    mass_kg: float,
    length_m: float,
    gyration_ratios: tuple[float, float, float],
) -> FloatArray:
    """Return the inertia tensor implied by de-Leva-style radius-of-gyration ratios.

    Given dimensionless ratios ``k = (k_x, k_y, k_z)`` and a segment
    length ``L``, the principal moments about the segment CoM are
    ``I_i = m * (k_i * L) ** 2``.

    Parameters
    ----------
    mass_kg
        Segment mass (kg). Must be strictly positive.
    length_m
        Segment length (m). Must be strictly positive.
    gyration_ratios
        Three dimensionless ratios ``(k_x, k_y, k_z)``. Each must be
        strictly positive and finite.

    Returns
    -------
    numpy.ndarray
        A ``(3, 3)`` symmetric, positive-definite inertia tensor in kg*m^2.

    Raises
    ------
    ValueError
        If any input is not a strictly positive finite float, or if
        ``gyration_ratios`` does not have exactly three elements.
    """
    _require_positive(mass_kg, "mass_kg")
    _require_positive(length_m, "length_m")

    ratios = tuple(gyration_ratios)
    if len(ratios) != 3:
        raise ValueError(
            f"gyration_ratios must have exactly 3 elements, got {len(ratios)}"
        )
    for axis_label, ratio in zip(("k_x", "k_y", "k_z"), ratios, strict=True):
        _require_positive(ratio, f"gyration_ratios[{axis_label}]")

    moments = tuple(float(mass_kg) * (float(k) * float(length_m)) ** 2 for k in ratios)
    return _diag_tensor(*moments)


# --------------------------------------------------------------------------- #
# Builder.                                                                    #
# --------------------------------------------------------------------------- #
_AllowedMethod = Literal["cylinder", "ellipsoid", "gyration_radii"]


def _inertia_for_method(
    method: _AllowedMethod,
    *,
    mass_kg: float,
    length_m: float,
    method_params: dict[str, Any],
) -> FloatArray:
    """Dispatch to the requested inertia computation.

    Parameters
    ----------
    method
        One of ``"cylinder"``, ``"ellipsoid"``, ``"gyration_radii"``.
    mass_kg, length_m
        Segment scalars forwarded to the chosen method.
    method_params
        Method-specific parameters:

        * ``cylinder``      — ``{"radius_m": float}``
        * ``ellipsoid``     — ``{"a_m": float, "b_m": float, "c_m": float}``
        * ``gyration_radii`` — ``{"gyration_ratios": tuple[float, float, float]}``

    Raises
    ------
    ValueError
        If *method* is not one of the supported strings, or
        if *method_params* is missing required keys.
    """
    if method == "cylinder":
        try:
            radius_m = method_params["radius_m"]
        except KeyError as exc:  # pragma: no cover - re-raised below
            raise ValueError(
                "cylinder method requires method_params['radius_m']"
            ) from exc
        return inertia_from_cylinder(mass_kg, length_m, float(radius_m))

    if method == "ellipsoid":
        missing = [k for k in ("a_m", "b_m", "c_m") if k not in method_params]
        if missing:
            raise ValueError(f"ellipsoid method requires method_params keys {missing}")
        return inertia_from_ellipsoid(
            mass_kg,
            float(method_params["a_m"]),
            float(method_params["b_m"]),
            float(method_params["c_m"]),
        )

    if method == "gyration_radii":
        try:
            gyration_ratios = method_params["gyration_ratios"]
        except KeyError as exc:  # pragma: no cover - re-raised below
            raise ValueError(
                "gyration_radii method requires method_params['gyration_ratios']"
            ) from exc
        return inertia_from_gyration_radii(mass_kg, length_m, tuple(gyration_ratios))

    raise ValueError(
        "method must be one of 'cylinder', 'ellipsoid', "
        f"'gyration_radii'; got {method!r}"
    )


def build_segment_properties_with_inertia(
    name: str,
    body_part_id: str,
    *,
    mass_kg: float,
    length_m: float,
    com_xyz_m: np.ndarray,
    method: _AllowedMethod,
    method_params: dict[str, Any],
    source_method: str,
    source_subject_height_m: float,
    source_subject_mass_kg: float,
    proximal_marker: str | None = None,
    distal_marker: str | None = None,
) -> SegmentProperties:
    """Compose a :class:`SegmentProperties` with an inertia tensor.

    The inertia tensor is computed by *method* using *method_params*.
    All other arguments are forwarded directly to the
    :class:`SegmentProperties` constructor, which performs the full
    physical-realisability validation suite.

    Parameters
    ----------
    name, body_part_id, source_method
        Identifier strings; must be non-empty.
    mass_kg, length_m, source_subject_height_m, source_subject_mass_kg
        Strictly positive finite scalars in SI units.
    com_xyz_m
        Center-of-mass vector with shape ``(3,)``.
    method
        ``"cylinder"``, ``"ellipsoid"``, or ``"gyration_radii"``.
    method_params
        Method-specific keyword payload (see
        :func:`_inertia_for_method`).
    proximal_marker, distal_marker
        Optional non-empty marker labels.

    Returns
    -------
    SegmentProperties
        Fully validated, frozen segment with the computed inertia.

    Raises
    ------
    ValueError
        If any precondition fails — propagated from the inertia
        computation or from the :class:`SegmentProperties`
        constructor.
    """
    inertia_tensor = _inertia_for_method(
        method,
        mass_kg=mass_kg,
        length_m=length_m,
        method_params=method_params,
    )

    return SegmentProperties(
        name=name,
        body_part_id=body_part_id,
        length_m=float(length_m),
        proximal_marker=proximal_marker,
        distal_marker=distal_marker,
        mass_kg=float(mass_kg),
        com_xyz_m=np.asarray(com_xyz_m, dtype=float),
        inertia_tensor=inertia_tensor,
        source_method=source_method,
        source_subject_height_m=float(source_subject_height_m),
        source_subject_mass_kg=float(source_subject_mass_kg),
    )
