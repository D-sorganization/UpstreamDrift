"""A drained shear cell driven through **F1's own constitutive model**.

Issue #8733 section 6
---------------------

ADR-0033 chose MPM over SPH because F1 shares its constitutive model with
the F2 reference (``SolverImplicitMPM``), so "the material calibration is
done once and carries between tiers".  Nothing had been calibrated.  The
harness in this package
(:class:`~bunkershot3d.calibration.optimizer.CalibrationOptimizer`,
:class:`~bunkershot3d.calibration.drained_shear_cell.DrainedShearCellExperiment`)
fitted *backend contact-model* parameters and contained no reference to
:class:`~bunkershot3d.solvers.mpm.constitutive.SandContinuum` at all, so
F1's friction angle stayed borrowed from the Quikrete analogue (#7999).

This module closes that gap on the quantity a shear cell actually
constrains: the friction angle.  It runs a **drained plane-strain
biaxial compression test** -- constant lateral stress, monotonic axial
compression -- as an *element* test on
:meth:`~bunkershot3d.solvers.mpm.constitutive.SandContinuum.project`, the
same return mapping every material point in an F1 solve is put through,
and the same one the F2 reference uses.  Nothing here is a
discretisation: what is fitted is the material, which is precisely the
part ADR-0033 says carries between the tiers.

What is fitted, and what it means
---------------------------------

**The target is a declared number, not a measurement of bunker sand.**
``target_phi_peak = 35 deg`` and ``target_phi_res = 30 deg`` are the
values this package's harness has always carried; issue #8616 found no
published measurement of any quantity this tier produces, and #8610
records that no bunker-sand friction angle exists in the literature
either.  Fitting F1 to them makes the model **self-consistent with a
stated experiment**.  It replaces "borrowed from a hardware-store
analogue" with "fitted to a declared target".  It is **not** validation:
NASA-STD-7009B validation for this package stays at **0 of 4**, every F1
verdict stays
:attr:`~bunkershot3d.solvers.envelope.EnvelopeStatus.BEYOND_VALIDATION`,
and ``MAX_VALIDATED_SPEED_M_S`` stays 1.44 m/s.

Two findings fall out of the model rather than out of the fit
-------------------------------------------------------------

1. **The angle F1 enforces is not the angle it is handed.**
   :func:`~bunkershot3d.solvers.mpm.constitutive.drucker_prager_alpha`
   fits the inner (compressive-meridian) cone to Mohr-Coulomb in *three*
   dimensions, but F1's yield surface is written on the **two** in-plane
   principal Kirchhoff stresses.  The plane-strain limit is therefore
   Mohr-Coulomb at ``phi* = asin(sqrt(2) alpha)``, which
   :mod:`bunkershot3d.solvers.mpm.limit_states` already derives and
   reports as 31.94 deg for the 34 deg every preset carries.  A shear
   cell measures ``phi*``, so the friction angle that reproduces a target
   ``phi*`` is *larger* than the target.

2. **The model has no peak-to-residual softening.**  Rate-independent
   perfect plasticity puts the stress on the cone and leaves it there, so
   ``phi_peak == phi_res`` identically.  The harness's objective asks for
   a 5 deg gap between them, and no parameter value can produce one.
   :attr:`F1DrainedShearCellExperiment.irreducible_residual_deg2` states
   how much of the final residual is that structural gap rather than a
   miss.

The elastic constants are not identifiable here
-----------------------------------------------

The drained limit ratio is ``q/p``, a ratio of two stresses that are both
linear in the elastic constants, so the shear modulus cancels *exactly*.
:attr:`F1DrainedShearCellExperiment.calibrated_parameters` therefore
declares only the friction angle, and constructing the experiment with
``include_shear_modulus=True`` exists so that
:meth:`~bunkershot3d.calibration.optimizer.CalibrationOptimizer.check_sensitivity`
can demonstrate the refusal rather than the point being left to
documentation.  The shear modulus keeps its Hardin & Richart (1963)
:attr:`~bunkershot3d.sand.provenance.ProvenanceBasis.ESTIMATED` label.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..sand import SandState
from ..sand.presets import PlayingCondition, playing_condition
from ..solvers.exceptions import CalibrationError
from ..solvers.mpm.constitutive import (
    PLANE_STRAIN_DIMENSION,
    SandContinuum,
    drucker_prager_alpha,
)

__all__ = [
    "F1_SHEAR_CELL_CONFINING_STRESSES_PA",
    "F1_SHEAR_CELL_TARGET_NOTE",
    "PLANE_STRAIN_CROSSOVER_ANGLE_DEG",
    "BiaxialPoint",
    "F1DrainedShearCellExperiment",
    "MohrCoulombEnvelope",
    "drained_biaxial_path",
    "friction_angle_for_plane_strain_angle_deg",
    "plane_strain_friction_angle_deg",
]


F1_SHEAR_CELL_CONFINING_STRESSES_PA: tuple[float, ...] = (2.0e3, 1.0e4, 5.0e4)
"""Lateral stresses the envelope is fitted over.

Three cells spanning the range a bunker bed and a laboratory apparatus
between them cover: 2 kPa is roughly the overburden at the bottom of a
125 mm USGA floor, 50 kPa is a normal laboratory consolidation.  A
cohesionless Drucker-Prager cone gives the same angle at every one of
them, which the test suite checks -- the spread is there so that a
*cohesive* tip shows up as an intercept instead of being folded into the
angle."""

F1_SHEAR_CELL_TARGET_NOTE = (
    "The drained-shear-cell targets (peak 35 deg, residual 30 deg) are the "
    "declared numbers this package's calibration harness has always carried. "
    "They are NOT a measurement of golf bunker sand: issue #8610 records that "
    "no published bunker-sand friction angle was found, and issue #8616 that "
    "no published measurement exists for any quantity F1 produces. Fitting "
    "F1's constitutive model to them makes it self-consistent with a stated "
    "experiment; it replaces a value borrowed from the Quikrete hardware-store "
    "analogue with a value fitted to a declared target. It does not validate "
    "the model. NASA-STD-7009B validation stays at 0 of 4, F1 stays "
    "BEYOND_VALIDATION, and MAX_VALIDATED_SPEED_M_S stays 1.44 m/s."
)
"""The honesty boundary, in the code rather than only in the pull request."""

_DEFAULT_AXIAL_STRAIN = -0.04
"""Axial strain the test is driven to; compressive, hence negative."""

_DEFAULT_INCREMENTS = 80
"""Load steps along the path. The plateau is reached long before the end."""

_BISECTION_ITERATIONS = 60
"""Bisection steps for the lateral strain at each load increment.

Sixty halvings takes a bracket of order ``1e-3`` strain down to ``1e-21``,
which is below the last representable bit of a strain of that size, so the
drained condition is held to machine precision and the measured angle does
not inherit a tolerance from the solve."""

_MIN_CONFINEMENTS = 2
"""Distinct lateral stresses needed before an envelope can be fitted."""

_SIN_PHI_CEILING = 1.0 - 1.0e-12
"""``sqrt(2) alpha`` above this has no usable ``asin``."""

PLANE_STRAIN_CROSSOVER_ANGLE_DEG = math.degrees(math.asin(3.0 - 4.0 / math.sqrt(3.0)))
"""Where ``phi*`` crosses ``phi``: 43.68 deg.

``sin(phi*) = sqrt(2) alpha = (4 / sqrt(3)) sin(phi) / (3 - sin(phi))``
equals ``sin(phi)`` when ``3 - sin(phi) = 4 / sqrt(3)``.  **Below** this
angle the plane-strain cone is softer than the angle it was handed --
34 deg gives 31.94 deg, which is the regime every sand preset and every
plausible calibration sits in -- and above it the cone is stronger.  The
crossover is stated rather than assumed because "plane strain is softer"
is only true on one side of it."""


# --------------------------------------------------------------- the angle


def plane_strain_friction_angle_deg(friction_angle_deg: float) -> float:
    """The Mohr-Coulomb angle F1 actually enforces, ``phi*``.

    F1's yield surface is ``||dev(tau)|| + alpha tr(tau) <= 0`` with both
    operators taken over the **two** in-plane principal Kirchhoff
    stresses.  For a two-dimensional state ``||dev(tau)|| = |tau_1 -
    tau_2| / sqrt(2)``, so the limit condition is Mohr-Coulomb with
    ``sin(phi*) = sqrt(2) alpha`` -- the same identity
    :class:`~bunkershot3d.solvers.mpm.limit_states.RankineLimits` derives
    for the passive-thrust case.

    ``phi*`` is strictly below the angle handed to
    :func:`~bunkershot3d.solvers.mpm.constitutive.drucker_prager_alpha`,
    because that function fits the inner cone to Mohr-Coulomb in three
    dimensions.  For the 34 deg every sand preset carries, ``phi*`` is
    31.94 deg.

    Args:
        friction_angle_deg: The angle handed to the cone, in degrees.

    Returns:
        The plane-strain equivalent friction angle in degrees.

    Raises:
        CalibrationError: If the angle is unusable, or if its cone is so
            steep that ``sqrt(2) alpha`` leaves the domain of ``asin``.
    """
    sin_phi_star = math.sqrt(2.0) * drucker_prager_alpha(friction_angle_deg)
    if sin_phi_star >= _SIN_PHI_CEILING:
        raise CalibrationError(
            f"a friction angle of {friction_angle_deg!r} deg gives sqrt(2) alpha "
            f"= {sin_phi_star!r}, which is not the sine of any angle: the "
            "plane-strain cone has closed. Keep the search below this angle "
            "rather than letting the objective return a nan"
        )
    return math.degrees(math.asin(sin_phi_star))


def friction_angle_for_plane_strain_angle_deg(plane_strain_angle_deg: float) -> float:
    """Invert :func:`plane_strain_friction_angle_deg` in closed form.

    ``sin(phi*) = sqrt(2) alpha`` and ``alpha = sqrt(2/3) 2 s / (3 - s)``
    with ``s = sin(phi)``, so
    ``s = 3 sin(phi*) / (4 / sqrt(3) + sin(phi*))``.

    This is the *answer* the incremental element test has to find on its
    own; it is exposed so that a test can check the simulation against it
    rather than the two being written from one expression.

    Args:
        plane_strain_angle_deg: The target ``phi*`` in degrees.

    Returns:
        The friction angle in degrees that produces it.

    Raises:
        CalibrationError: If the target is not an angle in ``(0, 90)``.
    """
    angle = float(plane_strain_angle_deg)
    if not math.isfinite(angle) or not 0.0 < angle < 90.0:
        raise CalibrationError(
            "plane-strain friction angle must lie strictly between 0 and 90 "
            f"deg, got {plane_strain_angle_deg!r} deg"
        )
    sin_star = math.sin(math.radians(angle))
    sin_phi = 3.0 * sin_star / (4.0 / math.sqrt(3.0) + sin_star)
    return math.degrees(math.asin(sin_phi))


# ------------------------------------------------------------ the element


@dataclass(frozen=True, slots=True)
class BiaxialPoint:
    """One load increment of the drained plane-strain compression test.

    Stresses are reported **compression positive**, the soil-mechanics
    convention, which is the opposite of the continuum sign the return
    map works in.

    Attributes:
        axial_strain: Hencky strain along the loading axis, negative.
        axial_stress_pa: Major principal stress ``sigma_1``.
        lateral_stress_pa: Minor principal stress ``sigma_3``; the
            controlled quantity of a drained test.
        volumetric_strain: ``tr(eps)`` of the elastic strain.
        yielded: Whether the return map moved this state.
    """

    axial_strain: float
    axial_stress_pa: float
    lateral_stress_pa: float
    volumetric_strain: float
    yielded: bool

    @property
    def mean_stress_pa(self) -> float:
        """``p = (sigma_1 + sigma_3) / 2``, the Mohr circle centre."""
        return 0.5 * (self.axial_stress_pa + self.lateral_stress_pa)

    @property
    def deviator_stress_pa(self) -> float:
        """``q = (sigma_1 - sigma_3) / 2``, the Mohr circle radius."""
        return 0.5 * (self.axial_stress_pa - self.lateral_stress_pa)

    @property
    def stress_ratio(self) -> float:
        """``q / p``; equals ``sin(phi_mobilised)`` for a cohesionless sand.

        Raises:
            CalibrationError: If the mean stress has reached zero, where
                the ratio is not defined.
        """
        mean = self.mean_stress_pa
        if mean <= 0.0:
            raise CalibrationError(
                f"mean stress is {mean!r} Pa, so no stress ratio exists at this "
                "point; the cell has gone into tension"
            )
        return self.deviator_stress_pa / mean

    @property
    def mobilised_friction_deg(self) -> float:
        """``asin(q / p)`` in degrees, the mobilised friction angle.

        Raises:
            CalibrationError: If the ratio is outside ``[-1, 1]``.
        """
        ratio = self.stress_ratio
        if not -1.0 <= ratio <= 1.0:
            raise CalibrationError(
                f"stress ratio {ratio!r} is not the sine of any angle"
            )
        return math.degrees(math.asin(ratio))


def _principal_cauchy_pa(
    material: SandContinuum, lateral_strain: float, axial_strain: float
) -> tuple[NDArray[np.float64], bool]:
    """Return-map one strain state and give back its Cauchy stresses.

    Args:
        material: The continuum.
        lateral_strain: Trial principal Hencky strain across the cell.
        axial_strain: Trial principal Hencky strain along the load axis.

    Returns:
        ``(principal_cauchy, yielded)`` with the stress in the continuum
        sign convention (compression negative).
    """
    trial = np.array([[float(lateral_strain), float(axial_strain)]])
    projected, yielded, capped = material.project(trial)
    stress = material.cauchy_from_hencky(projected)[0]
    return stress, bool(yielded[0] or capped[0])


def _solve_lateral_strain(
    material: SandContinuum,
    *,
    axial_strain: float,
    confining_stress_pa: float,
    seed: float,
    bracket_m: float,
) -> float:
    """Find the lateral strain that holds the drained lateral stress.

    The lateral Cauchy stress is monotone increasing in the lateral
    strain both inside and on the cone, so a bisection is exact and needs
    no derivative of the return map -- which is what makes this an
    element *test* rather than an inversion of the yield function.

    Args:
        material: The continuum.
        axial_strain: The imposed axial strain for this increment.
        confining_stress_pa: The lateral stress to hold, compression
            positive.
        seed: Previous increment's lateral strain, the bracket centre.
        bracket_m: Half-width of the initial bracket, in strain.

    Returns:
        The lateral strain.

    Raises:
        CalibrationError: If the drained condition cannot be bracketed.
    """

    def residual(lateral: float) -> float:
        stress, _ = _principal_cauchy_pa(material, lateral, axial_strain)
        return float(stress[0]) + confining_stress_pa

    low = seed - bracket_m
    high = seed + bracket_m
    low_value = residual(low)
    high_value = residual(high)
    for _ in range(60):
        if low_value <= 0.0 <= high_value:
            break
        low -= bracket_m
        high += bracket_m
        low_value = residual(low)
        high_value = residual(high)
    else:
        raise CalibrationError(
            "could not bracket the drained lateral stress "
            f"{confining_stress_pa!r} Pa at axial strain {axial_strain!r}; the "
            "cell has no state holding that confinement"
        )

    for _ in range(_BISECTION_ITERATIONS):
        middle = 0.5 * (low + high)
        value = residual(middle)
        if value <= 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def drained_biaxial_path(
    material: SandContinuum,
    *,
    confining_stress_pa: float,
    axial_strain: float = _DEFAULT_AXIAL_STRAIN,
    n_increments: int = _DEFAULT_INCREMENTS,
) -> tuple[BiaxialPoint, ...]:
    """Run one drained plane-strain compression test on the constitutive model.

    The cell is consolidated isotropically to ``confining_stress_pa``,
    then compressed along one principal axis in equal Hencky-strain
    increments while the lateral Cauchy stress is held at the
    consolidation value by a bisection on the lateral strain.  Every
    increment goes through
    :meth:`~bunkershot3d.solvers.mpm.constitutive.SandContinuum.project`,
    so what is being measured is F1's return mapping and nothing else.

    Args:
        material: The continuum under test.
        confining_stress_pa: Lateral stress held through the test,
            compression positive.
        axial_strain: Total axial Hencky strain, negative (compression).
        n_increments: Load steps along the path.

    Returns:
        One :class:`BiaxialPoint` per increment, in order.

    Raises:
        CalibrationError: If the confining stress is not compressive, the
            axial strain is not compressive, or the increment count is
            not positive.
    """
    if not isinstance(material, SandContinuum):
        raise CalibrationError(
            f"expected a SandContinuum, got {type(material).__name__}"
        )
    confining = float(confining_stress_pa)
    if not math.isfinite(confining) or confining <= 0.0:
        raise CalibrationError(
            "confining_stress_pa must be positive -- a drained shear cell is "
            f"consolidated in compression -- got {confining_stress_pa!r}"
        )
    total_axial = float(axial_strain)
    if not math.isfinite(total_axial) or total_axial >= 0.0:
        raise CalibrationError(
            "axial_strain must be negative: the cell is compressed, and an "
            f"extensional path measures a different limit, got {axial_strain!r}"
        )
    steps = int(n_increments)
    if steps < 1:
        raise CalibrationError(f"n_increments must be positive, got {n_increments!r}")

    bulk_term = (
        2.0 * material.shear_modulus_pa
        + PLANE_STRAIN_DIMENSION * material.lame_lambda_pa
    )
    # Isotropic consolidation: tr(tau) = -d sigma_3, deviator zero.
    volumetric = -PLANE_STRAIN_DIMENSION * confining / bulk_term
    if volumetric <= material.cap_volumetric_strain:
        raise CalibrationError(
            f"consolidating this sand to {confining:.4g} Pa needs a volumetric "
            f"strain of {volumetric:.4g}, which is at or past its compressive "
            f"cap of {material.cap_volumetric_strain:.4g}: the cell would be "
            "measuring plastic compaction of the packing, not the friction "
            "cone. Lower the confining stress or raise the shear modulus. "
            "This is a raise because the capped path returns a *negative* "
            "fitted angle, which would otherwise enter a calibration as a "
            "number rather than as a failure (issue #7999)."
        )
    lateral_strain = volumetric / PLANE_STRAIN_DIMENSION
    axial_seed = lateral_strain
    # The bracket has to span both the elastic scale of the confinement and
    # the plastic lateral expansion the cone drives, so it carries a floor
    # that does not vanish with a stiff material.
    bracket = max(abs(volumetric), 1.0e-3)

    increment = total_axial / steps
    points: list[BiaxialPoint] = []
    current_axial = axial_seed
    for _ in range(steps):
        current_axial += increment
        lateral_strain = _solve_lateral_strain(
            material,
            axial_strain=current_axial,
            confining_stress_pa=confining,
            seed=lateral_strain,
            bracket_m=bracket,
        )
        stress, yielded = _principal_cauchy_pa(material, lateral_strain, current_axial)
        projected, _, _ = material.project(np.array([[lateral_strain, current_axial]]))
        points.append(
            BiaxialPoint(
                axial_strain=float(current_axial),
                axial_stress_pa=float(-stress[1]),
                lateral_stress_pa=float(-stress[0]),
                volumetric_strain=float(projected[0].sum()),
                yielded=yielded,
            )
        )
    return tuple(points)


# ------------------------------------------------------------- the envelope


@dataclass(frozen=True, slots=True)
class MohrCoulombEnvelope:
    """A straight failure envelope fitted the way a laboratory fits one.

    Several cells at different consolidation stresses give several Mohr
    circles; the envelope tangent to all of them satisfies
    ``q = c cos(phi) + p sin(phi)`` (the Lambe ``p-q`` line).  A least
    squares fit of ``q`` against ``p`` therefore returns ``sin(phi)`` as
    its slope and ``c cos(phi)`` as its intercept, which is why the
    intercept is not forced through the origin: a damp sand has a real
    cohesive cone tip and folding it into the angle would overstate the
    friction.

    Attributes:
        friction_angle_deg: ``asin(slope)``.
        cohesion_pa: ``intercept / cos(phi)``.
        slope: Fitted ``dq/dp``.
        intercept_pa: Fitted ``q`` at ``p = 0``.
        n_points: Number of cells in the fit.
    """

    friction_angle_deg: float
    cohesion_pa: float
    slope: float
    intercept_pa: float
    n_points: int

    @classmethod
    def from_points(
        cls,
        mean_stress_pa: NDArray[np.float64],
        deviator_stress_pa: NDArray[np.float64],
    ) -> MohrCoulombEnvelope:
        """Fit the envelope through ``(p, q)`` pairs.

        Args:
            mean_stress_pa: ``(n,)`` Mohr circle centres.
            deviator_stress_pa: ``(n,)`` Mohr circle radii.

        Returns:
            The fitted envelope.

        Raises:
            CalibrationError: If fewer than two distinct centres are
                supplied, or if the fitted slope is not the sine of an
                angle.
        """
        centres = np.asarray(mean_stress_pa, dtype=np.float64).reshape(-1)
        radii = np.asarray(deviator_stress_pa, dtype=np.float64).reshape(-1)
        if centres.shape != radii.shape:
            raise CalibrationError(
                f"p and q must have the same shape, got {centres.shape!r} and "
                f"{radii.shape!r}"
            )
        if centres.size < _MIN_CONFINEMENTS or np.unique(centres).size < (
            _MIN_CONFINEMENTS
        ):
            raise CalibrationError(
                "a Mohr-Coulomb envelope needs at least two distinct "
                f"consolidation stresses, got {centres.size} point(s) at "
                f"{np.unique(centres).size} distinct stress(es); fitting a line "
                "through one circle would report an angle chosen by the "
                "intercept convention rather than by the material"
            )
        slope, intercept = np.polyfit(centres, radii, 1)
        if not -1.0 < float(slope) < 1.0:
            raise CalibrationError(
                f"the fitted p-q slope {slope!r} is not the sine of any angle"
            )
        angle_rad = math.asin(float(slope))
        return cls(
            friction_angle_deg=math.degrees(angle_rad),
            cohesion_pa=float(intercept) / math.cos(angle_rad),
            slope=float(slope),
            intercept_pa=float(intercept),
            n_points=int(centres.size),
        )


# ----------------------------------------------------------- the experiment


class F1DrainedShearCellExperiment:
    """Drives F1's constitutive model through the harness's shear-cell targets.

    Shaped to the contract
    :class:`~bunkershot3d.calibration.optimizer.CalibrationOptimizer`
    already reads: ``target_phi_peak`` / ``target_phi_res``,
    ``run_simulation``, ``calibrated_parameters`` and
    ``parameter_bounds``.  No change to the optimiser was needed, which is
    the point -- the harness was already general, it just had nothing
    pointed at F1.

    Attributes:
        sand: The bed whose continuum is being calibrated. Only its
            friction angle is varied; everything else (packing, moisture,
            gradation) is held so the fit cannot quietly move a second
            property.
        target_phi_peak: Declared peak friction angle, degrees.
        target_phi_res: Declared residual friction angle, degrees.
        confining_stresses_pa: Cells the envelope is fitted over.
        calibrated_parameters: What the optimiser may search.
        parameter_bounds: Search bounds, including for parameters that
            are **not** searched, so a caller can see the range that was
            considered and rejected.
    """

    #: ``shear_modulus_pa`` is deliberately absent; see the module docstring.
    calibrated_parameters: tuple[str, ...] = ("friction_angle_deg",)

    #: The friction-angle band is wide enough to contain any sand and
    #: narrow enough that ``sqrt(2) alpha`` stays inside ``asin``.
    #:
    #: The modulus band spans the Hardin & Richart estimate (about 7 MPa
    #: for the fluffy preset) by two decades either way. Its **lower** end
    #: is not a taste: below roughly 1 MPa the isotropic consolidation to
    #: the largest declared cell drives the volumetric strain past the
    #: packing state's compressive cap, and ``drained_biaxial_path``
    #: raises rather than reporting the capped path's angle. The band is
    #: declared but not searched; see the module docstring.
    parameter_bounds: dict[str, tuple[float, float]] = {
        "friction_angle_deg": (10.0, 60.0),
        "shear_modulus_pa": (1.0e6, 1.0e9),
    }

    #: This experiment simulates a *declared* target. It is not data.
    is_measured_on_bunker_sand: bool = False

    target_provenance_note: str = F1_SHEAR_CELL_TARGET_NOTE

    def __init__(
        self,
        sand: SandState | None = None,
        *,
        confining_stresses_pa: tuple[float, ...] = (
            F1_SHEAR_CELL_CONFINING_STRESSES_PA
        ),
        axial_strain: float = _DEFAULT_AXIAL_STRAIN,
        n_increments: int = _DEFAULT_INCREMENTS,
        target_phi_peak: float = 35.0,
        target_phi_res: float = 30.0,
        include_shear_modulus: bool = False,
    ) -> None:
        """Initialise the experiment.

        Args:
            sand: Bed to calibrate. Defaults to the fluffy USGA preset,
                which is the only one whose cone tip sits at the origin
                and therefore the only one whose envelope isolates the
                friction angle from the moisture model.
            confining_stresses_pa: Consolidation stresses, compression
                positive; at least two distinct values.
            axial_strain: Total axial Hencky strain per cell, negative.
            n_increments: Load steps per cell.
            target_phi_peak: Declared peak friction angle in degrees.
            target_phi_res: Declared residual friction angle in degrees.
            include_shear_modulus: Add ``shear_modulus_pa`` to
                :attr:`calibrated_parameters`. It is inert, so the
                optimiser's own #7999 guard refuses it; the flag exists so
                that refusal is demonstrable rather than asserted in prose.

        Raises:
            CalibrationError: If fewer than two distinct confining
                stresses are given, if any is not compressive, or if the
                targets are not usable angles.
        """
        self.sand = playing_condition(PlayingCondition.FLUFFY) if sand is None else sand
        if not isinstance(self.sand, SandState):
            raise CalibrationError(
                f"sand must be a SandState, got {type(self.sand).__name__}"
            )
        stresses = tuple(float(value) for value in confining_stresses_pa)
        if any(not math.isfinite(v) or v <= 0.0 for v in stresses):
            raise CalibrationError(
                f"every confining stress must be positive, got {stresses!r}"
            )
        if len(set(stresses)) < _MIN_CONFINEMENTS:
            raise CalibrationError(
                "at least two distinct confining stresses are needed to "
                f"separate the friction angle from the cohesion, got {stresses!r}"
            )
        for name, value in (
            ("target_phi_peak", target_phi_peak),
            ("target_phi_res", target_phi_res),
        ):
            if not math.isfinite(value) or not 0.0 < value < 90.0:
                raise CalibrationError(
                    f"{name} must be an angle in (0, 90) deg, got {value!r}"
                )
        self.confining_stresses_pa = stresses
        self.axial_strain = float(axial_strain)
        self.n_increments = int(n_increments)
        self.target_phi_peak = float(target_phi_peak)
        self.target_phi_res = float(target_phi_res)
        if include_shear_modulus:
            self.calibrated_parameters = (
                "friction_angle_deg",
                "shear_modulus_pa",
            )

    # --------------------------------------------------------------- model

    @property
    def irreducible_residual_deg2(self) -> float:
        """Objective value the model cannot go below, in ``deg^2``.

        Rate-independent perfect plasticity gives ``phi_peak == phi_res``
        identically, so the objective
        ``(phi_peak - t_peak)^2 + (phi_res - t_res)^2`` is minimised at
        the midpoint of the two targets and cannot fall below
        ``2 ((t_peak - t_res) / 2)^2``.  Reporting a residual without
        this number would make a structural limitation of the model look
        like a failure of the fit.
        """
        gap = self.target_phi_peak - self.target_phi_res
        return 2.0 * (0.5 * gap) ** 2

    def continuum(self, params: dict) -> SandContinuum:
        """Build the continuum a parameter vector describes.

        Args:
            params: May carry ``friction_angle_deg`` and
                ``shear_modulus_pa``; anything absent keeps the sand's
                own value.

        Returns:
            The continuum under test.

        Raises:
            CalibrationError: If the friction angle is unusable.
        """
        angle = float(params.get("friction_angle_deg", self.sand.friction_angle_deg))
        if not math.isfinite(angle) or not 0.0 < angle < 90.0:
            raise CalibrationError(
                f"friction_angle_deg must lie in (0, 90), got {angle!r}"
            )
        modulus = params.get("shear_modulus_pa")
        sand = dataclasses.replace(self.sand, friction_angle_deg=angle)
        return SandContinuum.from_sand_state(
            sand,
            shear_modulus_pa=None if modulus is None else float(modulus),
        )

    def envelopes(
        self, params: dict
    ) -> tuple[MohrCoulombEnvelope, MohrCoulombEnvelope]:
        """Fit the peak and residual envelopes from **one** pass of cells.

        Each cell is run once; the peak envelope reads it at its highest
        stress ratio and the residual envelope at the end of its path.
        Running the cells twice would double the cost for two numbers the
        same path already carries.

        Args:
            params: Parameter vector; see :meth:`continuum`.

        Returns:
            ``(peak_envelope, residual_envelope)``.

        Raises:
            CalibrationError: If a cell cannot be run or a fit fails.
        """
        material = self.continuum(params)
        peak_centres: list[float] = []
        peak_radii: list[float] = []
        end_centres: list[float] = []
        end_radii: list[float] = []
        for stress in self.confining_stresses_pa:
            path = drained_biaxial_path(
                material,
                confining_stress_pa=stress,
                axial_strain=self.axial_strain,
                n_increments=self.n_increments,
            )
            peak = max(path, key=lambda point: point.stress_ratio)
            peak_centres.append(peak.mean_stress_pa)
            peak_radii.append(peak.deviator_stress_pa)
            end_centres.append(path[-1].mean_stress_pa)
            end_radii.append(path[-1].deviator_stress_pa)
        return (
            MohrCoulombEnvelope.from_points(
                np.array(peak_centres), np.array(peak_radii)
            ),
            MohrCoulombEnvelope.from_points(np.array(end_centres), np.array(end_radii)),
        )

    def envelope(self, params: dict, *, at_peak: bool = True) -> MohrCoulombEnvelope:
        """Fit one Mohr-Coulomb envelope over every confining stress.

        Args:
            params: Parameter vector; see :meth:`continuum`.
            at_peak: Read each cell at its peak stress ratio when true,
                at the end of its path when false.

        Returns:
            The fitted envelope.

        Raises:
            CalibrationError: If a cell cannot be run or the fit fails.
        """
        peak, residual = self.envelopes(params)
        return peak if at_peak else residual

    # ---------------------------------------------------------- the harness

    def run_simulation(self, params: dict) -> tuple[float, float]:
        """Return ``(phi_peak, phi_res)`` in degrees for a parameter vector.

        Args:
            params: Parameter vector; see :meth:`continuum`.

        Returns:
            Peak and residual friction angles in degrees. They are equal
            to round-off: the model has no softening. See
            :attr:`irreducible_residual_deg2`.

        Raises:
            CalibrationError: If a cell cannot be run.
        """
        peak, residual = self.envelopes(params)
        return peak.friction_angle_deg, residual.friction_angle_deg

    def fit_friction_angle_deg(self) -> float:
        """The friction angle that minimises the harness objective.

        Solved in closed form rather than searched, because the closed
        form exists: ``phi_peak == phi_res == phi*``, the objective is a
        parabola in ``phi*`` with its minimum at the midpoint of the two
        targets, and
        :func:`friction_angle_for_plane_strain_angle_deg` inverts ``phi*``
        exactly.  The stochastic search
        :meth:`~bunkershot3d.calibration.optimizer.CalibrationOptimizer.optimize`
        runs is checked against this in the test suite; a global optimiser
        that disagreed with an available closed form would be reporting
        its own population, which is the #7999 failure mode.

        Returns:
            The fitted friction angle in degrees.
        """
        midpoint = 0.5 * (self.target_phi_peak + self.target_phi_res)
        return friction_angle_for_plane_strain_angle_deg(midpoint)
