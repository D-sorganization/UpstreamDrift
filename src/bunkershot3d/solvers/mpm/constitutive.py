"""The F1 constitutive model: Drucker-Prager sand with a compressive cap.

Why not the mu(I) rheology
--------------------------

The obvious constitutive choice for flowing sand is the ``mu(I)``
rheology, and it is the wrong one to ship here.  Barker, Schaeffer,
Bohorquez & Gray (2015), *"Well-posed and ill-posed behaviour of the
mu(I)-rheology for granular flow"*, J. Fluid Mech. **779**:794-818, show
that the incrementally-linearised equations **lose hyperbolicity** -- the
initial-value problem becomes Hadamard-unstable -- for inertial numbers
below ``I_1`` and above ``I_2``.  The perturbation growth rate then rises
without bound with wavenumber, so the computation does not converge under
refinement: **a finer grid makes the answer worse, not better**, and any
apparently converged result is the discretisation quietly acting as the
regulariser.  A tier whose entire deliverable is a *field* that will be
grid-refined and GCI-reported (ADR-0033) cannot be built on equations
that are ill-posed in exactly the regime being refined.

Both of ``mu(I)``'s ill-posed regimes are inside a bunker shot, not
outside it: the quiescent bed far from the club sits at ``I -> 0``, and
the leading edge reaches ``I ~ 11`` (see
:mod:`bunkershot3d.solvers.envelope`).  Regularising ``mu(I)`` into a
well-posed band (Barker & Gray 2017, JFM **828**:5-32) is possible, but
the regularisation constants are fitted, unmeasured for this sand, and
would become a second uncalibrated parameter set beside the one issue
#7999 already records.

What is shipped instead
-----------------------

**Rate-independent Drucker-Prager elastoplasticity with a compressive
cap**, integrated as a return mapping on the principal Hencky
(logarithmic) strains of the elastic deformation gradient:

* Drucker & Prager (1952), *Q. Appl. Math.* **10**:157-165 -- the yield
  surface.
* Klar, Gast, Pradhana, Fu, Schroeder, Jiang & Teran (2016),
  *"Drucker-Prager elastoplasticity for sand"*, ACM Trans. Graph.
  **35**(4):103 -- the principal-strain-space return mapping used here,
  and the same material model the F2 reference tier (Newton
  ``SolverImplicitMPM``) uses, which ADR-0033 requires so that one
  calibration carries between the tiers.
* Bonet & Wood (2008), *Nonlinear Continuum Mechanics for Finite Element
  Analysis*, section 6.6 -- Hencky (logarithmic) hyperelasticity.

Rate independence is the point.  The constitutive response contributes no
term that can lose hyperbolicity: the characteristic speeds of the
resulting system are the elastic wave speeds, which are real and positive
for any admissible ``(lambda, mu)``, so the problem stays well posed at
every inertial number and refinement is meaningful.  The price is that
this model has **no Bagnold profile** -- steady inclined flow of a
rate-independent plastic solid is plug flow with a basal shear band, not
the 3/2-power profile -- so Bagnold is not available as a verification
case for this tier, and :mod:`bunkershot3d.solvers.mpm.verification` uses
an elastic column and a free-fall case instead.

The compressive cap
-------------------

A bare Drucker-Prager cone is open in compression: the mean stress can
grow without bound while staying elastic, so nothing stops a bed being
compacted past the packing limit of its own grains.  The cap here is not
a tuning constant -- it is read off the sand's own packing state, which
already knows the densest reproducible packing:

    J_min = phi / phi_max,    eps_v_cap = ln(J_min) < 0

Below that volumetric strain, further compression is plastic compaction
rather than elastic storage.  A loose bed (``Dr = 0``) therefore has room
to compact by ~15%, a firm one (``Dr = 0.875``) by ~2%, which is
critical-state behaviour arriving directly from
:class:`bunkershot3d.sand.packing.PackingState` rather than from a second
set of invented constants.

Units are SI throughout, plane strain is ``d = 2``, and every contract
check is a ``raise`` rather than an ``assert`` because ``python -O``
strips assertions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...sand import Angularity, SandState
from ...sand.provenance import (
    PropertyProvenance,
    ProvenanceBasis,
    SandProvenance,
)
from ..envelope import GRAVITY_M_S2
from ..exceptions import CalibrationError

__all__ = [
    "HARDIN_RICHART_ANGULAR_COEFFICIENT_KPA",
    "HARDIN_RICHART_ROUND_COEFFICIENT_KPA",
    "PLANE_STRAIN_DIMENSION",
    "SAND_POISSON_RATIO",
    "SandContinuum",
    "drucker_prager_alpha",
    "hencky_kirchhoff_principal",
    "principal_stretches",
    "project_to_yield_surface",
    "reconstruct",
    "yield_function",
]

PLANE_STRAIN_DIMENSION = 2
"""Spatial dimension of the plane-strain problem, ``d`` in the return map."""

SAND_POISSON_RATIO = 0.30
"""Drained Poisson's ratio for a granular soil.

A textbook convention for sands (0.25-0.35), not a measurement on bunker
sand.  It is a weak lever here: the plastic limit state that sets the
forces is fixed by the friction angle and the cap, while ``nu`` and the
shear modulus set only the elastic wave speed and therefore the timestep.
"""

HARDIN_RICHART_ANGULAR_COEFFICIENT_KPA = 3230.0
"""``A`` in ``G_max = A (2.97 - e)^2 / (1 + e) sqrt(p')``, SI-converted.

Hardin & Richart (1963), *"Elastic wave velocities in granular soils"*,
J. Soil Mech. Found. Div. ASCE **89**(SM1):33-65, angular-grain form,
converted from the original 1230 psi coefficient.  USGA guidance
specifies angular bunker sand, so this is the branch the presets take."""

HARDIN_RICHART_ROUND_COEFFICIENT_KPA = 6908.0
"""The round-grain branch, ``G_max = A (2.17 - e)^2 / (1 + e) sqrt(p')``."""

_ANGULAR_VOID_OFFSET = 2.97
_ROUND_VOID_OFFSET = 2.17
_KPA_PER_PA = 1.0e-3
_PA_PER_KPA = 1.0e3

_SINGULAR_VALUE_FLOOR = 1.0e-8
"""Floor on a principal stretch before its logarithm is taken.

A deformation gradient that has been driven to (or through) zero
determinant has no logarithm.  Flooring rather than raising keeps one
pathological particle from destroying an otherwise healthy step, and the
solver reports how many particles were floored so the event is visible.
"""

_MIN_EARTH_PRESSURE_DEPTH_M = 1.0e-4
"""Shallowest bed depth the reference confining stress is formed on."""


def drucker_prager_alpha(friction_angle_deg: float) -> float:
    """Return the Drucker-Prager cone slope ``alpha`` for a friction angle.

    ``alpha = sqrt(2/3) * 2 sin(phi) / (3 - sin(phi))`` -- the inner-cone
    (compressive-meridian) match to Mohr-Coulomb, as used by Klar et al.
    (2016) eq. (26).  The yield surface is written on the Kirchhoff stress
    as ``||dev(tau)|| + alpha tr(tau) <= 0``, so ``alpha`` is
    dimensionless and the same number serves plane strain and 3-D, which
    is what lets F1 and the F2 reference share one calibration.

    Args:
        friction_angle_deg: Internal friction angle in degrees, strictly
            inside ``(0, 90)``.

    Returns:
        The cone slope.

    Raises:
        CalibrationError: If the friction angle is not a usable angle.
    """
    angle = float(friction_angle_deg)
    if not math.isfinite(angle) or not 0.0 < angle < 90.0:
        raise CalibrationError(
            "friction angle must lie strictly between 0 and 90 deg, got "
            f"{friction_angle_deg!r} deg"
        )
    sin_phi = math.sin(math.radians(angle))
    return math.sqrt(2.0 / 3.0) * (2.0 * sin_phi) / (3.0 - sin_phi)


def hencky_kirchhoff_principal(
    hencky_strain: NDArray[np.float64],
    *,
    shear_modulus_pa: float,
    lame_lambda_pa: float,
) -> NDArray[np.float64]:
    """Principal Kirchhoff stresses of the Hencky elastic model.

    ``tau_i = 2 mu eps_i + lambda tr(eps)``, the principal form of the
    St Venant-Kirchhoff model written in logarithmic strain (Bonet & Wood
    2008, section 6.6).  Kirchhoff rather than Cauchy because the return
    mapping is defined on ``tau``; divide by ``J`` for Cauchy.

    Args:
        hencky_strain: ``(n, d)`` principal logarithmic strains.
        shear_modulus_pa: Lame ``mu``.
        lame_lambda_pa: Lame ``lambda``.

    Returns:
        ``(n, d)`` principal Kirchhoff stresses in pascals.
    """
    trace = hencky_strain.sum(axis=-1, keepdims=True)
    return 2.0 * shear_modulus_pa * hencky_strain + lame_lambda_pa * trace


def yield_function(
    hencky_strain: NDArray[np.float64],
    *,
    shear_modulus_pa: float,
    lame_lambda_pa: float,
    alpha: float,
    tip_volumetric_strain: float,
) -> NDArray[np.float64]:
    """Drucker-Prager yield function on the elastic Hencky strain.

    ``y = ||dev(tau)|| + alpha (tr(tau) - tr(tau)_tip)``, expressed
    directly in strain because ``dev(tau) = 2 mu dev(eps)`` and
    ``tr(tau) = (2 mu + d lambda) tr(eps)`` for the Hencky model.  Values
    are pascals; ``y <= 0`` is admissible.

    Exposed because it is the sharpest available verification of the
    return mapping: after
    :func:`project_to_yield_surface` every particle must satisfy
    ``y <= 0``, and every particle that actually yielded must satisfy
    ``y = 0`` to round-off.

    Args:
        hencky_strain: ``(n, d)`` principal logarithmic strains.
        shear_modulus_pa: Lame ``mu``.
        lame_lambda_pa: Lame ``lambda``.
        alpha: Cone slope from :func:`drucker_prager_alpha`.
        tip_volumetric_strain: ``tr(eps)`` at the cone tip, which is
            positive when the sand carries cohesion and zero when it does
            not.

    Returns:
        ``(n,)`` yield-function values in pascals.
    """
    dimension = hencky_strain.shape[-1]
    trace = hencky_strain.sum(axis=-1)
    deviator = hencky_strain - (trace / dimension)[..., None]
    deviator_norm = np.sqrt(np.einsum("ij,ij->i", deviator, deviator))
    bulk_term = 2.0 * shear_modulus_pa + dimension * lame_lambda_pa
    return 2.0 * shear_modulus_pa * deviator_norm + alpha * bulk_term * (
        trace - tip_volumetric_strain
    )


def _validate_yield_projection_inputs(
    hencky_strain: NDArray[np.float64],
    *,
    cap_volumetric_strain: float,
) -> None:
    """Reject strain arrays and caps the return map cannot act on.

    Args:
        hencky_strain: Trial principal logarithmic strains.
        cap_volumetric_strain: Most compressive elastic ``tr(eps)``.

    Raises:
        CalibrationError: If the strain array is not ``(n, d)`` with
            ``d >= 1``, or if the cap is not compressive.
    """
    if hencky_strain.ndim != 2 or hencky_strain.shape[1] < 1:
        raise CalibrationError(
            "hencky_strain must have shape (n, d) with d >= 1, got "
            f"{hencky_strain.shape!r}"
        )
    if not math.isfinite(cap_volumetric_strain) or cap_volumetric_strain >= 0.0:
        raise CalibrationError(
            "cap_volumetric_strain must be negative -- the cap is a limit on "
            f"compaction, not on extension -- got {cap_volumetric_strain!r}"
        )


def project_to_yield_surface(
    hencky_strain: NDArray[np.float64],
    *,
    shear_modulus_pa: float,
    lame_lambda_pa: float,
    alpha: float,
    tip_volumetric_strain: float,
    cap_volumetric_strain: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Return-map the elastic Hencky strain onto the capped DP cone.

    Three cases, in the order they are applied:

    1. **Cap.**  ``tr(eps) < eps_cap`` means the bed has been compressed
       past the packing limit of its own grains; the excess volumetric
       strain becomes plastic compaction and the trace is clamped.  The
       deviator is untouched, so this is a purely volumetric projection.
    2. **Tip.**  ``tr(eps) >= tr(eps)_tip`` is a state the cone cannot
       carry at all -- isotropic tension beyond the cohesive apex -- and
       the whole deviatoric strain is released, leaving the particle at
       the tip.  This is what lets sand separate freely at a free surface
       instead of behaving like a stretched solid.  Note the condition is
       on the trace alone: Klar's published form also sends ``||dev|| =
       0`` to the tip, which would wrongly claim a purely isotropic
       *compression* -- an admissible state deep in the cone, and the
       commonest state in a settled bed.
    3. **Cone.**  Otherwise ``dgamma = ||dev|| + alpha (2 mu + d lambda) /
       (2 mu) * (tr - tr_tip)``; a non-positive ``dgamma`` is elastic and
       a positive one is returned along the deviatoric direction.

    Case 3 changes only the deviator, so plastic flow is **volume
    preserving** -- a non-associated flow rule, deliberately, because the
    associated rule for a frictional cone predicts unbounded dilation
    (Klar et al. 2016, section 4.2).  Dilatancy in this tier is carried by
    the cap and the packing state, not by the flow rule.

    Args:
        hencky_strain: ``(n, d)`` trial principal logarithmic strains.
        shear_modulus_pa: Lame ``mu``.
        lame_lambda_pa: Lame ``lambda``.
        alpha: Cone slope.
        tip_volumetric_strain: ``tr(eps)`` at the cone tip, ``>= 0``.
        cap_volumetric_strain: Most compressive ``tr(eps)`` the packing
            state can carry elastically, ``< 0``.

    Returns:
        ``(projected_strain, yielded, capped)`` where ``yielded`` marks
        particles moved by case 2 or 3 and ``capped`` marks those moved by
        case 1.

    Raises:
        CalibrationError: If the strain array is not ``(n, d)`` with
            ``d >= 1``, or if the cap is not compressive.
    """
    _validate_yield_projection_inputs(
        hencky_strain, cap_volumetric_strain=cap_volumetric_strain
    )
    dimension = hencky_strain.shape[1]
    strain = np.array(hencky_strain, dtype=np.float64, copy=True)

    trace = strain.sum(axis=1)
    capped = trace < cap_volumetric_strain
    if bool(capped.any()):
        correction = np.where(capped, cap_volumetric_strain - trace, 0.0) / dimension
        strain = strain + correction[:, None]
        trace = strain.sum(axis=1)

    deviator = strain - (trace / dimension)[:, None]
    deviator_norm = np.sqrt(np.einsum("ij,ij->i", deviator, deviator))

    at_tip = trace >= tip_volumetric_strain
    bulk_term = 2.0 * shear_modulus_pa + dimension * lame_lambda_pa
    delta_gamma = deviator_norm + alpha * (bulk_term / (2.0 * shear_modulus_pa)) * (
        trace - tip_volumetric_strain
    )
    # A state on the hydrostatic axis has no deviatoric direction to be
    # projected along.  It needs none: below the tip it is admissible, and
    # at or above the tip the tip branch above already claims it.
    on_cone = (~at_tip) & (deviator_norm > 0.0) & (delta_gamma > 0.0)

    projected = strain.copy()
    if bool(at_tip.any()):
        projected[at_tip] = tip_volumetric_strain / dimension
    if bool(on_cone.any()):
        # ``np.where`` would evaluate the quotient for every particle,
        # including the hydrostatic ones the mask is there to discard, so
        # a freshly seeded bed emits a divide-by-zero warning per call.
        # Dividing only where the norm is positive leaves those entries at
        # the zero they already hold.
        scale = np.zeros_like(deviator_norm)
        np.divide(delta_gamma, deviator_norm, out=scale, where=deviator_norm > 0.0)
        projected = np.where(
            on_cone[:, None], strain - scale[:, None] * deviator, projected
        )
    return projected, at_tip | on_cone, capped


@dataclass(frozen=True)
class SandContinuum:
    """The continuum material F1 solves, derived from a :class:`SandState`.

    Every constant here is either read from the sand package or derived
    from something it already carries.  Nothing is a second, parallel set
    of sand constants -- issue #7999 records that the sand package's own
    values are borrowed, and duplicating them somewhere else would make
    that record unfalsifiable.

    Attributes:
        density_kg_m3: Bulk density including pore water, from
            :attr:`~bunkershot3d.sand.state.SandState.bulk_density_kg_m3`.
        shear_modulus_pa: Lame ``mu``.
        lame_lambda_pa: Lame ``lambda``.
        alpha: Drucker-Prager cone slope.
        tip_volumetric_strain: ``tr(eps)`` at the cohesive cone tip.
        cap_volumetric_strain: Most compressive ``tr(eps)`` the packing
            state carries elastically.
        grain_diameter_m: ``d50``, for the validity envelope.
        friction_angle_deg: Retained for reporting.
        cohesion_pa: Moisture-derived apparent cohesion.
        provenance: Where each of the above came from.
    """

    density_kg_m3: float
    shear_modulus_pa: float
    lame_lambda_pa: float
    alpha: float
    tip_volumetric_strain: float
    cap_volumetric_strain: float
    grain_diameter_m: float
    friction_angle_deg: float
    cohesion_pa: float
    provenance: SandProvenance

    def __post_init__(self) -> None:
        positive = {
            "density_kg_m3": self.density_kg_m3,
            "shear_modulus_pa": self.shear_modulus_pa,
            "lame_lambda_pa": self.lame_lambda_pa,
            "alpha": self.alpha,
            "grain_diameter_m": self.grain_diameter_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise CalibrationError(f"{name} must be positive, got {value!r}")
        if not math.isfinite(self.cap_volumetric_strain) or (
            self.cap_volumetric_strain >= 0.0
        ):
            raise CalibrationError(
                "cap_volumetric_strain must be negative, got "
                f"{self.cap_volumetric_strain!r}"
            )
        if not math.isfinite(self.tip_volumetric_strain) or (
            self.tip_volumetric_strain < 0.0
        ):
            raise CalibrationError(
                "tip_volumetric_strain must be non-negative -- a cohesionless "
                f"sand has a tip at zero -- got {self.tip_volumetric_strain!r}"
            )

    # ------------------------------------------------------------- derived

    @property
    def youngs_modulus_pa(self) -> float:
        """``E`` implied by the Lame constants."""
        numerator = self.shear_modulus_pa * (
            3.0 * self.lame_lambda_pa + 2.0 * self.shear_modulus_pa
        )
        return numerator / (self.lame_lambda_pa + self.shear_modulus_pa)

    @property
    def p_wave_modulus_pa(self) -> float:
        """``lambda + 2 mu``: the constrained (oedometer) modulus."""
        return self.lame_lambda_pa + 2.0 * self.shear_modulus_pa

    @property
    def elastic_wave_speed_m_s(self) -> float:
        """Dilatational wave speed ``sqrt((lambda + 2 mu) / rho)``.

        This is the speed the CFL condition is formed on: it is the
        fastest signal the discretisation carries, and it is **computed**
        from the material rather than pinned, so a stiffer sand shortens
        the timestep by itself.
        """
        return math.sqrt(self.p_wave_modulus_pa / self.density_kg_m3)

    @property
    def tensile_strength_pa(self) -> float:
        """Isotropic tensile strength ``c cot(phi)`` at the cone apex."""
        return self.cohesion_pa / math.tan(math.radians(self.friction_angle_deg))

    def cauchy_from_hencky(
        self, hencky_strain: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Principal Cauchy stresses for a principal Hencky strain.

        Args:
            hencky_strain: ``(n, d)`` principal logarithmic strains.

        Returns:
            ``(n, d)`` principal Cauchy stresses; ``J = exp(tr(eps))``.
        """
        kirchhoff = hencky_kirchhoff_principal(
            hencky_strain,
            shear_modulus_pa=self.shear_modulus_pa,
            lame_lambda_pa=self.lame_lambda_pa,
        )
        jacobian = np.exp(hencky_strain.sum(axis=-1))[..., None]
        return kirchhoff / jacobian

    def project(
        self, hencky_strain: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
        """Return-map a trial strain with this material's constants."""
        return project_to_yield_surface(
            hencky_strain,
            shear_modulus_pa=self.shear_modulus_pa,
            lame_lambda_pa=self.lame_lambda_pa,
            alpha=self.alpha,
            tip_volumetric_strain=self.tip_volumetric_strain,
            cap_volumetric_strain=self.cap_volumetric_strain,
        )

    def yield_value(self, hencky_strain: NDArray[np.float64]) -> NDArray[np.float64]:
        """Yield-function values for this material's constants."""
        return yield_function(
            hencky_strain,
            shear_modulus_pa=self.shear_modulus_pa,
            lame_lambda_pa=self.lame_lambda_pa,
            alpha=self.alpha,
            tip_volumetric_strain=self.tip_volumetric_strain,
        )

    # ------------------------------------------------------------ factory

    @classmethod
    def from_sand_state(
        cls,
        sand: SandState,
        *,
        reference_depth_m: float | None = None,
        poisson_ratio: float = SAND_POISSON_RATIO,
        gravity_m_s2: float = GRAVITY_M_S2,
        shear_modulus_pa: float | None = None,
        dilation_suction_pa: float | None = None,
    ) -> SandContinuum:
        """Derive the continuum constants from a sand state.

        The small-strain shear modulus follows Hardin & Richart (1963),
        which consumes exactly the quantities the sand package already
        carries -- void ratio, angularity, and a confining stress formed
        from the bed's own depth and bulk density -- rather than
        introducing an independent stiffness constant.  The branch is
        chosen by :class:`~bunkershot3d.sand.packing.Angularity`, so an
        angular USGA sand and a rounded one do not silently share a
        modulus.

        The confining stress is the geostatic mean effective stress at
        mid-bed, ``p' = (1 + 2 K_0)/3 * rho g h/2`` with Jaky's
        ``K_0 = 1 - sin(phi)``.  It is a *reference* stress for the
        modulus only: the model is not pressure-dependent at runtime, and
        stating that plainly matters more than hiding it, because a
        pressure-dependent modulus would make the elastic wave speed --
        and therefore the timestep -- a function of the solution.

        Args:
            sand: The bed being struck.
            reference_depth_m: Bed depth the confining stress is formed
                over. Defaults to the sand state's own bed depth.
            poisson_ratio: Drained Poisson's ratio.
            gravity_m_s2: Gravitational acceleration.
            shear_modulus_pa: Override the derived modulus. Supplied by
                verification cases that need a stated stiffness, and by
                callers who have measured one.
            dilation_suction_pa: Forwarded to
                :meth:`~bunkershot3d.sand.state.SandState.cohesive_strength_pa`.
                A **saturated** bed has no default here and the sand
                package raises without it, which is deliberate and is
                left to propagate: a silent zero and a silent multi-MPa
                suction are both wrong, and F1 has no better answer than
                the sand model does.

        Returns:
            The continuum material.

        Raises:
            CalibrationError: If the sand state is not a
                :class:`~bunkershot3d.sand.state.SandState`, or if the
                Poisson ratio is outside ``(-1, 0.5)``.
            MoistureRegimeError: If the bed is saturated and no dilation
                suction was supplied.
        """
        _validate_continuum_inputs(sand, poisson_ratio=poisson_ratio)
        depth_m = sand.bed.depth_m if reference_depth_m is None else reference_depth_m
        modulus_pa = (
            _hardin_richart_shear_modulus_pa(
                sand, depth_m=depth_m, gravity_m_s2=gravity_m_s2
            )
            if shear_modulus_pa is None
            else float(shear_modulus_pa)
        )
        if not math.isfinite(modulus_pa) or modulus_pa <= 0.0:
            raise CalibrationError(
                f"shear modulus must be positive, got {modulus_pa!r} Pa"
            )
        lame_lambda = 2.0 * modulus_pa * poisson_ratio / (1.0 - 2.0 * poisson_ratio)

        alpha = drucker_prager_alpha(sand.friction_angle_deg)
        cohesion_pa = float(sand.cohesive_strength_pa(dilation_suction_pa))
        tensile_pa = cohesion_pa / math.tan(sand.friction_angle_rad)
        bulk_term = 2.0 * modulus_pa + PLANE_STRAIN_DIMENSION * lame_lambda
        tip = PLANE_STRAIN_DIMENSION * tensile_pa / bulk_term

        cap = _cap_volumetric_strain(sand)
        return cls(
            density_kg_m3=sand.bulk_density_kg_m3,
            shear_modulus_pa=modulus_pa,
            lame_lambda_pa=lame_lambda,
            alpha=alpha,
            tip_volumetric_strain=tip,
            cap_volumetric_strain=cap,
            grain_diameter_m=sand.d50_m,
            friction_angle_deg=sand.friction_angle_deg,
            cohesion_pa=cohesion_pa,
            provenance=_continuum_provenance(sand, derived=shear_modulus_pa is None),
        )


def _validate_continuum_inputs(sand: SandState, *, poisson_ratio: float) -> None:
    """Reject sand states and Poisson ratios the continuum cannot use.

    Args:
        sand: The bed being struck.
        poisson_ratio: Drained Poisson's ratio.

    Raises:
        CalibrationError: If the sand state is not a
            :class:`~bunkershot3d.sand.state.SandState`, or if the
            Poisson ratio is outside ``(-1, 0.5)``.
    """
    if not isinstance(sand, SandState):
        raise CalibrationError(f"expected a SandState, got {type(sand).__name__}")
    if not math.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise CalibrationError(
            f"poisson_ratio must lie strictly inside (-1, 0.5), got {poisson_ratio!r}"
        )


def _cap_volumetric_strain(sand: SandState) -> float:
    """Return the most compressive elastic ``tr(eps)`` the packing carries.

    Args:
        sand: The bed being struck.

    Returns:
        The compressive volumetric-strain cap, strictly negative.

    Raises:
        CalibrationError: If the packing is already at or past its
            densest state, leaving no compressive cap to place.
    """
    packing = sand.packing
    cap = math.log(packing.solid_fraction / packing.solid_fraction_max)
    if cap >= 0.0:
        raise CalibrationError(
            f"sand '{sand.name}' is already at or past its densest packing "
            f"(phi = {packing.solid_fraction:.4f}, phi_max = "
            f"{packing.solid_fraction_max:.4f}), so there is no compressive "
            "cap to place; the packing state is not physical"
        )
    return cap


def _hardin_richart_shear_modulus_pa(
    sand: SandState, *, depth_m: float, gravity_m_s2: float
) -> float:
    """Small-strain shear modulus from the void ratio and a confining stress."""
    depth = max(float(depth_m), _MIN_EARTH_PRESSURE_DEPTH_M)
    earth_pressure_coefficient = 1.0 - math.sin(sand.friction_angle_rad)
    vertical_stress_pa = sand.bulk_density_kg_m3 * gravity_m_s2 * depth / 2.0
    mean_stress_kpa = (
        (1.0 + 2.0 * earth_pressure_coefficient)
        / 3.0
        * vertical_stress_pa
        * _KPA_PER_PA
    )
    void_ratio = sand.void_ratio
    if sand.angularity.shape_index >= Angularity.SUBANGULAR.shape_index:
        coefficient = HARDIN_RICHART_ANGULAR_COEFFICIENT_KPA
        offset = _ANGULAR_VOID_OFFSET
    else:
        coefficient = HARDIN_RICHART_ROUND_COEFFICIENT_KPA
        offset = _ROUND_VOID_OFFSET
    shape_term = (offset - void_ratio) ** 2 / (1.0 + void_ratio)
    return coefficient * shape_term * math.sqrt(mean_stress_kpa) * _PA_PER_KPA


def _continuum_provenance(sand: SandState, *, derived: bool) -> SandProvenance:
    """Carry the sand's provenance forward and add the F1-specific entries."""
    entries = dict(sand.provenance.entries)
    entries["elastic_shear_modulus_pa"] = (
        PropertyProvenance(
            basis=ProvenanceBasis.ESTIMATED,
            source=(
                "Hardin & Richart (1963), Elastic wave velocities in granular "
                "soils, J. Soil Mech. Found. Div. ASCE 89(SM1):33-65"
            ),
            note=(
                "Small-strain modulus from the void ratio, the grain angularity "
                "and a geostatic reference confining stress. Not measured on "
                "bunker sand. In a rate-independent plastic model the modulus "
                "sets the elastic wave speed and hence the timestep; the limit "
                "state that sets the forces is fixed by the friction angle and "
                "the compressive cap."
            ),
        )
        if derived
        else PropertyProvenance(
            basis=ProvenanceBasis.CONVENTION,
            source="caller-supplied shear modulus",
            note="Supplied explicitly rather than derived; not a measurement.",
        )
    )
    entries["poisson_ratio"] = PropertyProvenance(
        basis=ProvenanceBasis.CONVENTION,
        source="textbook drained Poisson ratio band for sands, 0.25-0.35",
        note="A convention, not a measurement on bunker sand.",
    )
    entries["yield_surface"] = PropertyProvenance(
        basis=ProvenanceBasis.ESTIMATED,
        source=(
            "Drucker & Prager (1952) Q. Appl. Math. 10:157-165; return mapping "
            "per Klar et al. (2016) ACM TOG 35(4):103"
        ),
        note=(
            "Cone slope is the inner-cone Mohr-Coulomb match to the sand's own "
            "friction angle; the cohesive tip comes from the moisture model. "
            "Rate-independent by choice: the mu(I) rheology is ill-posed at "
            "both low and high inertial number (Barker et al. 2015, JFM "
            "779:794-818) and would not converge under the grid refinement "
            "ADR-0033 requires this tier to report."
        ),
    )
    entries["compressive_cap"] = PropertyProvenance(
        basis=ProvenanceBasis.ESTIMATED,
        source="random-close-packing limit of the sand's own PackingState",
        note=(
            "eps_v_cap = ln(phi / phi_max). Derived from the packing state "
            "rather than fitted, so a loose bed compacts further than a firm "
            "one without a second constant being introduced."
        ),
    )
    return SandProvenance(entries=entries)


def principal_stretches(
    deformation_gradient: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """SVD of a stack of deformation gradients, with a stretch floor.

    Plane strain makes every one of these a ``2 x 2``, so the closed form
    is used instead of ``np.linalg.svd``: LAPACK is called once per matrix
    and dominated the step profile at 29% of the runtime, while the ``2 x
    2`` decomposition is four ``arctan2`` calls over the whole array.
    Only three properties of the result are relied on downstream --
    ``U diag(s) V^T = F`` exactly, ``s >= 0``, and ``U`` orthogonal (the
    isotropic Kirchhoff stress is coaxial with it) -- and the test suite
    checks all three against LAPACK rather than checking that the factors
    happen to match it, which they need not.

    Args:
        deformation_gradient: ``(n, d, d)`` elastic deformation gradients.

    Returns:
        ``(left, stretches, right_transposed)`` with the singular values
        floored at :data:`_SINGULAR_VALUE_FLOOR` so that their logarithm
        exists even for an inverted or degenerate particle.
    """
    gradient = np.asarray(deformation_gradient, dtype=np.float64)
    if gradient.ndim != 3 or gradient.shape[1:] != (
        PLANE_STRAIN_DIMENSION,
        PLANE_STRAIN_DIMENSION,
    ):
        raise CalibrationError(
            f"deformation_gradient must have shape (n, 2, 2), got {gradient.shape!r}"
        )
    upper_left = gradient[:, 0, 0]
    upper_right = gradient[:, 0, 1]
    lower_left = gradient[:, 1, 0]
    lower_right = gradient[:, 1, 1]

    mean_diagonal = 0.5 * (upper_left + lower_right)
    diagonal_difference = 0.5 * (upper_left - lower_right)
    mean_off_diagonal = 0.5 * (lower_left + upper_right)
    off_diagonal_difference = 0.5 * (lower_left - upper_right)

    rotation_radius = np.hypot(mean_diagonal, off_diagonal_difference)
    shear_radius = np.hypot(diagonal_difference, mean_off_diagonal)
    larger = rotation_radius + shear_radius
    smaller = rotation_radius - shear_radius

    # Writing A = R(phi) diag(s) R(theta)^T and expanding gives
    # atan2(G, F) = phi + theta and atan2(H, E) = phi - theta, so the two
    # angles are the half-sum and the half-*difference* in that order.
    shear_angle = np.arctan2(mean_off_diagonal, diagonal_difference)
    rotation_angle = np.arctan2(off_diagonal_difference, mean_diagonal)
    right_angle = 0.5 * (shear_angle - rotation_angle)
    left_angle = 0.5 * (shear_angle + rotation_angle)

    left = _rotation(left_angle)
    right = _rotation(right_angle)
    # A negative determinant puts the second singular value below zero.
    # Flip it and absorb the sign into the matching right singular vector.
    flipped = smaller < 0.0
    if bool(flipped.any()):
        smaller = np.where(flipped, -smaller, smaller)
        right[:, :, 1] = np.where(flipped[:, None], -right[:, :, 1], right[:, :, 1])

    stretches = np.stack([larger, smaller], axis=1)
    return (
        left,
        np.maximum(stretches, _SINGULAR_VALUE_FLOOR),
        np.transpose(right, (0, 2, 1)),
    )


def _rotation(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(n, 2, 2)`` stack of plane rotations by ``angle``."""
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.stack(
        [
            np.stack([cosine, -sine], axis=1),
            np.stack([sine, cosine], axis=1),
        ],
        axis=1,
    )


def reconstruct(
    left: NDArray[np.float64],
    hencky_strain: NDArray[np.float64],
    right_transposed: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rebuild ``F = U exp(eps) V^T`` from a projected principal strain.

    Args:
        left: ``(n, d, d)`` left singular vectors.
        hencky_strain: ``(n, d)`` principal logarithmic strains.
        right_transposed: ``(n, d, d)`` transposed right singular vectors.

    Returns:
        ``(n, d, d)`` deformation gradients.
    """
    return (left * np.exp(hencky_strain)[:, None, :]) @ right_transposed
