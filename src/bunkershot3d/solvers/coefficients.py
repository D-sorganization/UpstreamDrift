"""The 3D-RFT generic response surface and its material scaling (#8611).

Source: Agarwal, Goldman & Kamrin, *PNAS* 120 (2023),
doi:10.1073/pnas.2214017120, as tabulated in
``docs/bunkershot3d/upgrade/research-digest-addendum.md`` section 3.

Formulation
-----------

In a local cylindrical frame with ``z_hat`` up, ``r_hat`` the horizontal
component of the velocity direction and ``theta_hat = z_hat x r_hat``::

    alpha_r     =  f1*sin(beta)*cos(psi) + f2*cos(gamma)
    alpha_theta =  f1*sin(beta)*sin(psi)
    alpha_z     = -f1*cos(beta) - f2*sin(gamma) - f3

    x1 = sin(gamma)
    x2 = cos(beta)
    x3 = cos(psi)*cos(gamma)*sin(beta) + sin(gamma)*cos(beta)

    f_i = sum_k c_i[k] * T_k          (20 polynomial terms in x1, x2, x3)

Reading the isotropic representation
------------------------------------

The three components above are one vector written in a basis of the only
three directions the problem has::

    alpha_generic = f1 * n_hat_fit  +  f2 * v_hat  -  f3 * z_hat

where ``n_hat_fit = (sin(beta) cos(psi), sin(beta) sin(psi), -cos(beta))``
in the local frame -- the *downward-facing* unit normal of the surface
element -- and ``v_hat = (cos(gamma), 0, -sin(gamma))`` by construction of
``r_hat``.  So ``f1`` weights the surface orientation, ``f2`` the motion
direction and ``f3`` the gravity direction, and the whole 20-term table
is a fit of three scalars, not of a nine-component tensor.  Writing it
this way is what makes the implementation vectorise cleanly and makes the
reflection symmetry below obvious.

Two consequences the solver relies on:

* ``beta`` and ``psi`` are restricted to ``[-pi/2, pi/2]``, so
  ``cos(beta) >= 0`` and the fitted domain covers only normals with
  ``n_z <= 0``.  An element whose outward normal points *upward* -- the
  lofted club face, for instance -- is outside the fit.  The solver
  clamps such elements to the vertical-wall limit and reports the clamped
  area fraction; it does not quietly extrapolate the polynomial.
* Only ``alpha_theta`` is odd in ``psi``; ``x1``, ``x2`` and ``x3`` are
  all even in ``psi``.  The response is therefore exactly covariant under
  reflection, which is what makes the metamorphic reflection test an
  identity rather than an approximation.

Anchors reproduced by this module (see the tests)
-------------------------------------------------

* ``alpha_z_generic(beta=0, gamma=pi/2, psi=0) = 0.87574`` -- the vertical
  flat-plate intrusion used for one-shot calibration.
* With ``rho_c = 1550 kg/m^3`` and ``Phi = 34 deg`` that gives
  ``2.11 N/cm^3``, against the ``2.02 N/cm^3`` measured on the Quikrete
  medium-sand analogue: a 4.6% independent cross-check between the
  material-scaling cubic and the plate measurement.
* The material-scaling table (``rho_c`` x ``mu``) reproduces to three
  significant figures with ``g = 9.81``.

Provenance
----------

**Every constant in this module is BORROWED, none is MEASURED.**  The
polynomial is fitted to simulations of a generic frictional-plastic
medium; the friction angle, packing fraction and grain density come from
the Quikrete medium-sand analogue; ``lambda`` comes from plate-drag and
wheel experiments on other materials.  Nothing here was measured on golf
bunker sand.  :data:`RFT_COEFFICIENT_PROVENANCE` records that per
constant, in the same form issue #7999 imposed on the sand presets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ..sand.provenance import (
    PropertyProvenance,
    ProvenanceBasis,
    SandProvenance,
)
from ..sand.state import SandState
from .envelope import GRAVITY_M_S2
from .exceptions import CalibrationError

__all__ = [
    "AGARWAL_2023_SOURCE",
    "AGARWAL_TERRAMECHANICS_SOURCE",
    "LAMBDA_BY_MOTION",
    "MATERIAL_SCALING_SOURCE",
    "PLATE_DRAG_LAMBDA",
    "RFT_COEFFICIENT_PROVENANCE",
    "RFT_POLYNOMIAL_COEFFICIENTS",
    "VERTICAL_PLATE_ALPHA_Z",
    "MaterialResponse",
    "generic_alpha",
    "internal_friction_mu",
    "material_scaling_pa_per_m",
    "scaling_shape_function",
    "polynomial_terms",
]

AGARWAL_2023_SOURCE = (
    "Agarwal, Goldman & Kamrin, PNAS 120 (2023), "
    "doi:10.1073/pnas.2214017120 (3D-RFT generic response surface)"
)

AGARWAL_TERRAMECHANICS_SOURCE = (
    "Agarwal, Senatore, Zhang, Kingsbury, Iagnemma, Goldman & Kamrin, "
    "J. Terramechanics (2019), arXiv:1901.10667; Agarwal, Karsai, Goldman & "
    "Kamrin, Science Advances (2021), arXiv:2005.10976 (DRFT inertial term)"
)

MATERIAL_SCALING_SOURCE = (
    "Material-scaling cubic xi_n = rho_c * g * f_hat(mu_int), "
    "f_hat = 894 mu^3 - 386 mu^2 + 89 mu (research digest addendum, section 3)"
)

_BORROWED_NOTE = (
    "Fitted to a generic frictional-plastic medium and to laboratory "
    "analogue sands. No coefficient in the F0 solver was measured on golf "
    "bunker sand; presenting one as measured is the failure mode issue "
    "#7999 corrected once already in this package."
)

# ---------------------------------------------------------------- the table

RFT_POLYNOMIAL_COEFFICIENTS: NDArray[np.float64] = np.array(
    [
        # c1        c2         c3           T_k
        [0.00212, -0.06796, -0.02634],  # 1
        [-0.02320, -0.10941, -0.03436],  # x1
        [-0.20890, 0.04725, 0.45256],  # x2
        [-0.43083, -0.06914, 0.00835],  # x3
        [-0.00259, -0.05835, 0.02553],  # x1^2
        [0.48872, -0.65880, -1.31290],  # x2^2
        [-0.00415, -0.11985, -0.05532],  # x3^2
        [0.07204, -0.25739, 0.06790],  # x1 x2
        [-0.02750, -0.26834, -0.16404],  # x2 x3
        [-0.08772, 0.02692, 0.02287],  # x3 x1
        [0.01992, -0.00736, 0.02927],  # x1^3
        [-0.45961, 0.63758, 0.95406],  # x2^3
        [0.40799, 0.08997, -0.00131],  # x3^3
        [-0.10107, 0.21069, -0.11028],  # x1 x2^2
        [-0.06576, 0.04748, 0.01487],  # x2 x1^2
        [0.05664, 0.20406, -0.02730],  # x2 x3^2
        [-0.09269, 0.18519, 0.10911],  # x3 x2^2
        [0.01892, 0.04934, -0.04097],  # x3 x1^2
        [0.01033, 0.13527, 0.07881],  # x1 x3^2
        [0.15120, -0.33207, -0.27519],  # x1 x2 x3
    ],
    dtype=np.float64,
)
RFT_POLYNOMIAL_COEFFICIENTS.flags.writeable = False
"""``(20, 3)`` table of ``c1``, ``c2``, ``c3`` against the 20 terms."""

VERTICAL_PLATE_ALPHA_Z = 0.8757399999999996
"""``alpha_z_generic(beta=0, gamma=pi/2, psi=0)``, the calibration anchor.

Pinned as a literal so a typo in :data:`RFT_POLYNOMIAL_COEFFICIENTS`
cannot pass silently; the test suite recomputes it from the table."""

PLATE_DRAG_LAMBDA = 1.1
"""``lambda`` for oblique horizontal plate drag -- the closest published
motion to a wedge sole planing through sand, and therefore the default."""

LAMBDA_BY_MOTION: Mapping[str, float] = MappingProxyType(
    {
        "grousered_wheel": 1.0,
        "oblique_horizontal_plate": 1.1,
        "sphere_vertical_impact": 1.4,
        "plane_strain_vertical_plate": 2.8,
    }
)
"""Measured ``lambda`` by motion type (addendum section 2).

A wedge is none of these.  The spread of 1.0-2.8 is the honest
uncertainty band on the term that carries ~90% of the load at greenside
delivery speed, which is why ADR-0032 makes ``lambda`` the primary
calibration target, ahead of ``alpha``."""

RFT_COEFFICIENT_PROVENANCE: Mapping[str, PropertyProvenance] = MappingProxyType(
    {
        "rft_polynomial": PropertyProvenance(
            basis=ProvenanceBasis.BORROWED_ANALOGUE,
            source=AGARWAL_2023_SOURCE,
            note=_BORROWED_NOTE,
        ),
        "material_scaling_cubic": PropertyProvenance(
            basis=ProvenanceBasis.BORROWED_ANALOGUE,
            source=MATERIAL_SCALING_SOURCE,
            note=_BORROWED_NOTE,
        ),
        "inertial_lambda": PropertyProvenance(
            basis=ProvenanceBasis.BORROWED_ANALOGUE,
            source=AGARWAL_TERRAMECHANICS_SOURCE,
            note=(
                "lambda = 1.1 is oblique horizontal plate drag. No wedge value "
                "exists; the published spread across motion types is 1.0-2.8. "
                + _BORROWED_NOTE
            ),
        ),
        "surface_friction_mu": PropertyProvenance(
            basis=ProvenanceBasis.ESTIMATED,
            source=(
                "Steel-on-sand interface friction, taken as about two thirds of "
                "the internal friction; the addendum records that the normal "
                "force is nearly independent of mu_surf over mu_int 0.3-0.9"
            ),
            note="Not measured on a wedge sole; the cutoff caps tangential "
            "traction and is deliberately a weak lever.",
        ),
        "gravity_m_s2": PropertyProvenance(
            basis=ProvenanceBasis.CONVENTION,
            source="g = 9.81 m/s^2",
            note="Pinned because the published material-scaling table "
            "reproduces with 9.81 and not with 9.80665.",
        ),
    }
)
"""Where every fitted constant in the F0 tier came from. All borrowed."""


# --------------------------------------------------------- the polynomial


def polynomial_terms(
    x1: NDArray[np.float64], x2: NDArray[np.float64], x3: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the 20 monomials ``T_k``, shape ``(..., 20)``.

    Args:
        x1: ``sin(gamma)``.
        x2: ``cos(beta)``.
        x3: ``cos(psi) cos(gamma) sin(beta) + sin(gamma) cos(beta)``.

    Returns:
        The design matrix, stacked on the last axis in the table's order.
    """
    # Filled into one preallocated block rather than stacked from 20
    # temporaries: this runs once per timestep on the shot path.
    design = np.empty((*np.shape(x1), 20), dtype=np.float64)
    x1_sq = x1 * x1
    x2_sq = x2 * x2
    x3_sq = x3 * x3
    design[..., 0] = 1.0
    design[..., 1] = x1
    design[..., 2] = x2
    design[..., 3] = x3
    design[..., 4] = x1_sq
    design[..., 5] = x2_sq
    design[..., 6] = x3_sq
    design[..., 7] = x1 * x2
    design[..., 8] = x2 * x3
    design[..., 9] = x3 * x1
    design[..., 10] = x1_sq * x1
    design[..., 11] = x2_sq * x2
    design[..., 12] = x3_sq * x3
    design[..., 13] = x1 * x2_sq
    design[..., 14] = x2 * x1_sq
    design[..., 15] = x2 * x3_sq
    design[..., 16] = x3 * x2_sq
    design[..., 17] = x3 * x1_sq
    design[..., 18] = x1 * x3_sq
    design[..., 19] = x1 * x2 * x3
    return design


def generic_alpha(
    beta_rad: NDArray[np.float64],
    gamma_rad: NDArray[np.float64],
    psi_rad: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The dimensionless generic response ``(alpha_r, alpha_theta, alpha_z)``.

    Fully vectorised: pass arrays of any common shape and get arrays of
    that shape back.  No per-element Python object is created.

    Args:
        beta_rad: Surface tilt, ``[-pi/2, pi/2]``. ``0`` is a horizontal
            surface whose normal points straight down.
        gamma_rad: Attack angle of the velocity below the horizontal,
            ``[-pi/2, pi/2]``. ``pi/2`` is straight down.
        psi_rad: Twist of the surface normal out of the ``r``-``z``
            plane, ``[-pi/2, pi/2]``.

    Returns:
        The three components in the local cylindrical frame.
    """
    sin_beta = np.sin(beta_rad)
    cos_beta = np.cos(beta_rad)
    sin_gamma = np.sin(gamma_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_psi = np.sin(psi_rad)
    cos_psi = np.cos(psi_rad)

    x1 = sin_gamma
    x2 = cos_beta
    x3 = cos_psi * cos_gamma * sin_beta + sin_gamma * cos_beta

    fitted = polynomial_terms(x1, x2, x3) @ RFT_POLYNOMIAL_COEFFICIENTS
    f1 = fitted[..., 0]
    f2 = fitted[..., 1]
    f3 = fitted[..., 2]

    alpha_r = f1 * sin_beta * cos_psi + f2 * cos_gamma
    alpha_theta = f1 * sin_beta * sin_psi
    alpha_z = -f1 * cos_beta - f2 * sin_gamma - f3
    return alpha_r, alpha_theta, alpha_z


# ------------------------------------------------------- material scaling


def scaling_shape_function(internal_friction_coefficient: float) -> float:
    """The material-scaling cubic ``f_hat = 894 mu^3 - 386 mu^2 + 89 mu``.

    Args:
        internal_friction_coefficient: ``mu_int = tan(Phi)``.

    Returns:
        The dimensionless shape factor.

    Raises:
        CalibrationError: If ``mu`` is non-finite, non-positive, or lands
            outside the 0.3-0.9 band the cubic was fitted over -- where a
            cubic through three points is an extrapolation, not a fit.
    """
    mu = float(internal_friction_coefficient)
    if not math.isfinite(mu) or mu <= 0.0:
        raise CalibrationError(
            f"internal friction coefficient must be positive and finite, got {mu!r}"
        )
    if not 0.3 <= mu <= 0.9:
        raise CalibrationError(
            f"internal friction coefficient {mu:.3f} (Phi = "
            f"{math.degrees(math.atan(mu)):.1f} deg) is outside the 0.3-0.9 band "
            "the material-scaling cubic was fitted over; extrapolating a cubic "
            "past its fit is how a solver invents a material"
        )
    return 894.0 * mu**3 - 386.0 * mu**2 + 89.0 * mu


def internal_friction_mu(friction_angle_deg: float) -> float:
    """``mu_int = tan(Phi)``.

    Args:
        friction_angle_deg: Internal friction angle in degrees.

    Returns:
        The friction coefficient.

    Raises:
        CalibrationError: If the angle is not strictly between 0 and 90.
    """
    angle = float(friction_angle_deg)
    if not math.isfinite(angle) or not 0.0 < angle < 90.0:
        raise CalibrationError(
            f"friction angle must lie strictly between 0 and 90 deg, got {angle!r}"
        )
    return math.tan(math.radians(angle))


def material_scaling_pa_per_m(
    *,
    bulk_density_kg_m3: float,
    friction_angle_deg: float,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> float:
    """``xi_n = rho_c * g * f_hat(mu_int)`` in Pa/m.

    Recalibrates the generic response surface to a specific sand from two
    measurable properties.  Reproduces the addendum's table to three
    significant figures, e.g. ``rho_c = 1550``, ``Phi = 35 deg`` gives
    ``2.73e6 Pa/m``.

    Args:
        bulk_density_kg_m3: Bulk density ``rho_c`` of the bed.
        friction_angle_deg: Internal friction angle ``Phi``.
        gravity_m_s2: Gravitational acceleration.

    Returns:
        The normal stress scale, Pa per metre of depth.

    Raises:
        CalibrationError: If any input is unusable.
    """
    density = float(bulk_density_kg_m3)
    if not math.isfinite(density) or density <= 0.0:
        raise CalibrationError(f"bulk density must be positive, got {density!r} kg/m^3")
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
        raise CalibrationError(f"gravity must be positive, got {gravity_m_s2!r}")
    shape = scaling_shape_function(internal_friction_mu(friction_angle_deg))
    return density * gravity_m_s2 * shape


@dataclass(frozen=True)
class MaterialResponse:
    """Everything the F0 solver needs to know about the sand.

    Attributes:
        normal_stress_scale_pa_per_m: ``xi_n``.
        bulk_density_kg_m3: ``rho`` in the inertial term.
        inertial_lambda: ``lambda`` in the inertial term.
        surface_friction_mu: Club-on-sand friction, used only by the
            tangential cutoff.
        grain_diameter_m: ``d50``, used by the validity envelope.
        friction_angle_deg: ``Phi``, retained for reporting.
        provenance: Per-constant paper trail. Every entry is borrowed.
    """

    normal_stress_scale_pa_per_m: float
    bulk_density_kg_m3: float
    inertial_lambda: float
    surface_friction_mu: float
    grain_diameter_m: float
    friction_angle_deg: float
    provenance: SandProvenance

    def __post_init__(self) -> None:
        checks = {
            "normal_stress_scale_pa_per_m": self.normal_stress_scale_pa_per_m,
            "bulk_density_kg_m3": self.bulk_density_kg_m3,
            "inertial_lambda": self.inertial_lambda,
            "grain_diameter_m": self.grain_diameter_m,
        }
        for name, value in checks.items():
            if not math.isfinite(value) or value <= 0.0:
                raise CalibrationError(f"{name} must be positive, got {value!r}")
        if (
            not math.isfinite(self.surface_friction_mu)
            or self.surface_friction_mu < 0.0
        ):
            raise CalibrationError(
                f"surface_friction_mu must be non-negative, got "
                f"{self.surface_friction_mu!r}"
            )

    @property
    def inertial_stress_scale_pa_s2_per_m2(self) -> float:
        """``lambda * rho``: the coefficient of ``v_n^2`` in the DRFT term."""
        return self.inertial_lambda * self.bulk_density_kg_m3

    @property
    def vertical_plate_alpha_z_n_per_cm3(self) -> float:
        """``alpha_z(0, pi/2)`` in the published N/cm^3 unit.

        The Quikrete analogue measured 2.02 N/cm^3.  This value is the
        material-scaling cubic's independent prediction of the same
        quantity, so the gap between them is a real cross-check rather
        than a restatement.
        """
        return self.normal_stress_scale_pa_per_m * VERTICAL_PLATE_ALPHA_Z / 1e6

    def crossover_speed_m_s(self, depth_m: float) -> float:
        """Speed at which the inertial term matches the depth term.

        ``sqrt(xi_n * alpha_z * |z| / (lambda rho))``.  About 7 m/s at a
        40 mm divot, against the 6.8 m/s quoted in the research digest --
        greenside delivery is 20-27 m/s, so the inertial term carries
        roughly 90% of the load.

        Args:
            depth_m: Depth below the free surface, positive downward.

        Returns:
            The crossover speed in m/s.

        Raises:
            CalibrationError: If ``depth_m`` is not positive and finite.
        """
        depth = float(depth_m)
        if not math.isfinite(depth) or depth <= 0.0:
            raise CalibrationError(f"depth must be positive, got {depth!r} m")
        numerator = self.normal_stress_scale_pa_per_m * VERTICAL_PLATE_ALPHA_Z * depth
        return math.sqrt(numerator / self.inertial_stress_scale_pa_s2_per_m2)

    @classmethod
    def from_sand_state(
        cls,
        sand: SandState,
        *,
        inertial_lambda: float = PLATE_DRAG_LAMBDA,
        surface_friction_mu: float | None = None,
        gravity_m_s2: float = GRAVITY_M_S2,
    ) -> MaterialResponse:
        """Scale the generic response surface to a :class:`SandState`.

        Args:
            sand: The bed being struck.
            inertial_lambda: DRFT inertial coefficient. Defaults to the
                oblique-plate value 1.1; the published spread is 1.0-2.8.
            surface_friction_mu: Club-on-sand friction. Defaults to two
                thirds of the internal friction coefficient.
            gravity_m_s2: Gravitational acceleration.

        Returns:
            A material response whose provenance carries every borrowed
            constant, the sand's own provenance included.

        Raises:
            CalibrationError: If the sand's properties fall outside the
                band the scaling cubic was fitted over.
        """
        if not isinstance(sand, SandState):
            raise CalibrationError(f"expected a SandState, got {type(sand).__name__}")
        mu_internal = internal_friction_mu(sand.friction_angle_deg)
        scale = material_scaling_pa_per_m(
            bulk_density_kg_m3=sand.bulk_density_kg_m3,
            friction_angle_deg=sand.friction_angle_deg,
            gravity_m_s2=gravity_m_s2,
        )
        entries = dict(sand.provenance.entries)
        entries.update(RFT_COEFFICIENT_PROVENANCE)
        return cls(
            normal_stress_scale_pa_per_m=scale,
            bulk_density_kg_m3=sand.bulk_density_kg_m3,
            inertial_lambda=float(inertial_lambda),
            surface_friction_mu=(
                (2.0 / 3.0) * mu_internal
                if surface_friction_mu is None
                else float(surface_friction_mu)
            ),
            grain_diameter_m=sand.d50_m,
            friction_angle_deg=sand.friction_angle_deg,
            provenance=SandProvenance(entries=entries),
        )

    @classmethod
    def from_vertical_plate_intrusion(
        cls,
        *,
        force_n: float,
        area_m2: float,
        depth_m: float,
        bulk_density_kg_m3: float,
        friction_angle_deg: float,
        grain_diameter_m: float,
        inertial_lambda: float = PLATE_DRAG_LAMBDA,
        surface_friction_mu: float | None = None,
    ) -> MaterialResponse:
        """One-shot calibration from a single vertical flat-plate intrusion.

        ``xi_n = F / (alpha_z_generic(0, pi/2, 0) * ds * |z|)`` -- the
        addendum's one-measurement recalibration.  Use it when a plate
        test on the actual sand exists; :meth:`from_sand_state` is the
        fallback that borrows from the analogue instead.

        Args:
            force_n: Measured vertical resistance.
            area_m2: Plate area.
            depth_m: Plate depth below the free surface, positive down.
            bulk_density_kg_m3: Bed bulk density, for the inertial term.
            friction_angle_deg: Internal friction angle, for reporting.
            grain_diameter_m: Median grain diameter.
            inertial_lambda: DRFT inertial coefficient.
            surface_friction_mu: Club-on-sand friction.

        Returns:
            A material response whose normal stress scale is measured on
            this material rather than borrowed.

        Raises:
            CalibrationError: If any measurement is not positive finite.
        """
        for name, value in (
            ("force_n", force_n),
            ("area_m2", area_m2),
            ("depth_m", depth_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise CalibrationError(
                    f"{name} must be positive and finite, got {value!r}"
                )
        scale = float(force_n) / (VERTICAL_PLATE_ALPHA_Z * area_m2 * depth_m)
        mu_internal = internal_friction_mu(friction_angle_deg)
        entries = dict(RFT_COEFFICIENT_PROVENANCE)
        entries["normal_stress_scale_pa_per_m"] = PropertyProvenance(
            basis=ProvenanceBasis.MEASURED,
            source=(
                f"vertical flat-plate intrusion: {force_n:.4g} N on "
                f"{area_m2:.4g} m^2 at {depth_m:.4g} m depth"
            ),
            note="One-shot calibration per the research digest addendum, "
            "section 3. The polynomial shape remains borrowed; only its "
            "scalar magnitude is measured.",
        )
        return cls(
            normal_stress_scale_pa_per_m=scale,
            bulk_density_kg_m3=float(bulk_density_kg_m3),
            inertial_lambda=float(inertial_lambda),
            surface_friction_mu=(
                (2.0 / 3.0) * mu_internal
                if surface_friction_mu is None
                else float(surface_friction_mu)
            ),
            grain_diameter_m=float(grain_diameter_m),
            friction_angle_deg=float(friction_angle_deg),
            provenance=SandProvenance(entries=entries),
        )

    def borrowed_constants(self) -> tuple[str, ...]:
        """Names of every constant taken from an analogue material."""
        return self.provenance.borrowed_properties()

    def measured_constants(self) -> tuple[str, ...]:
        """Names of every constant measured on the material itself."""
        return self.provenance.measured_properties()
