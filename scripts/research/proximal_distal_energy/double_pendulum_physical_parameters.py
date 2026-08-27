"""Exact physical-to-base parameter map for the planar double pendulum."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.shared.python.simulation_backends.model_params import GolfModelParams


FloatArray = npt.NDArray[np.float64]

BASE_COEFFICIENT_NAMES = (
    "alpha_proximal_inertia",
    "beta_coupling_inertia",
    "delta_distal_inertia",
    "gamma_proximal_gravity",
    "gamma_distal_gravity",
    "damping_proximal",
    "damping_distal",
)
BASE_COEFFICIENT_UNITS = (
    "kg*m^2",
    "kg*m^2",
    "kg*m^2",
    "N*m",
    "N*m",
    "N*m*s/rad",
    "N*m*s/rad",
)
PHYSICAL_PARAMETER_NAMES = (
    "upper_mass_kg",
    "upper_com_m",
    "upper_inertia_proximal_kg_m2",
    "lower_mass_kg",
    "upper_length_m",
    "lower_com_m",
    "lower_inertia_proximal_kg_m2",
    "gravity_m_s2",
    "plane_inclination_rad",
    "damping_proximal_nm_s_rad",
    "damping_distal_nm_s_rad",
)


@dataclass(frozen=True, slots=True)
class DoublePendulumPhysicalParameters:
    """Reduced physical parameterization upstream of the seven coefficients."""

    upper_mass_kg: float
    upper_com_m: float
    upper_inertia_proximal_kg_m2: float
    lower_mass_kg: float
    upper_length_m: float
    lower_com_m: float
    lower_inertia_proximal_kg_m2: float
    gravity_m_s2: float
    plane_inclination_rad: float
    damping_proximal_nm_s_rad: float
    damping_distal_nm_s_rad: float

    def __post_init__(self) -> None:
        positive = PHYSICAL_PARAMETER_NAMES[:8]
        nonnegative = PHYSICAL_PARAMETER_NAMES[9:]
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
            if item.name in positive and value <= 0.0:
                raise ValueError(f"{item.name} must be positive")
            if item.name in nonnegative and value < 0.0:
                raise ValueError(f"{item.name} must be nonnegative")
        if abs(self.plane_inclination_rad) >= math.pi / 2:
            raise ValueError("plane_inclination_rad must lie strictly within +/- pi/2")

    @classmethod
    def from_model(cls, model: GolfModelParams) -> DoublePendulumPhysicalParameters:
        """Extract exactly the physical combinations used by the ODE equations."""
        rendered = model.to_double_pendulum_parameters()
        if not rendered.gravity_enabled or not rendered.constrained_to_plane:
            raise ValueError(
                "the reduced map requires enabled gravity constrained to the plane"
            )
        return cls(
            upper_mass_kg=rendered.upper_segment.mass_kg,
            upper_com_m=rendered.upper_segment.center_of_mass_distance,
            upper_inertia_proximal_kg_m2=(
                rendered.upper_segment.inertia_about_proximal_joint
            ),
            lower_mass_kg=rendered.lower_segment.total_mass,
            upper_length_m=rendered.upper_segment.length_m,
            lower_com_m=rendered.lower_segment.center_of_mass_distance,
            lower_inertia_proximal_kg_m2=(
                rendered.lower_segment.inertia_about_proximal_joint
            ),
            gravity_m_s2=rendered.gravity_m_s2,
            plane_inclination_rad=rendered.plane_inclination_rad,
            damping_proximal_nm_s_rad=rendered.damping_shoulder,
            damping_distal_nm_s_rad=rendered.damping_wrist,
        )

    @property
    def projected_gravity_m_s2(self) -> float:
        """Return the only gravity/plane combination present in the equations."""
        return self.gravity_m_s2 * math.cos(self.plane_inclination_rad)

    def vector(self) -> FloatArray:
        """Return parameters in :data:`PHYSICAL_PARAMETER_NAMES` order."""
        return np.array([getattr(self, name) for name in PHYSICAL_PARAMETER_NAMES])

    def base_coefficients(self) -> FloatArray:
        """Map physical values to the seven inverse-dynamics coefficients."""
        m1 = self.upper_mass_kg
        r1 = self.upper_com_m
        i1 = self.upper_inertia_proximal_kg_m2
        m2 = self.lower_mass_kg
        l1 = self.upper_length_m
        r2 = self.lower_com_m
        i2 = self.lower_inertia_proximal_kg_m2
        gravity = self.projected_gravity_m_s2
        return np.array(
            [
                i1 + i2 + m2 * l1**2,
                m2 * l1 * r2,
                i2,
                (m1 * r1 + m2 * l1) * gravity,
                m2 * r2 * gravity,
                self.damping_proximal_nm_s_rad,
                self.damping_distal_nm_s_rad,
            ],
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class StructuralRankWitness:
    """Nonzero seven-by-seven minor proving full row rank of the map."""

    parameter_columns: tuple[str, ...]
    determinant: float
    closed_form: str


def parameter_map_jacobian(parameters: DoublePendulumPhysicalParameters) -> FloatArray:
    """Return the exact ``7 x 11`` Jacobian of the physical-to-base map."""
    p = parameters
    m1, r1 = p.upper_mass_kg, p.upper_com_m
    m2, l1, r2 = p.lower_mass_kg, p.upper_length_m, p.lower_com_m
    gravity = p.gravity_m_s2
    plane = p.plane_inclination_rad
    projected = p.projected_gravity_m_s2
    d_projected_d_gravity = math.cos(plane)
    d_projected_d_plane = -gravity * math.sin(plane)
    proximal_first_moment = m1 * r1 + m2 * l1
    distal_first_moment = m2 * r2
    jacobian = np.zeros((len(BASE_COEFFICIENT_NAMES), len(PHYSICAL_PARAMETER_NAMES)))
    jacobian[0, [2, 3, 4, 6]] = (1.0, l1**2, 2.0 * m2 * l1, 1.0)
    jacobian[1, [3, 4, 5]] = (l1 * r2, m2 * r2, m2 * l1)
    jacobian[2, 6] = 1.0
    jacobian[3, [0, 1, 3, 4]] = (
        r1 * projected,
        m1 * projected,
        l1 * projected,
        m2 * projected,
    )
    jacobian[3, 7] = proximal_first_moment * d_projected_d_gravity
    jacobian[3, 8] = proximal_first_moment * d_projected_d_plane
    jacobian[4, [3, 5]] = (r2 * projected, m2 * projected)
    jacobian[4, 7] = distal_first_moment * d_projected_d_gravity
    jacobian[4, 8] = distal_first_moment * d_projected_d_plane
    jacobian[5, 9] = 1.0
    jacobian[6, 10] = 1.0
    return jacobian


def physical_parameter_rank_witness(
    parameters: DoublePendulumPhysicalParameters,
) -> StructuralRankWitness:
    """Return an analytical nonzero minor proving row rank seven."""
    determinant = (
        parameters.lower_mass_kg**2
        * parameters.upper_com_m
        * parameters.lower_com_m
        * parameters.projected_gravity_m_s2**2
    )
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("the structural-rank witness must be positive and finite")
    return StructuralRankWitness(
        parameter_columns=(
            "upper_inertia_proximal_kg_m2",
            "upper_length_m",
            "lower_inertia_proximal_kg_m2",
            "upper_mass_kg",
            "lower_com_m",
            "damping_proximal_nm_s_rad",
            "damping_distal_nm_s_rad",
        ),
        determinant=determinant,
        closed_form="m2^2 * r1 * r2 * (g*cos(phi))^2",
    )


def exact_invariance_counterexamples(
    baseline: DoublePendulumPhysicalParameters,
) -> dict[str, DoublePendulumPhysicalParameters]:
    """Construct distinct physical parameters with identical base coefficients."""
    upper_scale = 1.10
    upper_tradeoff = replace(
        baseline,
        upper_mass_kg=baseline.upper_mass_kg * upper_scale,
        upper_com_m=baseline.upper_com_m / upper_scale,
    )
    alternate_plane = baseline.plane_inclination_rad * 0.8
    if math.isclose(alternate_plane, baseline.plane_inclination_rad, abs_tol=1e-12):
        alternate_plane = 0.1
    gravity_tradeoff = replace(
        baseline,
        gravity_m_s2=baseline.projected_gravity_m_s2 / math.cos(alternate_plane),
        plane_inclination_rad=alternate_plane,
    )
    lower_scale = 1.01
    lower_delta = (lower_scale - 1.0) * baseline.lower_mass_kg
    lower_coupling = replace(
        baseline,
        lower_mass_kg=baseline.lower_mass_kg * lower_scale,
        lower_com_m=baseline.lower_com_m / lower_scale,
        upper_inertia_proximal_kg_m2=(
            baseline.upper_inertia_proximal_kg_m2
            - lower_delta * baseline.upper_length_m**2
        ),
        upper_com_m=(
            baseline.upper_com_m
            - lower_delta * baseline.upper_length_m / baseline.upper_mass_kg
        ),
    )
    alternatives = {
        "gravity_plane_tradeoff": gravity_tradeoff,
        "lower_mass_com_coupling": lower_coupling,
        "upper_mass_com_tradeoff": upper_tradeoff,
    }
    reference = baseline.base_coefficients()
    for name, alternative in alternatives.items():
        if not np.allclose(
            alternative.base_coefficients(), reference, rtol=1e-12, atol=1e-12
        ):
            raise RuntimeError(f"{name} failed to preserve base coefficients")
    return alternatives
