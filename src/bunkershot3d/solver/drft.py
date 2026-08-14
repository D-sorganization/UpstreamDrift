"""Dynamic Resistive Force Theory solver (issue #8611, ADR-0032).

F0 tier: the default solver for design iteration. Computes granular resistance
forces on intruding geometries via local stress integration.

Key equations from research-digest-addendum.md section 3:

    t = alpha(beta, gamma) * H(-z_tilde) * |z_tilde|  -  n_hat * lambda * rho * v_n^2
    z_tilde = z + delta_h

Where:
- alpha is from the 20-term polynomial table
- lambda ~ 1.1 for oblique plates, 1.4 for vertical spheres
- delta_h is the dynamic structural correction (geometry-specific)

The stress is:
    sigma = xi_n * alpha_z * |z|  (quasi-static)
    sigma += lambda * rho * v_n^2  (inertial correction)
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass


from src.shared.python.contracts import require

from .coefficients import compute_alpha_components
from .envelope import (
    DRFT_LAMBDA_OBLIQUE,
    ValidityVerdict,
    compute_froude_number,
    compute_micro_inertial,
)
from .material import compute_xi_n

__all__ = [
    "DRFTResult",
    "DRFTSolver",
    "FidelityTier",
]


class FidelityTier(enum.Enum):
    """Solver fidelity tier per ADR-0032."""

    F0 = "F0"  # DRFT, ~ms/shot
    F1 = "F1"  # Reduced-order continuum, ~seconds
    F2 = "F2"  # MPM, ~30-90 min
    F3 = "F3"  # DEM, intractable at true scale


@dataclass(frozen=True, slots=True)
class DRFTResult:
    """Result from a DRFT force calculation.

    Attributes:
        force_x: Force in x direction [N].
        force_y: Force in y direction [N].
        force_z: Force in z direction [N] (positive = upward resistance).
        torque_x: Torque about x axis [N.m].
        torque_y: Torque about y axis [N.m].
        torque_z: Torque about z axis [N.m].
        fidelity_tier: Always F0 for DRFT.
        validity: Envelope assessment.
        quasi_static_fraction: Fraction of force from quasi-static term.
        inertial_fraction: Fraction of force from inertial term.
    """

    force_x: float
    force_y: float
    force_z: float
    torque_x: float
    torque_y: float
    torque_z: float
    fidelity_tier: FidelityTier
    validity: ValidityVerdict
    quasi_static_fraction: float
    inertial_fraction: float


class DRFTSolver:
    """Dynamic Resistive Force Theory solver.

    This is the F0 tier default solver. It integrates local stress responses
    over the intruder surface to compute net force and torque.

    Args:
        bulk_density_kg_m3: Dry bulk density of sand [kg/m^3].
        friction_angle_deg: Internal friction angle [deg].
        surface_friction: Surface friction coefficient mu_surf.
        enable_dynamic_terms: Whether to include the inertial correction.
        inertial_lambda: Lambda coefficient for inertial term (default 1.1).
        grain_diameter_m: Median grain diameter for envelope calculation [m].
    """

    def __init__(
        self,
        bulk_density_kg_m3: float,
        friction_angle_deg: float,
        surface_friction: float = 0.5,
        enable_dynamic_terms: bool = True,
        inertial_lambda: float = DRFT_LAMBDA_OBLIQUE,
        grain_diameter_m: float = 0.00033,  # USGA median
    ) -> None:
        require(bulk_density_kg_m3 > 0, "bulk density must be positive")
        require(0 < friction_angle_deg < 90, "friction angle must be in (0, 90) deg")
        require(surface_friction >= 0, "surface friction must be non-negative")
        require(inertial_lambda > 0, "inertial lambda must be positive")
        require(grain_diameter_m > 0, "grain diameter must be positive")

        self._bulk_density = bulk_density_kg_m3
        self._friction_angle_deg = friction_angle_deg
        self._surface_friction = surface_friction
        self._enable_dynamic = enable_dynamic_terms
        self._lambda = inertial_lambda
        self._grain_diameter = grain_diameter_m

        # Pre-compute material scaling factor
        self._xi_n = compute_xi_n(bulk_density_kg_m3, friction_angle_deg)

    def flat_plate_intrusion(
        self,
        width_m: float,
        height_m: float,
        depth_m: float,
        velocity_m_s: float,
        attack_angle_rad: float,
    ) -> DRFTResult:
        """Compute force on a flat plate intruding into sand.

        This is the simplest validation case: a rectangular plate at a given
        attack angle, moving at constant velocity to a specified depth.

        Args:
            width_m: Plate width [m].
            height_m: Plate height [m].
            depth_m: Depth below sand surface [m] (positive downward).
            velocity_m_s: Intrusion velocity [m/s].
            attack_angle_rad: Attack angle gamma [rad]. pi/2 = vertical.

        Returns:
            DRFTResult with force components and metadata.

        Raises:
            ValueError: If query is outside validity envelope and dynamic
                       terms are not active.
        """
        require(width_m > 0, "width must be positive")
        require(height_m > 0, "height must be positive")
        require(depth_m >= 0, "depth must be non-negative")
        require(velocity_m_s >= 0, "velocity must be non-negative")

        # Compute validity envelope
        length_scale = max(width_m, height_m)
        froude = compute_froude_number(velocity_m_s, length_scale)
        micro_inertial = compute_micro_inertial(
            velocity_m_s, self._grain_diameter, length_scale
        )
        depth_ratio = self._grain_diameter / length_scale

        validity = ValidityVerdict.evaluate(
            froude=froude,
            micro_inertial=micro_inertial,
            depth_ratio=depth_ratio,
            dynamic_terms_active=self._enable_dynamic,
        )

        if validity.should_refuse:
            raise ValueError(validity.reason)

        # Compute stress response
        # For a flat plate: beta = 0 (no surface tilt), psi = 0 (no twist)
        beta = 0.0
        gamma = attack_angle_rad
        psi = 0.0

        alpha_r, alpha_theta, alpha_z = compute_alpha_components(beta, gamma, psi)

        # Plate area
        area = width_m * height_m

        # Quasi-static term: F_qs = xi_n * |alpha_z| * z * A
        # The integral of depth over the plate. For uniform depth, this is:
        # F = xi_n * alpha_z * (z_avg) * A
        # where z_avg = depth / 2 for a uniformly submerged plate
        z_avg = depth_m / 2.0
        f_quasi_static = self._xi_n * abs(alpha_z) * z_avg * area

        # Inertial term: F_inertial = lambda * rho * v_n^2 * A
        # v_n is the normal component of velocity
        v_n = velocity_m_s * abs(math.sin(gamma))  # normal velocity
        f_inertial = 0.0
        if self._enable_dynamic:
            f_inertial = self._lambda * self._bulk_density * v_n**2 * area

        # Total force
        f_total = f_quasi_static + f_inertial

        # Decompose into components based on attack angle
        # For vertical intrusion (gamma = pi/2), force is purely vertical (z)
        # For horizontal (gamma = 0), force is horizontal (x)
        f_z = f_total * abs(math.sin(gamma))
        f_x = f_total * abs(math.cos(gamma))
        f_y = 0.0  # No lateral force for aligned plate

        # Torque is zero for a centered plate
        torque_x = 0.0
        torque_y = 0.0
        torque_z = 0.0

        # Compute fractions
        total_magnitude = f_quasi_static + f_inertial
        if total_magnitude > 0:
            qs_fraction = f_quasi_static / total_magnitude
            inertial_fraction = f_inertial / total_magnitude
        else:
            qs_fraction = 1.0
            inertial_fraction = 0.0

        return DRFTResult(
            force_x=f_x,
            force_y=f_y,
            force_z=f_z,
            torque_x=torque_x,
            torque_y=torque_y,
            torque_z=torque_z,
            fidelity_tier=FidelityTier.F0,
            validity=validity,
            quasi_static_fraction=qs_fraction,
            inertial_fraction=inertial_fraction,
        )
