"""Material scaling for 3D-RFT (issue #8611).

The stress response from the PNAS coefficient table must be scaled to the
actual sand being modelled. The scaling factor xi_n depends on bulk density
and internal friction angle:

    xi_n = rho_c * g * f_hat(mu)
    f_hat = 894*mu^3 - 386*mu^2 + 89*mu
    mu = tan(phi)

PROVENANCE: Agarwal, Goldman and Kamrin, PNAS 120 (2023), Eq. (3) and
            surrounding text. The f_hat cubic was fitted to DEM simulations
            over mu in [0.3, 0.9]. Values outside this range are extrapolation.
"""

from __future__ import annotations

import math


__all__ = [
    "GRAVITY_M_S2",
    "compute_f_hat",
    "compute_xi_n",
    "mu_from_friction_angle_deg",
]

#: Gravitational acceleration [m/s^2].
GRAVITY_M_S2: float = 9.81


def mu_from_friction_angle_deg(phi_deg: float) -> float:
    """Convert friction angle to friction coefficient.

    Args:
        phi_deg: Internal friction angle [deg].

    Returns:
        mu = tan(phi).

    Raises:
        ValueError: If phi is not in (0, 90).
    """
    if not 0.0 < phi_deg < 90.0:
        raise ValueError(f"friction angle must be in (0, 90) deg, got {phi_deg}")
    return math.tan(math.radians(phi_deg))


def compute_f_hat(mu: float) -> float:
    """Compute the material scaling cubic f_hat(mu).

    The cubic 894*mu^3 - 386*mu^2 + 89*mu was fitted to DEM simulations
    over mu in [0.3, 0.9]. Values outside this range are extrapolation.

    Args:
        mu: Internal friction coefficient, tan(phi).

    Returns:
        Dimensionless scaling factor f_hat.
    """
    return 894.0 * mu**3 - 386.0 * mu**2 + 89.0 * mu


def compute_xi_n(
    bulk_density_kg_m3: float,
    friction_angle_deg: float,
) -> float:
    """Compute the complete material scaling factor xi_n [Pa/m].

    The stress at depth z is: sigma = xi_n * alpha * |z|

    Args:
        bulk_density_kg_m3: Dry bulk density of the sand [kg/m^3].
        friction_angle_deg: Internal friction angle [deg].

    Returns:
        xi_n [Pa/m], the stress scaling factor.

    Raises:
        ValueError: If inputs are out of physical range.

    Note:
        Typical bunker sand: rho_c ~ 1450-1700 kg/m^3, phi ~ 30-40 deg.
        Reference condition: rho_c = 1550, phi = 35 deg => xi_n ~ 2.73e6 Pa/m.
    """
    if bulk_density_kg_m3 <= 0:
        raise ValueError(f"bulk density must be positive, got {bulk_density_kg_m3}")

    mu = mu_from_friction_angle_deg(friction_angle_deg)
    f_hat = compute_f_hat(mu)

    xi_n = bulk_density_kg_m3 * GRAVITY_M_S2 * f_hat

    return xi_n
