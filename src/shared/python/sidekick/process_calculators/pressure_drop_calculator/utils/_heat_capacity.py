import logging
import math

from ....utils.unit_constants import R_UNIVERSAL as R_UNIVERSAL_J_MOL_K
from ...constants import DEFAULT_GAMMA_DIATOMIC, GAMMA_UPPER_BOUND
from ._component_database import GAS_DATABASE, R_UNIVERSAL

logger = logging.getLogger(__name__)


def calculate_ideal_gas_cp(component: str, temperature: float) -> float:
    """Calculate ideal gas heat capacity using Shomate equation.

    Cp = A + B*t + C*t² + D*t³ + E/t²

    where t = T(K)/1000

    Args:
        component: Gas component name
        temperature: Temperature (K)

    Returns:
        Cp in J/(mol·K)

    Reference:
        NIST Chemistry WebBook, Shomate Equation
    """
    if not (component is not None):
        raise ValueError("component must be provided")
    if not (component is not None):
        raise ValueError("component must be provided")
    if component not in GAS_DATABASE:
        logger.warning(f"Component '{component}' not in database, using Air Cp")
        component = "Air"

    props = GAS_DATABASE[component]
    A, B, C, D, E = props.ideal_gas_cp_coeffs

    t = temperature / 1000.0  # Convert to kK for Shomate equation

    # Shomate equation
    cp = A + B * t + C * t**2 + D * t**3 + E / (t**2)

    return cp


def calculate_mixture_cp(composition: dict[str, float], temperature: float) -> float:
    """Calculate mixture ideal gas heat capacity using mole-fraction weighting.

    Cp_mix = Σ(y_i × Cp_i)

    Args:
        composition: Dictionary of {component: mole_fraction}
        temperature: Temperature (K)

    Returns:
        Mixture Cp in J/(mol·K)
    """
    if not (composition is not None):
        raise ValueError("composition must be provided")
    if not (composition is not None):
        raise ValueError("composition must be provided")
    cp_mix = 0.0

    for component, mole_frac in composition.items():
        if component not in GAS_DATABASE:
            logger.warning(f"Component '{component}' not in database, skipping Cp")
            continue
        cp_i = calculate_ideal_gas_cp(component, temperature)
        cp_mix += mole_frac * cp_i

    logger.debug(f"Mixture Cp = {cp_mix:.2f} J/(mol·K) at T = {temperature:.0f} K")
    return cp_mix


def calculate_heat_capacity_ratio(
    composition: dict[str, float], temperature: float
) -> float:
    """Calculate heat capacity ratio (gamma = Cp/Cv) for a gas mixture.

    For ideal gases: Cv = Cp - R
    Therefore: γ = Cp / (Cp - R)

    Args:
        composition: Dictionary of {component: mole_fraction}
        temperature: Temperature (K)

    Returns:
        Heat capacity ratio γ (dimensionless)

    Note:
        - For monatomic gases: γ ≈ 1.67
        - For diatomic gases (N2, O2, CO): γ ≈ 1.40
        - For triatomic gases (CO2, H2O): γ ≈ 1.30
        - For combustion/syngas mixtures: γ ≈ 1.25-1.40

    Reference:
        Ideal gas relations: Cp - Cv = R (universal gas constant per mole)
    """
    if not (composition is not None):
        raise ValueError("composition must be provided")
    if not (composition is not None):
        raise ValueError("composition must be provided")
    R_GAS = R_UNIVERSAL_J_MOL_K  # J/(mol·K)

    cp_mix = calculate_mixture_cp(composition, temperature)

    if cp_mix <= R_GAS:
        logger.error(f"Invalid Cp = {cp_mix:.2f}, must be > R = {R_GAS:.2f}")
        return float(DEFAULT_GAMMA_DIATOMIC)  # Default for diatomic gases

    cv_mix = cp_mix - R_GAS
    gamma = cp_mix / cv_mix

    # Physical bounds check
    if gamma < 1.0 or gamma > GAMMA_UPPER_BOUND:
        logger.warning(
            f"Calculated gamma = {gamma:.3f} outside physical bounds [1.0, 1.7]"
        )
        gamma = max(1.0, min(gamma, GAMMA_UPPER_BOUND))

    logger.debug(
        f"Heat capacity ratio γ = {gamma:.4f} (Cp = {cp_mix:.1f}, Cv = {cv_mix:.1f})"
    )
    return float(gamma)


def calculate_speed_of_sound(
    composition: dict[str, float],
    temperature: float,
    molecular_weight: float | None = None,
) -> float:
    """Calculate speed of sound in a gas mixture.

    a = √(γ × R × T / M)

    Args:
        composition: Dictionary of {component: mole_fraction}
        temperature: Temperature (K)
        molecular_weight: Optional pre-calculated MW (kg/kmol)

    Returns:
        Speed of sound (m/s)

    Reference:
        Ideal gas isentropic speed of sound formula
    """
    if not (composition is not None):
        raise ValueError("composition must be provided")
    if not (composition is not None):
        raise ValueError("composition must be provided")
    if molecular_weight is None:
        from ._mixture_properties import calculate_mixture_molecular_weight

        molecular_weight = calculate_mixture_molecular_weight(composition)

    gamma = calculate_heat_capacity_ratio(composition, temperature)

    # R_specific = R / M (J/(kg·K))
    R_specific = R_UNIVERSAL / molecular_weight

    speed_of_sound = math.sqrt(gamma * R_specific * temperature)

    logger.debug(
        f"Speed of sound = {speed_of_sound:.1f} m/s (γ = {gamma:.3f}, T = {temperature:.0f} K)"
    )
    return speed_of_sound
