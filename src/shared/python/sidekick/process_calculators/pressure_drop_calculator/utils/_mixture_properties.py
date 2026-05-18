import logging

from ._component_database import GAS_DATABASE, R_UNIVERSAL

logger = logging.getLogger(__name__)


def calculate_mixture_molecular_weight(composition: dict[str, float]) -> float:
    """Calculate mixture molecular weight using mole fractions.

    MW_mix = Σ(y_i × MW_i)

    Args:
        composition: Dictionary of {component: mole_fraction}

    Returns:
        Mixture molecular weight (kg/kmol)

    Example:
        >>> comp = {'H2': 0.3, 'CO': 0.4, 'CO2': 0.3}
        >>> mw = calculate_mixture_molecular_weight(comp)
        >>> print(f"MW = {mw:.2f} kg/kmol")
    """
    mw_mix = 0.0
    for component, mole_frac in composition.items():
        if component not in GAS_DATABASE:
            logger.warning(f"Component '{component}' not in database, skipping")
            continue
        mw_mix += mole_frac * GAS_DATABASE[component].molecular_weight

    logger.debug(f"Mixture MW = {mw_mix:.3f} kg/kmol")
    return mw_mix


def calculate_ideal_gas_density(
    molecular_weight: float, temperature: float, pressure: float
) -> float:
    """Calculate ideal gas density using ideal gas law.

    ρ = P × MW / (R × T)

    Args:
        molecular_weight: Molecular weight (kg/kmol)
        temperature: Temperature (K)
        pressure: Pressure (Pa)

    Returns:
        Density (kg/m³)

    Reference:
        Ideal Gas Law: PV = nRT
    """
    if not (molecular_weight is not None):
        raise ValueError("molecular_weight must be provided")
    if not (molecular_weight is not None):
        raise ValueError("molecular_weight must be provided")
    density = (pressure * molecular_weight) / (R_UNIVERSAL * temperature)
    logger.debug(f"Ideal gas density = {density:.4f} kg/m³")
    return float(density)


def calculate_compressibility_factor(
    composition: dict[str, float], temperature: float, pressure: float
) -> float:
    """Calculate compressibility factor (Z) using pseudocritical properties.

    Uses Kay's rule for mixture pseudocritical properties and
    Lee-Kesler correlation for Z-factor.

    Args:
        composition: Dictionary of {component: mole_fraction}
        temperature: Temperature (K)
        pressure: Pressure (Pa)

    Returns:
        Compressibility factor Z (dimensionless)

    References:
        - Kay, W.B. (1936): "Density of Hydrocarbon Gases and Vapors"
        - Lee, B.I., Kesler, M.G. (1975): "A Generalized Thermodynamic Correlation"

    Example:
        >>> comp = {'CH4': 0.9, 'CO2': 0.1}
        >>> z = calculate_compressibility_factor(comp, 300, 50e5)
        >>> print(f"Z = {z:.4f}")
    """
    # Calculate pseudocritical properties using Kay's rule
    if not (composition is not None):
        raise ValueError("composition must be provided")
    if not (composition is not None):
        raise ValueError("composition must be provided")
    T_pc = 0.0  # Pseudocritical temperature
    P_pc = 0.0  # Pseudocritical pressure
    omega_mix = 0.0  # Mixture acentric factor

    for component, mole_frac in composition.items():
        if component not in GAS_DATABASE:
            continue
        props = GAS_DATABASE[component]
        T_pc += mole_frac * props.critical_temp
        P_pc += mole_frac * props.critical_pressure
        omega_mix += mole_frac * props.acentric_factor

    # Reduced properties
    T_r = temperature / T_pc  # Reduced temperature
    P_r = pressure / P_pc  # Reduced pressure

    # Lee-Kesler correlation for simple fluid (ω = 0)
    B0 = 0.083 - 0.422 / (T_r**1.6)
    C0 = 0.139 - 0.172 / (T_r**4.2)
    D0 = 0.0

    Z0 = 1.0 + B0 * P_r / T_r + C0 * (P_r / T_r) ** 2 + D0 * (P_r / T_r) ** 5

    # Lee-Kesler correction for acentric factor
    B1 = 0.139 - 0.172 / (T_r**4.2)
    C1 = 0.0

    Z1 = B1 * P_r / T_r + C1 * (P_r / T_r) ** 2

    # Final Z-factor
    Z = Z0 + omega_mix * Z1

    # Physical bounds
    Z = max(0.1, min(Z, 1.5))

    logger.debug(f"Z-factor calculation: T_r={T_r:.3f}, P_r={P_r:.3f}, Z={Z:.4f}")
    return float(Z)


def calculate_real_gas_density(
    molecular_weight: float, temperature: float, pressure: float, compressibility: float
) -> float:
    """Calculate real gas density with compressibility correction.

    ρ = (P × MW) / (Z × R × T)

    Args:
        molecular_weight: Molecular weight (kg/kmol)
        temperature: Temperature (K)
        pressure: Pressure (Pa)
        compressibility: Z-factor

    Returns:
        Density (kg/m³)
    """
    if not (molecular_weight is not None):
        raise ValueError("molecular_weight must be provided")
    if not (molecular_weight is not None):
        raise ValueError("molecular_weight must be provided")
    density = (pressure * molecular_weight) / (
        compressibility * R_UNIVERSAL * temperature
    )
    logger.debug(f"Real gas density = {density:.4f} kg/m³ (Z = {compressibility:.4f})")
    return float(density)
