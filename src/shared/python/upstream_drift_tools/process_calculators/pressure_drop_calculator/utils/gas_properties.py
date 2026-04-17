# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

#!/usr/bin/env python3
"""Gas mixture property calculations for pressure drop analysis.

Comprehensive calculation of thermophysical properties for gas mixtures
including combustion and gasification gases.

References:
    - Reid, Prausnitz, Poling: "The Properties of Gases and Liquids", 5th Ed (2001)
    - Perry's Chemical Engineers' Handbook, 9th Edition
    - Poling, Prausnitz, O'Connell: "Properties of Gases and Liquids", 5th Ed
    - Chapman-Enskog theory for gas viscosity
    - Lucas method for gas viscosity
    - Lee-Kesler correlation for compressibility factor
"""

import logging
import math
from dataclasses import dataclass

from ....utils.unit_constants import R_UNIVERSAL as R_UNIVERSAL_J_MOL_K
from ....utils.unit_constants import (
    R_UNIVERSAL_KMOL,
)
from ...constants import DEFAULT_GAMMA_DIATOMIC, GAMMA_UPPER_BOUND

logger = logging.getLogger(__name__)


# ============================================================================
# GAS COMPONENT PROPERTIES DATABASE
# ============================================================================


@dataclass
class ComponentProperties:
    """Thermophysical properties of pure gas components."""

    name: str
    molecular_weight: float  # kg/kmol
    critical_temp: float  # K
    critical_pressure: float  # Pa
    acentric_factor: float  # dimensionless
    dipole_moment: float  # Debye
    ideal_gas_cp_coeffs: tuple[
        float, float, float, float, float
    ]  # Shomate equation coefficients


# Comprehensive gas component database
GAS_DATABASE: dict[str, ComponentProperties] = {
    # Component: (MW, Tc, Pc, ω, μ, Cp coefficients)
    # MW in kg/kmol, Tc in K, Pc in Pa, ω dimensionless, μ in Debye
    "H2": ComponentProperties(
        name="Hydrogen",
        molecular_weight=2.016,
        critical_temp=33.2,
        critical_pressure=1.296e6,
        acentric_factor=-0.216,
        dipole_moment=0.0,
        ideal_gas_cp_coeffs=(33.066178, -11.363417, 11.432816, -2.772874, -0.158558),
    ),
    "CO": ComponentProperties(
        name="Carbon Monoxide",
        molecular_weight=28.010,
        critical_temp=132.9,
        critical_pressure=3.494e6,
        acentric_factor=0.048,
        dipole_moment=0.112,
        ideal_gas_cp_coeffs=(25.56759, 6.096130, 4.054656, -2.671301, 0.131021),
    ),
    "CO2": ComponentProperties(
        name="Carbon Dioxide",
        molecular_weight=44.010,
        critical_temp=304.2,
        critical_pressure=7.382e6,
        acentric_factor=0.228,
        dipole_moment=0.0,
        ideal_gas_cp_coeffs=(24.99735, 55.18696, -33.69137, 7.948387, -0.136638),
    ),
    "CH4": ComponentProperties(
        name="Methane",
        molecular_weight=16.043,
        critical_temp=190.6,
        critical_pressure=4.599e6,
        acentric_factor=0.011,
        dipole_moment=0.0,
        ideal_gas_cp_coeffs=(-0.703029, 108.4773, -42.52157, 5.862788, 0.678565),
    ),
    "C2H6": ComponentProperties(
        name="Ethane",
        molecular_weight=30.070,
        critical_temp=305.4,
        critical_pressure=4.880e6,
        acentric_factor=0.099,
        dipole_moment=0.0,
        ideal_gas_cp_coeffs=(-4.335729, 178.6345, -112.8668, 30.09716, 0.630798),
    ),
    "C2H4": ComponentProperties(
        name="Ethylene",
        molecular_weight=28.054,
        critical_temp=282.4,
        critical_pressure=5.042e6,
        acentric_factor=0.087,
        dipole_moment=0.0,
        ideal_gas_cp_coeffs=(-6.387880, 184.4019, -112.9718, 28.49593, 0.315554),
    ),
    "N2": ComponentProperties(
        name="Nitrogen",
        molecular_weight=28.014,
        critical_temp=126.2,
        critical_pressure=3.394e6,
        acentric_factor=0.037,
        dipole_moment=0.0,
        ideal_gas_cp_coeffs=(19.50583, 19.88705, -8.598535, 1.369784, 0.527601),
    ),
    "O2": ComponentProperties(
        name="Oxygen",
        molecular_weight=31.999,
        critical_temp=154.6,
        critical_pressure=5.043e6,
        acentric_factor=0.022,
        dipole_moment=0.0,
        ideal_gas_cp_coeffs=(31.32234, -20.23531, 57.86644, -36.50624, -0.007374),
    ),
    "H2O": ComponentProperties(
        name="Water Vapor",
        molecular_weight=18.015,
        critical_temp=647.1,
        critical_pressure=22.064e6,
        acentric_factor=0.344,
        dipole_moment=1.85,
        ideal_gas_cp_coeffs=(30.09200, 6.832514, 6.793435, -2.534480, 0.082139),
    ),
    "Ar": ComponentProperties(
        name="Argon",
        molecular_weight=39.948,
        critical_temp=150.9,
        critical_pressure=4.898e6,
        acentric_factor=-0.002,
        dipole_moment=0.0,
        ideal_gas_cp_coeffs=(20.78600, 0.0, 0.0, 0.0, 0.0),
    ),
    "H2S": ComponentProperties(
        name="Hydrogen Sulfide",
        molecular_weight=34.082,
        critical_temp=373.5,
        critical_pressure=9.000e6,
        acentric_factor=0.094,
        dipole_moment=0.97,
        ideal_gas_cp_coeffs=(26.88412, 18.67809, 3.434203, -3.378702, 0.135882),
    ),
    "NH3": ComponentProperties(
        name="Ammonia",
        molecular_weight=17.031,
        critical_temp=405.7,
        critical_pressure=11.357e6,
        acentric_factor=0.253,
        dipole_moment=1.47,
        ideal_gas_cp_coeffs=(19.99563, 49.77119, -15.37599, 1.921168, 0.189174),
    ),
    "Air": ComponentProperties(
        name="Air (Pseudo-component)",
        molecular_weight=28.97,
        critical_temp=132.5,
        critical_pressure=3.774e6,
        acentric_factor=0.035,
        dipole_moment=0.0,
        ideal_gas_cp_coeffs=(28.11, 0.1967e-2, 0.4802e-5, -1.966e-9, 0.0),
    ),
}


# Universal gas constant
# Note: This module uses J/(kmol·K) for R_UNIVERSAL
R_UNIVERSAL = R_UNIVERSAL_KMOL  # J/(kmol·K)


# ============================================================================
# HEAT CAPACITY AND GAMMA CALCULATIONS
# ============================================================================


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
    if component is None:
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
    if composition is None:
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
    if composition is None:
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
    if composition is None:
        raise ValueError("composition must be provided")
    if molecular_weight is None:
        molecular_weight = calculate_mixture_molecular_weight(composition)

    gamma = calculate_heat_capacity_ratio(composition, temperature)

    # R_specific = R / M (J/(kg·K))
    R_specific = R_UNIVERSAL / molecular_weight

    speed_of_sound = math.sqrt(gamma * R_specific * temperature)

    logger.debug(
        f"Speed of sound = {speed_of_sound:.1f} m/s (γ = {gamma:.3f}, T = {temperature:.0f} K)"
    )
    return speed_of_sound


# ============================================================================
# MIXTURE PROPERTY CALCULATIONS
# ============================================================================


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
    if molecular_weight is None:
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
    if composition is None:
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
    if molecular_weight is None:
        raise ValueError("molecular_weight must be provided")
    density = (pressure * molecular_weight) / (
        compressibility * R_UNIVERSAL * temperature
    )
    logger.debug(f"Real gas density = {density:.4f} kg/m³ (Z = {compressibility:.4f})")
    return float(density)


# ============================================================================
# VISCOSITY CALCULATIONS
# ============================================================================


def calculate_pure_gas_viscosity_lucas(
    temperature: float, pressure: float, props: ComponentProperties
) -> float:
    """Calculate pure gas viscosity using Lucas method.

    Accurate method for pure component gas viscosity at any temperature and pressure.

    Args:
        temperature: Temperature (K)
        pressure: Pressure (Pa)
        props: Component properties

    Returns:
        Dynamic viscosity (Pa·s)

    Reference:
        Lucas, K. (1981): "Die Druckabhängigkeit der Viskosität von Flüssigkeiten"
        Reid, Prausnitz, Poling (2001), Chapter 9

    Note:
        More accurate than Sutherland's law for high-temperature applications.
    """
    # Low pressure viscosity (dilute gas)
    if temperature is None:
        raise ValueError("temperature must be provided")
    T_r = temperature / props.critical_temp
    M = props.molecular_weight

    # Dimensionless reduced dipole moment
    if props.critical_temp > 0 and props.critical_pressure > 0:
        mu_r = (
            52.46
            * (props.dipole_moment**2)
            * props.critical_pressure
            / (props.critical_temp**2)
        )
    else:
        mu_r = 0.0

    # Correlation for ξ
    if mu_r < 0.022:
        F_p = 1.0
    elif mu_r < 0.075:
        F_p = 1.0 + 30.55 * (0.292 - T_r) ** 1.72
    else:
        F_p = 1.0 + 30.55 * (0.292 - T_r) ** 1.72 * abs(mu_r - 0.022)

    # Low pressure viscosity correlation
    if T_r <= 1.5:
        eta_low = (
            0.807 * (T_r**0.618)
            - 0.357 * math.exp(-0.449 * T_r)
            + 0.340 * math.exp(-4.058 * T_r)
            + 0.018
        ) * F_p
    else:
        eta_low = (
            0.807 * (T_r**0.618)
            - 0.357 * math.exp(-0.449 * T_r)
            + 0.340 * math.exp(-4.058 * T_r)
            + 0.018
        )

    # Convert to Pa·s
    # Formula: μ = 0.176 × (M × T_c / (V_c^(2/3))) × η
    # Simplified using critical properties
    T_c = props.critical_temp
    # Z_c = 0.29  # Approximate critical compressibility (unused)

    # Characteristic viscosity
    mu_low = (
        0.807
        * ((M * T_c) ** 0.5)
        / ((props.critical_pressure / 1e6) ** (2 / 3))
        * eta_low
        * 1e-7
    )

    # High pressure correction (simplified)
    P_r = pressure / props.critical_pressure
    if P_r > 1.0:
        # Jossi-Stiel-Thodos correlation for high pressure
        rho_r = P_r / T_r  # Approximate reduced density
        # xi = (rho_r**0.25) / (T_r ** (1 / 6))  # Unused
        delta_mu = (
            0.1023
            + 0.023364 * rho_r
            + 0.058533 * (rho_r**2)
            - 0.040758 * (rho_r**3)
            + 0.0093324 * (rho_r**4)
        )
        mu = mu_low * (1.0 + delta_mu)
    else:
        mu = mu_low

    return float(mu)


def calculate_pure_gas_viscosity_sutherland(
    temperature: float,
    T_ref: float = 273.15,
    mu_ref: float = 1.716e-5,
    S: float = 110.4,
) -> float:
    """Calculate gas viscosity using Sutherland's law.

    Simpler method, accurate for air and similar gases at moderate temperatures.

    μ/μ_ref = (T/T_ref)^(3/2) × (T_ref + S)/(T + S)

    Args:
        temperature: Temperature (K)
        T_ref: Reference temperature (K), default 273.15 K
        mu_ref: Reference viscosity (Pa·s), default for air
        S: Sutherland constant (K), default 110.4 for air

    Returns:
        Dynamic viscosity (Pa·s)

    Reference:
        Sutherland, W. (1893): "The viscosity of gases and molecular force"

    Common values:
        Air: S = 110.4 K, μ_ref = 1.716e-5 Pa·s at 273 K
        N2: S = 111 K
        O2: S = 127 K
        CO2: S = 240 K
    """
    if temperature is None:
        raise ValueError("temperature must be provided")
    mu = mu_ref * ((temperature / T_ref) ** 1.5) * (T_ref + S) / (temperature + S)
    return float(mu)


# Sutherland constants for common gases
SUTHERLAND_CONSTANTS: dict[str, dict[str, float]] = {
    "Air": {"S": 110.4, "T_ref": 273.15, "mu_ref": 1.716e-5},
    "N2": {"S": 111.0, "T_ref": 273.15, "mu_ref": 1.663e-5},
    "O2": {"S": 127.0, "T_ref": 273.15, "mu_ref": 1.919e-5},
    "CO2": {"S": 240.0, "T_ref": 273.15, "mu_ref": 1.370e-5},
    "H2": {"S": 72.0, "T_ref": 273.15, "mu_ref": 8.411e-6},
    "CO": {"S": 136.0, "T_ref": 273.15, "mu_ref": 1.657e-5},
    "CH4": {"S": 164.0, "T_ref": 273.15, "mu_ref": 1.027e-5},
}


def _compute_pure_viscosities(
    composition: dict[str, float], temperature: float, pressure: float
) -> dict[str, float]:
    """Compute pure-component viscosities for each species in the mixture.

    Uses Sutherland's law when constants are available, otherwise the Lucas method.

    Args:
        composition: Dictionary of {component: mole_fraction}
        temperature: Temperature (K)
        pressure: Pressure (Pa)

    Returns:
        Dictionary of {component: viscosity_Pa_s}
    """
    if composition is None:
        raise ValueError("composition must be provided")
    pure_viscosities: dict[str, float] = {}
    for component in composition:
        if component not in GAS_DATABASE:
            logger.warning(f"Component '{component}' not found, using air properties")
            pure_viscosities[component] = float(
                calculate_pure_gas_viscosity_sutherland(temperature)
            )
            continue

        props = GAS_DATABASE[component]

        if component in SUTHERLAND_CONSTANTS:
            params = SUTHERLAND_CONSTANTS[component]
            mu_i = calculate_pure_gas_viscosity_sutherland(
                temperature, params["T_ref"], params["mu_ref"], params["S"]
            )
        else:
            mu_i = calculate_pure_gas_viscosity_lucas(temperature, pressure, props)

        pure_viscosities[component] = float(mu_i)

    return pure_viscosities


def _wilke_mixing_rule(
    composition: dict[str, float],
    pure_viscosities: dict[str, float],
) -> float:
    """Apply Wilke's mixing rule to calculate mixture viscosity.

    Builds the Φ interaction matrix and computes the weighted mixture viscosity.

    Args:
        composition: Dictionary of {component: mole_fraction}
        pure_viscosities: Dictionary of {component: viscosity_Pa_s}

    Returns:
        Mixture dynamic viscosity (Pa·s)
    """
    if composition is None:
        raise ValueError("composition must be provided")
    components = list(composition.keys())
    component_data: dict[str, dict[str, float]] = {}
    for comp in components:
        if comp in GAS_DATABASE:
            component_data[comp] = {
                "M": GAS_DATABASE[comp].molecular_weight,
                "mu": pure_viscosities[comp],
            }

    phi: dict[tuple[str, str], float] = {}
    for i, comp_i in enumerate(components):
        if comp_i not in component_data:
            continue
        M_i = component_data[comp_i]["M"]
        mu_i = component_data[comp_i]["mu"]

        for j, comp_j in enumerate(components):
            if comp_j not in component_data:
                continue
            M_j = component_data[comp_j]["M"]
            mu_j = component_data[comp_j]["mu"]

            if i == j:
                phi[(comp_i, comp_j)] = 1.0
            else:
                numerator = (1.0 + (mu_i / mu_j) ** 0.5 * (M_j / M_i) ** 0.25) ** 2
                denominator = (8.0 * (1.0 + M_i / M_j)) ** 0.5
                phi[(comp_i, comp_j)] = numerator / denominator

    mu_mix = 0.0
    for _, comp_i in enumerate(components):
        if comp_i not in component_data:
            continue

        y_i = composition[comp_i]
        mu_i = component_data[comp_i]["mu"]

        denominator_sum = 0.0
        for _, comp_j in enumerate(components):
            if comp_j not in component_data:
                continue
            y_j = composition[comp_j]
            denominator_sum += y_j * phi.get((comp_i, comp_j), 1.0)

        if denominator_sum > 0:
            mu_mix += y_i * mu_i / denominator_sum

    return float(mu_mix)


def calculate_mixture_viscosity_wilke(
    composition: dict[str, float], temperature: float, pressure: float
) -> float:
    """Calculate gas mixture viscosity using Wilke's mixing rule.

    Most accurate method for gas mixture viscosity.

    μ_mix = Σ [y_i × μ_i / Σ(y_j × Φ_ij)]

    where Φ_ij = [1 + (μ_i/μ_j)^0.5 × (M_j/M_i)^0.25]^2 / [8(1 + M_i/M_j)]^0.5

    Note on the Φ_ij formula:
        The numerator uses a constant coefficient of 1.0 in front of the bracketed
        term: [1 + (μ_i/μ_j)^0.5 × (M_j/M_i)^0.25]^2. This is the original Wilke
        (1950) formulation. Some literature sources include an additional
        correction factor based on Sutherland constants, but we assume the
        simpler form with constant numerator coefficient = 1.0 for all species
        pairs. This assumption:
        - Matches the original Wilke derivation
        - Provides adequate accuracy (typically within 2-5%) for most gas mixtures
        - Avoids requiring Sutherland constants for all species
        - Is standard practice in process simulation software

    Args:
        composition: Dictionary of {component: mole_fraction}
        temperature: Temperature (K)
        pressure: Pressure (Pa)

    Returns:
        Mixture dynamic viscosity (Pa·s)

    Reference:
        Wilke, C.R. (1950): "A Viscosity Equation for Gas Mixtures"
        J. Chem. Phys. 18(4), 517-519

    Example:
        >>> comp = {'H2': 0.3, 'CO': 0.3, 'N2': 0.4}
        >>> mu = calculate_mixture_viscosity_wilke(comp, 800, 1e5)
        >>> print(f"Viscosity = {mu:.6f} Pa·s = {mu*1e6:.2f} µPa·s")
    """
    if composition is None:
        raise ValueError("composition must be provided")
    pure_viscosities = _compute_pure_viscosities(composition, temperature, pressure)
    mu_mix = _wilke_mixing_rule(composition, pure_viscosities)
    logger.debug(f"Mixture viscosity = {mu_mix:.6e} Pa·s = {mu_mix * 1e6:.3f} µPa·s")
    return mu_mix


def calculate_mixture_viscosity_simple(
    composition: dict[str, float], temperature: float
) -> float:
    """Calculate mixture viscosity using simple mole-fraction averaging.

    Simpler but less accurate than Wilke's method.
    μ_mix = Σ(y_i × μ_i)

    Args:
        composition: Dictionary of {component: mole_fraction}
        temperature: Temperature (K)

    Returns:
        Mixture dynamic viscosity (Pa·s)
    """
    if composition is None:
        raise ValueError("composition must be provided")
    mu_mix = 0.0
    for component, mole_frac in composition.items():
        if component in SUTHERLAND_CONSTANTS:
            params = SUTHERLAND_CONSTANTS[component]
            mu_i = calculate_pure_gas_viscosity_sutherland(
                temperature, params["T_ref"], params["mu_ref"], params["S"]
            )
            mu_mix += mole_frac * mu_i
        else:
            logger.warning(f"No Sutherland data for {component}, using air properties")
            mu_mix += mole_frac * calculate_pure_gas_viscosity_sutherland(temperature)

    return mu_mix


# ============================================================================
# COMPLETE PROPERTY CALCULATION
# ============================================================================


def calculate_gas_properties(
    composition: dict[str, float],
    temperature: float,
    pressure: float,
    use_compressibility: bool = True,
) -> dict[str, float]:
    """Calculate complete set of gas mixture properties.

    Args:
        composition: Dictionary of {component: mole_fraction}
        temperature: Temperature (K)
        pressure: Pressure (Pa)
        use_compressibility: Whether to use real gas corrections

    Returns:
        Dictionary with properties:
            - molecular_weight (kg/kmol)
            - density (kg/m³)
            - viscosity (Pa·s)
            - compressibility_factor (dimensionless)
            - heat_capacity_ratio (γ = Cp/Cv)
            - speed_of_sound (m/s)
            - cp (J/(mol·K))

    Example:
        >>> comp = {'H2': 0.25, 'CO': 0.35, 'CO2': 0.15, 'N2': 0.25}
        >>> props = calculate_gas_properties(comp, 700, 5e5)
        >>> print(f"Density: {props['density']:.3f} kg/m³")
        >>> print(f"Gamma: {props['heat_capacity_ratio']:.3f}")
    """
    # Molecular weight
    if composition is None:
        raise ValueError("composition must be provided")
    mw = calculate_mixture_molecular_weight(composition)

    # Compressibility factor
    if use_compressibility:
        Z = calculate_compressibility_factor(composition, temperature, pressure)
        density = calculate_real_gas_density(mw, temperature, pressure, Z)
    else:
        Z = 1.0
        density = calculate_ideal_gas_density(mw, temperature, pressure)

    # Viscosity
    viscosity = calculate_mixture_viscosity_wilke(composition, temperature, pressure)

    # Heat capacity and gamma
    cp = calculate_mixture_cp(composition, temperature)
    gamma = calculate_heat_capacity_ratio(composition, temperature)

    # Speed of sound
    speed_of_sound = calculate_speed_of_sound(composition, temperature, mw)

    properties = {
        "molecular_weight": mw,
        "density": density,
        "viscosity": viscosity,
        "compressibility_factor": Z,
        "heat_capacity_ratio": gamma,
        "speed_of_sound": speed_of_sound,
        "cp": cp,
    }

    logger.info(f"Gas properties at T={temperature}K, P={pressure / 1e5:.1f}bar:")
    logger.info(f"  MW = {mw:.2f} kg/kmol")
    logger.info(f"  ρ = {density:.4f} kg/m³")
    logger.info(f"  μ = {viscosity:.6e} Pa·s")
    logger.info(f"  Z = {Z:.4f}")
    logger.info(f"  γ = {gamma:.4f}")
    logger.info(f"  a = {speed_of_sound:.1f} m/s")

    return properties


if __name__ == "__main__":
    # Demonstration
    logging.basicConfig(level=logging.INFO)

    logger.info("\n" + "=" * 80)
    logger.info("GAS MIXTURE PROPERTY CALCULATOR - EXAMPLES")
    logger.info("=" * 80)

    # Example 1: Syngas composition
    logger.info("\nExample 1: Syngas from coal gasification")
    logger.info("-" * 80)
    syngas = {
        "H2": 0.30,
        "CO": 0.40,
        "CO2": 0.15,
        "N2": 0.10,
        "CH4": 0.05,
    }
    T = 800  # K
    P = 25e5  # Pa (25 bar)

    props = calculate_gas_properties(syngas, T, P)
    logger.info(f"\nComposition: {syngas}")
    logger.info(f"Temperature: {T} K ({T - 273.15:.0f}°C)")
    logger.info(f"Pressure: {P / 1e5:.1f} bar")
    logger.info("\nCalculated Properties:")
    logger.info(f"  Molecular Weight: {props['molecular_weight']:.2f} kg/kmol")
    logger.info(f"  Density: {props['density']:.4f} kg/m³")
    logger.info(
        f"  Viscosity: {props['viscosity']:.6e} Pa·s ({props['viscosity'] * 1e6:.2f} µPa·s)"
    )
    logger.info(f"  Z-factor: {props['compressibility_factor']:.4f}")

    # Example 2: Air at different conditions
    logger.info("\n\nExample 2: Air at various temperatures")
    logger.info("-" * 80)
    air = {"Air": 1.0}
    for temp in [300, 500, 800, 1200]:
        props_air = calculate_gas_properties(air, temp, 1e5, use_compressibility=False)
        logger.info(
            f"T = {temp}K: ρ = {props_air['density']:.4f} kg/m³, "
            f"μ = {props_air['viscosity'] * 1e6:.2f} µPa·s"
        )
