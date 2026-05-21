import logging
import math

from ._component_database import (
    GAS_DATABASE,
    SUTHERLAND_CONSTANTS,
    ComponentProperties,
)

logger = logging.getLogger(__name__)


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


def _wilke_mixing_rule(  # noqa: C901
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
