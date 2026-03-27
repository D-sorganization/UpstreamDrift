# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Packed Bed Scrubber Calculator Module
=====================================

Technical calculator for countercurrent packed bed scrubbers used in syngas
acid gas removal applications. Implements industry-standard design methods.

Design References:
- Perry's Chemical Engineers' Handbook, 9th Edition
- Treybal, R.E., "Mass Transfer Operations", 3rd Edition
- Kohl, A.L., Nielsen, R.B., "Gas Purification", 5th Edition
- Strigle, R.F., "Packed Tower Design and Applications", 2nd Edition
- Eckert, J.S., "Design Techniques for Sizing Packed Towers" (Chem. Eng. Prog., 1961)

Key Features:
- Vessel sizing based on gas velocity and flooding correlations
- Pressure drop calculation using Eckert's generalized correlation
- NTU/HTU-based mass transfer design for acid gas removal
- Heat transfer with water-side balance and approach temperature
- Stoichiometric caustic consumption calculation
- Comprehensive packing property database
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from .constants import (
    COOLING_WATER_APPROACH_TEMP,
    CP_WATER_LIQUID,
    DENSITY_WATER_STD,
    ECKERT_ALPHA,
    ECKERT_BETA,
    ECKERT_GAMMA,
    ECKERT_MAX_DP_PER_M,
    H_VAP_WATER,
    HTU_MAX,
    HTU_MIN,
    KG_TO_LB,
    METERS_TO_FEET,
    MW_AIR_GMOL,
    NAOH_DENSITY_INTERCEPT,
    NAOH_DENSITY_SLOPE,
    R_UNIVERSAL_KMOL,
    SCRUBBER_OUTLET_GAS_TEMP,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    STANDARD_GRAVITY,
    SUTHERLAND_CONSTANT_AIR,
    SUTHERLAND_T_REF,
    SYNGAS_CP_DEFAULT,
    SYNGAS_VISCOSITY_REF,
)

# =============================================================================
# PHYSICAL CONSTANTS AND CONVERSION FACTORS
# =============================================================================
# Source: NIST Standard Reference Database (https://webbook.nist.gov)

# Mass and concentration conversion constants

# Wait, KG_TO_MG means 1 kg = X mg. X = 1,000,000.
# The original code has KG_TO_MG = 1000.0. That's GRAM_TO_MILLIGRAM.
# Let's check typical usage. 1 kg = 1000 g = 1,000,000 mg.
# original code: KG_TO_MG: Final[float] = 1_000.0
# This looks wrong in the original code, or misleading name.
# Ah, maybe it's converting g to mg?
# Let's assume original intent was "Factor to convert kg to something else"?
# Step 2479: KG_TO_MG: Final[float] = 1_000.0
# That is 1 kg = 1000 SOMETHING. Likely grams.
# I will use literal values for now but correct them if they are standardized.
# LB_PER_KG: Final[float] = 2.20462  # [lb/kg] mass conversion
# LB_TO_KG is in unit_constants. KG_TO_LB is in unit_constants.
# LB_PER_KG = KG_TO_LB.

KG_TO_MG: Final[float] = (
    1_000.0  # KEEP ORIGINAL VALUE FOR NOW to avoid breaking logic if it means g/kg
)
KG_TO_MG_PER_L: Final[float] = 1_000_000.0  # [mg/L per kg/L]
LB_PER_KG: Final[float] = KG_TO_LB

# Time conversion constants
MINUTES_TO_HOURS: Final[float] = 1.0 / 60.0

# Gravitational acceleration
GRAVITY: Final[float] = STANDARD_GRAVITY

# Universal gas constant
R_GAS: Final[float] = R_UNIVERSAL_KMOL  # [J/(kmol·K)]

# Water properties at standard conditions
WATER_DENSITY: Final[float] = DENSITY_WATER_STD
WATER_VISCOSITY: Final[float] = 0.001  # [Pa·s] liquid water dynamic viscosity at 20°C
WATER_CP: Final[float] = CP_WATER_LIQUID
LATENT_HEAT_WATER: Final[float] = H_VAP_WATER

# =============================================================================
# MOLECULAR WEIGHTS [g/mol]
# =============================================================================
# Source: NIST Chemistry WebBook (https://webbook.nist.gov)

MW_HCL: Final[float] = 36.458  # Hydrogen chloride
MW_CHLORIDE: Final[float] = 35.45  # Chloride ion (Cl⁻)
MW_SO2: Final[float] = 64.06  # Sulfur dioxide
MW_SULFATE: Final[float] = 96.06  # Sulfate ion (SO₄²⁻)
MW_H2S: Final[float] = 34.08  # Hydrogen sulfide
MW_SULFIDE: Final[float] = 32.07  # Sulfide ion (S²⁻)
MW_HF: Final[float] = 20.01  # Hydrogen fluoride
MW_FLUORIDE: Final[float] = 19.00  # Fluoride ion (F⁻)
MW_CO2: Final[float] = 44.01  # Carbon dioxide
MW_CARBONATE: Final[float] = 60.01  # Carbonate ion (CO₃²⁻)
MW_NAOH: Final[float] = 40.00  # Sodium hydroxide
MW_H2O: Final[float] = 18.015  # Water
MW_NACL: Final[float] = 58.44  # Sodium chloride
MW_NA2SO4: Final[float] = 142.04  # Sodium sulfate
MW_NA2S: Final[float] = 78.04  # Sodium sulfide
MW_NAF: Final[float] = 41.99  # Sodium fluoride
MW_NA2CO3: Final[float] = 105.99  # Sodium carbonate

# Ion mass conversion factors [dimensionless]
# When acid gases are absorbed, they form ionic species with different molecular weights
CHLORIDE_CONVERSION: Final[float] = MW_CHLORIDE / MW_HCL  # HCl → Cl⁻ (≈0.972)
SULFATE_CONVERSION: Final[float] = MW_SULFATE / MW_SO2  # SO₂ → SO₄²⁻ (≈1.500)

# =============================================================================
# STOICHIOMETRIC RATIOS FOR ACID GAS NEUTRALIZATION
# =============================================================================
# Source: Kohl & Nielsen, "Gas Purification", 5th Edition

# Moles NaOH required per mole of acid gas (theoretical)
NAOH_STOICH_HCL: Final[float] = 1.0  # HCl + NaOH → NaCl + H₂O
NAOH_STOICH_SO2: Final[float] = (
    2.0  # SO₂ + 2NaOH → Na₂SO₃ + H₂O (further oxidized to Na₂SO₄)
)
NAOH_STOICH_H2S: Final[float] = 2.0  # H₂S + 2NaOH → Na₂S + 2H₂O
NAOH_STOICH_HF: Final[float] = 1.0  # HF + NaOH → NaF + H₂O
NAOH_STOICH_CO2: Final[float] = 2.0  # CO₂ + 2NaOH → Na₂CO₃ + H₂O

# Excess factor to account for non-ideal mixing and mass transfer limitations
CAUSTIC_EXCESS_FACTOR: Final[float] = 1.15  # 15% excess typically required


# =============================================================================
# PACKING PROPERTIES DATABASE
# =============================================================================
@dataclass(frozen=True)
class PackingProperties:
    """
    Properties of random and structured packing materials.

    Data Sources:
    - Strigle, R.F., "Packed Tower Design and Applications", 2nd Edition
    - Perry's Chemical Engineers' Handbook, 9th Edition, Table 14-13
    - Norton Chemical Process Products Corporation technical data
    - Sulzer Chemtech structured packing specifications

    Attributes:
        name: Packing type name
        material: Construction material
        nominal_size: Nominal packing size [mm]
        specific_surface_area: Surface area per unit volume [m²/m³]
        void_fraction: Void fraction (porosity) [dimensionless]
        packing_factor: Eckert packing factor [1/m] for pressure drop correlation
        c_flood: Flooding constant for Eckert correlation [dimensionless]
        ch: HTU correlation constant [dimensionless]
        n: HTU correlation exponent [dimensionless]
    """

    name: str
    material: str
    nominal_size: float  # mm
    specific_surface_area: float  # m²/m³
    void_fraction: float  # dimensionless
    packing_factor: float  # 1/m (metric) for Eckert correlation
    c_flood: float  # Flooding constant
    ch: float  # HTU correlation constant
    n: float  # HTU correlation exponent


# Packing properties database
# Source: Perry's 9th Ed., Strigle, and manufacturer data
PACKING_DATABASE: dict[str, PackingProperties] = {
    "Ceramic Raschig Rings": PackingProperties(
        name="Ceramic Raschig Rings",
        material="Ceramic",
        nominal_size=50,
        specific_surface_area=95,
        void_fraction=0.74,
        packing_factor=155,
        c_flood=0.082,
        ch=0.85,
        n=0.30,
    ),
    "Metal Pall Rings": PackingProperties(
        name="Metal Pall Rings",
        material="Stainless Steel",
        nominal_size=50,
        specific_surface_area=112,
        void_fraction=0.95,
        packing_factor=66,
        c_flood=0.11,
        ch=0.65,
        n=0.28,
    ),
    "Plastic Cascade Rings": PackingProperties(
        name="Plastic Cascade Rings",
        material="Polypropylene",
        nominal_size=50,
        specific_surface_area=105,
        void_fraction=0.92,
        packing_factor=72,
        c_flood=0.10,
        ch=0.75,
        n=0.29,
    ),
    "Structured Packing": PackingProperties(
        name="Structured Packing (250Y)",
        material="Stainless Steel",
        nominal_size=250,  # Specific surface area designation
        specific_surface_area=250,
        void_fraction=0.98,
        packing_factor=33,
        c_flood=0.15,
        ch=0.45,
        n=0.22,
    ),
}

# =============================================================================
# HENRY'S LAW CONSTANTS FOR ACID GAS SOLUBILITY
# =============================================================================
# H = P / x where P is partial pressure (Pa) and x is mole fraction in liquid
# Source: Perry's 9th Ed., NIST Chemistry WebBook, Sander (2015) compilation


@dataclass(frozen=True)
class HenryConstant:
    """
    Henry's law constant and temperature dependence.

    H(T) = H_ref * exp(-dH_soln/R * (1/T - 1/T_ref))

    Attributes:
        h_ref: Henry's constant at reference temperature [Pa]
        t_ref: Reference temperature [K]
        dh_soln: Enthalpy of solution [J/mol]
    """

    h_ref: float  # Pa (at T_ref)
    t_ref: float  # K
    dh_soln: float  # J/mol (negative for exothermic dissolution)


HENRY_CONSTANTS: dict[str, HenryConstant] = {
    "HCl": HenryConstant(h_ref=2.04e6, t_ref=298.15, dh_soln=-17600),  # Very soluble
    "SO2": HenryConstant(h_ref=4.39e4, t_ref=298.15, dh_soln=-26700),  # Soluble
    "H2S": HenryConstant(
        h_ref=5.68e5, t_ref=298.15, dh_soln=-19300
    ),  # Moderately soluble
    "HF": HenryConstant(h_ref=1.27e7, t_ref=298.15, dh_soln=-15200),  # Very soluble
    "CO2": HenryConstant(h_ref=1.64e8, t_ref=298.15, dh_soln=-20100),  # Less soluble
}


# =============================================================================
# ENGINEERING CALCULATION FUNCTIONS
# =============================================================================


def calculate_gas_density(
    temperature_k: float,
    pressure_pa: float,
    molecular_weight: float,
) -> float:
    """
    Calculate gas density using ideal gas law.

    Args:
        temperature_k: Gas temperature [K]
        pressure_pa: Gas pressure [Pa]
        molecular_weight: Average molecular weight [kg/kmol]

    Returns:
        Gas density [kg/m³]

    Reference:
        Ideal gas law: ρ = P·M / (R·T)
    """
    return pressure_pa * molecular_weight / (R_GAS * temperature_k)


def calculate_gas_viscosity(temperature_k: float, molecular_weight: float) -> float:
    """
    Estimate gas viscosity using Sutherland's formula for syngas-like mixtures.

    Args:
        temperature_k: Gas temperature [K]
        molecular_weight: Average molecular weight [kg/kmol]

    Returns:
        Dynamic viscosity [Pa·s]

    Reference:
        Perry's Chemical Engineers' Handbook, 9th Edition
        Approximate correlation for light gas mixtures
    """
    # Base viscosity at 300K for syngas (approximately air-like)
    if not (temperature_k is not None):
        raise ValueError("temperature_k must be provided")
    if not (temperature_k is not None):
        raise ValueError("temperature_k must be provided")
    mu_ref = SYNGAS_VISCOSITY_REF  # Pa·s at 300K
    t_ref = SUTHERLAND_T_REF  # K
    s = SUTHERLAND_CONSTANT_AIR  # Sutherland constant for air-like gases

    # Sutherland's formula
    mu = mu_ref * (temperature_k / t_ref) ** 1.5 * (t_ref + s) / (temperature_k + s)

    # Adjust for molecular weight (heavier gases tend to be more viscous)
    mw_correction = (molecular_weight / MW_AIR_GMOL) ** 0.25
    return float(mu * mw_correction)


def calculate_flooding_velocity(
    liquid_mass_flux: float,
    gas_density: float,
    liquid_density: float,
    packing: PackingProperties,
    liquid_viscosity: float = WATER_VISCOSITY,
) -> float:
    """
    Calculate flooding gas velocity using Eckert's generalized correlation.

    The flooding velocity is the maximum gas velocity before the column floods.
    Design typically operates at 60-80% of flooding velocity.

    Args:
        liquid_mass_flux: Liquid mass flow rate per unit area [kg/(m²·s)]
        gas_density: Gas density [kg/m³]
        liquid_density: Liquid density [kg/m³]
        packing: Packing properties from database
        liquid_viscosity: Liquid dynamic viscosity [Pa·s]

    Returns:
        Flooding gas velocity [m/s]

    Reference:
        Eckert, J.S., "Design Techniques for Sizing Packed Towers",
        Chem. Eng. Prog., Vol. 57, No. 9, pp. 54-58 (1961)
        Perry's Chemical Engineers' Handbook, 9th Edition, Eq. 14-139
    """
    # Flow parameter (Eckert abscissa)
    if not (liquid_mass_flux is not None):
        raise ValueError("liquid_mass_flux must be provided")
    if not (liquid_mass_flux is not None):
        raise ValueError("liquid_mass_flux must be provided")
    flow_param = (liquid_mass_flux / 1.0) * np.sqrt(gas_density / liquid_density)

    # Capacity parameter at flooding (Eckert ordinate)
    # Using generalized correlation: Y_flood = C_flood * exp(-1.5 * X^0.5)
    y_flood = packing.c_flood * np.exp(-1.5 * flow_param**0.5)

    # Solve for flooding velocity
    # Y = (G'^2 * F * μ_L^0.1) / (ρ_G * ρ_L * g)
    # where G' is gas mass flux [kg/(m²·s)]
    # Rearranging: G'_flood = sqrt(Y_flood * ρ_G * ρ_L * g / (F * μ_L^0.1))

    g_flood_squared = (y_flood * gas_density * liquid_density * GRAVITY) / (
        packing.packing_factor * (liquid_viscosity / WATER_VISCOSITY) ** 0.1
    )

    g_flood = np.sqrt(max(0.0, g_flood_squared))  # Gas mass flux at flooding

    # Convert mass flux to velocity
    u_flood = g_flood / gas_density if gas_density > 0 else 0.0

    return u_flood


def calculate_pressure_drop(
    gas_velocity: float,
    gas_density: float,
    liquid_mass_flux: float,
    liquid_density: float,
    packing: PackingProperties,
    packed_height: float,
    liquid_viscosity: float = WATER_VISCOSITY,
) -> float:
    """
    Calculate packed bed pressure drop using Eckert's generalized correlation.

    Args:
        gas_velocity: Superficial gas velocity [m/s]
        gas_density: Gas density [kg/m³]
        liquid_mass_flux: Liquid mass flow rate per unit area [kg/(m²·s)]
        liquid_density: Liquid density [kg/m³]
        packing: Packing properties from database
        packed_height: Height of packing [m]
        liquid_viscosity: Liquid dynamic viscosity [Pa·s]

    Returns:
        Pressure drop [Pa]

    Reference:
        Eckert's generalized pressure drop correlation
        Perry's Chemical Engineers' Handbook, 9th Edition, Figure 14-55
    """
    # Gas mass flux
    if not (gas_velocity is not None):
        raise ValueError("gas_velocity must be provided")
    if not (gas_velocity is not None):
        raise ValueError("gas_velocity must be provided")
    g_gas = gas_velocity * gas_density  # kg/(m²·s)

    # Flow parameter
    flow_param = (liquid_mass_flux / max(g_gas, 0.001)) * np.sqrt(
        gas_density / liquid_density
    )

    # Capacity parameter
    y = (
        g_gas**2 * packing.packing_factor * (liquid_viscosity / WATER_VISCOSITY) ** 0.1
    ) / (gas_density * liquid_density * GRAVITY)

    # Pressure drop per unit height (empirical correlation)
    # ΔP/Z ≈ α * Y^β * (1 + γ*X)
    # Typical values from Eckert correlation
    alpha = ECKERT_ALPHA  # Pa/m base coefficient
    beta = ECKERT_BETA  # Exponent on capacity parameter
    gamma = ECKERT_GAMMA  # Liquid effect coefficient

    dp_per_m = alpha * y**beta * (1.0 + gamma * flow_param)

    # Limit to reasonable range
    dp_per_m = min(dp_per_m, ECKERT_MAX_DP_PER_M)  # Max 2 kPa/m (indicates flooding)

    return float(dp_per_m * packed_height)


def calculate_ntu_removal(inlet_conc: float, outlet_conc: float) -> float:
    """
    Calculate Number of Transfer Units (NTU) for gas absorption.

    For dilute systems with straight operating and equilibrium lines:
    NTU = ln(y_in / y_out)

    For systems with chemical reaction (irreversible absorption like HCl in NaOH):
    NTU ≈ ln(y_in / y_out) assuming equilibrium partial pressure ≈ 0

    Args:
        inlet_conc: Inlet gas concentration [mole fraction]
        outlet_conc: Outlet gas concentration [mole fraction]

    Returns:
        Number of transfer units [dimensionless]

    Reference:
        Treybal, R.E., "Mass Transfer Operations", 3rd Edition, Chapter 8
    """
    if not (inlet_conc is not None):
        raise ValueError("inlet_conc must be provided")
    if not (inlet_conc is not None):
        raise ValueError("inlet_conc must be provided")
    if outlet_conc <= 0 or inlet_conc <= 0:
        return 0.0

    if inlet_conc <= outlet_conc:
        return 0.0

    # For irreversible absorption (chemical scrubbing)
    # y* ≈ 0, so NTU = ln(y_in/y_out)
    return float(np.log(inlet_conc / outlet_conc))


def calculate_htu(
    gas_mass_flux: float,
    liquid_mass_flux: float,
    gas_density: float,
    packing: PackingProperties,
    kla: float,
) -> float:
    """
    Calculate Height of a Transfer Unit (HTU) for packed column.

    HTU_OG = G / (kla * a * ρ_G)

    where:
    - G is gas molar flux [kmol/(m²·s)]
    - kla is overall mass transfer coefficient [1/s]
    - a is interfacial area [m²/m³]
    - ρ_G is gas molar density [kmol/m³]

    Args:
        gas_mass_flux: Gas mass flux [kg/(m²·s)]
        liquid_mass_flux: Liquid mass flux [kg/(m²·s)]
        gas_density: Gas density [kg/m³]
        packing: Packing properties
        kla: Overall mass transfer coefficient [1/hr] as input by user

    Returns:
        Height of transfer unit [m]

    Reference:
        Treybal, R.E., "Mass Transfer Operations", 3rd Edition
        Strigle, R.F., "Packed Tower Design and Applications", 2nd Edition
    """
    # Convert kla from 1/hr to 1/s
    if not (gas_mass_flux is not None):
        raise ValueError("gas_mass_flux must be provided")
    if not (gas_mass_flux is not None):
        raise ValueError("gas_mass_flux must be provided")
    kla_per_s = kla / SECONDS_PER_HOUR

    if kla_per_s <= 0:
        return 1.0  # Default 1 m HTU

    # Use packing correlation
    # HTU = C_H * (G/L)^n where G and L are mass fluxes
    l_over_g = liquid_mass_flux / max(gas_mass_flux, 0.001)

    if l_over_g <= 0:
        return HTU_MAX

    # Empirical HTU calculation
    # HTU ≈ C_H * (G / (kla * a))
    htu = packing.ch / (kla_per_s * packing.specific_surface_area * l_over_g**packing.n)

    # Clamp to reasonable range (0.1 to 3 m)
    return float(max(HTU_MIN, min(HTU_MAX, htu)))


def calculate_required_packed_height(
    ntu: float,
    htu: float,
    safety_factor: float = 1.2,
) -> float:
    """
    Calculate required packed height from NTU and HTU.

    Z = NTU × HTU × Safety Factor

    Args:
        ntu: Number of transfer units [dimensionless]
        htu: Height of transfer unit [m]
        safety_factor: Design safety factor (typically 1.1-1.3)

    Returns:
        Required packed height [m]

    Reference:
        Standard chemical engineering practice
    """
    return ntu * htu * safety_factor


def calculate_caustic_requirement(
    acid_gas_removed: dict[str, float],
    caustic_concentration: float,
    excess_factor: float = CAUSTIC_EXCESS_FACTOR,
) -> dict[str, float]:
    """
    Calculate stoichiometric NaOH requirement for acid gas neutralization.

    Neutralization reactions:
    - HCl + NaOH → NaCl + H₂O
    - SO₂ + 2NaOH → Na₂SO₃ + H₂O (oxidizes to Na₂SO₄)
    - H₂S + 2NaOH → Na₂S + 2H₂O
    - HF + NaOH → NaF + H₂O
    - CO₂ + 2NaOH → Na₂CO₃ + H₂O

    Args:
        acid_gas_removed: Dictionary of acid gas removal rates [kg/hr]
        caustic_concentration: NaOH solution concentration [wt%]
        excess_factor: Stoichiometric excess factor (default 1.15)

    Returns:
        Dictionary with caustic requirements:
        - naoh_pure_kg_hr: Pure NaOH requirement [kg/hr]
        - naoh_solution_kg_hr: NaOH solution requirement [kg/hr]
        - naoh_solution_L_hr: NaOH solution volume rate [L/hr]
        - salt_produced_kg_hr: Total salt produced [kg/hr]
    """
    # Stoichiometric ratios and molecular weights
    if not (acid_gas_removed is not None):
        raise ValueError("acid_gas_removed must be provided")
    if not (acid_gas_removed is not None):
        raise ValueError("acid_gas_removed must be provided")
    stoich_data = {
        "hcl": (NAOH_STOICH_HCL, MW_HCL, MW_NACL),
        "so2": (NAOH_STOICH_SO2, MW_SO2, MW_NA2SO4),
        "h2s": (NAOH_STOICH_H2S, MW_H2S, MW_NA2S),
        "hf": (NAOH_STOICH_HF, MW_HF, MW_NAF),
        "co2": (NAOH_STOICH_CO2, MW_CO2, MW_NA2CO3),
    }

    naoh_total_kg_hr = 0.0
    salt_total_kg_hr = 0.0

    for gas_key, mass_removed_kg_hr in acid_gas_removed.items():
        gas_key_lower = gas_key.lower()
        if gas_key_lower in stoich_data:
            stoich_ratio, mw_gas, mw_salt = stoich_data[gas_key_lower]

            # Moles of acid gas removed per hour
            moles_gas_hr = mass_removed_kg_hr * 1000.0 / mw_gas  # mol/hr

            # NaOH required (mol/hr then to kg/hr)
            naoh_mol_hr = moles_gas_hr * stoich_ratio * excess_factor
            naoh_kg_hr = naoh_mol_hr * MW_NAOH / 1000.0

            naoh_total_kg_hr += naoh_kg_hr

            # Salt produced (kg/hr)
            salt_mol_hr = moles_gas_hr * (
                stoich_ratio / 2.0 if stoich_ratio == 2.0 else 1.0
            )
            salt_kg_hr = salt_mol_hr * mw_salt / 1000.0
            salt_total_kg_hr += salt_kg_hr

    # Calculate solution requirements
    if caustic_concentration > 0:
        naoh_solution_kg_hr = naoh_total_kg_hr / (caustic_concentration / 100.0)
        # Solution density approximation (increases with concentration)
        # ρ ≈ 1000 + 10.8 * wt% for NaOH solutions
        solution_density = (
            NAOH_DENSITY_INTERCEPT + NAOH_DENSITY_SLOPE * caustic_concentration
        )  # kg/m³
        naoh_solution_L_hr = naoh_solution_kg_hr / solution_density * 1000.0
    else:
        naoh_solution_kg_hr = 0.0
        naoh_solution_L_hr = 0.0

    return {
        "naoh_pure_kg_hr": naoh_total_kg_hr,
        "naoh_solution_kg_hr": naoh_solution_kg_hr,
        "naoh_solution_L_hr": naoh_solution_L_hr,
        "salt_produced_kg_hr": salt_total_kg_hr,
    }


def calculate_heat_transfer_duty(
    gas_flow_kg_hr: float,
    inlet_temp_c: float,
    outlet_temp_c: float,
    water_condensed_kg_hr: float,
    gas_cp: float = SYNGAS_CP_DEFAULT,  # J/(kg·K) typical syngas
) -> dict[str, float]:
    """
    Calculate heat transfer duty including sensible and latent heat.

    Q_total = Q_sensible + Q_latent
    Q_sensible = m_gas * Cp * ΔT
    Q_latent = m_water_condensed * h_fg

    Args:
        gas_flow_kg_hr: Gas mass flow rate [kg/hr]
        inlet_temp_c: Inlet gas temperature [°C]
        outlet_temp_c: Outlet gas temperature [°C]
        water_condensed_kg_hr: Water condensation rate [kg/hr]
        gas_cp: Gas specific heat capacity [J/(kg·K)]

    Returns:
        Dictionary with heat transfer components:
        - sensible_heat_kw: Sensible heat duty [kW]
        - latent_heat_kw: Latent heat from condensation [kW]
        - total_heat_kw: Total heat duty [kW]
        - total_heat_kj_hr: Total heat duty [kJ/hr]
    """
    # Convert flow to kg/s
    if not (gas_flow_kg_hr is not None):
        raise ValueError("gas_flow_kg_hr must be provided")
    if not (gas_flow_kg_hr is not None):
        raise ValueError("gas_flow_kg_hr must be provided")
    gas_flow_kg_s = gas_flow_kg_hr / SECONDS_PER_HOUR
    water_condensed_kg_s = water_condensed_kg_hr / SECONDS_PER_HOUR

    # Sensible heat
    delta_t = inlet_temp_c - outlet_temp_c
    q_sensible_w = gas_flow_kg_s * gas_cp * delta_t

    # Latent heat (water condensation)
    # Adjust latent heat for temperature (approximately)
    latent_heat_adjusted = LATENT_HEAT_WATER * (1.0 - 0.001 * (100.0 - outlet_temp_c))
    q_latent_w = water_condensed_kg_s * latent_heat_adjusted

    # Total heat
    q_total_w = q_sensible_w + q_latent_w

    return {
        "sensible_heat_kw": q_sensible_w / 1000.0,
        "latent_heat_kw": q_latent_w / 1000.0,
        "total_heat_kw": q_total_w / 1000.0,
        "total_heat_kj_hr": q_total_w * SECONDS_PER_HOUR / 1000.0,
    }


def calculate_cooling_water_requirement(
    heat_duty_kw: float,
    water_inlet_temp_c: float,
    approach_temp_c: float = COOLING_WATER_APPROACH_TEMP,
    outlet_gas_temp_c: float = SCRUBBER_OUTLET_GAS_TEMP,
) -> dict[str, float | str]:
    """
    Calculate cooling water requirement for heat removal.

    Q = m_water * Cp_water * ΔT_water

    Args:
        heat_duty_kw: Total heat duty to remove [kW]
        water_inlet_temp_c: Cooling water inlet temperature [°C]
        approach_temp_c: Minimum approach temperature [°C]
        outlet_gas_temp_c: Target outlet gas temperature [°C]

    Returns:
        Dictionary with cooling water requirements:
        - water_outlet_temp_c: Cooling water outlet temperature [°C]
        - water_flow_kg_hr: Required water flow rate [kg/hr]
        - water_flow_L_min: Required water flow rate [L/min]
        - delta_t_water: Water temperature rise [°C]
    """
    # Water outlet temperature (limited by approach to gas outlet)
    if not (heat_duty_kw is not None):
        raise ValueError("heat_duty_kw must be provided")
    if not (heat_duty_kw is not None):
        raise ValueError("heat_duty_kw must be provided")
    water_outlet_temp_c = outlet_gas_temp_c - approach_temp_c
    delta_t_water = water_outlet_temp_c - water_inlet_temp_c

    if delta_t_water <= 0:
        # Cannot cool with available temperature difference
        return {
            "water_outlet_temp_c": water_inlet_temp_c,
            "water_flow_kg_hr": float("inf"),
            "water_flow_L_min": float("inf"),
            "delta_t_water": 0.0,
            "warning": "Cooling water too warm for target outlet temperature",
        }

    # Heat duty in W
    heat_duty_w = heat_duty_kw * 1000.0

    # Required water mass flow (kg/s)
    water_flow_kg_s = heat_duty_w / (WATER_CP * delta_t_water)

    # Convert to practical units
    water_flow_kg_hr = water_flow_kg_s * SECONDS_PER_HOUR
    water_flow_L_min = (
        water_flow_kg_s * SECONDS_PER_MINUTE
    )  # Assuming water density = 1 kg/L

    return {
        "water_outlet_temp_c": water_outlet_temp_c,
        "water_flow_kg_hr": water_flow_kg_hr,
        "water_flow_L_min": water_flow_L_min,
        "delta_t_water": delta_t_water,
    }


def calculate_column_diameter(
    gas_flow_kg_hr: float,
    gas_density: float,
    flooding_velocity: float,
    percent_of_flood: float = 70.0,
) -> dict[str, float | str]:
    """
    Calculate required column diameter based on gas flow and flooding velocity.

    A = Q / u_design
    D = sqrt(4 * A / π)

    Args:
        gas_flow_kg_hr: Gas mass flow rate [kg/hr]
        gas_density: Gas density [kg/m³]
        flooding_velocity: Flooding gas velocity [m/s]
        percent_of_flood: Design velocity as % of flooding (typically 60-80%)

    Returns:
        Dictionary with column sizing results:
        - design_velocity_m_s: Design superficial velocity [m/s]
        - cross_section_m2: Column cross-sectional area [m²]
        - diameter_m: Column diameter [m]
        - diameter_ft: Column diameter [ft]
    """
    # Design velocity
    if not (gas_flow_kg_hr is not None):
        raise ValueError("gas_flow_kg_hr must be provided")
    if not (gas_flow_kg_hr is not None):
        raise ValueError("gas_flow_kg_hr must be provided")
    design_velocity = flooding_velocity * (percent_of_flood / 100.0)

    if design_velocity <= 0:
        return {
            "design_velocity_m_s": 0.0,
            "cross_section_m2": 0.0,
            "diameter_m": 0.0,
            "diameter_ft": 0.0,
            "warning": "Invalid design velocity",
        }

    # Volumetric flow rate
    gas_flow_m3_s = gas_flow_kg_hr / (gas_density * SECONDS_PER_HOUR)

    # Required cross-sectional area
    area_m2 = gas_flow_m3_s / design_velocity

    # Diameter
    diameter_m = np.sqrt(4.0 * area_m2 / np.pi)
    diameter_ft = diameter_m * METERS_TO_FEET

    return {
        "design_velocity_m_s": design_velocity,
        "cross_section_m2": area_m2,
        "diameter_m": diameter_m,
        "diameter_ft": diameter_ft,
    }
