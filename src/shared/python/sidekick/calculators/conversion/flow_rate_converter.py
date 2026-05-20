# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

#!/usr/bin/env python3
"""Flow rate unit conversions for pressure drop calculator.

Comprehensive conversions between mass, molar, and volumetric flow rates
with support for standard and actual conditions.

References:
    - Perry's Chemical Engineers' Handbook, 9th Edition
    - GPSA Engineering Data Book, 14th Edition
    - ASME standards for standard reference conditions
"""

import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

R_UNIVERSAL = 8314.46  # J/(kmol·K) = Pa·m³/(kmol·K)

# Standard reference conditions (multiple standards exist)
STANDARD_CONDITIONS = {
    # Name: (Temperature_K, Pressure_Pa, description)
    "STP": (273.15, 101325, "Standard Temperature and Pressure (NIST)"),
    "NTP": (293.15, 101325, "Normal Temperature and Pressure"),
    "SATP": (298.15, 101325, "Standard Ambient Temperature and Pressure"),
    "SCFM": (288.71, 101325, "Standard Cubic Feet per Minute (60°F, 14.696 psia)"),
    "ISO": (288.15, 101325, "ISO 5024"),
    "API": (288.71, 101560, "API 14.73 psia standard"),
}

# Conversion factors to SI
MASS_FLOW_CONVERSIONS = {
    # Unit: factor to convert to kg/s
    "kg/s": 1.0,
    "kg/h": 1.0 / 3600.0,
    "kg/hr": 1.0 / 3600.0,
    "kg/min": 1.0 / 60.0,
    "g/s": 1.0 / 1000.0,
    "g/h": 1.0 / 3.6e6,
    "lb/s": 0.453592,
    "lb/h": 0.453592 / 3600.0,
    "lb/hr": 0.453592 / 3600.0,
    "lb/min": 0.453592 / 60.0,
    "ton/h": 1000.0 / 3600.0,  # metric ton
}

MOLAR_FLOW_CONVERSIONS = {
    # Unit: factor to convert to mol/s
    "mol/s": 1.0,
    "mol/h": 1.0 / 3600.0,
    "mol/hr": 1.0 / 3600.0,
    "mol/min": 1.0 / 60.0,
    "kmol/s": 1000.0,
    "kmol/h": 1000.0 / 3600.0,
    "kmol/hr": 1000.0 / 3600.0,
    "kmol/min": 1000.0 / 60.0,
    "lbmol/s": 453.592,  # lb-mole
    "lbmol/h": 453.592 / 3600.0,
    "lbmol/hr": 453.592 / 3600.0,
    "lbmol/min": 453.592 / 60.0,
}

VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S = {
    # Unit: factor to convert to m³/s (actual conditions)
    "m3/s": 1.0,
    "m³/s": 1.0,
    "m3/h": 1.0 / 3600.0,
    "m³/h": 1.0 / 3600.0,
    "m3/hr": 1.0 / 3600.0,
    "m³/hr": 1.0 / 3600.0,
    "m3/min": 1.0 / 60.0,
    "m³/min": 1.0 / 60.0,
    "L/s": 1.0 / 1000.0,
    "L/min": 1.0 / 60000.0,
    "L/h": 1.0 / 3.6e6,
    "ft3/s": 0.0283168,
    "ft³/s": 0.0283168,
    "ft3/min": 0.0283168 / 60.0,
    "ft³/min": 0.0283168 / 60.0,
    "CFM": 0.0283168 / 60.0,  # Cubic feet per minute
    "ft3/h": 0.0283168 / 3600.0,
    "ft³/h": 0.0283168 / 3600.0,
    "GPM": 0.00378541 / 60.0,  # US Gallons per minute
    "gal/min": 0.00378541 / 60.0,
}


def _require_finite(value: float, name: str) -> None:
    """Ensure a scalar input is finite."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _require_positive_finite(value: float, name: str) -> None:
    """Ensure a scalar input is positive and finite."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _require_known_unit(unit: str, units: dict[str, float], kind: str) -> None:
    """Ensure a unit exists in a conversion table."""
    if unit not in units:
        raise ValueError(f"Unknown {kind} unit: {unit}")


def _require_known_standard(standard: str) -> None:
    """Ensure a standard condition label is supported."""
    if standard not in STANDARD_CONDITIONS:
        raise ValueError(f"Unknown standard condition: {standard}")


def _normalize_prefixed_volume_unit(unit: str) -> str:
    """Normalize optional N/S prefixed volumetric units."""
    if unit.startswith(("N", "S")):
        base = unit[1:]
        if base in VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S:
            return base
    return unit


def _volume_unit_to_m3_per_s(unit: str) -> float:
    """Resolve volumetric unit factor to m³/s."""
    normalized = _normalize_prefixed_volume_unit(unit)
    _require_known_unit(
        normalized, VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S, "volumetric flow"
    )
    return VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S[normalized]


# ============================================================================
# FLOW RATE CONVERSION FUNCTIONS
# ============================================================================


def mass_to_mass(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between mass flow rate units.

    Args:
        value: Flow rate value (must be finite)
        from_unit: Source unit (e.g., 'kg/h', 'lb/hr')
        to_unit: Target unit

    Returns:
        Converted flow rate

    Raises:
        ValueError: If value is not finite or units are unknown.

    Example:
        >>> flow_lbhr = mass_to_mass(1000, 'kg/h', 'lb/hr')
        >>> print(f"{flow_lbhr:.1f} lb/hr")
    """
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value}")
    if from_unit not in MASS_FLOW_CONVERSIONS:
        raise ValueError(f"Unknown mass flow unit: {from_unit}")
    if to_unit not in MASS_FLOW_CONVERSIONS:
        raise ValueError(f"Unknown mass flow unit: {to_unit}")

    # Convert to kg/s, then to target unit
    kg_per_s = value * MASS_FLOW_CONVERSIONS[from_unit]
    result = kg_per_s / MASS_FLOW_CONVERSIONS[to_unit]

    logger.debug(f"Mass flow: {value} {from_unit} = {result:.6f} {to_unit}")
    return result


def molar_to_molar(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between molar flow rate units.

    Args:
        value: Flow rate value (must be finite)
        from_unit: Source unit (e.g., 'kmol/h', 'lbmol/hr')
        to_unit: Target unit

    Returns:
        Converted flow rate

    Raises:
        ValueError: If value is not finite or units are unknown.

    Example:
        >>> flow_kmol = molar_to_molar(100, 'lbmol/hr', 'kmol/h')
        >>> print(f"{flow_kmol:.2f} kmol/h")
    """
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value}")
    if from_unit not in MOLAR_FLOW_CONVERSIONS:
        raise ValueError(f"Unknown molar flow unit: {from_unit}")
    if to_unit not in MOLAR_FLOW_CONVERSIONS:
        raise ValueError(f"Unknown molar flow unit: {to_unit}")

    # Convert to mol/s, then to target unit
    mol_per_s = value * MOLAR_FLOW_CONVERSIONS[from_unit]
    result = mol_per_s / MOLAR_FLOW_CONVERSIONS[to_unit]

    logger.debug(f"Molar flow: {value} {from_unit} = {result:.6f} {to_unit}")
    return result


def mass_to_molar(
    mass_flow: float, mass_unit: str, molecular_weight: float, molar_unit: str = "mol/s"
) -> float:
    """Convert mass flow rate to molar flow rate.

    n_dot = m_dot / MW

    Args:
        mass_flow: Mass flow rate (must be finite)
        mass_unit: Mass flow unit (e.g., 'kg/h')
        molecular_weight: Molecular weight in kg/kmol (must be > 0)
        molar_unit: Target molar flow unit

    Returns:
        Molar flow rate

    Raises:
        ValueError: If inputs are not finite or molecular_weight is not positive.

    Example:
        >>> # 100 kg/h of air (MW = 29 kg/kmol)
        >>> n_dot = mass_to_molar(100, 'kg/h', 29.0, 'kmol/h')
        >>> print(f"{n_dot:.2f} kmol/h")
    """
    if mass_flow is None:
        raise ValueError("mass_flow must be provided")
    _require_finite(mass_flow, "mass_flow")
    _require_positive_finite(molecular_weight, "molecular_weight")
    _require_known_unit(mass_unit, MASS_FLOW_CONVERSIONS, "mass flow")
    _require_known_unit(molar_unit, MOLAR_FLOW_CONVERSIONS, "molar flow")
    # Convert to kg/s
    kg_per_s = mass_flow * MASS_FLOW_CONVERSIONS[mass_unit]

    # Convert to mol/s
    mol_per_s = (kg_per_s / molecular_weight) * 1000.0  # kg/kmol -> mol/s

    # Convert to target unit
    result = mol_per_s / MOLAR_FLOW_CONVERSIONS[molar_unit]

    logger.debug(
        f"Mass to molar: {mass_flow} {mass_unit} = {result:.6f} {molar_unit} (MW={molecular_weight})"
    )
    return result


def molar_to_mass(
    molar_flow: float, molar_unit: str, molecular_weight: float, mass_unit: str = "kg/s"
) -> float:
    """Convert molar flow rate to mass flow rate.

    m_dot = n_dot * MW

    Args:
        molar_flow: Molar flow rate (must be finite)
        molar_unit: Molar flow unit (e.g., 'kmol/h')
        molecular_weight: Molecular weight in kg/kmol (must be > 0)
        mass_unit: Target mass flow unit

    Returns:
        Mass flow rate

    Raises:
        ValueError: If inputs are not finite or molecular_weight is not positive.

    Example:
        >>> # 10 kmol/h of CO2 (MW = 44 kg/kmol)
        >>> m_dot = molar_to_mass(10, 'kmol/h', 44.0, 'kg/h')
        >>> print(f"{m_dot:.1f} kg/h")
    """
    if molar_flow is None:
        raise ValueError("molar_flow must be provided")
    _require_finite(molar_flow, "molar_flow")
    _require_positive_finite(molecular_weight, "molecular_weight")
    _require_known_unit(molar_unit, MOLAR_FLOW_CONVERSIONS, "molar flow")
    _require_known_unit(mass_unit, MASS_FLOW_CONVERSIONS, "mass flow")
    # Convert to mol/s
    mol_per_s = molar_flow * MOLAR_FLOW_CONVERSIONS[molar_unit]

    # Convert to kg/s
    kg_per_s = (mol_per_s * molecular_weight) / 1000.0  # mol/s -> kg/s

    # Convert to target unit
    result = kg_per_s / MASS_FLOW_CONVERSIONS[mass_unit]

    logger.debug(
        f"Molar to mass: {molar_flow} {molar_unit} = {result:.6f} {mass_unit} (MW={molecular_weight})"
    )
    return result


def volumetric_actual_to_mass(
    vol_flow: float, vol_unit: str, density: float, mass_unit: str = "kg/s"
) -> float:
    """Convert actual volumetric flow rate to mass flow rate.

    m_dot = Q × ρ

    Args:
        vol_flow: Volumetric flow rate at actual conditions
        vol_unit: Volumetric flow unit (e.g., 'm3/h', 'CFM')
        density: Gas density at actual conditions (kg/m³)
        mass_unit: Target mass flow unit

    Returns:
        Mass flow rate

    Raises:
        ValueError: If inputs are not finite, density is not positive, or units unknown.

    Example:
        >>> # 1000 m³/h at ρ = 1.2 kg/m³
        >>> m_dot = volumetric_actual_to_mass(1000, 'm3/h', 1.2, 'kg/h')
        >>> print(f"{m_dot:.1f} kg/h")
    """
    if vol_flow is None:
        raise ValueError("vol_flow must be provided")
    _require_finite(vol_flow, "vol_flow")
    _require_positive_finite(density, "density")
    _require_known_unit(
        vol_unit, VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S, "volumetric flow"
    )
    _require_known_unit(mass_unit, MASS_FLOW_CONVERSIONS, "mass flow")

    # Convert to m³/s
    m3_per_s = vol_flow * VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S[vol_unit]

    # Convert to kg/s
    kg_per_s = m3_per_s * density

    # Convert to target unit
    result = kg_per_s / MASS_FLOW_CONVERSIONS[mass_unit]

    logger.debug(
        f"Volumetric to mass: {vol_flow} {vol_unit} = {result:.6f} {mass_unit} (ρ={density})"
    )
    return result


def mass_to_volumetric_actual(
    mass_flow: float, mass_unit: str, density: float, vol_unit: str = "m3/s"
) -> float:
    """Convert mass flow rate to actual volumetric flow rate.

    Q = m_dot / ρ

    Args:
        mass_flow: Mass flow rate
        mass_unit: Mass flow unit
        density: Gas density at actual conditions (kg/m³)
        vol_unit: Target volumetric flow unit

    Returns:
        Volumetric flow rate at actual conditions

    Raises:
        ValueError: If inputs are not finite, density is not positive, or units unknown.

    Example:
        >>> # 100 kg/h at ρ = 1.2 kg/m³
        >>> Q = mass_to_volumetric_actual(100, 'kg/h', 1.2, 'm3/h')
        >>> print(f"{Q:.1f} m³/h")
    """
    if mass_flow is None:
        raise ValueError("mass_flow must be provided")
    _require_finite(mass_flow, "mass_flow")
    _require_positive_finite(density, "density")
    _require_known_unit(mass_unit, MASS_FLOW_CONVERSIONS, "mass flow")
    _require_known_unit(
        vol_unit, VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S, "volumetric flow"
    )

    # Convert to kg/s
    kg_per_s = mass_flow * MASS_FLOW_CONVERSIONS[mass_unit]

    # Convert to m³/s
    m3_per_s = kg_per_s / density

    # Convert to target unit
    result = m3_per_s / VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S[vol_unit]

    logger.debug(
        f"Mass to volumetric: {mass_flow} {mass_unit} = {result:.6f} {vol_unit} (ρ={density})"
    )
    return result


def standard_volumetric_to_mass(
    vol_flow_std: float,
    vol_unit: str,
    molecular_weight: float,
    standard: str = "STP",
    mass_unit: str = "kg/s",
) -> float:
    """Convert standard volumetric flow rate to mass flow rate.

    Uses ideal gas law at standard conditions:
    m_dot = (Q_std × P_std × MW) / (R × T_std)

    Args:
        vol_flow_std: Volumetric flow rate at standard conditions
        vol_unit: Volumetric flow unit (e.g., 'SCFM', 'Nm3/h')
        molecular_weight: Molecular weight (kg/kmol)
        standard: Standard condition reference ('STP', 'NTP', 'SCFM', etc.)
        mass_unit: Target mass flow unit

    Returns:
        Mass flow rate

    Example:
        >>> # 1000 SCFM of air (MW = 29 kg/kmol)
        >>> m_dot = standard_volumetric_to_mass(1000, 'ft3/min', 29.0, 'SCFM', 'lb/hr')
        >>> print(f"{m_dot:.1f} lb/hr")

    Raises:
        ValueError: If inputs are not valid.

    Notes:
        - SCFM refers to "Standard" CFM at 60°F, 14.696 psia
        - Nm³/h refers to "Normal" m³/h at 0°C, 1 atm
        - The standard parameter specifies which reference conditions to use
    """
    if vol_flow_std is None:
        raise ValueError("vol_flow_std must be provided")
    _require_finite(vol_flow_std, "vol_flow_std")
    _require_positive_finite(molecular_weight, "molecular_weight")
    _require_known_unit(mass_unit, MASS_FLOW_CONVERSIONS, "mass flow")
    _require_known_standard(standard)
    T_std, P_std, _ = STANDARD_CONDITIONS[standard]
    m3_per_s_std = vol_flow_std * _volume_unit_to_m3_per_s(vol_unit)
    kg_per_s = _standard_density(P_std, molecular_weight, T_std) * m3_per_s_std
    result = _from_kg_per_s(kg_per_s, mass_unit)

    logger.debug(
        f"Std volumetric to mass: {vol_flow_std} {vol_unit} @ {standard} = {result:.6f} {mass_unit}"
    )

    return result


def mass_to_standard_volumetric(
    mass_flow: float,
    mass_unit: str,
    molecular_weight: float,
    standard: str = "STP",
    vol_unit: str = "Nm3/h",
) -> float:
    """Convert mass flow rate to standard volumetric flow rate.

    Q_std = (m_dot × R × T_std) / (P_std × MW)

    Args:
        mass_flow: Mass flow rate
        mass_unit: Mass flow unit
        molecular_weight: Molecular weight (kg/kmol)
        standard: Standard condition reference
        vol_unit: Target volumetric flow unit

    Returns:
        Volumetric flow rate at standard conditions

    Raises:
        ValueError: If inputs are not valid.

    Example:
        >>> # 100 kg/h of CH4 (MW = 16 kg/kmol)
        >>> Q_std = mass_to_standard_volumetric(100, 'kg/h', 16.0, 'STP', 'Nm3/h')
        >>> print(f"{Q_std:.1f} Nm³/h")
    """
    if mass_flow is None:
        raise ValueError("mass_flow must be provided")
    _require_finite(mass_flow, "mass_flow")
    _require_positive_finite(molecular_weight, "molecular_weight")
    _require_known_unit(mass_unit, MASS_FLOW_CONVERSIONS, "mass flow")
    _require_known_standard(standard)
    T_std, P_std, _ = STANDARD_CONDITIONS[standard]
    kg_per_s = mass_flow * MASS_FLOW_CONVERSIONS[mass_unit]
    rho_std = _standard_density(P_std, molecular_weight, T_std)
    m3_per_s_std = kg_per_s / rho_std
    result = m3_per_s_std / _volume_unit_to_m3_per_s(vol_unit)

    logger.debug(
        f"Mass to std volumetric: {mass_flow} {mass_unit} = {result:.6f} {vol_unit} @ {standard}"
    )

    return result


def scfm_to_acfm(
    scfm: float, temperature: float, pressure: float, standard: str = "SCFM"
) -> float:
    """Convert SCFM to ACFM (Actual Cubic Feet per Minute).

    ACFM = SCFM × (T_actual/T_std) × (P_std/P_actual)

    Args:
        scfm: Standard cubic feet per minute
        temperature: Actual temperature (K)
        pressure: Actual pressure (Pa)
        standard: Standard condition reference

    Returns:
        Actual cubic feet per minute

    Raises:
        ValueError: If temperature or pressure is not positive and finite.

    Example:
        >>> # 1000 SCFM at 500°F (533 K) and 5 bar
        >>> acfm = scfm_to_acfm(1000, 533, 5e5, 'SCFM')
        >>> print(f"{acfm:.0f} ACFM")
    """
    if scfm is None:
        raise ValueError("scfm must be provided")
    _require_finite(scfm, "scfm")
    _require_positive_finite(temperature, "temperature")
    _require_positive_finite(pressure, "pressure")
    _require_known_standard(standard)
    T_std, P_std, _ = STANDARD_CONDITIONS[standard]

    acfm = scfm * (temperature / T_std) * (P_std / pressure)

    logger.debug(
        f"SCFM to ACFM: {scfm} SCFM = {acfm:.2f} ACFM @ T={temperature}K, P={pressure / 1e5:.1f}bar"
    )
    return acfm


def acfm_to_scfm(
    acfm: float, temperature: float, pressure: float, standard: str = "SCFM"
) -> float:
    """Convert ACFM to SCFM.

    SCFM = ACFM × (T_std/T_actual) × (P_actual/P_std)

    Args:
        acfm: Actual cubic feet per minute
        temperature: Actual temperature (K)
        pressure: Actual pressure (Pa)
        standard: Standard condition reference

    Returns:
        Standard cubic feet per minute

    Raises:
        ValueError: If temperature or pressure is not positive and finite.
    """
    if acfm is None:
        raise ValueError("acfm must be provided")
    _require_finite(acfm, "acfm")
    _require_positive_finite(temperature, "temperature")
    _require_positive_finite(pressure, "pressure")
    _require_known_standard(standard)
    T_std, P_std, _ = STANDARD_CONDITIONS[standard]

    scfm = acfm * (T_std / temperature) * (pressure / P_std)

    logger.debug(
        f"ACFM to SCFM: {acfm} ACFM = {scfm:.2f} SCFM @ T={temperature}K, P={pressure / 1e5:.1f}bar"
    )
    return scfm


# ============================================================================
# UNIVERSAL CONVERTER
# ============================================================================


def convert_flow_rate_to_mass(
    value: float,
    from_unit: str,
    molecular_weight: float,
    temperature: float | None = None,
    pressure: float | None = None,
    density: float | None = None,
    standard: str = "STP",
) -> float:
    """Universal converter: any flow rate unit to kg/s.

    Args:
        value: Flow rate value
        from_unit: Source unit
        molecular_weight: Molecular weight (kg/kmol)
        temperature: Temperature (K) - required for ACFM conversion
        pressure: Pressure (Pa) - required for ACFM conversion
        density: Density (kg/m³) - required for ACFM conversion
        standard: Standard conditions for SCFM/Nm³

    Returns:
        Mass flow rate in kg/s

    Example:
        >>> # Convert 1000 SCFM to kg/s for air
        >>> m_dot = convert_flow_rate_to_mass(1000, 'SCFM', 29.0, standard='SCFM')
        >>> print(f"{m_dot:.3f} kg/s")
    """
    _require_finite(value, "value")
    if from_unit in MASS_FLOW_CONVERSIONS:
        return value * MASS_FLOW_CONVERSIONS[from_unit]
    if from_unit in MOLAR_FLOW_CONVERSIONS:
        _require_positive_finite(molecular_weight, "molecular_weight")
        mol_per_s = value * MOLAR_FLOW_CONVERSIONS[from_unit]
        return (mol_per_s * molecular_weight) / 1000.0
    if _is_standard_volumetric_unit(from_unit):
        return standard_volumetric_to_mass(
            value, from_unit, molecular_weight, standard, "kg/s"
        )
    if _is_actual_volumetric_unit(from_unit):
        if density is None:
            raise ValueError(
                f"Density required for actual volumetric flow unit '{from_unit}'"
            )
        return volumetric_actual_to_mass(value, from_unit, density, "kg/s")
    raise ValueError(f"Unknown or unsupported flow rate unit: {from_unit}")


def _is_standard_volumetric_unit(unit: str) -> bool:
    """Return True when unit uses standard-condition volumetric notation."""
    return unit.upper() in {"SCFM", "NM3/H", "NM³/H", "SM3/H", "SM³/H"}


def _is_actual_volumetric_unit(unit: str) -> bool:
    """Return True when unit is an actual-condition volumetric flow unit."""
    return (
        unit.upper() in {"ACFM", "CFM"} or unit in VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S
    )


def _standard_density(
    pressure_pa: float, molecular_weight: float, temperature_k: float
) -> float:
    """Calculate standard density with ideal gas law."""
    return (pressure_pa * molecular_weight) / (R_UNIVERSAL * temperature_k)


def _from_kg_per_s(value_kg_per_s: float, mass_unit: str) -> float:
    """Convert kg/s to a target mass flow unit."""
    return value_kg_per_s / MASS_FLOW_CONVERSIONS[mass_unit]


if __name__ == "__main__":
    # Demonstration
    logging.basicConfig(level=logging.INFO)

    logger.info("\n" + "=" * 80)
    logger.info("FLOW RATE CONVERSION EXAMPLES")
    logger.info("=" * 80)

    # Example 1: Mass flow conversions
    logger.info("\nExample 1: Mass flow rate conversions")
    logger.info("-" * 80)
    mass_kg_h: float = 1000.0  # kg/h
    mass_lb_hr = mass_to_mass(mass_kg_h, "kg/h", "lb/hr")
    logger.info(f"{mass_kg_h} kg/h = {mass_lb_hr:.1f} lb/hr")

    # Example 2: Molar to mass
    logger.info("\nExample 2: Molar to mass flow rate")
    logger.info("-" * 80)
    molar_kmol_h = 10  # kmol/h
    MW_air = 29.0  # kg/kmol
    mass_kg_h = molar_to_mass(molar_kmol_h, "kmol/h", MW_air, "kg/h")
    logger.info(f"{molar_kmol_h} kmol/h of air (MW={MW_air}) = {mass_kg_h:.1f} kg/h")

    # Example 3: SCFM to mass flow
    logger.info("\nExample 3: SCFM to mass flow rate")
    logger.info("-" * 80)
    scfm = 1000  # SCFM
    mass_lb_hr = standard_volumetric_to_mass(scfm, "ft3/min", MW_air, "SCFM", "lb/hr")
    mass_kg_s = standard_volumetric_to_mass(scfm, "ft3/min", MW_air, "SCFM", "kg/s")
    logger.info(f"{scfm} SCFM of air = {mass_lb_hr:.1f} lb/hr = {mass_kg_s:.3f} kg/s")

    # Example 4: SCFM to ACFM
    logger.info("\nExample 4: SCFM to ACFM conversion")
    logger.info("-" * 80)
    T_actual = 800  # K (~527°C)
    P_actual = 5e5  # Pa (5 bar)
    acfm = scfm_to_acfm(scfm, T_actual, P_actual, "SCFM")
    logger.info(
        f"{scfm} SCFM @ {T_actual}K, {P_actual / 1e5:.0f} bar = {acfm:.0f} ACFM"
    )

    # Example 5: Universal converter
    logger.info("\nExample 5: Universal converter")
    logger.info("-" * 80)
    inputs = [
        (1000, "kg/h", 29.0),
        (100, "lbmol/hr", 29.0),
        (5000, "Nm3/h", 29.0),
    ]
    for val, unit, mw in inputs:
        mass_kg_s = convert_flow_rate_to_mass(val, unit, mw)
        logger.info(f"{val} {unit} = {mass_kg_s:.3f} kg/s")
