#!/usr/bin/env python3
"""Data models for advanced pressure drop calculator.

This module defines comprehensive data structures for pressure drop calculations
in combustion and gasification systems with variable gas composition support.

References:
- Crane Technical Paper 410, "Flow of Fluids Through Valves, Fittings, and Pipe"
- Perry's Chemical Engineers' Handbook, 9th Edition
- GPSA Engineering Data Book, 14th Edition
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GasComposition:
    """Gas composition in mole fractions.

    Attributes:
        components: Dictionary mapping gas component names to mole fractions

    Common components:
        H2, CO, CO2, CH4, N2, H2O, O2, Ar, C2H6, C2H4, H2S, NH3
    """

    components: dict[str, float] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate composition sums to 1.0 within tolerance."""
        try:
            total = sum(self.components.values())
            if not (0.99 <= total <= 1.01):
                logger.warning(f"Composition sum = {total:.4f}, expected 1.0")
                return False
            if any(x < 0 or x > 1 for x in self.components.values()):
                logger.error("Mole fractions must be between 0 and 1")
                return False
            return True
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.error(f"Composition validation error: {e}")
            return False

    def normalize(self) -> None:
        """Normalize composition to sum to 1.0."""
        total = sum(self.components.values())
        if total > 0:
            self.components = {k: v / total for k, v in self.components.items()}


@dataclass
class PipeFitting:
    """Pipe fitting with associated pressure loss coefficient.

    Attributes:
        fitting_type: Type of fitting (elbow, tee, valve, etc.)
        quantity: Number of fittings
        k_factor: Resistance coefficient (dimensionless)
        description: Optional description

    References:
        Crane TP-410: Table A-29, Resistance Coefficients
    """

    fitting_type: str
    quantity: int = 1
    k_factor: float = 0.0
    description: str = ""


@dataclass
class PressureDropInputs:
    """Comprehensive input parameters for pressure drop calculation.

    All inputs should be in SI units unless otherwise specified.
    The calculator handles unit conversions from user inputs.

    Attributes:
        # Pipe geometry
        pipe_diameter: Internal diameter (m)
        pipe_length: Total pipe length (m)
        pipe_roughness: Absolute roughness (m)

        # Flow conditions
        mass_flow_rate: Mass flow rate (kg/s)
        inlet_pressure: Inlet pressure (Pa, absolute)
        inlet_temperature: Inlet temperature (K)

        # Optional parameters with defaults
        elevation_change: Elevation change (m, positive = upward)
        gas_composition: Gas composition as GasComposition object
        fittings: List of pipe fittings and valves
        compressibility_correction: Apply compressible flow corrections
        friction_method: Friction factor correlation method

    References:
        - Darcy-Weisbach equation: ΔP = f × (L/D) × (ρV²/2)
        - Colebrook-White equation for friction factor
    """

    # Pipe geometry (required)
    pipe_diameter: float
    pipe_length: float
    pipe_roughness: float

    # Flow conditions (required)
    mass_flow_rate: float
    inlet_pressure: float
    inlet_temperature: float

    # Optional parameters (with defaults)
    elevation_change: float = 0.0
    gas_composition: GasComposition = field(default_factory=GasComposition)
    fittings: list[PipeFitting] = field(default_factory=list)
    compressibility_correction: bool = True
    friction_method: str = "colebrook"  # Options: colebrook, swamee-jain, churchill

    def validate(self) -> tuple[bool, str]:
        """Validate all input parameters.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Geometry validation
            if self.pipe_diameter <= 0:
                return False, "Pipe diameter must be positive"
            if self.pipe_length <= 0:
                return False, "Pipe length must be positive"
            if self.pipe_roughness < 0:
                return False, "Pipe roughness must be non-negative"

            # Flow condition validation
            if self.mass_flow_rate <= 0:
                return False, "Mass flow rate must be positive"
            if self.inlet_pressure <= 0:
                return False, "Inlet pressure must be positive"
            if self.inlet_temperature <= 0:
                return False, "Temperature must be positive (Kelvin)"

            # Composition validation
            if not self.gas_composition.validate():
                return False, "Gas composition validation failed"

            # Check for reasonable values
            if self.pipe_diameter > 10:  # 10 m seems unreasonable
                logger.warning(f"Large pipe diameter: {self.pipe_diameter} m")
            if self.inlet_pressure > 100e5:  # 100 bar
                logger.warning(f"High inlet pressure: {self.inlet_pressure / 1e5:.1f} bar")

            return True, "All validations passed"

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error(f"Input validation error: {e}")
            return False, f"Validation error: {str(e)}"


@dataclass
class FlowProperties:
    """Calculated flow properties for the gas mixture.

    Attributes:
        density: Gas density (kg/m³)
        viscosity: Dynamic viscosity (Pa·s)
        velocity: Flow velocity (m/s)
        reynolds_number: Reynolds number (dimensionless)
        mach_number: Mach number (dimensionless)
        compressibility_factor: Z-factor (dimensionless)
        molecular_weight: Mixture molecular weight (kg/kmol)

    References:
        - Reid, Prausnitz, Poling: "The Properties of Gases and Liquids", 5th Ed
        - Chapman-Enskog theory for gas viscosity
    """

    density: float
    viscosity: float
    velocity: float
    reynolds_number: float
    mach_number: float
    compressibility_factor: float
    molecular_weight: float
    mass_flux: float  # kg/(m²·s)
    volumetric_flow_rate: float  # m³/s


@dataclass
class PressureDropResults:
    """Comprehensive pressure drop calculation results.

    All pressures in Pa, velocities in m/s unless otherwise noted.

    Attributes:
        # Primary results
        total_pressure_drop: Total pressure drop (Pa)
        outlet_pressure: Outlet pressure (Pa, absolute)

        # Pressure drop components
        friction_pressure_drop: Frictional losses (Pa)
        fitting_pressure_drop: Fitting and valve losses (Pa)
        elevation_pressure_drop: Hydrostatic pressure change (Pa)
        acceleration_pressure_drop: Momentum change due to density variation (Pa)

        # Flow characteristics
        friction_factor: Darcy friction factor (dimensionless)
        flow_properties: Calculated flow properties

        # Performance metrics
        pressure_drop_per_100ft: Pressure drop per 100 ft (Pa/100ft)
        velocity_pressure: Dynamic pressure (Pa)
        erosional_velocity: Erosional velocity limit (m/s)
        erosion_ratio: Actual/erosional velocity ratio

        # Additional information
        flow_regime: Flow regime classification
        warnings: List of warning messages

    References:
        - API RP 14E: Erosional velocity = C/sqrt(ρ), C ≈ 100-150 for continuous service
    """

    # Primary results
    total_pressure_drop: float
    outlet_pressure: float

    # Pressure drop components
    friction_pressure_drop: float
    fitting_pressure_drop: float
    elevation_pressure_drop: float
    acceleration_pressure_drop: float

    # Flow characteristics
    friction_factor: float
    flow_properties: FlowProperties

    # Performance metrics
    pressure_drop_per_100ft: float
    velocity_pressure: float
    erosional_velocity: float
    erosion_ratio: float

    # Additional information
    flow_regime: str  # "laminar", "transitional", "turbulent"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert results to dictionary for display/export."""
        return {
            "Pressure Drop (Pa)": self.total_pressure_drop,
            "Pressure Drop (bar)": self.total_pressure_drop / 1e5,
            "Pressure Drop (psi)": self.total_pressure_drop / 6894.76,
            "Outlet Pressure (Pa)": self.outlet_pressure,
            "Outlet Pressure (bar)": self.outlet_pressure / 1e5,
            "Outlet Pressure (psi)": self.outlet_pressure / 6894.76,
            "Friction Loss (Pa)": self.friction_pressure_drop,
            "Fitting Loss (Pa)": self.fitting_pressure_drop,
            "Elevation Loss (Pa)": self.elevation_pressure_drop,
            "Acceleration Loss (Pa)": self.acceleration_pressure_drop,
            "Friction Factor": self.friction_factor,
            "Reynolds Number": self.flow_properties.reynolds_number,
            "Flow Velocity (m/s)": self.flow_properties.velocity,
            "Mach Number": self.flow_properties.mach_number,
            "Flow Regime": self.flow_regime,
            "Density (kg/m³)": self.flow_properties.density,
            "Viscosity (Pa·s)": self.flow_properties.viscosity,
            "Erosional Velocity (m/s)": self.erosional_velocity,
            "Erosion Ratio (%)": self.erosion_ratio * 100,
        }


@dataclass
class PipeSpecification:
    """Standard pipe specification from ASME/ANSI standards.

    Attributes:
        nominal_size: Nominal pipe size (e.g., "2", "4", "6")
        schedule: Pipe schedule (e.g., "40", "80", "160")
        outer_diameter: Outside diameter (mm)
        wall_thickness: Wall thickness (mm)
        inner_diameter: Inside diameter (mm)
        material: Pipe material
        max_pressure: Maximum rated pressure (bar)

    References:
        - ASME B36.10M: Welded and Seamless Wrought Steel Pipe
        - ASME B36.19M: Stainless Steel Pipe
    """

    nominal_size: str
    schedule: str
    outer_diameter: float  # mm
    wall_thickness: float  # mm
    inner_diameter: float  # mm
    material: str = "Carbon Steel"
    max_pressure: float | None = None  # bar

    def get_id_meters(self) -> float:
        """Get inner diameter in meters."""
        return self.inner_diameter / 1000.0

    def get_od_meters(self) -> float:
        """Get outer diameter in meters."""
        return self.outer_diameter / 1000.0


@dataclass
class FlowRateInput:
    """Flow rate input with flexible units.

    Supports conversion from multiple flow rate specifications:
    - Mass flow: kg/s, kg/h, lb/hr
    - Molar flow: mol/s, kmol/h, lbmol/hr
    - Volumetric flow: m³/s, m³/h, SCFM, ACFM, Nm³/h

    Attributes:
        value: Numerical value
        unit: Unit specification
        reference_conditions: For volumetric flow (T, P)
    """

    value: float
    unit: str
    reference_temperature: float | None = 273.15  # K for standard conditions
    reference_pressure: float | None = 101325.0  # Pa for standard conditions

    def to_mass_flow(
        self, molecular_weight: float, actual_temp: float, actual_pressure: float
    ) -> float:
        """Convert to mass flow rate in kg/s.

        Args:
            molecular_weight: Molecular weight (kg/kmol)
            actual_temp: Actual temperature (K)
            actual_pressure: Actual pressure (Pa)

        Returns:
            Mass flow rate (kg/s)
        """
        # This will be implemented in the utils module
        # Conversion logic deferred to unit_conversion service
        return 0.0
