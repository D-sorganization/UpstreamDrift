"""Flow-property and primary pressure-loss helpers for the pressure drop engine."""

from __future__ import annotations

import logging
import math

from src.shared.python.contracts import require_positive

from ....utils.unit_constants import STANDARD_GRAVITY
from ...constants import (
    API_14E_C_CONTINUOUS,
    API_14E_C_INTERMITTENT,
    FT_S_TO_M_S,
    KG_M3_TO_LB_FT3,
    RE_LAMINAR_UPPER,
    RE_TURBULENT_LOWER,
)
from ..models.pressure_drop_data_models import FlowProperties, PressureDropInputs
from ..utils.gas_properties import calculate_gas_properties

logger = logging.getLogger(__name__)

GRAVITY = STANDARD_GRAVITY
PI = math.pi


def calculate_flow_properties(inputs: PressureDropInputs) -> FlowProperties:
    """Calculate comprehensive flow properties for the gas mixture."""
    require_positive(inputs.pipe_diameter, "pipe_diameter")
    require_positive(inputs.mass_flow_rate, "mass_flow_rate")
    require_positive(inputs.inlet_temperature, "inlet_temperature")
    require_positive(inputs.inlet_pressure, "inlet_pressure")

    gas_props = calculate_gas_properties(
        inputs.gas_composition.components,
        inputs.inlet_temperature,
        inputs.inlet_pressure,
        inputs.compressibility_correction,
    )

    density = gas_props["density"]
    viscosity = gas_props["viscosity"]
    molecular_weight = gas_props["molecular_weight"]
    compressibility_factor = gas_props["compressibility_factor"]
    speed_of_sound = gas_props["speed_of_sound"]
    heat_capacity_ratio = gas_props["heat_capacity_ratio"]

    require_positive(density, "gas_density")
    require_positive(viscosity, "gas_viscosity")
    require_positive(speed_of_sound, "speed_of_sound")

    pipe_area = PI * (inputs.pipe_diameter**2) / 4.0
    velocity = inputs.mass_flow_rate / (density * pipe_area)
    mass_flux = inputs.mass_flow_rate / pipe_area
    reynolds_number = (density * velocity * inputs.pipe_diameter) / viscosity
    mach_number = velocity / speed_of_sound
    volumetric_flow_rate = inputs.mass_flow_rate / density

    flow_props = FlowProperties(
        density=density,
        viscosity=viscosity,
        velocity=velocity,
        reynolds_number=reynolds_number,
        mach_number=mach_number,
        compressibility_factor=compressibility_factor,
        molecular_weight=molecular_weight,
        mass_flux=mass_flux,
        volumetric_flow_rate=volumetric_flow_rate,
    )

    if not (flow_props.velocity > 0):
        raise ValueError(f"Flow velocity must be positive, got {flow_props.velocity}")
    if not (flow_props.reynolds_number > 0):
        raise ValueError(
            f"Reynolds number must be positive, got {flow_props.reynolds_number}"
        )
    if not (0 <= flow_props.mach_number < 50):
        raise ValueError(
            f"Mach number out of physical range, got {flow_props.mach_number}"
        )

    logger.info("Flow properties calculated:")
    logger.info("  Velocity: %.2f m/s", velocity)
    logger.info("  Reynolds: %.0f", reynolds_number)
    logger.info("  Mach: %.4f", mach_number)
    logger.info("  Density: %.4f kg/m³", density)
    logger.info("  γ (Cp/Cv): %.3f", heat_capacity_ratio)
    logger.info("  Speed of sound: %.1f m/s", speed_of_sound)
    return flow_props


def classify_flow_regime(reynolds_number: float) -> str:
    """Classify flow regime based on Reynolds number."""
    if reynolds_number < RE_LAMINAR_UPPER:
        return "laminar"
    if reynolds_number < RE_TURBULENT_LOWER:
        return "transitional"
    return "turbulent"


def calculate_frictional_pressure_drop(
    friction_factor: float,
    length: float,
    diameter: float,
    density: float,
    velocity: float,
) -> float:
    """Calculate frictional pressure drop using Darcy-Weisbach."""
    if not (friction_factor > 0):
        raise ValueError(f"friction_factor must be positive, got {friction_factor}")
    if not (length > 0):
        raise ValueError(f"length must be positive, got {length}")
    if not (diameter > 0):
        raise ValueError(f"diameter must be positive, got {diameter}")
    if not (density > 0):
        raise ValueError(f"density must be positive, got {density}")
    if not (velocity > 0):
        raise ValueError(f"velocity must be positive, got {velocity}")

    velocity_head = 0.5 * density * (velocity**2)
    dp_friction = friction_factor * (length / diameter) * velocity_head
    if not (dp_friction >= 0):
        raise ValueError(f"Pressure drop must be non-negative, got {dp_friction}")

    logger.debug(
        "Darcy-Weisbach: f=%.6f, L/D=%.1f, ΔP=%.1f Pa",
        friction_factor,
        length / diameter,
        dp_friction,
    )
    return dp_friction


def calculate_elevation_pressure_drop(density: float, elevation_change: float) -> float:
    """Calculate hydrostatic pressure change due to elevation."""
    if density is None:
        raise ValueError("density must be provided")
    dp_elevation = density * GRAVITY * elevation_change
    logger.debug("Elevation: Δh=%.1fm, ΔP=%.1f Pa", elevation_change, dp_elevation)
    return float(dp_elevation)


def calculate_erosional_velocity(
    density: float, service_type: str = "continuous"
) -> float:
    """Calculate erosional velocity limit using API RP 14E."""
    if density is None:
        raise ValueError("density must be provided")
    if service_type == "continuous":
        coefficient = API_14E_C_CONTINUOUS
    elif service_type in {"intermittent", "non_corrosive"}:
        coefficient = API_14E_C_INTERMITTENT
    else:
        coefficient = API_14E_C_CONTINUOUS

    velocity_erosion = coefficient / math.sqrt(density * KG_M3_TO_LB_FT3)
    velocity_erosion_si = velocity_erosion * FT_S_TO_M_S
    logger.debug(
        "Erosional velocity: %.2f m/s (C=%s)", velocity_erosion_si, coefficient
    )
    return velocity_erosion_si
