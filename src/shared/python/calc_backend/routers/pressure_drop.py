"""Pressure drop calculator router.  See issue #613."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException

from ..contracts.pressure_drop import PressureDropRequest, PressureDropResponse

router = APIRouter(prefix="/api/calc/pressure-drop", tags=["pressure-drop"])

# Universal gas constant [J/(mol*K)]
_R_UNIVERSAL = 8.314462618


@router.post("", response_model=PressureDropResponse)
def calculate_pressure_drop(request: PressureDropRequest) -> PressureDropResponse:
    """Calculate pressure drop using Darcy-Weisbach equation.

    Uses an inline implementation matching the legacy PressureDropCalculator
    to avoid import issues with the advanced pressure_drop_calculator package.
    """
    try:
        # Gas density (ideal gas)
        density = (request.pressure_pa * request.molecular_weight_kg_mol) / (
            _R_UNIVERSAL * request.temperature_k
        )

        # Viscosity estimate (Sutherland's formula for air-like gases)
        t_ref = 291.15
        mu_ref = 1.827e-5
        s_const = 120.0
        viscosity = (
            mu_ref
            * ((t_ref + s_const) / (request.temperature_k + s_const))
            * (request.temperature_k / t_ref) ** 1.5
        )

        # Velocity
        vol_flow = request.flow_rate_kg_s / density if density > 0 else 0.0
        area = math.pi * (request.pipe_diameter_m / 2) ** 2
        velocity = vol_flow / area if area > 0 else 0.0

        # Reynolds number
        re = (density * velocity * request.pipe_diameter_m) / viscosity if viscosity > 0 else 0.0

        # Friction factor
        rel_roughness = (
            request.roughness_m / request.pipe_diameter_m if request.pipe_diameter_m > 0 else 0.0
        )

        if re > 4000:
            a_val = rel_roughness / 3.7
            b_val = 5.74 / (re**0.9) if re > 0 else 0.01
            try:
                friction_factor = 0.25 / (math.log10(a_val + b_val) ** 2)
            except ValueError:
                friction_factor = 0.02
        elif re > 2300:
            friction_factor = 0.03
        else:
            friction_factor = 64 / re if re > 0 else 0.05

        # Darcy-Weisbach pressure drop
        if request.pipe_diameter_m > 0:
            pressure_drop_pa = (
                friction_factor
                * (request.pipe_length_m / request.pipe_diameter_m)
                * (density * velocity**2 / 2)
            )
        else:
            pressure_drop_pa = 0.0

        # Flow regime
        if re < 2300:
            flow_regime = "Laminar"
        elif re < 4000:
            flow_regime = "Transitional"
        else:
            flow_regime = "Turbulent"

    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PressureDropResponse(
        pressure_drop_pa=pressure_drop_pa,
        reynolds_number=re,
        friction_factor=friction_factor,
        velocity_m_s=velocity,
        flow_regime=flow_regime,
        density_kg_m3=density,
        viscosity_pa_s=viscosity,
    )
