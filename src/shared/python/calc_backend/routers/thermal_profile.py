"""Thermal profile predictor router.  See issue #608."""

from __future__ import annotations  # noqa: E402, F404

import math  # noqa: E402

from fastapi import APIRouter, HTTPException  # noqa: E402

from ..contracts.thermal_profile import (  # noqa: E402
    ThermalProfileDataPoint,
    ThermalProfileRequest,
    ThermalProfileResponse,
)

router = APIRouter(prefix="/api/calc/thermal-profile", tags=["thermal-profile"])


@router.post("", response_model=ThermalProfileResponse)
def predict_thermal_profile(
    request: ThermalProfileRequest,
) -> ThermalProfileResponse:
    """Predict temperature profile for a heated vessel."""
    try:
        result = _solve_thermal_profile(request)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


def _solve_thermal_profile(
    request: ThermalProfileRequest,
) -> ThermalProfileResponse:
    """Solve the thermal ODE using RK4 integration."""
    # Build power function
    power_w = request.power_w
    profile = request.power_profile

    def power_func(t: float) -> float:
        if profile == "constant":
            return power_w
        if profile == "linear_ramp":
            return power_w + request.ramp_rate_w_per_s * t
        if profile == "step":
            return power_w if t < request.step_time_s else 0.0
        return power_w

    # ODE: dT/dt = (Q_in - h*(T - T_amb)) / C_th
    thermal_mass = request.thermal_mass_j_per_k
    h = request.heat_loss_coeff_w_per_k
    t_amb = request.ambient_temp_c

    def deriv(t: float, temp: float) -> float:
        if t is None:
            raise ValueError("t must be provided")
        q_in = power_func(t)
        q_loss = h * (temp - t_amb)
        return (q_in - q_loss) / thermal_mass

    # RK4 integration
    dt = (request.t_end_s - request.t_start_s) / (request.num_points - 1)
    data: list[ThermalProfileDataPoint] = []
    temp = request.initial_temp_c

    for i in range(request.num_points):
        t = request.t_start_s + i * dt
        q = power_func(t)
        q_loss = h * (temp - t_amb)
        if not all(math.isfinite(value) for value in (t, temp, q, q_loss)):
            raise ValueError(
                "Thermal profile diverged with non-finite values; reduce the "
                "time span or check the thermal inputs"
            )

        data.append(
            ThermalProfileDataPoint(
                time_s=round(t, 4),
                temperature_c=round(temp, 4),
                power_w=round(q, 4),
                heat_loss_w=round(q_loss, 4),
            )
        )

        if i < request.num_points - 1:
            k1 = deriv(t, temp)
            k2 = deriv(t + dt / 2, temp + dt * k1 / 2)
            k3 = deriv(t + dt / 2, temp + dt * k2 / 2)
            k4 = deriv(t + dt, temp + dt * k3)
            temp += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if not math.isfinite(temp):
                raise ValueError(
                    "Thermal profile diverged with non-finite values; reduce the "
                    "time span or check the thermal inputs"
                )

    temps = [d.temperature_c for d in data]
    final_temp = temps[-1]
    max_temp = max(temps)
    min_temp = min(temps)

    steady_state = None
    time_constant = None
    if profile == "constant" and h > 0:
        steady_state = round(power_w / h + t_amb, 4)
        time_constant = round(thermal_mass / h, 4)

    return ThermalProfileResponse(
        data=data,
        final_temp_c=round(final_temp, 4),
        max_temp_c=round(max_temp, 4),
        min_temp_c=round(min_temp, 4),
        temp_change_c=round(final_temp - request.initial_temp_c, 4),
        steady_state_temp_c=steady_state,
        time_constant_s=time_constant,
    )
