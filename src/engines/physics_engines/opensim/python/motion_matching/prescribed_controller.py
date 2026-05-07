"""Pure helpers for OpenSim prescribed polynomial torque controller setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    evaluate_polynomial_torque,
)

UnavailableMode = Literal["raise", "fallback"]


class PrescribedControllerUnavailableError(RuntimeError):
    """Raised when an OpenSim PrescribedController boundary is unavailable."""


@dataclass(frozen=True, slots=True)
class PrescribedPolynomialTorquePlan:
    """Validated polynomial torque samples for a prescribed-controller backend."""

    time_grid: npt.NDArray[np.float64]
    coeffs: npt.NDArray[np.float64]
    actuator_names: tuple[str, ...]
    sampled_tau: npt.NDArray[np.float64]


def build_prescribed_polynomial_torque_plan(
    *,
    theta: npt.ArrayLike,
    time_grid: npt.ArrayLike,
    actuator_names: tuple[str, ...] | list[str],
) -> PrescribedPolynomialTorquePlan:
    """Validate inputs and sample polynomial torques for prescribed controls."""

    coeffs = _coerce_coeffs(theta)
    times = _coerce_time_grid(time_grid)
    names = _coerce_actuator_names(actuator_names, n_actuators=coeffs.shape[0])
    sampled_tau = np.vstack(
        [evaluate_polynomial_torque(coeffs, float(t)) for t in times]
    )

    return PrescribedPolynomialTorquePlan(
        time_grid=times.copy(),
        coeffs=coeffs.copy(),
        actuator_names=names,
        sampled_tau=sampled_tau,
    )


def build_prescribed_polynomial_controller(
    *,
    theta: npt.ArrayLike,
    time_grid: npt.ArrayLike,
    actuator_names: tuple[str, ...] | list[str],
    opensim_module: Any | None = None,
    unavailable: UnavailableMode = "raise",
) -> tuple[Any | None, PrescribedPolynomialTorquePlan]:
    """Construct the OpenSim PrescribedController boundary when available.

    This pure boundary validates and pre-samples the torque law. It intentionally
    does not wire model actuators or run a live OpenSim integration.
    """

    if unavailable not in ("raise", "fallback"):
        msg = "unavailable must be 'raise' or 'fallback'"
        raise ValueError(msg)

    plan = build_prescribed_polynomial_torque_plan(
        theta=theta,
        time_grid=time_grid,
        actuator_names=actuator_names,
    )
    osim = opensim_module if opensim_module is not None else _import_opensim()
    constructor = (
        getattr(osim, "PrescribedController", None) if osim is not None else None
    )
    if constructor is None:
        return _handle_unavailable(
            "OpenSim PrescribedController is not available",
            plan=plan,
            unavailable=unavailable,
        )

    try:
        controller = constructor()
    except Exception as exc:  # noqa: BLE001 -- optional OpenSim bindings vary
        return _handle_unavailable(
            f"OpenSim PrescribedController construction failed: {exc}",
            plan=plan,
            unavailable=unavailable,
        )
    return controller, plan


def _coerce_coeffs(theta: npt.ArrayLike) -> npt.NDArray[np.float64]:
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim == 1:
        if theta_arr.size == 0:
            msg = "theta must contain at least one actuator coefficient row"
            raise ValueError(msg)
        if theta_arr.size % COEFFS_PER_JOINT != 0:
            msg = f"flat theta length must be divisible by {COEFFS_PER_JOINT}"
            raise ValueError(msg)
        coeffs = theta_arr.reshape(-1, COEFFS_PER_JOINT)
    elif theta_arr.ndim == 2:
        if theta_arr.shape[0] == 0:
            msg = "theta must contain at least one actuator coefficient row"
            raise ValueError(msg)
        if theta_arr.shape[1] != COEFFS_PER_JOINT:
            msg = f"theta coefficient matrix must have {COEFFS_PER_JOINT} coefficients"
            raise ValueError(msg)
        coeffs = theta_arr
    else:
        msg = f"theta must be 1-D or 2-D; got ndim={theta_arr.ndim}"
        raise ValueError(msg)

    if not np.all(np.isfinite(coeffs)):
        msg = "theta must be finite"
        raise ValueError(msg)
    return np.asarray(coeffs, dtype=np.float64)


def _coerce_time_grid(time_grid: npt.ArrayLike) -> npt.NDArray[np.float64]:
    times = np.asarray(time_grid, dtype=np.float64)
    if times.ndim != 1:
        msg = "time_grid must be 1-D"
        raise ValueError(msg)
    if times.size == 0:
        msg = "time_grid must contain at least one sample"
        raise ValueError(msg)
    if not np.all(np.isfinite(times)):
        msg = "time_grid must be finite"
        raise ValueError(msg)
    if np.any(np.diff(times) <= 0.0):
        msg = "time_grid must be strictly increasing"
        raise ValueError(msg)
    return times


def _coerce_actuator_names(
    actuator_names: tuple[str, ...] | list[str],
    *,
    n_actuators: int,
) -> tuple[str, ...]:
    names = tuple(actuator_names)
    if len(names) != n_actuators:
        msg = f"actuator name count must match coefficient rows: {len(names)} != {n_actuators}"
        raise ValueError(msg)
    if any(not isinstance(name, str) or not name for name in names):
        msg = "actuator names must be non-empty strings"
        raise ValueError(msg)
    if len(set(names)) != len(names):
        msg = "actuator names must be unique"
        raise ValueError(msg)
    return names


def _import_opensim() -> Any | None:
    try:
        import opensim  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError:
        return None
    return opensim


def _handle_unavailable(
    message: str,
    *,
    plan: PrescribedPolynomialTorquePlan,
    unavailable: UnavailableMode,
) -> tuple[None, PrescribedPolynomialTorquePlan]:
    if unavailable == "fallback":
        return None, plan
    raise PrescribedControllerUnavailableError(message)


__all__ = [
    "PrescribedControllerUnavailableError",
    "PrescribedPolynomialTorquePlan",
    "build_prescribed_polynomial_controller",
    "build_prescribed_polynomial_torque_plan",
]
