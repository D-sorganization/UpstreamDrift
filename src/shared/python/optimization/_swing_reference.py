"""Private adaptive endpoint refinement for smooth angular swing ODEs.

Two tolerance/step settings provide numerical refinement evidence, not a
rigorous global-error bound, independent dynamics oracle or physical validation.
State order is angular positions (rad), then angular velocities (rad/s).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.integrate import solve_ivp

from src.shared.python.core.contracts.validators import check_finite

Rhs = Callable[[float, np.ndarray], object]
_TOLERANCE_REFINEMENT = 10.0
_SOLVER_TOLERANCE_MARGIN = 10.0
_STEP_REFINEMENT = 2.0
_MIN_SOLVER_RTOL = 100 * np.finfo(float).eps


class ReferenceIntegrationError(RuntimeError):
    """The requested reference integration or refinement was not established."""


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def finite_real_array(value: object, name: str) -> np.ndarray:
    """Own finite real data without string/bool coercion; callers check shape."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or any(
        isinstance(item, (bool, np.bool_))
        for item in np.asarray(value, dtype=object).flat
    ):
        raise TypeError(f"{name} must contain real numbers, not booleans or strings")
    result = np.array(array, dtype=float, copy=True)
    if not check_finite(result):
        raise ValueError(f"{name} must be finite")
    return result


def finite_vector(value: object, name: str) -> np.ndarray:
    """Own a finite nonempty vector with strict numeric element types."""
    result = finite_real_array(value, name)
    if result.ndim != 1 or not result.size:
        raise ValueError(f"{name} must be a nonempty vector")
    return result


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True)
class ReferenceControls:
    """Explicit numerical tolerances, step cap and RHS budget per integration.

    Position/velocity absolute tolerances have units rad/rad/s respectively;
    rtol is dimensionless. These are numerical settings, not measured error.
    These tolerances govern endpoint agreement. Local integration tolerances
    are ten times tighter initially and one hundred times tighter on refinement;
    the step cap is halved. This margin does not guarantee endpoint convergence.
    Values that require SciPy's silent relative-tolerance clipping are refused.
    """

    rtol: float
    position_atol: float
    velocity_atol: float
    max_step_s: float
    max_rhs_evaluations: int = 100_000

    def __post_init__(self) -> None:
        for name in ("rtol", "position_atol", "velocity_atol", "max_step_s"):
            value = _positive_real(getattr(self, name), name)
            if value / (_SOLVER_TOLERANCE_MARGIN * _TOLERANCE_REFINEMENT) == 0:
                raise ValueError(f"{name} is not representable after refinement")
            object.__setattr__(self, name, value)
        refined_rtol = self.rtol / (_SOLVER_TOLERANCE_MARGIN * _TOLERANCE_REFINEMENT)
        if refined_rtol < _MIN_SOLVER_RTOL or self.rtol >= 1:
            raise ValueError("rtol must be below one and resolve the refined tolerance")
        budget = positive_integer(self.max_rhs_evaluations, "max_rhs_evaluations")
        object.__setattr__(self, "max_rhs_evaluations", budget)

    def absolute_tolerances(self, joint_count: int) -> np.ndarray:
        """Position block followed by velocity block; no mixed-unit norm."""
        return np.r_[
            np.full(joint_count, self.position_atol),
            np.full(joint_count, self.velocity_atol),
        ]


@dataclass(frozen=True)
class ReferenceEndpoint:
    """Owned refined endpoint and componentwise coarse/fine separation.

    Separation is divided by atol+rtol*max(abs(coarse),abs(fine)).
    Support requires every component ratio <= 1. This is a refinement check,
    not a certified absolute error bound or a whole-trajectory accuracy claim.
    """

    state: tuple[float, ...]
    normalized_refinement: tuple[float, ...]
    rhs_evaluations: tuple[int, int]
    controls: ReferenceControls
    status: str = "refinement_supported"


def _checked_rhs(
    rhs: Rhs, size: int, budget: int
) -> tuple[Callable[[float, np.ndarray], np.ndarray], list[int]]:
    calls = [0]

    def evaluate(time: float, state: np.ndarray) -> np.ndarray:
        calls[0] += 1
        if calls[0] > budget:
            raise ReferenceIntegrationError("reference RHS evaluation budget exhausted")
        owned = finite_vector(state, "RHS state")
        owned.setflags(write=False)
        try:
            result = finite_vector(rhs(time, owned), "RHS result")
            if result.shape != (size,):
                raise ValueError("RHS result shape differs from the state")
        except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
            raise ReferenceIntegrationError(
                f"invalid reference RHS: {error}"
            ) from error
        return result

    return evaluate, calls


def _solve_endpoint(
    rhs: Rhs, initial: np.ndarray, duration_s: float, options: dict
) -> tuple[np.ndarray, int]:
    settings = dict(options)
    checked_rhs, calls = _checked_rhs(
        rhs, initial.size, settings.pop("max_rhs_evaluations")
    )
    result = solve_ivp(
        checked_rhs, (0.0, duration_s), initial.copy(), t_eval=[duration_s], **settings
    )
    if not result.success:
        raise ReferenceIntegrationError(
            f"reference integration failed: {result.message}"
        )
    if np.asarray(result.t).shape != (1,) or result.t[0] != duration_s:
        raise ReferenceIntegrationError(
            "reference did not reach the requested endpoint"
        )
    if np.asarray(result.y).shape != (initial.size, 1):
        raise ReferenceIntegrationError("invalid reference endpoint shape")
    return finite_vector(result.y[:, -1], "reference endpoint"), calls[0]


def _solver_settings(controls: ReferenceControls, size: int, refined: bool) -> dict:
    tolerance_factor = _SOLVER_TOLERANCE_MARGIN * (
        _TOLERANCE_REFINEMENT if refined else 1.0
    )
    step_factor = _STEP_REFINEMENT if refined else 1.0
    return {
        "method": "DOP853",
        "rtol": controls.rtol / tolerance_factor,
        "atol": controls.absolute_tolerances(size // 2) / tolerance_factor,
        "max_step": controls.max_step_s / step_factor,
        "max_rhs_evaluations": controls.max_rhs_evaluations,
    }


def _refinement_ratios(
    coarse: np.ndarray,
    fine: np.ndarray,
    controls: ReferenceControls,
) -> np.ndarray:
    scale = np.maximum(np.abs(coarse), np.abs(fine))
    denominator = controls.absolute_tolerances(coarse.size // 2) + controls.rtol * scale
    ratios = finite_vector(np.abs(coarse - fine) / denominator, "refinement ratios")
    if np.any(ratios > 1):
        raise ReferenceIntegrationError("reference refinement is unresolved")
    return ratios


def resolved_endpoint(
    rhs: Rhs, state: object, duration_s: float, controls: ReferenceControls
) -> ReferenceEndpoint:
    """Integrate a smooth angular ODE independently of a fixed shooting map.

    Input contract failures raise TypeError/ValueError. Integration, output or
    refinement failure raises ReferenceIntegrationError, never an accuracy flag.
    Each pass has a bounded number of RHS evaluations; callback wall time is
    outside this numerical budget. This path does not resolve hybrid events.
    """
    if not callable(rhs) or not isinstance(controls, ReferenceControls):
        raise TypeError("expected a callable RHS and ReferenceControls")
    initial = finite_vector(state, "initial state")
    if initial.size % 2:
        raise ValueError("state requires equal position and velocity blocks")
    duration = _positive_real(duration_s, "duration_s")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            coarse, coarse_calls = _solve_endpoint(
                rhs, initial, duration, _solver_settings(controls, initial.size, False)
            )
            fine, fine_calls = _solve_endpoint(
                rhs, initial, duration, _solver_settings(controls, initial.size, True)
            )
            ratios = _refinement_ratios(coarse, fine, controls)
    except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
        raise ReferenceIntegrationError(
            f"reference evaluation failed: {error}"
        ) from error
    return ReferenceEndpoint(
        tuple(float(value) for value in fine),
        tuple(float(value) for value in ratios),
        (coarse_calls, fine_calls),
        controls,
    )


__all__ = ()
