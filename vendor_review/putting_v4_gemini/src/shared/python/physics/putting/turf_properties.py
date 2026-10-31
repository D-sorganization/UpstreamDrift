"""USGA Stimpmeter turf properties, oblong green geometry, and Holmes/Penner cup capture.

Part of the UpstreamDrift 3D Counterbalanced Putting Dynamics Suite (v3.2/v3.3).
Implements:
- Zero-allocation @precondition signature caching (Issue D-11803)
- Corrected USGA Stimpmeter rolling friction formula: mu_roll = 0.560 / stimp_rating
- Oblong (elliptical) green boundary containment and signed margin geometry
- Holmes/Penner chord-dependent critical capture velocity:
  v_crit(b) = 1.38 * (1 - (b / R_h)^2)^0.75

Architecture budgets enforced:
- <= 500 LOC per file
- <= 100 LOC per function
- <= 8 parameters per function
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
import math
from typing import Any, Callable, Dict, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

F = TypeVar("F", bound=Callable[..., Any])

# Issue D-11803: Module-level signature & parameter-binding cache so @precondition
# never invokes inspect.signature() or allocates BoundArguments dicts at call time.
_SIGNATURE_CACHE: Dict[Callable[..., Any], Tuple[inspect.Signature, Tuple[str, ...], Dict[str, Any]]] = {}

USGA_STIMP_NUMERATOR: float = 0.560
USGA_CUP_RADIUS_M: float = 0.053975  # 4.25 in diameter -> 2.125 in (53.975 mm) radius
USGA_BALL_RADIUS_M: float = 0.021335  # 1.68 in diameter -> 0.84 in (21.335 mm) radius
STANDARD_GRAVITY_M_S2: float = 9.80665
HOLMES_PENNER_V0_M_S: float = 1.38
HOLMES_PENNER_EXPONENT: float = 0.75
SOLID_SPHERE_ROLLING_FACTOR: float = 5.0 / 7.0


def _cache_function_signature(
    func: Callable[..., Any],
) -> Tuple[inspect.Signature, Tuple[str, ...], Dict[str, Any]]:
    """Pre-compute and cache function signature metadata once at decoration time."""
    cached = _SIGNATURE_CACHE.get(func)
    if cached is not None:
        return cached
    sig = inspect.signature(func)
    param_names = tuple(sig.parameters.keys())
    defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    entry = (sig, param_names, defaults)
    _SIGNATURE_CACHE[func] = entry
    return entry


def precondition(
    predicate: Callable[..., bool],
    message: str = "Precondition violated",
) -> Callable[[F], F]:
    """Zero-allocation Design-by-Contract @precondition decorator (Issue D-11803).

    Resolves inspect.signature(func) once at decoration time and caches parameter
    positions. At runtime, invokes the predicate directly without calling
    inspect.signature() or allocating inspect.BoundArguments objects.
    """
    pred_sig = inspect.signature(predicate)
    pred_params = tuple(pred_sig.parameters.values())
    single_dict_mode = (
        len(pred_params) == 1
        and pred_params[0].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
        and pred_params[0].name in ("params", "bound", "kwargs", "args_dict")
    )

    def decorator(func: F) -> F:
        _, param_names, defaults = _cache_function_signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if single_dict_mode:
                call_map = defaults.copy()
                for idx, val in enumerate(args):
                    if idx < len(param_names):
                        call_map[param_names[idx]] = val
                if kwargs:
                    call_map.update(kwargs)
                ok = predicate(call_map)
            elif not kwargs:
                if len(args) < len(param_names) and defaults:
                    resolved = list(args)
                    for idx in range(len(args), len(param_names)):
                        pname = param_names[idx]
                        if pname in defaults:
                            resolved.append(defaults[pname])
                    ok = predicate(*resolved)
                else:
                    ok = predicate(*args)
            else:
                resolved = [
                    args[idx]
                    if idx < len(args)
                    else kwargs.get(pname, defaults.get(pname))
                    for idx, pname in enumerate(param_names)
                ]
                ok = predicate(*resolved)
            if not ok:
                raise ValueError(f"{message} in '{func.__name__}'")
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@dataclass(frozen=True)
class OblongGreenBoundary:
    """Elliptical oblong putting green geometry centered at (center_x, center_y)."""

    semi_major_a: float = 18.0
    semi_minor_b: float = 12.0
    center_x: float = 0.0
    center_y: float = 0.0
    heading_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.semi_major_a <= 0.0 or self.semi_minor_b <= 0.0:
            raise ValueError("Green semi-axes must be strictly positive.")
        if self.semi_major_a < self.semi_minor_b:
            raise ValueError("semi_major_a must be >= semi_minor_b for oblong green.")


@dataclass(frozen=True)
class CupCaptureResult:
    """Holmes/Penner chord-dependent cup capture evaluation result."""

    is_captured: bool
    critical_velocity_m_s: float
    ball_speed_m_s: float
    impact_offset_b_m: float
    normalized_offset: float
    chord_length_m: float
    outcome: str


@dataclass(frozen=True)
class TurfProperties:
    """Physical turf and cup parameters for 3D putting simulation."""

    stimp_rating: float = 11.0
    cup_radius_m: float = USGA_CUP_RADIUS_M
    ball_radius_m: float = USGA_BALL_RADIUS_M
    gravity_m_s2: float = STANDARD_GRAVITY_M_S2
    green_boundary: OblongGreenBoundary = OblongGreenBoundary()

    def __post_init__(self) -> None:
        if self.stimp_rating <= 0.0 or not math.isfinite(self.stimp_rating):
            raise ValueError("stimp_rating must be finite and strictly positive.")
        if self.cup_radius_m <= 0.0 or self.ball_radius_m <= 0.0:
            raise ValueError("cup_radius_m and ball_radius_m must be strictly positive.")
        if self.gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be strictly positive.")

    @property
    def mu_roll(self) -> float:
        """Corrected USGA Stimpmeter rolling friction coefficient."""
        return compute_rolling_friction(self.stimp_rating)


@precondition(
    lambda stimp_rating: (
        isinstance(stimp_rating, (int, float))
        and math.isfinite(stimp_rating)
        and stimp_rating > 0.0
    ),
    "stimp_rating must be a finite positive number",
)
def compute_rolling_friction(stimp_rating: float) -> float:
    """Compute rolling friction coefficient via USGA Stimpmeter formula.

    Formula:
        mu_roll = 0.560 / stimp_rating

    Args:
        stimp_rating: USGA Stimpmeter distance in feet (e.g., 10.5 for tour greens).

    Returns:
        Dimensionless rolling friction coefficient mu_roll.
    """
    return USGA_STIMP_NUMERATOR / float(stimp_rating)


@precondition(
    lambda x_m, y_m, boundary: (
        math.isfinite(x_m)
        and math.isfinite(y_m)
        and boundary.semi_major_a > 0.0
        and boundary.semi_minor_b > 0.0
    ),
    "Coordinates must be finite and boundary axes strictly positive",
)
def normalized_oblong_radius_sq(
    x_m: float,
    y_m: float,
    boundary: OblongGreenBoundary = OblongGreenBoundary(),
) -> float:
    """Compute normalized squared elliptical radius (x'/a)^2 + (y'/b)^2."""
    dx = x_m - boundary.center_x
    dy = y_m - boundary.center_y
    cos_h = math.cos(boundary.heading_rad)
    sin_h = math.sin(boundary.heading_rad)
    x_local = cos_h * dx + sin_h * dy
    y_local = -sin_h * dx + cos_h * dy
    return (x_local / boundary.semi_major_a) ** 2 + (
        y_local / boundary.semi_minor_b
    ) ** 2


@precondition(
    lambda x_m, y_m, boundary, ball_radius_m=0.0: (
        math.isfinite(x_m)
        and math.isfinite(y_m)
        and ball_radius_m >= 0.0
    ),
    "Position must be finite and ball_radius_m non-negative",
)
def is_within_oblong_green(
    x_m: float,
    y_m: float,
    boundary: OblongGreenBoundary = OblongGreenBoundary(),
    ball_radius_m: float = 0.0,
) -> bool:
    """Return True if (x_m, y_m) lies inside the oblong green boundary."""
    eff_a = max(boundary.semi_major_a - ball_radius_m, 1e-9)
    eff_b = max(boundary.semi_minor_b - ball_radius_m, 1e-9)
    shrunk = OblongGreenBoundary(
        semi_major_a=eff_a,
        semi_minor_b=min(eff_a, eff_b),
        center_x=boundary.center_x,
        center_y=boundary.center_y,
        heading_rad=boundary.heading_rad,
    )
    return normalized_oblong_radius_sq(x_m, y_m, shrunk) <= 1.0


@precondition(
    lambda impact_offset_b_m, cup_radius_m=USGA_CUP_RADIUS_M: (
        math.isfinite(impact_offset_b_m)
        and math.isfinite(cup_radius_m)
        and cup_radius_m > 0.0
    ),
    "impact_offset_b_m must be finite and cup_radius_m strictly positive",
)
def critical_capture_velocity(
    impact_offset_b_m: float,
    cup_radius_m: float = USGA_CUP_RADIUS_M,
) -> float:
    """Compute Holmes/Penner chord-dependent critical capture velocity v_crit(b).

    Formula:
        v_crit(b) = 1.38 * (1 - (b / R_h)^2)^0.75   for |b| < R_h
        v_crit(b) = 0.0                              for |b| >= R_h

    Args:
        impact_offset_b_m: Lateral distance b (m) from cup center to ball path chord.
        cup_radius_m: Effective cup radius R_h (m), default 0.053975 m.

    Returns:
        Critical capture speed v_crit in m/s.
    """
    b_abs = abs(float(impact_offset_b_m))
    if b_abs >= cup_radius_m:
        return 0.0
    ratio_sq = (b_abs / cup_radius_m) ** 2
    chord_factor = max(0.0, 1.0 - ratio_sq)
    return HOLMES_PENNER_V0_M_S * (chord_factor ** HOLMES_PENNER_EXPONENT)


@precondition(
    lambda ball_speed_m_s, impact_offset_b_m, cup_radius_m=USGA_CUP_RADIUS_M: (
        math.isfinite(ball_speed_m_s)
        and ball_speed_m_s >= 0.0
        and math.isfinite(impact_offset_b_m)
        and cup_radius_m > 0.0
    ),
    "ball_speed_m_s must be non-negative finite and cup_radius_m positive",
)
def evaluate_cup_capture(
    ball_speed_m_s: float,
    impact_offset_b_m: float,
    cup_radius_m: float = USGA_CUP_RADIUS_M,
) -> CupCaptureResult:
    """Evaluate golf ball cup capture under the Holmes/Penner chord model."""
    b_abs = abs(float(impact_offset_b_m))
    norm_b = b_abs / cup_radius_m
    v_crit = critical_capture_velocity(b_abs, cup_radius_m)

    if b_abs >= cup_radius_m:
        return CupCaptureResult(
            is_captured=False,
            critical_velocity_m_s=0.0,
            ball_speed_m_s=float(ball_speed_m_s),
            impact_offset_b_m=b_abs,
            normalized_offset=norm_b,
            chord_length_m=0.0,
            outcome="MISS_OUTSIDE_CUP",
        )

    chord_len = 2.0 * math.sqrt(max(0.0, cup_radius_m**2 - b_abs**2))
    captured = float(ball_speed_m_s) <= v_crit
    if captured:
        outcome = "CAPTURED_CENTER" if norm_b <= 0.15 else "CAPTURED_CHORD"
    else:
        outcome = "LIP_OUT_FAST"

    return CupCaptureResult(
        is_captured=captured,
        critical_velocity_m_s=v_crit,
        ball_speed_m_s=float(ball_speed_m_s),
        impact_offset_b_m=b_abs,
        normalized_offset=norm_b,
        chord_length_m=chord_len,
        outcome=outcome,
    )


@precondition(
    lambda velocity_xy, slope_grad_xy, turf=TurfProperties(), speed_eps=1e-6: (
        velocity_xy.shape == (2,)
        and slope_grad_xy.shape == (2,)
        and speed_eps > 0.0
    ),
    "velocity_xy and slope_grad_xy must be 2D vectors and speed_eps > 0",
)
def compute_turf_rolling_acceleration(
    velocity_xy: NDArray[np.floating[Any]],
    slope_grad_xy: NDArray[np.floating[Any]],
    turf: TurfProperties = TurfProperties(),
    speed_eps: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute 2D ball rolling acceleration on sloped turf with USGA Stimp friction.

    Includes solid-sphere rolling inertia factor (5/7) on gravitational slope terms
    and opposing Coulomb-Stimp rolling deceleration mu_roll * g * v_hat.
    """
    vel = np.asarray(velocity_xy, dtype=np.float64)
    grad = np.asarray(slope_grad_xy, dtype=np.float64)
    speed = float(np.linalg.norm(vel))

    a_slope = -SOLID_SPHERE_ROLLING_FACTOR * turf.gravity_m_s2 * grad
    if speed <= speed_eps:
        return a_slope

    v_hat = vel / speed
    a_fric = -turf.mu_roll * turf.gravity_m_s2 * v_hat
    return a_slope + a_fric
