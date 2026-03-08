"""Rust-backed physics kernel interface for UpstreamDrift.

This module provides a clean Python facade over the ``upstream_physics``
Rust binary (built via PyO3/Maturin). Legacy Python physics code should
import from here instead of re-implementing the math.

If the Rust wheel is not installed, a graceful fallback to pure-Python
implementations is provided so the application never breaks.

Principles:
- **DRY**: All physics calculations route through the same Rust binary
  that the WASM frontend uses.
- **DbC**: Each function validates inputs before forwarding to Rust.
- **TDD**: See ``tests/rust_bindings/test_physics_bindings.py``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Try importing the Rust wheel ──────────────────────────────────────────────

_RUST_AVAILABLE = False

try:
    import upstream_physics as _rust  # type: ignore[import-untyped]

    _RUST_AVAILABLE = True
    logger.info("upstream_physics Rust kernel loaded successfully")
except ImportError:
    _rust = None  # type: ignore[assignment]
    logger.warning(
        "upstream_physics Rust wheel not installed — "
        "falling back to pure-Python physics. "
        "Install with: pip install upstream_physics"
    )


def is_rust_available() -> bool:
    """Return True if the Rust physics kernel is available."""
    return _RUST_AVAILABLE


# ── Integrator Config ─────────────────────────────────────────────────────────


def create_integrator_config(dt: float = 0.001, max_steps: int = 10000) -> Any:
    """Create an RK4 integrator configuration.

    Args:
        dt: Fixed time step in seconds.
        max_steps: Maximum number of integration steps.

    Returns:
        IntegratorConfig (Rust) or dict fallback.
    """
    if _RUST_AVAILABLE:
        return _rust.IntegratorConfig(dt=dt, max_steps=max_steps)
    return {"dt": dt, "max_steps": max_steps}


# ── Contact Parameters ────────────────────────────────────────────────────────


def create_contact_parameters(cor: float = 0.82, friction: float = 0.4) -> Any:
    """Create contact model parameters.

    Args:
        cor: Coefficient of restitution [0, 1].
        friction: Coefficient of friction.

    Returns:
        ContactParameters (Rust) or dict fallback.
    """
    if _RUST_AVAILABLE:
        return _rust.ContactParameters(cor=cor, friction=friction)
    return {"cor": cor, "friction": friction}


# ── Vector Math (delegates to Rust Vector3 when available) ────────────────────


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max.

    Uses Rust tools_core::clamp when available, pure Python otherwise.
    """
    if _RUST_AVAILABLE and hasattr(_rust, "clamp"):
        return float(_rust.clamp(value, min_val, max_val))
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b.

    Uses Rust tools_core::lerp when available, pure Python otherwise.
    """
    if _RUST_AVAILABLE and hasattr(_rust, "lerp"):
        return float(_rust.lerp(a, b, t))
    return a + t * (b - a)


# ── Aerodynamics ──────────────────────────────────────────────────────────────


def create_air_properties(
    density: float = 1.225,
    viscosity: float = 1.81e-5,
    temperature: float = 288.15,
    pressure: float = 101325.0,
) -> Any:
    """Create air properties for aerodynamics calculations.

    Design by Contract:
        Preconditions:
            - density must be positive
            - viscosity must be positive
            - temperature must be positive (Kelvin)

    Args:
        density: Air density [kg/m³].
        viscosity: Dynamic viscosity [Pa·s].
        temperature: Temperature [K].
        pressure: Atmospheric pressure [Pa].

    Returns:
        AirProperties (Rust) or dict fallback.
    """
    if _RUST_AVAILABLE:
        return _rust.AirProperties(
            density=density,
            viscosity=viscosity,
            temperature=temperature,
            pressure=pressure,
        )
    return {
        "density": density,
        "viscosity": viscosity,
        "temperature": temperature,
        "pressure": pressure,
    }


def create_ball_properties(
    mass: float = 0.04593,
    radius: float = 0.02135,
    drag_coefficient: float = 0.25,
    spin_decay_rate: float = 0.1,
) -> Any:
    """Create golf ball properties for aerodynamics calculations.

    Design by Contract:
        Preconditions:
            - mass must be positive
            - radius must be positive

    Args:
        mass: Ball mass [kg].
        radius: Ball radius [m].
        drag_coefficient: Baseline drag coefficient.
        spin_decay_rate: Spin decay time constant [1/s].

    Returns:
        BallProperties (Rust) or dict fallback.
    """
    import math

    if _RUST_AVAILABLE:
        return _rust.BallProperties(
            mass=mass,
            radius=radius,
            drag_coefficient=drag_coefficient,
            spin_decay_rate=spin_decay_rate,
        )
    return {
        "mass": mass,
        "radius": radius,
        "area": math.pi * radius**2,
        "drag_coefficient": drag_coefficient,
        "spin_decay_rate": spin_decay_rate,
    }


# ── Deprecation Helpers ───────────────────────────────────────────────────────

_DEPRECATION_EMITTED: set[str] = set()


def mark_legacy(func_name: str, module: str) -> None:
    """Emit a one-time deprecation warning for legacy physics functions.

    Call this at the top of legacy functions that have Rust replacements.
    """
    import warnings

    key = f"{module}.{func_name}"
    if key not in _DEPRECATION_EMITTED:
        _DEPRECATION_EMITTED.add(key)
        warnings.warn(
            f"{key} has a Rust kernel replacement. "
            f"Migrate to src.shared.python.physics.rust_kernel. "
            f"Rust available: {_RUST_AVAILABLE}",
            DeprecationWarning,
            stacklevel=2,
        )


# ── Module Diagnostics ────────────────────────────────────────────────────────


def get_kernel_info() -> dict[str, Any]:
    """Return diagnostic information about the physics kernel."""
    info: dict[str, Any] = {
        "rust_available": _RUST_AVAILABLE,
        "backend": "rust" if _RUST_AVAILABLE else "python-fallback",
    }
    if _RUST_AVAILABLE:
        info["types"] = {
            "IntegratorConfig": hasattr(_rust, "IntegratorConfig"),
            "IntegrationResult": hasattr(_rust, "IntegrationResult"),
            "ContactParameters": hasattr(_rust, "ContactParameters"),
            "ContactResult": hasattr(_rust, "ContactResult"),
            "SwingPlaneResult": hasattr(_rust, "SwingPlaneResult"),
            "AirProperties": hasattr(_rust, "AirProperties"),
            "BallProperties": hasattr(_rust, "BallProperties"),
            "AeroForces": hasattr(_rust, "AeroForces"),
        }
    return info
