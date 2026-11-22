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

import importlib.metadata
from dataclasses import dataclass
from typing import Any

from src.shared.python.core.physics_constants import (
    GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    SPIN_DECAY_RATE_S,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# ── Try importing the Rust wheel ──────────────────────────────────────────────

#: Minimum ``upstream_physics`` version this module accepts.  It must equal the
#: crate version in ``rust_core/upstream-physics/Cargo.toml`` (enforced by
#: ``tests/unit/physics/test_rust_kernel_version_guard.py``).  Bump both when
#: the kernel's physics changes (2.2.0: spin-dependent drag, #15460).
MIN_KERNEL_VERSION = "2.2.0"

_REBUILD_HINT = (
    "maturin develop (or maturin build --features python --release) "
    "in rust_core/upstream-physics"
)


def _version_tuple(version: str | None) -> tuple[int, ...] | None:
    """Parse ``"2.2.0"`` into ``(2, 2, 0)``; ``None`` if unparseable."""
    parts = (version or "").split(".")[:3]
    if len(parts) < 2 or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def _installed_kernel_version(module: Any) -> str | None:
    """Return the installed wheel version, or ``None`` when it is unknown."""
    try:
        return importlib.metadata.version("upstream-physics")
    except importlib.metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


def _kernel_is_current(installed: str | None) -> bool:
    """True when ``installed`` is at least ``MIN_KERNEL_VERSION``."""
    have = _version_tuple(installed)
    need = _version_tuple(MIN_KERNEL_VERSION)
    return have is not None and need is not None and have >= need


_RUST_AVAILABLE = False

try:
    import upstream_physics as _rust  # type: ignore[import-untyped]

    _installed = _installed_kernel_version(_rust)
    if _kernel_is_current(_installed):
        _RUST_AVAILABLE = True
        logger.info("upstream_physics Rust kernel loaded successfully")
    else:
        logger.warning(
            "upstream_physics %s is older than the required %s (it lacks "
            "recent physics such as spin-dependent drag) - using the "
            "pure-Python fallback. Rebuild with: %s",
            _installed or "of unknown version",
            MIN_KERNEL_VERSION,
            _REBUILD_HINT,
        )
        _rust = None  # type: ignore[assignment]
except ImportError:
    _rust = None  # type: ignore[assignment]
    logger.warning(
        "upstream_physics Rust wheel not installed — "
        "falling back to pure-Python physics. "
        "Install with: pip install upstream_physics"
    )


def is_rust_available() -> bool:
    """Return True if a current Rust physics kernel is available.

    Postcondition: never raises.  False when the wheel is missing or older
    than ``MIN_KERNEL_VERSION``; callers then use the Python fallback.
    """
    return _RUST_AVAILABLE


# ── Fallback Types ────────────────────────────────────────────────────────────


@dataclass
class IntegratorConfigFallback:
    dt: float
    max_steps: int


@dataclass
class ContactParametersFallback:
    cor: float
    friction: float


@dataclass
class AirPropertiesFallback:
    density: float
    viscosity: float
    temperature: float
    pressure: float


@dataclass
class BallPropertiesFallback:
    mass: float
    radius: float
    area: float
    drag_coefficient: float
    spin_decay_rate: float


# ── Integrator Config ─────────────────────────────────────────────────────────


def create_integrator_config(dt: float = 0.001, max_steps: int = 10000) -> Any:
    """Create an RK4 integrator configuration.

    Args:
        dt: Fixed time step in seconds.
        max_steps: Maximum number of integration steps.

    Returns:
        IntegratorConfig (Rust) or dict fallback.
    """
    if dt is None:
        raise ValueError("dt must be provided")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if _RUST_AVAILABLE:
        return _rust.IntegratorConfig(dt=dt, max_steps=max_steps)
    return IntegratorConfigFallback(dt=dt, max_steps=max_steps)


# ── Contact Parameters ────────────────────────────────────────────────────────


def create_contact_parameters(cor: float = 0.82, friction: float = 0.4) -> Any:
    """Create contact model parameters.

    Args:
        cor: Coefficient of restitution [0, 1].
        friction: Coefficient of friction.

    Returns:
        ContactParameters (Rust) or dict fallback.
    """
    if cor is None:
        raise ValueError("cor must be provided")
    if cor < 0 or cor > 1:
        raise ValueError("cor must be between 0 and 1")
    if friction < 0:
        raise ValueError("friction must be non-negative")
    if _RUST_AVAILABLE:
        return _rust.ContactParameters(cor=cor, friction=friction)
    return ContactParametersFallback(cor=cor, friction=friction)


# ── Vector Math (delegates to Rust Vector3 when available) ────────────────────


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max.

    Uses Rust tools_core::clamp when available, pure Python otherwise.
    """
    if value is None:
        raise ValueError("value must be provided")
    if _RUST_AVAILABLE and hasattr(_rust, "clamp"):
        return float(_rust.clamp(value, min_val, max_val))
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b.

    Uses Rust tools_core::lerp when available, pure Python otherwise.
    """
    if a is None:
        raise ValueError("a must be provided")
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
    if density is None:
        raise ValueError("density must be provided")
    if density <= 0:
        raise ValueError("density must be positive")
    if viscosity <= 0:
        raise ValueError("viscosity must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive (Kelvin)")
    if pressure <= 0:
        raise ValueError("pressure must be positive")
    if _RUST_AVAILABLE:
        return _rust.AirProperties(
            density=density,
            viscosity=viscosity,
            temperature=temperature,
            pressure=pressure,
        )
    return AirPropertiesFallback(
        density=density,
        viscosity=viscosity,
        temperature=temperature,
        pressure=pressure,
    )


def create_ball_properties(
    mass: float = float(GOLF_BALL_MASS_KG),
    radius: float = float(GOLF_BALL_RADIUS_M),
    drag_coefficient: float = float(GOLF_BALL_POST_CRISIS_DRAG_COEFFICIENT),
    spin_decay_rate: float = float(SPIN_DECAY_RATE_S),
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
    if mass is None:
        raise ValueError("mass must be provided")
    if mass <= 0:
        raise ValueError("mass must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if drag_coefficient < 0:
        raise ValueError("drag_coefficient must be non-negative")
    if spin_decay_rate < 0:
        raise ValueError("spin_decay_rate must be non-negative")
    import math

    if _RUST_AVAILABLE:
        return _rust.BallProperties(
            mass=mass,
            radius=radius,
            drag_coefficient=drag_coefficient,
            spin_decay_rate=spin_decay_rate,
        )
    return BallPropertiesFallback(
        mass=mass,
        radius=radius,
        area=math.pi * radius**2,
        drag_coefficient=drag_coefficient,
        spin_decay_rate=spin_decay_rate,
    )


# ── Deprecation Helpers ───────────────────────────────────────────────────────

_DEPRECATION_EMITTED: set[str] = set()


def mark_legacy(func_name: str, module: str) -> None:
    """Emit a one-time deprecation warning for legacy physics functions.

    Call this at the top of legacy functions that have Rust replacements.
    """
    if func_name is None:
        raise ValueError("func_name must be provided")
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
