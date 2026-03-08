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


# ── Deprecation Helpers ───────────────────────────────────────────────────────

_DEPRECATION_EMITTED: set[str] = set()


def mark_legacy(func_name: str, module: str) -> None:
    """Emit a one-time deprecation warning for legacy physics functions.

    Call this at the top of legacy functions that have Rust replacements.
    """
    key = f"{module}.{func_name}"
    if key not in _DEPRECATION_EMITTED:
        _DEPRECATION_EMITTED.add(key)
        logger.info(
            "DEPRECATION: %s has a Rust kernel replacement. "
            "Migrate to src.shared.python.physics.rust_kernel. "
            "Rust available: %s",
            key,
            _RUST_AVAILABLE,
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
        }
    return info
