"""Rust-backed aerodynamics facade with pure-Python fallback.

This module exposes a single :func:`compute_total_force` entry point that
routes through the ``upstream_physics`` Rust kernel when it is installed,
and falls back to the canonical Python implementation otherwise.

The pure-Python fallback intentionally re-uses the same
:class:`DragModel`, :class:`LiftModel`, and :class:`MagnusModel` objects
as the legacy :class:`AerodynamicsEngine` so behaviour is byte-identical
between fallback and reference paths.

This is the per-step entry point used by hot integrator loops to avoid
the Python/C boundary on every RK4 stage (issue #5265). The full
:class:`AerodynamicsEngine` (with environment randomization and stochastic
gusts) remains the public top-level API; it can optionally delegate the
deterministic drag/lift/magnus + wind subset to this facade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# ── Lazy Rust import ─────────────────────────────────────────────────────────

_RUST_AVAILABLE = False

try:
    import upstream_physics as _rust  # type: ignore[import-untyped]

    _RUST_AVAILABLE = bool(
        hasattr(_rust, "AerodynamicsEngine")
        and hasattr(_rust, "AeroEngineConfig")
        and hasattr(_rust, "AeroBallProperties")
        and hasattr(_rust, "AirProperties")
    )
    if _RUST_AVAILABLE:
        logger.info("upstream_physics AerodynamicsEngine loaded (#5265 Rust kernel)")
    else:
        logger.info(
            "upstream_physics installed but AerodynamicsEngine missing — "
            "falling back to pure Python aerodynamics."
        )
except ImportError:
    _rust = None  # type: ignore[assignment]
    logger.info(
        "upstream_physics not installed — using pure-Python aerodynamics. "
        "Install with `maturin develop -m rust_core/upstream-physics`."
    )


def is_rust_available() -> bool:
    """Return True when the Rust aerodynamics kernel is loaded."""
    return _RUST_AVAILABLE


# ── Spec for the per-step facade ─────────────────────────────────────────────


@dataclass(frozen=True)
class AerodynamicsSpec:
    """Immutable per-step aerodynamics specification.

    Captures everything the deterministic Rust kernel needs: ball physical
    properties, air properties, toggle flags, and (optionally) a constant
    wind vector for the relative-velocity correction. Stochastic wind gusts
    are not handled here — callers wanting gusts should pre-compute the
    wind vector in Python and pass it via ``wind``.
    """

    mass: float
    radius: float
    drag_coefficient: float
    spin_decay_rate: float
    air_density: float
    air_viscosity: float = 1.81e-5
    air_temperature: float = 288.15
    air_pressure: float = 101_325.0
    drag_enabled: bool = True
    lift_enabled: bool = True
    magnus_enabled: bool = True
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ── Rust path ────────────────────────────────────────────────────────────────


def _build_rust_engine(spec: AerodynamicsSpec) -> Any:
    """Construct a Rust ``AerodynamicsEngine`` from a spec."""
    if not _RUST_AVAILABLE:
        raise RuntimeError("upstream_physics Rust kernel is not available")
    cfg = _rust.AeroEngineConfig(
        enabled=True,
        drag_enabled=spec.drag_enabled,
        lift_enabled=spec.lift_enabled,
        magnus_enabled=spec.magnus_enabled,
    )
    ball = _rust.AeroBallProperties(
        mass=spec.mass,
        radius=spec.radius,
        drag_coefficient=spec.drag_coefficient,
        spin_decay_rate=spec.spin_decay_rate,
    )
    air = _rust.AirProperties(
        density=spec.air_density,
        viscosity=spec.air_viscosity,
        temperature=spec.air_temperature,
        pressure=spec.air_pressure,
    )
    wind_model = None
    wx, wy, wz = spec.wind
    if wx != 0.0 or wy != 0.0 or wz != 0.0:
        wind_cfg = _rust.WindConfig(
            base_velocity=[wx, wy, wz],
            altitude_gradient=False,
            gradient_factor=0.0,
            turbulence_intensity=0.0,
        )
        wind_model = _rust.WindModel(
            wind_cfg,
            [0.0] * 30,
            [0.0] * 30,
            [1.0] * 10,
        )
    return _rust.AerodynamicsEngine(cfg, ball, air, wind_model)


# ── Python fallback ──────────────────────────────────────────────────────────


def _python_fallback_total(
    spec: AerodynamicsSpec,
    velocity: np.ndarray,
    spin: np.ndarray,
) -> np.ndarray:
    """Pure-Python total aerodynamic force matching the Rust kernel.

    Coefficient definitions are intentionally identical to
    ``rust_core/upstream-physics/src/aerodynamics.rs`` so the fallback
    produces byte-identical results to the Rust kernel. This is the
    invariant the parity test asserts.

    The legacy :class:`DragModel` / :class:`LiftModel` / :class:`MagnusModel`
    in ``_models.py`` use slightly different coefficient formulas (in
    particular ``MagnusModel`` scales by 1.25 in the small-spin regime
    while the Rust kernel scales by 0.4). The Rust formulation is the
    new canonical reference per issue #5265.
    """
    # Local import avoids a circular import: this module is consumed by
    # _engine.py which itself imports from the aerodynamics package.
    from src.shared.python.physics.atmosphere import cd_dimpled_sphere

    area = math.pi * spec.radius**2
    rel_velocity = velocity - np.asarray(spec.wind, dtype=float)
    speed = float(
        math.sqrt(np.dot(rel_velocity, rel_velocity))
    )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
    spin_mag = float(
        math.sqrt(np.dot(spin, spin))
    )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm

    total: np.ndarray = np.zeros(3)
    if speed < 1e-6:
        return total

    # ── Drag ────────────────────────────────────────────────────────────
    if spec.drag_enabled:
        diameter = 2.0 * spec.radius
        re = spec.air_density * speed * diameter / spec.air_viscosity
        re_clamped = max(1.0e3, min(1.0e7, re))
        cd = cd_dimpled_sphere(re_clamped, base_cd=spec.drag_coefficient)
        f_mag = 0.5 * spec.air_density * cd * area * speed * speed
        total = total - f_mag * rel_velocity / speed

    # ── Lift ────────────────────────────────────────────────────────────
    if spec.lift_enabled and spin_mag > 1e-6:
        spin_ratio = spec.radius * spin_mag / (speed + 1e-10)
        # Rust: cl_max = 0.4, cl = cl_max * (1 - exp(-spin_ratio / 0.1))
        cl = 0.4 * (1.0 - math.exp(-spin_ratio / 0.1))
        spin_axis = spin / (spin_mag + 1e-10)
        lift_dir = np.cross(spin_axis, rel_velocity)
        lift_norm = float(
            math.sqrt(np.dot(lift_dir, lift_dir))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        if lift_norm > 1e-6:
            f_mag = 0.5 * spec.air_density * cl * area * speed * speed
            total = total + f_mag * lift_dir / lift_norm

    # ── Magnus ──────────────────────────────────────────────────────────
    if spec.magnus_enabled and spin_mag > 1e-6:
        magnus_dir = np.cross(spin, rel_velocity)
        magnus_norm = float(
            math.sqrt(np.dot(magnus_dir, magnus_dir))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        if magnus_norm > 1e-6:
            spin_param = spec.radius * spin_mag / speed
            # Rust: 0.4 * min(spin_param, 0.5)
            cm = 0.4 * min(spin_param, 0.5)
            f_mag = 0.5 * spec.air_density * cm * area * speed * speed
            total = total + f_mag * magnus_dir / magnus_norm

    return total


# ── Public API ───────────────────────────────────────────────────────────────


def compute_total_force(
    spec: AerodynamicsSpec,
    velocity: np.ndarray,
    spin: np.ndarray,
) -> np.ndarray:
    """Return the total aerodynamic force for `(velocity, spin)`.

    Uses the Rust kernel when available, otherwise the pure-Python
    fallback. Both paths apply ``spec.wind`` as a relative-velocity
    subtraction before evaluating drag/lift/magnus.

    Args:
        spec: Immutable aerodynamics specification.
        velocity: Ball velocity in world frame [m/s], shape (3,).
        spin: Ball angular velocity [rad/s], shape (3,).

    Returns:
        Total aerodynamic force [N], shape (3,).
    """
    if velocity is None:
        raise ValueError("velocity must be provided")
    if spin is None:
        raise ValueError("spin must be provided")
    velocity = np.asarray(velocity, dtype=float)
    spin = np.asarray(spin, dtype=float)
    if velocity.shape != (3,):
        raise ValueError(f"velocity must be shape (3,), got {velocity.shape}")
    if spin.shape != (3,):
        raise ValueError(f"spin must be shape (3,), got {spin.shape}")

    if _RUST_AVAILABLE:
        engine = _build_rust_engine(spec)
        f = engine.compute_total_force(
            [float(velocity[0]), float(velocity[1]), float(velocity[2])],
            [float(spin[0]), float(spin[1]), float(spin[2])],
            0.0,
            [0.0, 0.0, 0.0],
        )
        return np.asarray(f, dtype=float)

    return _python_fallback_total(spec, velocity, spin)


def compute_acceleration(
    spec: AerodynamicsSpec,
    velocity: np.ndarray,
    spin: np.ndarray,
) -> np.ndarray:
    """Return acceleration from aerodynamic forces.

    Convenience wrapper: F_total / mass.
    """
    if spec.mass <= 0:
        raise ValueError(f"mass must be positive, got {spec.mass}")
    return compute_total_force(spec, velocity, spin) / spec.mass


__all__ = [
    "AerodynamicsSpec",
    "compute_acceleration",
    "compute_total_force",
    "is_rust_available",
]
