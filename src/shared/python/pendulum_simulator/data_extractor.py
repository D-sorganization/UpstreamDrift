"""
Reusable data extractor for simulation results.

Provides a registry of named data series that can be extracted from any
simulation result object.  Used by the pop-out chart data selector and
the data table module.

Design by Contract
------------------
- extract_series() returns (values, description, unit_label) or raises KeyError.
- list_available_series() returns all valid series names for a result type.
- All returned arrays are shape (N,) and finite.

DRY
---
Dispatch table pattern replaces sequential if-elif chains (O(1) lookup).
Vectorized extractors avoid per-frame method calls where possible.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extractor type alias
# ---------------------------------------------------------------------------

Extractor = Callable[[Any], np.ndarray]


def _make_state_extractor(col: int) -> Extractor:
    """Factory: extract a single column from the states array."""

    def _extract(result: Any) -> np.ndarray:
        return np.asarray(result.states[:, col], dtype=float)

    return _extract


def _make_torque_extractor(joint: int) -> Extractor:
    """Factory: extract driving torque for a specific joint."""

    def _extract(result: Any) -> np.ndarray:
        n = result.n_steps
        return np.array([result.torques_at(i)[joint] for i in range(n)], dtype=float)

    return _extract


def _make_total_torque_extractor(joint: int) -> Extractor:
    """Factory: extract total torque (drive + friction) for a joint."""

    def _extract(result: Any) -> np.ndarray:
        return np.asarray(result.all_total_torques()[:, joint], dtype=float)

    return _extract


def _make_energy_extractor(component: str) -> Extractor:
    """Factory: extract energy component ('kinetic', 'potential', 'total')."""

    def _extract(result: Any) -> np.ndarray:
        return np.asarray(result.all_energies()[component], dtype=float)

    return _extract


def _make_velocity_extractor(key: str) -> Extractor:
    """Factory: extract a named velocity from joint_velocities_at."""

    def _extract(result: Any) -> np.ndarray:
        n = result.n_steps
        return np.array(
            [result.joint_velocities_at(i)[key] for i in range(n)], dtype=float
        )

    return _extract


def _make_accel_extractor(joint: int) -> Extractor:
    """Factory: extract acceleration for a specific joint."""

    def _extract(result: Any) -> np.ndarray:
        return np.asarray(result.all_accelerations()[:, joint], dtype=float)

    return _extract


def _make_coriolis_extractor(joint: int) -> Extractor:
    """Factory: extract Coriolis term for a specific joint."""

    def _extract(result: Any) -> np.ndarray:
        n = result.n_steps
        return np.array([result.coriolis_at(i)[joint] for i in range(n)], dtype=float)

    return _extract


def _make_gravity_extractor(joint: int) -> Extractor:
    """Factory: extract gravity term for a specific joint."""

    def _extract(result: Any) -> np.ndarray:
        n = result.n_steps
        return np.array([result.gravity_at(i)[joint] for i in range(n)], dtype=float)

    return _extract


def _make_friction_extractor(joint: int) -> Extractor:
    """Factory: extract friction torque for a specific joint."""

    def _extract(result: Any) -> np.ndarray:
        return np.asarray(result.all_friction_torques()[:, joint], dtype=float)

    return _extract


def _make_base_force_extractor(component: str) -> Extractor:
    """Factory: extract base force component ('fx', 'fy', 'magnitude')."""

    def _extract(result: Any) -> np.ndarray:
        n = result.n_steps
        return np.array(
            [result.base_force_at(i)[component] for i in range(n)], dtype=float
        )

    return _extract


# ---------------------------------------------------------------------------
# Series registry: (description, unit, extractor)
# ---------------------------------------------------------------------------

_DOUBLE_SERIES: dict[str, tuple[str, str, Extractor]] = {
    "time": ("Time", "s", lambda r: np.asarray(r.t, dtype=float)),
    "theta1": ("Shoulder angle θ₁", "rad", _make_state_extractor(0)),
    "phi": ("Wrist angle φ", "rad", _make_state_extractor(1)),
    "dtheta1": ("Shoulder velocity θ̇₁", "rad/s", _make_state_extractor(2)),
    "dphi": ("Wrist velocity φ̇", "rad/s", _make_state_extractor(3)),
    "torque_shoulder": ("Shoulder torque", "N·m", _make_torque_extractor(0)),
    "torque_wrist": ("Wrist torque", "N·m", _make_torque_extractor(1)),
    "total_torque_shoulder": (
        "Total shoulder torque",
        "N·m",
        _make_total_torque_extractor(0),
    ),
    "total_torque_wrist": (
        "Total wrist torque",
        "N·m",
        _make_total_torque_extractor(1),
    ),
    "kinetic_energy": ("Kinetic energy", "J", _make_energy_extractor("kinetic")),
    "potential_energy": ("Potential energy", "J", _make_energy_extractor("potential")),
    "total_energy": ("Total energy", "J", _make_energy_extractor("total")),
    "wrist_speed": ("Wrist speed", "m/s", _make_velocity_extractor("wrist_speed")),
    "tip_speed": ("Tip speed", "m/s", _make_velocity_extractor("tip_speed")),
    "accel_shoulder": ("Shoulder acceleration", "rad/s²", _make_accel_extractor(0)),
    "accel_wrist": ("Wrist acceleration", "rad/s²", _make_accel_extractor(1)),
    "coriolis_shoulder": (
        "Coriolis term (shoulder)",
        "N·m",
        _make_coriolis_extractor(0),
    ),
    "coriolis_wrist": ("Coriolis term (wrist)", "N·m", _make_coriolis_extractor(1)),
    "gravity_shoulder": ("Gravity term (shoulder)", "N·m", _make_gravity_extractor(0)),
    "gravity_wrist": ("Gravity term (wrist)", "N·m", _make_gravity_extractor(1)),
    "friction_shoulder": (
        "Friction torque (shoulder)",
        "N·m",
        _make_friction_extractor(0),
    ),
    "friction_wrist": ("Friction torque (wrist)", "N·m", _make_friction_extractor(1)),
    "base_force_x": ("Base force Fx", "N", _make_base_force_extractor("fx")),
    "base_force_y": ("Base force Fy", "N", _make_base_force_extractor("fy")),
    "base_force_mag": (
        "Base force magnitude",
        "N",
        _make_base_force_extractor("magnitude"),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_available_series(model_type: str = "double") -> list[tuple[str, str, str]]:
    """List all available data series for a model type.

    Returns
    -------
    list of (key, description, unit_label)
    """
    registry = _DOUBLE_SERIES  # extend for triple/golfer later
    return [(k, desc, unit) for k, (desc, unit, _) in registry.items()]


def extract_series(
    result: Any,
    key: str,
    model_type: str = "double",
) -> tuple[np.ndarray, str, str]:
    """Extract a named data series from a simulation result.

    Parameters
    ----------
    result : SimulationResult (or triple/golfer variant)
    key : str
        Series name (e.g., "torque_shoulder", "tip_speed").
    model_type : str
        "double", "triple", or "golfer".

    Returns
    -------
    (values, description, unit_label)

    Raises
    ------
    KeyError if key is not recognized.

    Design by Contract
    ------------------
    Pre: key in _DOUBLE_SERIES
    Post: returned array is 1-D
    """
    if key not in _DOUBLE_SERIES:
        raise KeyError(f"Unknown series key: {key!r}")
    desc, unit, extractor = _DOUBLE_SERIES[key]
    values = extractor(result)
    assert values.ndim == 1, f"Expected 1-D array, got shape {values.shape}"
    return values, desc, unit
