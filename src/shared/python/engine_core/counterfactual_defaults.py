"""Default counterfactual accelerations for engines without native hooks (#14943)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:

    class _MassGravityEngine(Protocol):
        def compute_mass_matrix(self) -> np.ndarray: ...
        def compute_gravity_forces(self) -> np.ndarray: ...
        def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray: ...


def default_gravity_only_acceleration(
    engine: _MassGravityEngine, q: np.ndarray
) -> np.ndarray:
    """Legacy gravity-only diagnostic at fixed q with v = u = 0 (not SPEC ZVCF)."""
    _ = np.asarray(q, dtype=float)  # state at q is engine-specific
    mass = engine.compute_mass_matrix()
    gravity = engine.compute_gravity_forces()
    return np.linalg.solve(mass, -gravity)


def default_control_acceleration_at_state(
    engine: _MassGravityEngine, q: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """SPEC ZVCF control component M(q)⁻¹ τ at fixed state (per §4)."""
    _ = np.asarray(q, dtype=float)
    tau_v = np.asarray(tau, dtype=float).reshape(-1)
    return engine.compute_control_acceleration(tau_v)


def deprecated_compute_zvcf_shim(
    engine: _MassGravityEngine, q: np.ndarray
) -> np.ndarray:
    """One-release shim: old compute_zvcf → gravity-only diagnostic."""
    msg = "compute_zvcf(q) is legacy; use compute_control_acceleration_at_state or compute_gravity_only_acceleration"
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    return default_gravity_only_acceleration(engine, q)
