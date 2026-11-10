"""Shared conformance helpers for counterfactual reporting (#14943)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.shared.python.engine_core.counterfactual_contract import (
    CounterfactualKind,
    wrap_counterfactual,
)


class _SuperpositionEngine(Protocol):
    def compute_mass_matrix(self) -> np.ndarray: ...
    def compute_bias_forces(self) -> np.ndarray: ...
    def compute_drift_acceleration(self) -> np.ndarray: ...
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray: ...


def assert_drift_control_superposition(
    engine: _SuperpositionEngine,
    tau: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-9,
) -> None:
    """SECTION F / SPEC §4: drift + control equals full generalized acceleration."""
    mass = engine.compute_mass_matrix()
    bias = engine.compute_bias_forces()
    drift = engine.compute_drift_acceleration()
    control = engine.compute_control_acceleration(tau)
    full = np.linalg.solve(mass, np.asarray(tau, dtype=float).reshape(-1) - bias)
    np.testing.assert_allclose(drift + control, full, rtol=rtol, atol=atol)


def stamped_drift(engine: _SuperpositionEngine) -> object:
    return wrap_counterfactual(CounterfactualKind.DRIFT, engine.compute_drift_acceleration())


def stamped_control(
    engine: _SuperpositionEngine, tau: np.ndarray
) -> object:
    return wrap_counterfactual(
        CounterfactualKind.CONTROL, engine.compute_control_acceleration(tau)
    )
