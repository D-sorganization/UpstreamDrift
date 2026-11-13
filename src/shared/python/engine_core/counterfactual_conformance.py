"""Shared conformance helpers for counterfactual reporting (#14943)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.shared.python.engine_core.counterfactual_contract import (
    CounterfactualKind,
    CounterfactualResult,
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
    full = np.linalg.solve(
        engine.compute_mass_matrix(),
        np.asarray(tau, float).reshape(-1) - engine.compute_bias_forces(),
    )
    np.testing.assert_allclose(
        engine.compute_drift_acceleration() + engine.compute_control_acceleration(tau),
        full,
        rtol=rtol,
        atol=atol,
    )


def stamped_drift(engine: _SuperpositionEngine) -> CounterfactualResult:
    return wrap_counterfactual(
        CounterfactualKind.DRIFT, engine.compute_drift_acceleration()
    )


def stamped_control(
    engine: _SuperpositionEngine, tau: np.ndarray
) -> CounterfactualResult:
    return wrap_counterfactual(
        CounterfactualKind.CONTROL, engine.compute_control_acceleration(tau)
    )
