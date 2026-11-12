"""Issue #14943: counterfactual contract and mock-engine conformance (slice 1)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.shared.python.engine_core.counterfactual_contract import (
    CounterfactualKind,
    CounterfactualResult,
    ensure_comparable_counterfactuals,
    wrap_counterfactual,
)
from src.shared.python.engine_core.mock_engine import MockPhysicsEngine

pytestmark = pytest.mark.unit


def test_mixed_counterfactual_kind_comparison_raises() -> None:
    a = wrap_counterfactual(CounterfactualKind.CONTROL, np.ones(3))
    b = wrap_counterfactual(CounterfactualKind.DRIFT, np.ones(3))
    with pytest.raises(ValueError, match="Cannot compare counterfactuals"):
        ensure_comparable_counterfactuals(a, b)


def test_mixed_definition_version_comparison_raises() -> None:
    a = wrap_counterfactual(CounterfactualKind.CONTROL, np.ones(3))
    b = CounterfactualResult(
        kind=CounterfactualKind.CONTROL,
        definition_version="legacy-v0",
        qdd=np.ones(3),
    )
    with pytest.raises(ValueError, match="definition_version"):
        ensure_comparable_counterfactuals(a, b)


def test_mock_engine_drift_plus_control_matches_inverse_dynamics() -> None:
    engine = MockPhysicsEngine()
    tau = np.array([2.0, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0], dtype=float)
    m = engine.compute_mass_matrix()
    bias = engine.compute_bias_forces()
    drift = engine.compute_drift_acceleration()
    control = engine.compute_control_acceleration(tau)
    full = np.linalg.solve(m, tau - bias)
    np.testing.assert_allclose(drift + control, full, rtol=1e-9, atol=1e-9)


def test_mock_compute_zvcf_shim_warns_and_matches_gravity_only() -> None:
    engine = MockPhysicsEngine()
    q = np.zeros(engine.num_joints)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = engine.compute_zvcf(q)
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    gravity_only = engine.compute_gravity_only_acceleration(q)
    np.testing.assert_allclose(legacy, gravity_only)
