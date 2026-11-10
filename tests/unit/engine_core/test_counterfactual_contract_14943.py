from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.engine_core.counterfactual_contract import (
    CounterfactualKind,
    CounterfactualResult,
    ensure_comparable_counterfactuals,
    wrap_counterfactual,
)
from src.shared.python.engine_core.counterfactual_defaults import (
    default_control_acceleration_at_state,
    default_gravity_only_acceleration,
)
from src.shared.python.engine_core.mock_engine import MockPhysicsEngine

pytestmark = pytest.mark.unit


def test_mixed_counterfactual_kind_comparison_raises() -> None:
    with pytest.raises(ValueError, match="Cannot compare counterfactuals"):
        ensure_comparable_counterfactuals(
            wrap_counterfactual(CounterfactualKind.CONTROL, np.ones(3)),
            wrap_counterfactual(CounterfactualKind.DRIFT, np.ones(3)),
        )


def test_mixed_definition_version_comparison_raises() -> None:
    with pytest.raises(ValueError, match="definition_version"):
        ensure_comparable_counterfactuals(
            wrap_counterfactual(CounterfactualKind.CONTROL, np.ones(3)),
            CounterfactualResult(CounterfactualKind.CONTROL, "legacy-v0", np.ones(3)),
        )


def test_mock_engine_drift_plus_control_matches_inverse_dynamics() -> None:
    engine = MockPhysicsEngine()
    tau = np.array([2.0, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0], dtype=float)
    full = np.linalg.solve(
        engine.compute_mass_matrix(), tau - engine.compute_bias_forces()
    )
    res = engine.compute_drift_acceleration() + engine.compute_control_acceleration(tau)
    np.testing.assert_allclose(res, full, rtol=1e-9, atol=1e-9)


def test_mock_compute_zvcf_shim_warns_and_matches_gravity_only() -> None:
    engine, q = MockPhysicsEngine(), np.zeros(7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = engine.compute_zvcf(q)
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    np.testing.assert_allclose(legacy, engine.compute_gravity_only_acceleration(q))


def test_default_counterfactuals_evaluate_at_q_and_restore_state() -> None:
    engine = MagicMock()
    engine.get_state.return_value = (np.array([1.0, 2.0]), np.array([0.5, -0.5]))
    engine.compute_mass_matrix.side_effect = lambda: np.diag(engine.q)
    engine.compute_gravity_forces.side_effect = lambda: -engine.q * 9.81
    engine.compute_control_acceleration.side_effect = lambda tau: tau / engine.q

    def set_state(q: np.ndarray, v: np.ndarray) -> None:
        engine.q, engine.v = np.asarray(q, float), np.asarray(v, float)

    engine.set_state.side_effect = set_state
    engine.q, engine.v = np.array([1.0, 2.0]), np.array([0.5, -0.5])

    target_q, tau = np.array([4.0, 5.0]), np.array([8.0, 10.0])
    g = default_gravity_only_acceleration(engine, target_q)
    np.testing.assert_allclose(g, [9.81, 9.81])

    c = default_control_acceleration_at_state(engine, target_q, tau)
    np.testing.assert_allclose(c, [2.0, 2.0])

    assert engine.set_state.call_count == 4
    np.testing.assert_allclose(engine.set_state.call_args[0][0], [1.0, 2.0])
    np.testing.assert_allclose(engine.set_state.call_args[0][1], [0.5, -0.5])
