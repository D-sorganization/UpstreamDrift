from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
    PendulumPhysicsEngine,
)
from src.shared.python.engine_core.counterfactual_conformance import (
    assert_drift_control_superposition,
    stamped_control,
    stamped_drift,
)
from src.shared.python.engine_core.counterfactual_contract import CounterfactualKind

pytestmark = [pytest.mark.unit, pytest.mark.scientific]


@pytest.fixture
def pendulum() -> PendulumPhysicsEngine:
    engine = PendulumPhysicsEngine()
    engine.set_state(np.array([0.35, -0.2]), np.array([0.5, -0.3]))
    return engine


def test_pendulum_drift_control_superposition(pendulum: PendulumPhysicsEngine) -> None:
    assert_drift_control_superposition(pendulum, np.array([1.2, -0.4]))


def test_pendulum_stamped_kinds_and_zvcf_shim(pendulum: PendulumPhysicsEngine) -> None:
    drift = stamped_drift(pendulum)
    control = stamped_control(pendulum, np.array([0.2, 0.1]))
    assert (
        drift.kind == CounterfactualKind.DRIFT
        and control.kind == CounterfactualKind.CONTROL
    )
    q = np.array([0.2, -0.15])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = pendulum.compute_zvcf(q)
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    np.testing.assert_allclose(legacy, pendulum.compute_gravity_only_acceleration(q))
    zvcf_control = pendulum.compute_control_acceleration_at_state(
        q, np.array([0.3, 0.0])
    )
    assert zvcf_control.shape == (2,)


def test_zvcf_shim_evaluates_supplied_q_not_current_state(
    pendulum: PendulumPhysicsEngine,
) -> None:
    pendulum.set_state(np.array([0.9, 0.1]), np.array([0.4, -0.2]))
    q_arg = np.array([0.15, -0.35])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shim = pendulum.compute_zvcf(q_arg)
    expected = pendulum.compute_gravity_only_acceleration(q_arg)
    at_current = pendulum.compute_gravity_only_acceleration(np.array([0.9, 0.1]))
    np.testing.assert_allclose(shim, expected)
    assert not np.allclose(shim, at_current)
