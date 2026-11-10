"""Pendulum engine conformance for counterfactual contract (issue #14943 slice 2)."""

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
    tau = np.array([1.2, -0.4])
    assert_drift_control_superposition(pendulum, tau)


def test_pendulum_stamped_kinds_and_zvcf_shim(pendulum: PendulumPhysicsEngine) -> None:
    drift = stamped_drift(pendulum)
    control = stamped_control(pendulum, np.array([0.2, 0.1]))
    assert drift.kind == CounterfactualKind.DRIFT
    assert control.kind == CounterfactualKind.CONTROL
    q = np.array([0.2, -0.15])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = pendulum.compute_zvcf(q)
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    gravity_only = pendulum.compute_gravity_only_acceleration(q)
    np.testing.assert_allclose(legacy, gravity_only)
    zvcf_control = pendulum.compute_control_acceleration_at_state(q, np.array([0.3, 0.0]))
    assert zvcf_control.shape == (2,)
