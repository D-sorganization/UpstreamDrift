from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python.shared_engine import (
    SharedPuttingEngine,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_putting_green_zvcf_emits_deprecation() -> None:
    engine = SharedPuttingEngine()
    q = np.array([0.1, -0.2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = engine.compute_zvcf(q)
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)
    gravity_only = engine.compute_gravity_only_acceleration(q)
    np.testing.assert_allclose(legacy, gravity_only)


def test_putting_green_control_acceleration_at_state() -> None:
    engine = SharedPuttingEngine()
    q = np.array([0.1, -0.2])
    tau = np.array([1.0, 0.5])
    accel = engine.compute_control_acceleration_at_state(q, tau)
    np.testing.assert_allclose(accel, tau / engine.ball_mass)
    with pytest.raises(ValueError):
        engine.compute_control_acceleration_at_state(None, tau)  # type: ignore[arg-type]
