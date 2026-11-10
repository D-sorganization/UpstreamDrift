from __future__ import annotations

import pytest

from src.engines.simscape import SimscapeAdapter, SimscapeStateError
from src.shared.python.core.contracts.exceptions import PreconditionError
from src.shared.python.engine_core.interfaces import PhysicsEngine

pytestmark = pytest.mark.unit


def test_simscape_adapter_satisfies_physics_engine_protocol() -> None:
    adapter = SimscapeAdapter()
    assert isinstance(adapter, PhysicsEngine)
    assert hasattr(adapter, "compute_gravity_only_acceleration")
    assert hasattr(adapter, "compute_control_acceleration_at_state")


def test_simscape_counterfactual_lifecycle_and_preconditions() -> None:
    import numpy as np

    adapter = SimscapeAdapter()
    q = np.array([0.1, 0.2])
    tau = np.array([1.0, 2.0])
    with pytest.raises(SimscapeStateError):
        adapter.compute_gravity_only_acceleration(q)
    with pytest.raises(SimscapeStateError):
        adapter.compute_control_acceleration_at_state(q, tau)
    with pytest.raises(PreconditionError):
        adapter.compute_gravity_only_acceleration(np.zeros((2, 2)))
