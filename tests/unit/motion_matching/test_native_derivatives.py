"""Native derivative boundary refreshes dynamics and preserves coordinate identity."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)

pytestmark = pytest.mark.unit


def test_derivatives_refresh_map_and_detach_native_storage() -> None:
    model = object.__new__(NativePinocchioModel)
    model._velocity_indices = {"x": 1, "y": 0}
    model.model = SimpleNamespace(nv=2)
    model.data, model.constraints, model.constraint_data = object(), [], []
    raw = np.arange(4.0).reshape(2, 2)
    calls = []
    model.accelerations = lambda *args: calls.append("forward")

    def native(*args):
        assert calls == ["forward"]
        return (raw, raw + 10, raw + 20, None, None, None)

    model._pin = SimpleNamespace(computeConstraintDynamicsDerivatives=native)
    result = model.acceleration_derivatives({"x": 0, "y": 0}, {}, {})
    assert result.names == ("x", "y")
    np.testing.assert_array_equal(result.dq, [[3, 2], [1, 0]])
    assert not result.dq.flags.writeable
    raw[:] = -1
    assert result.dq[0, 0] == 3


def test_nonfinite_derivatives_fail() -> None:
    model = object.__new__(NativePinocchioModel)
    model._velocity_indices = {"x": 0}
    model.model = SimpleNamespace(nv=1)
    model.data, model.constraints, model.constraint_data = object(), [], []
    model.accelerations = lambda *args: None
    model._pin = SimpleNamespace(
        computeConstraintDynamicsDerivatives=lambda *a: (np.array([[np.nan]]),) * 6
    )
    with pytest.raises(ValueError):
        model.acceleration_derivatives({"x": 0}, {}, {})
