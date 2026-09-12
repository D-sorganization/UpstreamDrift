"""Public closure diagnostics preserve native data and reject invalid evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)

pytestmark = pytest.mark.unit


def test_closure_errors_are_detached_from_native_storage() -> None:
    model = object.__new__(NativePinocchioModel)
    pose, velocity = np.arange(6.0), np.arange(6.0) + 1
    model.constraint_data = [
        SimpleNamespace(
            contact_placement_error=SimpleNamespace(vector=pose),
            contact_velocity_error=SimpleNamespace(vector=velocity),
        )
    ]
    p, v = model.closure_errors()
    np.testing.assert_array_equal(p, pose)
    np.testing.assert_array_equal(v, velocity)
    p[:] = 0
    assert pose[-1] == 5


def test_missing_closure_is_not_zero_error() -> None:
    model = object.__new__(NativePinocchioModel)
    model.constraint_data = []
    with pytest.raises(ValueError, match="closure"):
        model.closure_errors()
