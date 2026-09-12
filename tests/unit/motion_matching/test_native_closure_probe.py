"""Contracts for querying a native weld residual at one candidate state."""

from __future__ import annotations

from types import MethodType

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)


@pytest.mark.unit
def test_closure_probe_uses_zero_effort_and_default_zero_rates() -> None:
    """A static pose query refreshes constraint data without hidden actuation."""
    model = object.__new__(NativePinocchioModel)
    model._velocity_indices = {"hip": 0, "shoulder": 1}
    captured: dict[str, dict[str, float]] = {}

    def accelerations(
        self: NativePinocchioModel,
        coordinates: dict[str, float],
        rates: dict[str, float],
        efforts: dict[str, float],
    ) -> dict[str, float]:
        captured.update(coordinates=coordinates, rates=rates, efforts=efforts)
        return {"hip": 0.0, "shoulder": 0.0}

    model.accelerations = MethodType(accelerations, model)  # type: ignore[method-assign]
    model.closure_errors = MethodType(  # type: ignore[method-assign]
        lambda self: (np.array([0.1, 0.2]), np.array([0.3, 0.4])), model
    )

    pose, rate = model.closure_residuals({"hip": 1.0, "shoulder": -2.0})

    assert captured == {
        "coordinates": {"hip": 1.0, "shoulder": -2.0},
        "rates": {"hip": 0.0, "shoulder": 0.0},
        "efforts": {"hip": 0.0, "shoulder": 0.0},
    }
    np.testing.assert_array_equal(pose, [0.1, 0.2])
    np.testing.assert_array_equal(rate, [0.3, 0.4])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("coordinates", "rates"),
    [
        ({"hip": 1.0}, None),
        ({"hip": 1.0, "shoulder": np.nan}, None),
        ({"hip": 1.0, "shoulder": 2.0}, {"hip": 0.0}),
        ({"hip": 1.0, "shoulder": 2.0}, {"hip": 0.0, "shoulder": np.inf}),
    ],
)
def test_closure_probe_rejects_incomplete_or_nonfinite_state(
    coordinates: dict[str, float], rates: dict[str, float] | None
) -> None:
    """The probe cannot silently substitute a coordinate or rate."""
    model = object.__new__(NativePinocchioModel)
    model._velocity_indices = {"hip": 0, "shoulder": 1}

    with pytest.raises(ValueError, match="exactly native coordinate and rate"):
        model.closure_residuals(coordinates, rates)
