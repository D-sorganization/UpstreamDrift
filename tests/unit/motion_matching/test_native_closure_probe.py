"""Contracts for querying a native weld residual at one candidate state."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativeClosurePositionLinearization,
    NativeClosureTrajectoryResiduals,
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


@pytest.mark.unit
def test_closure_linearization_uses_refreshed_native_weld_jacobian() -> None:
    """The returned Jacobian follows named native coordinate order and is detached."""
    model = object.__new__(NativePinocchioModel)
    model._velocity_indices = {"hip": 2, "shoulder": 0}
    model.model = SimpleNamespace(nv=3)
    model.data = object()
    model.constraints = [object()]
    model.constraint_data = [object()]
    captured: dict[str, object] = {}

    class Pin:
        @staticmethod
        def getConstraintsJacobian(
            model_value: object,
            data_value: object,
            constraints_value: list[object],
            constraint_data_value: list[object],
        ) -> np.ndarray:
            captured.update(
                model=model_value,
                data=data_value,
                constraints=constraints_value,
                constraint_data=constraint_data_value,
            )
            return np.arange(18.0).reshape(6, 3)

    model._pin = Pin()
    model.closure_residuals = MethodType(  # type: ignore[method-assign]
        lambda self, coordinates, rates=None: (np.arange(6.0), np.zeros(6)), model
    )

    result = model.closure_position_linearization({"hip": 1.0, "shoulder": -2.0})

    assert result.names == ("hip", "shoulder")
    np.testing.assert_array_equal(result.position, np.arange(6.0))
    np.testing.assert_array_equal(
        result.jacobian,
        [[2.0, 0.0], [5.0, 3.0], [8.0, 6.0], [11.0, 9.0], [14.0, 12.0], [17.0, 15.0]],
    )
    assert not result.jacobian.flags.writeable
    assert captured["model"] is model.model


@pytest.mark.unit
def test_trajectory_closure_uses_zero_effort_constrained_acceleration_as_drift() -> (
    None
):
    """Acceleration closure is evaluated without an inverse-dynamics solve."""
    model = object.__new__(NativePinocchioModel)
    model._velocity_indices = {"hip": 1, "shoulder": 0}
    model.model = SimpleNamespace(nv=2)
    model.data = object()
    model.constraints = [object()]
    model.constraint_data = [object()]

    class Pin:
        @staticmethod
        def getConstraintsJacobian(*_: object) -> np.ndarray:
            return np.array([[2.0, 3.0], [5.0, 7.0]])

    model._pin = Pin()
    model.closure_errors = MethodType(  # type: ignore[method-assign]
        lambda self: (np.array([0.1, 0.2]), np.zeros(2)), model
    )
    captured: dict[str, dict[str, float]] = {}

    def accelerations(
        self: NativePinocchioModel,
        coordinates: dict[str, float],
        rates: dict[str, float],
        efforts: dict[str, float],
    ) -> dict[str, float]:
        captured.update(coordinates=coordinates, rates=rates, efforts=efforts)
        return {"hip": 11.0, "shoulder": 13.0}

    model.accelerations = MethodType(accelerations, model)  # type: ignore[method-assign]

    residual = model.closure_trajectory_residuals(
        {"hip": 1.0, "shoulder": -2.0},
        {"hip": 4.0, "shoulder": -5.0},
        {"hip": 17.0, "shoulder": 19.0},
    )

    np.testing.assert_array_equal(residual.position, [0.1, 0.2])
    np.testing.assert_array_equal(residual.rate, [2.0, 3.0])
    np.testing.assert_array_equal(residual.acceleration, [30.0, 72.0])
    assert captured["efforts"] == {"hip": 0.0, "shoulder": 0.0}


@pytest.mark.unit
def test_trajectory_linearization_uses_local_differences_and_exact_acceleration() -> (
    None
):
    """Only local q/v sensitivity is differenced; the acceleration block is J."""
    model = object.__new__(NativePinocchioModel)

    def residual(
        self: NativePinocchioModel,
        coordinates: dict[str, float],
        rates: dict[str, float],
        accelerations: dict[str, float],
    ) -> NativeClosureTrajectoryResiduals:
        q = np.asarray([coordinates["hip"], coordinates["shoulder"]])
        v = np.asarray([rates["hip"], rates["shoulder"]])
        a = np.asarray([accelerations["hip"], accelerations["shoulder"]])
        return NativeClosureTrajectoryResiduals(
            q, q + 2.0 * v, 3.0 * q + 5.0 * v + 7.0 * a
        )

    model.closure_trajectory_residuals = MethodType(residual, model)  # type: ignore[method-assign]
    model.closure_position_linearization = MethodType(  # type: ignore[method-assign]
        lambda self, coordinates: NativeClosurePositionLinearization(
            ("hip", "shoulder"), np.zeros(2), 7.0 * np.eye(2)
        ),
        model,
    )

    result = model.closure_trajectory_linearization(
        {"hip": 1.0, "shoulder": 2.0},
        {"hip": 3.0, "shoulder": 5.0},
        {"hip": 7.0, "shoulder": 11.0},
        finite_difference_step=1e-5,
    )

    np.testing.assert_allclose(
        result.dq, np.vstack((np.eye(2), np.eye(2), 3.0 * np.eye(2)))
    )
    np.testing.assert_allclose(
        result.dv, np.vstack((np.zeros((2, 2)), 2.0 * np.eye(2), 5.0 * np.eye(2)))
    )
    np.testing.assert_allclose(
        result.da, np.vstack((np.zeros((4, 2)), 7.0 * np.eye(2)))
    )
