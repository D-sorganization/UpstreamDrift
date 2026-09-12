"""World marker derivatives preserve offset rotation and native column order."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def engine() -> NativePinocchioModel:
    model = object.__new__(NativePinocchioModel)
    model.model = SimpleNamespace(nv=2)
    model.data = object()
    model._velocity_indices = {"shift": 1, "spin": 0}
    model._frames = {"club": 7}
    frame = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    model.frame_poses = lambda coordinates: {"club": frame}
    model.configuration = lambda coordinates: np.zeros(2)
    calls = []
    jacobian = np.zeros((6, 2))
    jacobian[5, 0] = 1
    jacobian[0, 1] = 1

    def get_frame(*args):
        assert calls == ["jacobians"]
        assert args[-2:] == (7, "aligned")
        return jacobian

    model._pin = SimpleNamespace(
        ReferenceFrame=SimpleNamespace(LOCAL_WORLD_ALIGNED="aligned"),
        computeJointJacobians=lambda *args: calls.append("jacobians"),
        getFrameJacobian=get_frame,
    )
    return model


def test_rotated_offset_and_column_order(engine: NativePinocchioModel) -> None:
    result = engine.marker_derivatives(
        {"shift": 0.0, "spin": 0.0}, ["club"], [[1, 0, 0]]
    )
    assert result.names == ("shift", "spin")
    np.testing.assert_allclose(result.positions_m, [[0, 1, 0]])
    np.testing.assert_allclose(result.dposition_dq, [[[1, -1], [0, 0], [0, 0]]])
    assert not result.positions_m.flags.writeable
    assert not result.dposition_dq.flags.writeable


def test_unknown_marker_frame_rejected(engine: NativePinocchioModel) -> None:
    with pytest.raises(ValueError, match="missing"):
        engine.marker_derivatives({"shift": 0.0, "spin": 0.0}, ["missing"], [[1, 0, 0]])


def test_nonfinite_native_jacobian_rejected(engine: NativePinocchioModel) -> None:
    engine._pin.getFrameJacobian = lambda *args: np.full((6, 2), np.nan)
    with pytest.raises(ValueError, match="Jacobian"):
        engine.marker_derivatives({"shift": 0.0, "spin": 0.0}, ["club"], [[1, 0, 0]])
