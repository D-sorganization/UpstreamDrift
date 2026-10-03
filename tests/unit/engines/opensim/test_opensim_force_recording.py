import pytest

osim = pytest.importorskip("opensim", reason="OpenSim not installed")
if not hasattr(osim, "Model"):
    pytest.skip("real opensim is unavailable", allow_module_level=True)

from unittest.mock import MagicMock, patch

import numpy as np

from src.engines.physics_engines.opensim.python.opensim_force_recording import (
    record_force_and_segment_series,
    record_force_series,
)
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    ForceTorqueSeries,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.playback import SegmentSeries

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_record_force_series_mocked() -> None:
    mock_model = MagicMock()
    mock_states = [MagicMock(), MagicMock(), MagicMock()]

    mock_source = MagicMock()
    mock_joint = MagicMock()
    mock_joint.getChildFrame().getPositionInGround.return_value = (0.0, 1.0, 0.0)
    mock_joint.getChildFrame().findBaseFrame().getName.return_value = "link"
    mock_joint.getParentFrame().findBaseFrame().getName.return_value = "ground"
    mock_source._joints.return_value = [mock_joint]
    mock_source._world.side_effect = lambda v: np.asarray(v, dtype=np.float64)
    mock_source._segment_distal_point.return_value = np.array([0.0, 0.0, 0.0])

    def _sample(s: Any) -> ForceTorqueFrame:
        idx = mock_states.index(s)
        w = OverlayWrench(
            kind=WrenchKind.JOINT_REACTION,
            label="reaction:pin",
            body="link",
            point_m=(0.0, 0.0, 1.0),
            force_n=(0.0, 0.0, 10.0),
            torque_nm=None,
            source="mock",
        )
        return ForceTorqueFrame(
            time_s=float(idx) * 0.1,
            engine="opensim",
            wrenches=(w,),
        )

    mock_source.sample.side_effect = _sample

    with patch(
        "src.engines.physics_engines.opensim.python.opensim_force_recording.OpenSimForceTorqueSource",
        return_value=mock_source,
    ):
        series, segments = record_force_and_segment_series(mock_model, mock_states)

    assert isinstance(series, ForceTorqueSeries)
    assert len(series) == 3
    assert series.times_s == (0.0, 0.1, 0.2)
    assert isinstance(segments, SegmentSeries)
    assert segments.names == ("link",)
    assert segments.proximal.shape == (3, 1, 3)
    assert segments.distal.shape == (3, 1, 3)


try:
    import opensim as osim

    has_opensim = hasattr(osim, "Model")
except ImportError:
    has_opensim = False


def _build_test_pendulum() -> tuple[osim.Model, list[osim.State]]:
    """Build a hanging pendulum and a list of 10 states with advancing times."""
    model = osim.Model()
    model.setName("synthetic_pendulum")
    body = osim.Body(
        "link",
        2.0,
        osim.Vec3(0, -0.5, 0),
        osim.Inertia(0.01, 0.01, 0.01),
    )
    model.addBody(body)
    joint = osim.PinJoint(
        "pin",
        model.getGround(),
        osim.Vec3(0, 1.0, 0),
        osim.Vec3(0, 0, 0),
        body,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
    )
    model.addJoint(joint)
    initial_state = model.initSystem()

    states = []
    # Create 10 distinct states with monotone increasing times
    for i in range(10):
        s = osim.State(initial_state)
        s.setTime(i * 0.05)
        # Give a small displacement
        joint.getCoordinate().setValue(s, 0.1 * i)
        model.realizeAcceleration(s)
        states.append(s)

    return model, states


from typing import Any
import pytest


@pytest.mark.skipif(not has_opensim, reason="OpenSim not installed")
def test_record_force_series_length_and_monotone_times() -> None:
    model, states = _build_test_pendulum()
    series = record_force_series(model, states)

    assert isinstance(series, ForceTorqueSeries)
    assert len(series) == 10
    times = series.times_s
    assert len(times) == 10
    assert all(times[i] < times[i + 1] for i in range(len(times) - 1))


@pytest.mark.skipif(not has_opensim, reason="OpenSim not installed")
def test_record_force_and_segment_series() -> None:
    model, states = _build_test_pendulum()
    series, segments = record_force_and_segment_series(model, states)

    assert isinstance(series, ForceTorqueSeries)
    assert isinstance(segments, SegmentSeries)
    assert len(series) == 10
    assert segments.proximal.shape[0] == 10
    assert segments.distal.shape[0] == 10
    assert "link" in segments.names
