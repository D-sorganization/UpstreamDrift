"""Tests for src/api/models/requests.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models import requests as r

pytestmark = pytest.mark.unit


# ----- SimulationRequest -----


def test_simulation_request_minimum_valid() -> None:
    sim = r.SimulationRequest(engine_type="MuJoCo")
    assert sim.engine_type == "mujoco"
    assert sim.duration == 1.0
    assert sim.timestep is None


def test_simulation_request_unknown_engine() -> None:
    with pytest.raises(ValidationError):
        r.SimulationRequest(engine_type="nope")


def test_simulation_request_duration_bounds() -> None:
    with pytest.raises(ValidationError):
        r.SimulationRequest(engine_type="mujoco", duration=0)
    with pytest.raises(ValidationError):
        r.SimulationRequest(
            engine_type="mujoco", duration=r.MAX_SIMULATION_DURATION + 1
        )


def test_simulation_request_timestep_too_small() -> None:
    with pytest.raises(ValidationError):
        r.SimulationRequest(engine_type="mujoco", timestep=r.MIN_TIMESTEP / 10)


def test_simulation_request_timestep_too_large() -> None:
    with pytest.raises(ValidationError):
        r.SimulationRequest(engine_type="mujoco", timestep=r.MAX_TIMESTEP * 2)


def test_simulation_request_initial_state_qv_normalized() -> None:
    sim = r.SimulationRequest(
        engine_type="mujoco",
        initial_state={"q": [0, "1.5"], "v": (2, 3.5)},
    )
    assert sim.initial_state == {"q": [0.0, 1.5], "v": [2.0, 3.5]}


def test_simulation_request_initial_state_qv_rejects_bad_values() -> None:
    with pytest.raises(ValidationError):
        r.SimulationRequest(
            engine_type="mujoco",
            initial_state={"q": "not-a-list"},
        )
    with pytest.raises(ValidationError):
        r.SimulationRequest(
            engine_type="mujoco",
            initial_state={"v": [0.0, float("nan")]},
        )


def test_simulation_request_control_inputs_length_capped() -> None:
    """control_inputs over MAX_CONTROL_INPUTS is rejected (issue #6948)."""
    over = [{"t": i} for i in range(r.MAX_CONTROL_INPUTS + 1)]
    with pytest.raises(ValidationError):
        r.SimulationRequest(engine_type="mujoco", control_inputs=over)


def test_simulation_request_control_inputs_at_limit_ok() -> None:
    """A small control sequence within bounds is accepted."""
    sim = r.SimulationRequest(
        engine_type="mujoco", control_inputs=[{"t": 0.0}, {"t": 0.1}]
    )
    assert sim.control_inputs is not None
    assert len(sim.control_inputs) == 2


def test_simulation_request_initial_state_length_capped() -> None:
    """initial_state q/v vectors over MAX_STATE_VECTOR_LEN are rejected (#6948)."""
    over = [0.0] * (r.MAX_STATE_VECTOR_LEN + 1)
    with pytest.raises(ValidationError):
        r.SimulationRequest(engine_type="mujoco", initial_state={"q": over})


def test_normalize_initial_state_component_length_check() -> None:
    """The helper raises ValueError on an over-long component (#6948)."""
    over = [0.0] * (r.MAX_STATE_VECTOR_LEN + 1)
    with pytest.raises(ValueError, match="exceeds maximum length"):
        r._normalize_initial_state_component("q", over)


# ----- AnalysisRequest -----


def test_analysis_request_normalizes() -> None:
    a = r.AnalysisRequest(
        analysis_type=" Kinematics ", data_source="sim", export_format="JSON"
    )
    assert a.analysis_type == "kinematics"
    assert a.export_format == "json"


def test_analysis_request_unknown_type() -> None:
    with pytest.raises(ValidationError):
        r.AnalysisRequest(analysis_type="nope", data_source="sim")


def test_analysis_request_unknown_format() -> None:
    with pytest.raises(ValidationError):
        r.AnalysisRequest(
            analysis_type="kinematics", data_source="sim", export_format="xml"
        )


# ----- VideoAnalysisRequest -----


def test_video_request_defaults() -> None:
    v = r.VideoAnalysisRequest()
    assert v.estimator_type == "mediapipe"
    assert 0 <= v.min_confidence <= 1


def test_video_request_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        r.VideoAnalysisRequest(min_confidence=-0.1)
    with pytest.raises(ValidationError):
        r.VideoAnalysisRequest(min_confidence=1.1)


# ----- ActuatorUpdateRequest -----


def test_actuator_update_strategy_normalized() -> None:
    a = r.ActuatorUpdateRequest(strategy=" PD ")
    assert a.strategy == "pd"


def test_actuator_update_unknown_strategy() -> None:
    with pytest.raises(ValidationError):
        r.ActuatorUpdateRequest(strategy="quantum")


def test_actuator_update_strategy_none_ok() -> None:
    a = r.ActuatorUpdateRequest(strategy=None)
    assert a.strategy is None


def test_actuator_update_kp_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        r.ActuatorUpdateRequest(kp=0)


def test_actuator_update_ki_zero_ok() -> None:
    a = r.ActuatorUpdateRequest(ki=0)
    assert a.ki == 0


# ----- SpeedControlRequest -----


def test_speed_control_bounds() -> None:
    assert r.SpeedControlRequest().speed_factor == 1.0
    with pytest.raises(ValidationError):
        r.SpeedControlRequest(speed_factor=0.0)
    with pytest.raises(ValidationError):
        r.SpeedControlRequest(speed_factor=11.0)


# ----- CameraPresetRequest -----


def test_camera_preset_valid() -> None:
    assert r.CameraPresetRequest(preset="SIDE").preset == "side"


def test_camera_preset_invalid() -> None:
    with pytest.raises(ValidationError):
        r.CameraPresetRequest(preset="behind")


# ----- TrajectoryRecordRequest -----


def test_trajectory_record_actions() -> None:
    for action in ["start", "STOP", " export "]:
        req = r.TrajectoryRecordRequest(action=action)
        assert req.action in {"start", "stop", "export"}


def test_trajectory_record_invalid_action() -> None:
    with pytest.raises(ValidationError):
        r.TrajectoryRecordRequest(action="pause")


# ----- DataExportRequest -----


def test_data_export_format_normalized() -> None:
    assert r.DataExportRequest(format="JSON").format == "json"


def test_data_export_invalid() -> None:
    with pytest.raises(ValidationError):
        r.DataExportRequest(format="xml")


# ----- BodyPositionUpdateRequest -----


def test_body_position_valid() -> None:
    req = r.BodyPositionUpdateRequest(
        body_name="club", position=[0.0, 1.0, 2.0], rotation=[0.0, 0.0, 0.0]
    )
    assert req.position == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("bad", [[0.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
def test_body_position_wrong_size(bad: list[float]) -> None:
    with pytest.raises(ValidationError):
        r.BodyPositionUpdateRequest(body_name="club", position=bad)


@pytest.mark.parametrize("bad", [[0.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
def test_body_rotation_wrong_size(bad: list[float]) -> None:
    with pytest.raises(ValidationError):
        r.BodyPositionUpdateRequest(body_name="club", rotation=bad)


# ----- MeasurementRequest -----


def test_measurement_request_valid() -> None:
    m = r.MeasurementRequest(body_a="A", body_b="B")
    assert m.body_a == "A"


# ----- ForceOverlayRequest -----


def test_force_overlay_defaults() -> None:
    fo = r.ForceOverlayRequest()
    assert fo.force_types == ["applied"]


def test_force_overlay_normalizes_force_types() -> None:
    fo = r.ForceOverlayRequest(force_types=["APPLIED", " gravity "])
    assert fo.force_types == ["applied", "gravity"]


def test_force_overlay_unknown_type() -> None:
    with pytest.raises(ValidationError):
        r.ForceOverlayRequest(force_types=["magic"])


def test_force_overlay_scale_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        r.ForceOverlayRequest(scale_factor=0)


# ----- ActuatorCommandRequest -----


def test_actuator_command_valid() -> None:
    c = r.ActuatorCommandRequest(actuator_index=2, value=1.0, control_type="PD_GAINS")
    assert c.control_type == "pd_gains"


def test_actuator_command_negative_index() -> None:
    with pytest.raises(ValidationError):
        r.ActuatorCommandRequest(actuator_index=-1, value=0.0)


def test_actuator_command_unknown_control() -> None:
    with pytest.raises(ValidationError):
        r.ActuatorCommandRequest(actuator_index=0, value=0.0, control_type="nope")


# ----- ActuatorBatchCommandRequest -----


def test_actuator_batch_requires_one_command() -> None:
    with pytest.raises(ValidationError):
        r.ActuatorBatchCommandRequest(commands=[])


def test_actuator_batch_valid() -> None:
    batch = r.ActuatorBatchCommandRequest(
        commands=[r.ActuatorCommandRequest(actuator_index=0, value=1.0)]
    )
    assert len(batch.commands) == 1


# ----- ModelExplorerRequest / ModelCompareRequest -----


def test_model_explorer_request_minimal() -> None:
    m = r.ModelExplorerRequest(model_path="x.urdf")
    assert m.model_path == "x.urdf"
    assert m.joint_values is None


def test_model_compare_request_minimal() -> None:
    mc = r.ModelCompareRequest(model_a_path="a", model_b_path="b")
    assert mc.model_a_path == "a"


# ----- AIPJsonRpcRequest -----


def test_aip_request_default_version() -> None:
    req = r.AIPJsonRpcRequest(method="ping")
    assert req.jsonrpc == "2.0"
    assert req.id is None


def test_aip_request_bad_version() -> None:
    with pytest.raises(ValidationError):
        r.AIPJsonRpcRequest(jsonrpc="1.0", method="ping")


def test_aip_request_with_params_and_id() -> None:
    req = r.AIPJsonRpcRequest(method="x", params={"a": 1}, id=7)
    assert req.id == 7


# ----- ModelFittingRequest -----


def test_model_fitting_minimal() -> None:
    f = r.ModelFittingRequest(model_path="m", data_path="d")
    assert f.fitting_method == "least_squares"
