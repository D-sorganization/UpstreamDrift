"""Tests for src/api/models/responses.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models import responses as m

pytestmark = pytest.mark.unit


def test_engine_status_minimal() -> None:
    e = m.EngineStatusResponse(
        name="mujoco",
        available=True,
        engine_type="mujoco",
        status="ok",
        is_available=True,
    )
    assert e.loaded is False
    assert e.capabilities == []


def test_simulation_response_success_must_have_data() -> None:
    with pytest.raises(ValidationError):
        m.SimulationResponse(success=True, duration=1.0, frames=10, data={})


def test_simulation_response_success_with_data_ok() -> None:
    r = m.SimulationResponse(
        success=True, duration=1.0, frames=10, data={"states": [0]}
    )
    assert r.success is True


def test_simulation_response_failure_empty_data_ok() -> None:
    r = m.SimulationResponse(success=False, duration=0.0, frames=0, data={})
    assert r.success is False


def test_simulation_response_negative_duration_rejected() -> None:
    with pytest.raises(ValidationError):
        m.SimulationResponse(
            success=True, duration=-1.0, frames=10, data={"states": [0]}
        )


def test_analysis_response_success_requires_results() -> None:
    with pytest.raises(ValidationError):
        m.AnalysisResponse(analysis_type="kinematics", success=True, results={})


def test_analysis_response_failure_empty_ok() -> None:
    r = m.AnalysisResponse(analysis_type="kinematics", success=False, results={})
    assert r.success is False


def test_task_status_response_minimal() -> None:
    t = m.TaskStatusResponse(task_id="t1", status="pending")
    assert t.progress is None


def test_joint_info_response_required_fields() -> None:
    j = m.JointInfoResponse(
        index=0,
        name="shoulder",
        torque_limit=10.0,
        position_limit_lower=-1.0,
        position_limit_upper=1.0,
        velocity_limit=5.0,
        current_torque=0.0,
    )
    assert j.name == "shoulder"


def test_actuator_state_response_minimal() -> None:
    a = m.ActuatorStateResponse(
        strategy="pd",
        n_joints=0,
        joint_names=[],
        torques=[],
        kp=[],
        kd=[],
        ki=[],
        joints=[],
        available_strategies=[],
    )
    assert a.n_joints == 0


def test_force_vector_response_required() -> None:
    f = m.ForceVectorResponse(sim_time=0.0, applied_torques=[1.0, 2.0])
    assert f.applied_torques == [1.0, 2.0]


def test_biomechanics_metrics_response_minimal() -> None:
    b = m.BiomechanicsMetricsResponse(
        sim_time=0.0,
        joint_positions=[0.0],
        joint_velocities=[0.0],
    )
    assert b.club_head_speed is None


def test_capability_level_and_engine_capabilities() -> None:
    lvl = m.CapabilityLevelResponse(name="dyn", level="full", supported=True)
    e = m.EngineCapabilitiesResponse(
        engine_name="mujoco",
        engine_type="mujoco",
        capabilities=[lvl],
        summary={"full": 1, "partial": 0, "none": 0},
    )
    assert e.capabilities[0].level == "full"


def test_urdf_response_defaults() -> None:
    link = m.URDFLinkGeometry(link_name="root", geometry_type="box")
    assert link.origin == [0.0, 0.0, 0.0]
    assert link.color == [0.5, 0.5, 0.5, 1.0]


def test_urdf_joint_descriptor_defaults() -> None:
    j = m.URDFJointDescriptor(
        name="j1", joint_type="revolute", parent_link="a", child_link="b"
    )
    assert j.axis == [0.0, 0.0, 1.0]


def test_urdf_model_and_model_list_response() -> None:
    model = m.URDFModelResponse(model_name="foo", links=[], joints=[], root_link="root")
    assert model.urdf_raw is None
    lr = m.ModelListResponse(models=[{"name": "a", "format": "urdf"}])
    assert lr.models[0]["name"] == "a"


def test_analysis_metrics_summary_default_std() -> None:
    s = m.AnalysisMetricsSummary(
        metric_name="x", current=1.0, minimum=0.0, maximum=2.0, mean=1.0
    )
    assert s.std_dev == 0.0


def test_data_export_response_required_fields() -> None:
    r = m.DataExportResponse(
        format="csv",
        filename="x.csv",
        size_bytes=10,
        download_url="/dl",
        record_count=1,
    )
    assert r.format == "csv"


def test_measurement_result_required() -> None:
    res = m.MeasurementResult(
        body_a="a",
        body_b="b",
        distance=1.0,
        position_a=[0.0, 0.0, 0.0],
        position_b=[1.0, 0.0, 0.0],
        delta=[1.0, 0.0, 0.0],
    )
    assert res.distance == 1.0


def test_force_vector3d_defaults() -> None:
    fv = m.ForceVector3D(
        body_name="a",
        force_type="applied",
        origin=[0, 0, 0],
        direction=[1, 0, 0],
        magnitude=1.0,
    )
    assert fv.color == [1.0, 0.0, 0.0, 1.0]


def test_force_overlay_response_defaults() -> None:
    r = m.ForceOverlayResponse(sim_time=0.0)
    assert r.vectors == []
    assert r.total_force_magnitude == 0.0


def test_actuator_info_defaults() -> None:
    info = m.ActuatorInfo(index=0, name="j")
    assert info.units == "N*m"
    assert info.min_value < info.max_value


def test_aip_handshake_response_defaults() -> None:
    cap = m.AIPCapability(name="x", methods=["m"])
    h = m.AIPHandshakeResponse(capabilities=[cap], supported_methods=["m"])
    assert h.server_name.startswith("UpstreamDrift")


def test_aip_jsonrpc_response_defaults() -> None:
    r = m.AIPJsonRpcResponse()
    assert r.jsonrpc == "2.0"
    assert r.result is None
    assert r.error is None
