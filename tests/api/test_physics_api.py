"""Tests for Phase 2 physics API routes.

Validates the shared physics backend REST API (#1209),
engine capabilities API (#1204), and simulation controls (#1202).

Tests cover:
- Pydantic contract validation for all new request/response models
- Actuator control endpoints
- Force/torque query endpoints
- Biomechanics metrics endpoints
- Control features registry endpoint
- Engine capabilities endpoint
- Simulation controls (speed, camera, recording, stats)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models.requests import (
    VALID_CAMERA_PRESETS,
    VALID_CONTROL_STRATEGIES,
    ActuatorUpdateRequest,
    CameraPresetRequest,
    SpeedControlRequest,
    TrajectoryRecordRequest,
)
from src.api.models.responses import (
    ActuatorStateResponse,
    BiomechanicsMetricsResponse,
    CameraPresetResponse,
    CapabilityLevelResponse,
    ControlFeaturesResponse,
    EngineCapabilitiesResponse,
    ForceVectorResponse,
    SimulationStatsResponse,
    SpeedControlResponse,
    TrajectoryRecordResponse,
)

try:
    from fastapi.testclient import TestClient

    from src.api.server import app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ──────────────────────────────────────────────────────────────
#  Contract Tests: ActuatorUpdateRequest
# ──────────────────────────────────────────────────────────────
class TestActuatorUpdateRequestContract:
    """Validate ActuatorUpdateRequest preconditions."""

    def test_empty_request_valid(self) -> None:
        """An empty update (no changes) is valid."""
        req = ActuatorUpdateRequest()
        assert req.strategy is None
        assert req.torques is None

    def test_valid_strategy(self) -> None:
        """All valid strategies are accepted."""
        for strategy in VALID_CONTROL_STRATEGIES:
            req = ActuatorUpdateRequest(strategy=strategy)
            assert req.strategy == strategy

    def test_invalid_strategy_rejected(self) -> None:
        """Unknown strategy raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown strategy"):
            ActuatorUpdateRequest(strategy="nonexistent_strategy")

    def test_strategy_normalized(self) -> None:
        """Strategy is lowercased and stripped."""
        req = ActuatorUpdateRequest(strategy="  PD  ")
        assert req.strategy == "pd"

    def test_torques_accepted(self) -> None:
        """Torque list is accepted."""
        req = ActuatorUpdateRequest(torques=[1.0, -2.0, 3.5])
        assert req.torques == [1.0, -2.0, 3.5]

    def test_gains_must_be_positive(self) -> None:
        """Negative kp/kd gains are rejected."""
        with pytest.raises(ValidationError):
            ActuatorUpdateRequest(kp=-1.0)
        with pytest.raises(ValidationError):
            ActuatorUpdateRequest(kd=-1.0)

    def test_ki_allows_zero(self) -> None:
        """ki gain allows zero (ge=0)."""
        req = ActuatorUpdateRequest(ki=0.0)
        assert req.ki == 0.0


# ──────────────────────────────────────────────────────────────
#  Contract Tests: SpeedControlRequest
# ──────────────────────────────────────────────────────────────
class TestSpeedControlRequestContract:
    """Validate SpeedControlRequest preconditions."""

    def test_default_speed(self) -> None:
        """Default speed factor is 1.0."""
        req = SpeedControlRequest()
        assert req.speed_factor == 1.0

    def test_valid_range(self) -> None:
        """Speed within [0.1, 10.0] is valid."""
        req = SpeedControlRequest(speed_factor=5.0)
        assert req.speed_factor == 5.0

    def test_below_minimum_rejected(self) -> None:
        """Speed below 0.1 is rejected."""
        with pytest.raises(ValidationError):
            SpeedControlRequest(speed_factor=0.05)

    def test_above_maximum_rejected(self) -> None:
        """Speed above 10.0 is rejected."""
        with pytest.raises(ValidationError):
            SpeedControlRequest(speed_factor=15.0)

    def test_boundary_values(self) -> None:
        """Boundary values 0.1 and 10.0 are accepted."""
        req_min = SpeedControlRequest(speed_factor=0.1)
        req_max = SpeedControlRequest(speed_factor=10.0)
        assert req_min.speed_factor == 0.1
        assert req_max.speed_factor == 10.0


# ──────────────────────────────────────────────────────────────
#  Contract Tests: CameraPresetRequest
# ──────────────────────────────────────────────────────────────
class TestCameraPresetRequestContract:
    """Validate CameraPresetRequest preconditions."""

    def test_valid_presets(self) -> None:
        """All valid camera presets are accepted."""
        for preset in VALID_CAMERA_PRESETS:
            req = CameraPresetRequest(preset=preset)
            assert req.preset == preset

    def test_invalid_preset_rejected(self) -> None:
        """Unknown preset raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown preset"):
            CameraPresetRequest(preset="orbital")

    def test_preset_normalized(self) -> None:
        """Preset is lowercased and stripped."""
        req = CameraPresetRequest(preset="  FRONT  ")
        assert req.preset == "front"


# ──────────────────────────────────────────────────────────────
#  Contract Tests: TrajectoryRecordRequest
# ──────────────────────────────────────────────────────────────
class TestTrajectoryRecordRequestContract:
    """Validate TrajectoryRecordRequest preconditions."""

    def test_valid_actions(self) -> None:
        """start, stop, export are valid actions."""
        for action in ("start", "stop", "export"):
            req = TrajectoryRecordRequest(action=action)
            assert req.action == action

    def test_invalid_action_rejected(self) -> None:
        """Unknown action raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown action"):
            TrajectoryRecordRequest(action="reset")

    def test_default_export_format(self) -> None:
        """Default export format is json."""
        req = TrajectoryRecordRequest(action="export")
        assert req.export_format == "json"


# ──────────────────────────────────────────────────────────────
#  Response Model Tests
# ──────────────────────────────────────────────────────────────
class TestResponseModels:
    """Validate response model construction."""

    def test_actuator_state_response(self) -> None:
        """ActuatorStateResponse can be constructed with valid data."""
        resp = ActuatorStateResponse(
            strategy="pd",
            n_joints=3,
            joint_names=["j0", "j1", "j2"],
            torques=[0.0, 0.0, 0.0],
            kp=[100.0, 100.0, 100.0],
            kd=[10.0, 10.0, 10.0],
            ki=[0.0, 0.0, 0.0],
            joints=[],
            available_strategies=[{"name": "pd", "description": "PD control"}],
        )
        assert resp.n_joints == 3

    def test_force_vector_response(self) -> None:
        """ForceVectorResponse can be constructed."""
        resp = ForceVectorResponse(
            sim_time=1.5,
            applied_torques=[0.0, 0.0],
        )
        assert resp.sim_time == 1.5
        assert resp.gravity_forces is None

    def test_biomechanics_metrics_response(self) -> None:
        """BiomechanicsMetricsResponse can be constructed."""
        resp = BiomechanicsMetricsResponse(
            sim_time=2.0,
            joint_positions=[0.1, 0.2],
            joint_velocities=[0.0, 0.0],
        )
        assert resp.sim_time == 2.0
        assert resp.club_head_speed is None

    def test_engine_capabilities_response(self) -> None:
        """EngineCapabilitiesResponse can be constructed."""
        resp = EngineCapabilitiesResponse(
            engine_name="MuJoCo",
            engine_type="mujoco",
            capabilities=[
                CapabilityLevelResponse(
                    name="mass_matrix", level="full", supported=True
                ),
            ],
            summary={"full": 1, "partial": 0, "none": 0},
        )
        assert resp.engine_name == "MuJoCo"
        assert len(resp.capabilities) == 1

    def test_control_features_response(self) -> None:
        """ControlFeaturesResponse can be constructed."""
        resp = ControlFeaturesResponse(
            engine="PendulumPhysicsEngine",
            total_features=15,
            available_features=10,
            categories=[],
            features=[],
        )
        assert resp.total_features == 15

    def test_simulation_stats_response(self) -> None:
        """SimulationStatsResponse can be constructed."""
        resp = SimulationStatsResponse(
            sim_time=5.0,
            wall_time=5.2,
            fps=500.0,
            real_time_factor=0.96,
            speed_factor=1.0,
            is_recording=False,
            frame_count=2500,
        )
        assert resp.frame_count == 2500

    def test_speed_control_response(self) -> None:
        """SpeedControlResponse can be constructed."""
        resp = SpeedControlResponse(speed_factor=2.0, status="Speed set to 2.0x")
        assert resp.speed_factor == 2.0

    def test_camera_preset_response(self) -> None:
        """CameraPresetResponse can be constructed."""
        resp = CameraPresetResponse(
            preset="side",
            position=[3.0, 0.0, 1.5],
            target=[0.0, 0.0, 1.0],
            up=[0.0, 0.0, 1.0],
        )
        assert resp.preset == "side"

    def test_trajectory_record_response(self) -> None:
        """TrajectoryRecordResponse can be constructed."""
        resp = TrajectoryRecordResponse(
            recording=True,
            frame_count=100,
            status="Recording started",
        )
        assert resp.recording is True


# ──────────────────────────────────────────────────────────────
#  API Integration Tests (require FastAPI)
# ──────────────────────────────────────────────────────────────
@pytest.fixture()
def client():
    """Create test client."""
    if not HAS_FASTAPI:
        pytest.skip("FastAPI not available")
    with TestClient(app) as c:
        yield c


class TestCameraPresetAPI:
    """Test camera preset endpoint."""

    def test_valid_preset_returns_200(self, client) -> None:
        """POST /simulation/camera with valid preset returns 200."""
        resp = client.post("/simulation/camera", json={"preset": "side"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["preset"] == "side"
        assert len(data["position"]) == 3
        assert len(data["target"]) == 3
        assert len(data["up"]) == 3

    def test_all_presets_return_200(self, client) -> None:
        """All camera presets return valid responses."""
        for preset in VALID_CAMERA_PRESETS:
            resp = client.post("/simulation/camera", json={"preset": preset})
            assert resp.status_code == 200, f"Preset {preset} failed"

    def test_invalid_preset_returns_422(self, client) -> None:
        """Invalid preset returns 422 validation error."""
        resp = client.post("/simulation/camera", json={"preset": "orbital"})
        assert resp.status_code == 422


class TestSpeedControlAPI:
    """Test simulation speed control endpoint."""

    def test_set_speed_returns_200(self, client) -> None:
        """POST /simulation/speed with valid factor returns 200."""
        resp = client.post("/simulation/speed", json={"speed_factor": 2.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["speed_factor"] == 2.0
        assert "status" in data

    def test_default_speed(self, client) -> None:
        """Default speed factor 1.0 is accepted."""
        resp = client.post("/simulation/speed", json={"speed_factor": 1.0})
        assert resp.status_code == 200

    def test_invalid_speed_rejected(self, client) -> None:
        """Speed out of range returns 422."""
        resp = client.post("/simulation/speed", json={"speed_factor": 100.0})
        assert resp.status_code == 422


class TestRecordingAPI:
    """Test trajectory recording endpoint."""

    def test_start_recording(self, client) -> None:
        """POST /simulation/recording with start action works."""
        resp = client.post(
            "/simulation/recording",
            json={"action": "start", "export_format": "json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["recording"] is True

    def test_stop_recording(self, client) -> None:
        """POST /simulation/recording with stop action works."""
        # Start then stop
        client.post(
            "/simulation/recording",
            json={"action": "start", "export_format": "json"},
        )
        resp = client.post(
            "/simulation/recording",
            json={"action": "stop", "export_format": "json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["recording"] is False

    def test_export_empty(self, client) -> None:
        """Export with no recorded frames returns message."""
        resp = client.post(
            "/simulation/recording",
            json={"action": "export", "export_format": "json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "No frames" in data["status"] or data["frame_count"] == 0


class TestSimulationStatsAPI:
    """Test simulation stats endpoint."""

    def test_stats_returns_200(self, client) -> None:
        """GET /simulation/stats returns 200."""
        resp = client.get("/simulation/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "sim_time" in data
        assert "fps" in data
        assert "real_time_factor" in data
        assert "speed_factor" in data
        assert "is_recording" in data
        assert "frame_count" in data


class TestEngineCapabilitiesAPI:
    """Test engine capabilities endpoint. See issue #1204."""

    def test_pendulum_capabilities(self, client) -> None:
        """GET /engines/pendulum/capabilities returns 200."""
        resp = client.get("/engines/pendulum/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "engine_type" in data
        assert "capabilities" in data
        assert "summary" in data
        assert isinstance(data["capabilities"], list)

    def test_unknown_engine_returns_400(self, client) -> None:
        """Unknown engine type returns 400."""
        resp = client.get("/engines/nonexistent/capabilities")
        assert resp.status_code == 400

    def test_capabilities_have_required_fields(self, client) -> None:
        """Each capability entry has name, level, supported fields."""
        resp = client.get("/engines/pendulum/capabilities")
        if resp.status_code == 200:
            data = resp.json()
            for cap in data["capabilities"]:
                assert "name" in cap
                assert "level" in cap
                assert "supported" in cap
                assert cap["level"] in ("full", "partial", "none")


class TestActuatorEndpoints:
    """Test actuator control endpoints (require loaded engine)."""

    def test_get_actuators_no_engine_returns_400(self, client) -> None:
        """GET /simulation/actuators returns 400 when no engine loaded."""
        resp = client.get("/simulation/actuators")
        # Should be 400 since no engine is loaded
        assert resp.status_code == 400

    def test_post_actuators_no_engine_returns_400(self, client) -> None:
        """POST /simulation/actuators returns 400 when no engine loaded."""
        resp = client.post("/simulation/actuators", json={"strategy": "pd"})
        assert resp.status_code == 400


class TestForceEndpoints:
    """Test force/torque query endpoints."""

    def test_get_forces_no_engine_returns_400(self, client) -> None:
        """GET /simulation/forces returns 400 when no engine loaded."""
        resp = client.get("/simulation/forces")
        assert resp.status_code == 400


class TestMetricsEndpoints:
    """Test biomechanics metrics endpoints."""

    def test_get_metrics_no_engine_returns_400(self, client) -> None:
        """GET /simulation/metrics returns 400 when no engine loaded."""
        resp = client.get("/simulation/metrics")
        assert resp.status_code == 400


class TestControlFeaturesEndpoints:
    """Test control features registry endpoint."""

    def test_get_features_no_engine_returns_400(self, client) -> None:
        """GET /simulation/control-features returns 400 when no engine loaded."""
        resp = client.get("/simulation/control-features")
        assert resp.status_code == 400
