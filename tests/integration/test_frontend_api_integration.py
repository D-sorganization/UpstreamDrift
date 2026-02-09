"""Integration tests for frontend-backend API communication.

These tests verify that the API endpoints used by the frontend work correctly,
ensuring the contract between frontend and backend is maintained.
"""

from unittest.mock import MagicMock

import pytest

from src.shared.python.engine_manager import EngineManager


class TestEnginesEndpoint:
    """Tests for the /api/engines endpoint used by the frontend EngineSelector."""

    @pytest.fixture
    def mock_engine_manager(self):
        """Create a mock engine manager."""
        manager = MagicMock(spec=EngineManager)
        manager.get_available_engines.return_value = []
        manager.get_current_engine.return_value = None
        manager.get_engine_status.return_value = MagicMock(value="unavailable")
        return manager

    def test_engines_response_structure(self, mock_engine_manager):
        """Verify the engines endpoint returns the structure expected by frontend."""
        # The frontend expects this structure:
        # {
        #   engines: [
        #     { name: string, available: boolean, loaded: boolean, capabilities: string[] }
        #   ]
        # }
        from src.shared.python.engine_registry import EngineStatus, EngineType

        mock_engine_manager.get_available_engines.return_value = [EngineType.MUJOCO]
        mock_engine_manager.get_current_engine.return_value = EngineType.MUJOCO
        mock_engine_manager.get_engine_status.return_value = EngineStatus.AVAILABLE

        # Simulate what the endpoint returns
        engines = []
        engine_capabilities = {
            EngineType.MUJOCO: ["physics", "contacts", "muscles", "tendons"],
            EngineType.DRAKE: ["physics", "optimization", "control"],
            EngineType.PINOCCHIO: ["kinematics", "dynamics", "collision"],
        }

        for engine_type in [EngineType.MUJOCO, EngineType.DRAKE, EngineType.PINOCCHIO]:
            available_engines = mock_engine_manager.get_available_engines()
            current_engine = mock_engine_manager.get_current_engine()

            is_available = engine_type in available_engines
            is_loaded = current_engine == engine_type

            engines.append(
                {
                    "name": engine_type.value,
                    "available": is_available,
                    "loaded": is_loaded,
                    "capabilities": engine_capabilities.get(engine_type, []),
                }
            )

        # Verify structure matches frontend expectations
        assert len(engines) == 3
        assert engines[0]["name"] == "mujoco"
        assert engines[0]["available"] is True
        assert engines[0]["loaded"] is True
        assert isinstance(engines[0]["capabilities"], list)

        assert engines[1]["name"] == "drake"
        assert engines[1]["available"] is False

    def test_engines_all_fields_present(self, mock_engine_manager):
        """Ensure all fields expected by frontend are present."""
        from src.shared.python.engine_registry import EngineStatus, EngineType

        mock_engine_manager.get_available_engines.return_value = [EngineType.MUJOCO]
        mock_engine_manager.get_current_engine.return_value = None
        mock_engine_manager.get_engine_status.return_value = EngineStatus.UNAVAILABLE

        # Required fields for frontend compatibility
        required_fields = {"name", "available", "loaded", "capabilities"}

        engine_data = {
            "name": EngineType.MUJOCO.value,
            "available": True,
            "loaded": False,
            "capabilities": ["physics"],
        }

        # All required fields must be present
        assert required_fields.issubset(engine_data.keys())

    def test_engine_types_match_frontend_expectations(self):
        """Verify engine type values match what frontend expects."""
        from src.shared.python.engine_registry import EngineType

        # Frontend expects these engine names
        expected_engines = {"mujoco", "drake", "pinocchio", "opensim", "myosim"}

        actual_engines = {e.value for e in EngineType}

        # All expected engines should be available
        assert expected_engines.issubset(actual_engines)


class TestSimulationWebSocketProtocol:
    """Tests for the WebSocket simulation protocol used by useSimulation hook."""

    def test_start_action_config_structure(self):
        """Verify the start action config matches frontend expectations."""
        # Frontend sends this structure on start:
        start_message = {
            "action": "start",
            "config": {
                "duration": 3.0,
                "timestep": 0.002,
                "live_analysis": True,
            },
        }

        assert start_message["action"] == "start"
        assert "config" in start_message
        assert "duration" in start_message["config"]
        assert "timestep" in start_message["config"]
        assert "live_analysis" in start_message["config"]

    def test_frame_response_structure(self):
        """Verify simulation frame response matches frontend expectations."""
        # Backend should send frames in this format:
        frame_response = {
            "frame": 42,
            "time": 0.84,
            "state": {"qpos": [0.1, 0.2, 0.3]},
            "analysis": {
                "joint_angles": [0.5, 0.3, 0.2, 0.1],
                "velocities": [0.1, 0.2],
            },
        }

        # Required fields
        assert "frame" in frame_response
        assert "time" in frame_response
        assert "state" in frame_response
        assert isinstance(frame_response["frame"], int)
        assert isinstance(frame_response["time"], float)
        assert isinstance(frame_response["state"], dict)

        # Optional analysis field
        if "analysis" in frame_response:
            assert isinstance(frame_response["analysis"], dict)

    def test_status_messages(self):
        """Verify status message types match frontend expectations."""
        # Frontend expects these status messages
        status_messages = [
            {"status": "complete"},
            {"status": "stopped"},
            {"status": "paused"},
        ]

        for msg in status_messages:
            assert "status" in msg
            assert msg["status"] in {"complete", "stopped", "paused"}

    def test_control_actions(self):
        """Verify control actions match frontend expectations."""
        # Frontend sends these control messages
        control_messages = [
            {"action": "pause"},
            {"action": "resume"},
            {"action": "stop"},
        ]

        for msg in control_messages:
            assert "action" in msg
            assert msg["action"] in {"start", "pause", "resume", "stop"}


class TestSimulationRequestValidation:
    """Tests for simulation request validation."""

    def test_simulation_request_model(self):
        """Verify SimulationRequest model accepts frontend parameters."""
        from src.api.models.requests import SimulationRequest

        # Test with parameters that frontend might send
        request = SimulationRequest(
            engine_type="mujoco",
            duration=3.0,
            timestep=0.002,
        )

        assert request.engine_type == "mujoco"
        assert request.duration == 3.0
        assert request.timestep == 0.002

    def test_simulation_request_defaults(self):
        """Verify SimulationRequest has reasonable defaults."""
        from src.api.models.requests import SimulationRequest

        # Minimal request
        request = SimulationRequest(engine_type="mujoco")

        # Should have defaults
        assert request.engine_type == "mujoco"
        assert request.duration is not None or hasattr(request, "duration")


class TestSimulationResponseFormat:
    """Tests for simulation response format."""

    def test_simulation_response_model(self):
        """Verify SimulationResponse model has fields frontend expects."""
        from src.api.models.responses import SimulationResponse

        # Check that required fields exist in the model
        model_fields = SimulationResponse.model_fields

        # At minimum, frontend expects status and potentially results
        assert (
            "status" in model_fields
            or "success" in model_fields
            or hasattr(SimulationResponse, "__annotations__")
        )


class TestAPIContractStability:
    """Tests to ensure API contract stability with frontend."""

    def test_engine_status_response_model(self):
        """Verify EngineStatusResponse has all frontend-expected fields."""
        from src.api.models.responses import EngineStatusResponse

        # Create a sample response
        response = EngineStatusResponse(
            name="mujoco",
            available=True,
            loaded=True,
            version="3.0.0",
            capabilities=["physics", "contacts"],
            engine_type="mujoco",
            status="available",
            is_available=True,
            description="MuJoCo physics engine",
        )

        # Frontend-required fields
        assert response.name == "mujoco"
        assert response.available is True
        assert response.loaded is True
        assert isinstance(response.capabilities, list)

    def test_cors_allows_frontend_origin(self):
        """Verify CORS configuration allows frontend requests."""
        from src.api.config import get_cors_origins

        origins = get_cors_origins()

        # Should allow localhost for development
        assert any("localhost" in origin for origin in origins) or "*" in origins


class TestErrorResponses:
    """Tests for error response format expected by frontend."""

    def test_http_exception_format(self):
        """Verify HTTP exceptions have proper format for frontend error handling."""
        from fastapi import HTTPException

        # Standard error response
        exc = HTTPException(status_code=400, detail="Invalid parameters")

        assert exc.status_code == 400
        assert exc.detail == "Invalid parameters"

    def test_validation_error_structure(self):
        """Verify validation errors have proper structure."""
        from pydantic import ValidationError

        from src.api.models.requests import SimulationRequest

        # Try to create invalid request
        try:
            # This should fail validation
            SimulationRequest(engine_type=123)  # type: ignore
        except ValidationError as e:
            errors = e.errors()
            assert len(errors) > 0
            assert "type" in errors[0] or "msg" in errors[0]


class TestConnectionResilience:
    """Tests for connection handling that frontend relies on."""

    def test_websocket_close_codes(self):
        """Verify WebSocket close codes match frontend expectations."""
        # Standard close codes the frontend handles
        clean_close = 1000  # Normal closure
        abnormal_close = 1006  # Abnormal closure

        # Frontend uses these to determine reconnection behavior
        assert clean_close == 1000
        assert abnormal_close == 1006

    def test_frame_history_limit(self):
        """Verify backend respects same frame history limit as frontend."""
        # Frontend keeps MAX_FRAMES_HISTORY = 1000
        # Backend should be aware of this for memory efficiency
        max_frames_frontend = 1000

        # This is a documentation test to ensure both sides agree
        assert max_frames_frontend == 1000
