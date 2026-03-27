"""Contract parity tests for API request/response models.

Validates that Pydantic precondition validators on request models and
postcondition validators on response models enforce the documented contracts.
Also verifies consistency between the API model constants and the engine
registry.

Fixes #1131 (Phase 2)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models.requests import (
    VALID_ANALYSIS_TYPES,
    VALID_ENGINE_TYPES,
    VALID_EXPORT_FORMATS,
    AnalysisRequest,
    SimulationRequest,
)
from src.api.models.responses import (
    AnalysisResponse,
    SimulationResponse,
)
from src.shared.python.engine_core.engine_registry import EngineType


# ──────────────────────────────────────────────────────────────
#  SimulationRequest Precondition Tests
# ──────────────────────────────────────────────────────────────
class TestSimulationRequestPreconditions:
    """SimulationRequest validators enforce input contracts."""

    def test_valid_request(self) -> None:
        """Standard valid request passes all validators."""
        req = SimulationRequest(engine_type="mujoco")
        assert req.engine_type == "mujoco"
        assert req.duration == 1.0

    def test_engine_type_normalized(self) -> None:
        """Engine type is lowercased and stripped."""
        req = SimulationRequest(engine_type="  MuJoCo  ")
        assert req.engine_type == "mujoco"

    def test_invalid_engine_type_rejected(self) -> None:
        """Unknown engine type raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown engine_type"):
            SimulationRequest(engine_type="nonexistent_engine")

    def test_all_engine_types_accepted(self) -> None:
        """Every VALID_ENGINE_TYPES entry is accepted."""
        for engine in VALID_ENGINE_TYPES:
            req = SimulationRequest(engine_type=engine)
            assert req.engine_type == engine

    def test_zero_duration_rejected(self) -> None:
        """Duration of 0 is rejected (gt=0 constraint)."""
        with pytest.raises(ValidationError):
            SimulationRequest(engine_type="mujoco", duration=0)

    def test_negative_duration_rejected(self) -> None:
        """Negative duration is rejected."""
        with pytest.raises(ValidationError):
            SimulationRequest(engine_type="mujoco", duration=-1.0)

    def test_excessive_duration_rejected(self) -> None:
        """Duration exceeding MAX_SIMULATION_DURATION is rejected."""
        with pytest.raises(ValidationError):
            SimulationRequest(engine_type="mujoco", duration=500.0)

    def test_valid_timestep(self) -> None:
        """Valid timestep is accepted."""
        req = SimulationRequest(engine_type="mujoco", timestep=0.01)
        assert req.timestep == 0.01

    def test_excessive_timestep_rejected(self) -> None:
        """Timestep exceeding MAX_TIMESTEP is rejected."""
        with pytest.raises(ValidationError):
            SimulationRequest(engine_type="mujoco", timestep=1.0)

    def test_negative_timestep_rejected(self) -> None:
        """Negative timestep is rejected."""
        with pytest.raises(ValidationError):
            SimulationRequest(engine_type="mujoco", timestep=-0.01)

    def test_none_timestep_allowed(self) -> None:
        """None timestep (default) is allowed."""
        req = SimulationRequest(engine_type="mujoco")
        assert req.timestep is None


# ──────────────────────────────────────────────────────────────
#  AnalysisRequest Precondition Tests
# ──────────────────────────────────────────────────────────────
class TestAnalysisRequestPreconditions:
    """AnalysisRequest validators enforce input contracts."""

    def test_valid_request(self) -> None:
        """Standard valid request passes."""
        req = AnalysisRequest(analysis_type="kinematics", data_source="simulation")
        assert req.analysis_type == "kinematics"
        assert req.export_format == "json"

    def test_analysis_type_normalized(self) -> None:
        """Analysis type is lowercased and stripped."""
        req = AnalysisRequest(analysis_type="  Kinematics  ", data_source="sim")
        assert req.analysis_type == "kinematics"

    def test_invalid_analysis_type_rejected(self) -> None:
        """Unknown analysis type raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown analysis_type"):
            AnalysisRequest(analysis_type="invalid_type", data_source="sim")

    def test_all_analysis_types_accepted(self) -> None:
        """Every VALID_ANALYSIS_TYPES entry is accepted."""
        for atype in VALID_ANALYSIS_TYPES:
            req = AnalysisRequest(analysis_type=atype, data_source="sim")
            assert req.analysis_type == atype

    def test_valid_export_format(self) -> None:
        """All valid export formats are accepted."""
        for fmt in VALID_EXPORT_FORMATS:
            req = AnalysisRequest(
                analysis_type="kinematics", data_source="sim", export_format=fmt
            )
            assert req.export_format == fmt

    def test_invalid_export_format_rejected(self) -> None:
        """Unknown export format raises ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported export_format"):
            AnalysisRequest(
                analysis_type="kinematics",
                data_source="sim",
                export_format="xlsx",
            )


# ──────────────────────────────────────────────────────────────
#  SimulationResponse Postcondition Tests
# ──────────────────────────────────────────────────────────────
class TestSimulationResponsePostconditions:
    """SimulationResponse validators ensure data integrity."""

    def test_valid_success_response(self) -> None:
        """Successful response with data passes."""
        resp = SimulationResponse(
            success=True,
            duration=1.0,
            frames=100,
            data={"states": [[0, 0, 0]]},
        )
        assert resp.success is True

    def test_success_with_empty_data_rejected(self) -> None:
        """Successful response with empty data is rejected."""
        with pytest.raises(ValidationError, match="non-empty data"):
            SimulationResponse(
                success=True,
                duration=1.0,
                frames=100,
                data={},
            )

    def test_failure_with_empty_data_allowed(self) -> None:
        """Failed response with empty data is allowed."""
        resp = SimulationResponse(
            success=False,
            duration=0.0,
            frames=0,
            data={},
        )
        assert resp.success is False

    def test_negative_duration_rejected(self) -> None:
        """Negative duration is rejected."""
        with pytest.raises(ValidationError):
            SimulationResponse(
                success=False,
                duration=-1.0,
                frames=0,
                data={},
            )

    def test_negative_frames_rejected(self) -> None:
        """Negative frame count is rejected."""
        with pytest.raises(ValidationError):
            SimulationResponse(
                success=False,
                duration=0.0,
                frames=-1,
                data={},
            )


# ──────────────────────────────────────────────────────────────
#  AnalysisResponse Postcondition Tests
# ──────────────────────────────────────────────────────────────
class TestAnalysisResponsePostconditions:
    """AnalysisResponse validators ensure data integrity."""

    def test_valid_success_response(self) -> None:
        """Successful analysis with results passes."""
        resp = AnalysisResponse(
            analysis_type="kinematics",
            success=True,
            results={"joint_angles": [0.1, 0.2, 0.3]},
        )
        assert resp.success is True

    def test_success_with_empty_results_rejected(self) -> None:
        """Successful analysis with empty results is rejected."""
        with pytest.raises(ValidationError, match="non-empty results"):
            AnalysisResponse(
                analysis_type="kinematics",
                success=True,
                results={},
            )

    def test_failure_with_empty_results_allowed(self) -> None:
        """Failed analysis with empty results is allowed."""
        resp = AnalysisResponse(
            analysis_type="kinematics",
            success=False,
            results={},
        )
        assert resp.success is False


# ──────────────────────────────────────────────────────────────
#  Engine Registry ↔ API Model Consistency
# ──────────────────────────────────────────────────────────────
class TestRegistryModelConsistency:
    """Verify API model constants stay in sync with engine registry."""

    def test_all_engine_types_covered(self) -> None:
        """Every EngineType enum value has a matching entry in VALID_ENGINE_TYPES."""
        for et in EngineType:
            assert et.value in VALID_ENGINE_TYPES, (
                f"EngineType.{et.name} ({et.value}) missing from "
                f"VALID_ENGINE_TYPES in requests.py"
            )

    def test_no_phantom_engine_types(self) -> None:
        """VALID_ENGINE_TYPES doesn't contain phantom entries."""
        registry_values = {et.value for et in EngineType}
        # 'myosuite' is an alias for 'myosim', so exclude from check
        aliases = {"myosuite"}
        for vt in VALID_ENGINE_TYPES:
            if vt in aliases:
                continue
            assert vt in registry_values, (
                f"'{vt}' in VALID_ENGINE_TYPES but not in EngineType enum"
            )
