"""Tests for analysis_service - Biomechanical analysis service.

These tests verify the analysis service using Design by Contract principles.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

# Configure async tests to use asyncio backend only
pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend():
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


@pytest.fixture
def mock_engine_manager():
    """Create a mock engine manager."""
    manager = MagicMock()
    manager.get_active_physics_engine = MagicMock(return_value=None)
    return manager


@pytest.fixture
def analysis_service(mock_engine_manager):
    """Create an analysis service instance."""
    from src.api.services.analysis_service import AnalysisService
    return AnalysisService(mock_engine_manager)


class TestAnalysisServiceContract:
    """Design by Contract tests for AnalysisService class."""

    def test_instantiates(self, mock_engine_manager):
        """Postcondition: AnalysisService can be instantiated."""
        from src.api.services.analysis_service import AnalysisService
        service = AnalysisService(mock_engine_manager)
        assert service is not None

    def test_has_engine_manager(self, analysis_service):
        """Postcondition: AnalysisService has engine_manager attribute."""
        assert hasattr(analysis_service, "engine_manager")

    def test_has_analyze_biomechanics_method(self, analysis_service):
        """Postcondition: AnalysisService has analyze_biomechanics method."""
        assert hasattr(analysis_service, "analyze_biomechanics")
        assert callable(analysis_service.analyze_biomechanics)


class TestAnalyzeKinematicsInternal:
    """Tests for _analyze_kinematics internal method."""

    async def test_kinematics_with_no_engine(self, analysis_service):
        """Test kinematics analysis with no engine loaded."""
        from src.api.models.requests import AnalysisRequest

        request = AnalysisRequest(
            analysis_type="kinematics",
            data_source="simulation",
            parameters={},
        )

        result = await analysis_service._analyze_kinematics(request, None)

        assert result["analysis_type"] == "kinematics"
        assert "metadata" in result
        assert result["metadata"]["data_source"] == "none"
        assert "joint_angles" in result

    async def test_kinematics_with_mock_engine(self, mock_engine_manager):
        """Test kinematics analysis with mock engine."""
        from src.api.models.requests import AnalysisRequest
        from src.api.services.analysis_service import AnalysisService

        mock_engine = MagicMock()
        mock_engine.get_joint_positions = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))
        mock_engine.get_joint_velocities = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        mock_engine.get_joint_accelerations = MagicMock(return_value=np.array([0.5, 1.0, 1.5]))
        mock_engine.get_state = MagicMock(return_value={"time": 0.5})

        service = AnalysisService(mock_engine_manager)
        request = AnalysisRequest(
            analysis_type="kinematics",
            data_source="simulation",
            parameters={},
        )

        result = await service._analyze_kinematics(request, mock_engine)

        assert result["joint_angles"] == [0.1, 0.2, 0.3]
        assert result["angular_velocities"] == [1.0, 2.0, 3.0]
        assert result["metadata"]["data_source"] == "engine"


class TestAnalyzeKineticsInternal:
    """Tests for _analyze_kinetics internal method."""

    async def test_kinetics_with_no_engine(self, analysis_service):
        """Test kinetics analysis with no engine loaded."""
        from src.api.models.requests import AnalysisRequest

        request = AnalysisRequest(
            analysis_type="kinetics",
            data_source="simulation",
            parameters={},
        )

        result = await analysis_service._analyze_kinetics(request, None)

        assert result["analysis_type"] == "kinetics"
        assert "joint_torques" in result
        assert "muscle_forces" in result

    async def test_kinetics_with_mock_engine(self, mock_engine_manager):
        """Test kinetics analysis with mock engine."""
        from src.api.models.requests import AnalysisRequest
        from src.api.services.analysis_service import AnalysisService

        mock_engine = MagicMock()
        mock_engine.get_joint_torques = MagicMock(return_value=np.array([10.0, 20.0]))
        mock_engine.get_actuator_forces = MagicMock(return_value=np.array([100.0, 200.0]))

        service = AnalysisService(mock_engine_manager)
        request = AnalysisRequest(
            analysis_type="kinetics",
            data_source="simulation",
            parameters={},
        )

        result = await service._analyze_kinetics(request, mock_engine)

        assert result["joint_torques"] == [10.0, 20.0]
        assert result["muscle_forces"] == [100.0, 200.0]


class TestAnalyzeEnergeticsInternal:
    """Tests for _analyze_energetics internal method."""

    async def test_energetics_with_no_engine(self, analysis_service):
        """Test energetics analysis with no engine loaded."""
        from src.api.models.requests import AnalysisRequest

        request = AnalysisRequest(
            analysis_type="energetics",
            data_source="simulation",
            parameters={},
        )

        result = await analysis_service._analyze_energetics(request, None)

        assert result["analysis_type"] == "energetics"
        assert "kinetic_energy" in result
        assert "potential_energy" in result
        assert "total_energy" in result

    async def test_energetics_with_mock_engine(self, mock_engine_manager):
        """Test energetics analysis with mock engine."""
        from src.api.models.requests import AnalysisRequest
        from src.api.services.analysis_service import AnalysisService

        mock_engine = MagicMock()
        mock_engine.get_kinetic_energy = MagicMock(return_value=150.0)
        mock_engine.get_potential_energy = MagicMock(return_value=50.0)
        mock_engine.get_total_energy = MagicMock(return_value=200.0)

        service = AnalysisService(mock_engine_manager)
        request = AnalysisRequest(
            analysis_type="energetics",
            data_source="simulation",
            parameters={},
        )

        result = await service._analyze_energetics(request, mock_engine)

        assert result["kinetic_energy"] == 150.0
        assert result["potential_energy"] == 50.0
        assert result["total_energy"] == 200.0


class TestAnalyzeSwingSequenceInternal:
    """Tests for _analyze_swing_sequence internal method."""

    async def test_swing_sequence_with_no_engine(self, analysis_service):
        """Test swing sequence analysis with no engine loaded."""
        from src.api.models.requests import AnalysisRequest

        request = AnalysisRequest(
            analysis_type="swing_sequence",
            data_source="simulation",
            parameters={},
        )

        result = await analysis_service._analyze_swing_sequence(request, None)

        assert result["analysis_type"] == "swing_sequence"
        assert "phases" in result
        assert len(result["phases"]) == 8

    async def test_swing_sequence_phases_list(self, analysis_service):
        """Test that swing sequence contains correct phases."""
        from src.api.models.requests import AnalysisRequest

        request = AnalysisRequest(
            analysis_type="swing_sequence",
            data_source="simulation",
            parameters={},
        )

        result = await analysis_service._analyze_swing_sequence(request, None)

        expected_phases = [
            "address",
            "takeaway",
            "backswing",
            "transition",
            "downswing",
            "impact",
            "follow_through",
            "finish",
        ]
        assert result["phases"] == expected_phases


class TestToListHelper:
    """Tests for _to_list helper method."""

    def test_converts_numpy_array(self, analysis_service):
        """Test converting numpy array to list."""
        result = analysis_service._to_list(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_converts_list(self, analysis_service):
        """Test converting list."""
        result = analysis_service._to_list([1, 2, 3])
        assert result == [1, 2, 3]

    def test_converts_tuple(self, analysis_service):
        """Test converting tuple."""
        result = analysis_service._to_list((1, 2, 3))
        assert result == [1, 2, 3]

    def test_converts_scalar(self, analysis_service):
        """Test converting scalar value."""
        result = analysis_service._to_list(42)
        assert result == [42]

    def test_returns_empty_for_none(self, analysis_service):
        """Test returning empty list for None."""
        result = analysis_service._to_list(None)
        assert result == []


class TestDetectSwingPhase:
    """Tests for _detect_swing_phase helper method."""

    def test_returns_address_at_time_zero(self, analysis_service):
        """Test returning 'address' phase at time zero."""
        result = analysis_service._detect_swing_phase({"time": 0})
        assert result == "address"

    def test_returns_none_for_empty_state(self, analysis_service):
        """Test returning None for empty state."""
        result = analysis_service._detect_swing_phase({})
        assert result is None

    def test_returns_none_for_nonzero_time(self, analysis_service):
        """Test returning None for non-zero time (needs more context)."""
        result = analysis_service._detect_swing_phase({"time": 0.5})
        assert result is None
