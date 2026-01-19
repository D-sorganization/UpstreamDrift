"""Tests for API service layer.

This module tests the SimulationService and AnalysisService classes
with mocked dependencies to ensure business logic is correct.
"""

from unittest.mock import MagicMock, patch

import pytest

# Skip if API dependencies not available
try:
    from api.models.requests import AnalysisRequest, SimulationRequest
    from api.services.analysis_service import AnalysisService
    from api.services.simulation_service import SimulationService
except ImportError as e:
    pytest.skip(f"Cannot import API services: {e}", allow_module_level=True)


class TestSimulationService:
    """Tests for SimulationService."""

    @pytest.fixture
    def mock_engine_manager(self) -> MagicMock:
        """Create a mock engine manager."""
        manager = MagicMock()
        mock_engine = MagicMock()
        mock_engine.step = MagicMock()
        mock_engine.set_state = MagicMock()
        mock_engine.set_control = MagicMock()
        mock_engine.reset = MagicMock()
        mock_engine.load_from_path = MagicMock()
        manager.get_active_physics_engine.return_value = mock_engine
        return manager

    @pytest.fixture
    def service(self, mock_engine_manager: MagicMock) -> SimulationService:
        """Create a simulation service with mocked dependencies."""
        return SimulationService(mock_engine_manager)

    def test_init(self, mock_engine_manager: MagicMock) -> None:
        """Test service initialization."""
        service = SimulationService(mock_engine_manager)
        assert service.engine_manager is mock_engine_manager

    @pytest.mark.skip(reason="pytest-asyncio missing")
    @pytest.mark.asyncio
    async def test_run_simulation_basic(
        self, service: SimulationService, mock_engine_manager: MagicMock
    ) -> None:
        """Test basic simulation run."""
        # Mock the recorder to avoid complex setup
        with patch(
            "api.services.simulation_service.GenericPhysicsRecorder"
        ) as MockRecorder:
            mock_recorder = MagicMock()
            mock_recorder.is_recording = False
            mock_recorder.get_time_series.return_value = ([], [])
            MockRecorder.return_value = mock_recorder

            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.1,
                timestep=0.01,
                model_path=None,
                initial_state=None,
                control_inputs=None,
                analysis_config=None,
            )

            result = await service.run_simulation(request)

            # Should return a response (success depends on engine availability)
            assert result is not None
            assert hasattr(result, "success")

    @pytest.mark.skip(reason="pytest-asyncio missing")
    @pytest.mark.asyncio
    async def test_run_simulation_with_initial_state(
        self, service: SimulationService, mock_engine_manager: MagicMock
    ) -> None:
        """Test simulation with initial state."""
        with patch(
            "api.services.simulation_service.GenericPhysicsRecorder"
        ) as MockRecorder:
            mock_recorder = MagicMock()
            mock_recorder.is_recording = False
            mock_recorder.get_time_series.return_value = ([], [])
            MockRecorder.return_value = mock_recorder

            request = SimulationRequest(
                engine_type="mujoco",
                duration=0.1,
                timestep=0.01,
                model_path=None,
                initial_state={"positions": [0.0, 0.0], "velocities": [0.0, 0.0]},
                control_inputs=None,
                analysis_config=None,
            )

            result = await service.run_simulation(request)
            assert result is not None

    @pytest.mark.skip(reason="pytest-asyncio missing")
    @pytest.mark.asyncio
    async def test_run_simulation_engine_failure(
        self, mock_engine_manager: MagicMock
    ) -> None:
        """Test simulation when engine fails to load."""
        mock_engine_manager.get_active_physics_engine.return_value = None
        service = SimulationService(mock_engine_manager)

        request = SimulationRequest(
            engine_type="mujoco",
            duration=0.1,
            model_path=None,
            timestep=None,
            initial_state=None,
            control_inputs=None,
            analysis_config=None,
        )

        result = await service.run_simulation(request)

        # Should return failure response, not raise exception
        assert result is not None
        assert result.success is False

    def test_extract_simulation_data(self, service: SimulationService) -> None:
        """Test data extraction from recorder."""
        mock_recorder = MagicMock()
        mock_recorder.get_time_series.side_effect = [
            ([0.0, 0.1], [[0.0], [0.1]]),  # positions
            ([0.0, 0.1], [[0.0], [0.1]]),  # velocities
            ([0.0, 0.1], [[0.0], [0.1]]),  # accelerations
            ([0.0, 0.1], [[0.0], [0.1]]),  # control
        ]

        data = service._extract_simulation_data(mock_recorder)

        assert "times" in data
        assert "joint_positions" in data
        assert "joint_velocities" in data

    def test_extract_simulation_data_handles_errors(
        self, service: SimulationService
    ) -> None:
        """Test that data extraction handles missing data gracefully."""
        mock_recorder = MagicMock()
        mock_recorder.get_time_series.side_effect = KeyError("no data")

        # Should not raise, just return empty/partial data
        data = service._extract_simulation_data(mock_recorder)
        assert isinstance(data, dict)


class TestAnalysisService:
    """Tests for AnalysisService."""

    @pytest.fixture
    def mock_engine_manager(self) -> MagicMock:
        """Create a mock engine manager."""
        return MagicMock()

    @pytest.fixture
    def service(self, mock_engine_manager: MagicMock) -> AnalysisService:
        """Create an analysis service with mocked dependencies."""
        return AnalysisService(mock_engine_manager)

    def test_init(self, mock_engine_manager: MagicMock) -> None:
        """Test service initialization."""
        service = AnalysisService(mock_engine_manager)
        assert service.engine_manager is mock_engine_manager

    @pytest.mark.skip(reason="pytest-asyncio missing")
    @pytest.mark.asyncio
    async def test_analyze_biomechanics_basic(self, service: AnalysisService) -> None:
        """Test basic biomechanics analysis."""
        request = AnalysisRequest(
            analysis_type="kinematic_sequence",
            data_source="simulation",
            parameters={
                "joint_positions": [[0.0, 0.1]],
                "joint_velocities": [[0.0, 0.1]],
                "times": [0.0, 0.1],
            },
            export_format="json",
        )

        result = await service.analyze_biomechanics(request)
        assert result is not None

    @pytest.mark.skip(reason="pytest-asyncio missing")
    @pytest.mark.asyncio
    async def test_analyze_biomechanics_missing_data(
        self, service: AnalysisService
    ) -> None:
        """Test analysis with missing required data."""
        request = AnalysisRequest(
            analysis_type="kinematic_sequence",
            data_source="simulation",
            parameters={},  # Empty data
            export_format="json",
        )

        # Should handle gracefully
        result = await service.analyze_biomechanics(request)
        assert result is not None


class TestServiceIntegration:
    """Integration tests for services working together."""

    @pytest.fixture
    def mock_engine_manager(self) -> MagicMock:
        """Create a mock engine manager."""
        manager = MagicMock()
        mock_engine = MagicMock()
        manager.get_active_physics_engine.return_value = mock_engine
        return manager

    @pytest.mark.skip(reason="pytest-asyncio missing")
    @pytest.mark.asyncio
    async def test_simulation_to_analysis_flow(
        self, mock_engine_manager: MagicMock
    ) -> None:
        """Test that simulation output can feed into analysis."""
        with patch(
            "api.services.simulation_service.GenericPhysicsRecorder"
        ) as MockRecorder:
            mock_recorder = MagicMock()
            mock_recorder.is_recording = False
            mock_recorder.get_time_series.return_value = ([0.0, 0.1], [[0.0], [0.1]])
            MockRecorder.return_value = mock_recorder

            sim_service = SimulationService(mock_engine_manager)
            analysis_service = AnalysisService(mock_engine_manager)

            # Run simulation
            sim_request = SimulationRequest(
                engine_type="mujoco",
                duration=0.1,
                model_path=None,
                timestep=None,
                initial_state=None,
                control_inputs=None,
                analysis_config=None,
            )
            sim_result = await sim_service.run_simulation(sim_request)

            # Use simulation data for analysis
            if sim_result.data:
                analysis_request = AnalysisRequest(
                    analysis_type="kinematic_sequence",
                    data_source="simulation",
                    parameters=sim_result.data,
                    export_format="json",
                )
                analysis_result = await analysis_service.analyze_biomechanics(
                    analysis_request
                )
                assert analysis_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
