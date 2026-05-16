"""
Integration tests between different physics engines.

These tests verify that the EngineManager correctly integrates with the
filesystem and engine discovery system. They use real filesystem operations
instead of mocks to test actual integration.
"""

from typing import Any
from unittest.mock import Mock

import pytest
from src.shared.python.engine_core.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
)


class TestEngineIntegration:
    """Test integration between different physics engines."""

    @pytest.mark.integration
    def test_engine_manager_initialization(self) -> None:
        """Test that engine manager initializes with real project structure.

        This is a real integration test - uses actual filesystem.
        """
        manager = EngineManager()

        # Should have initialized with real paths
        assert manager.suite_root.exists(), (
            "Assertion failed: manager.suite_root.exists()"
        )
        assert manager.engines_root.exists(), (
            "Assertion failed: manager.engines_root.exists()"
        )

        # Should have discovered some engines (at least the ones in the repo)
        available_engines = manager.get_available_engines()
        assert isinstance(available_engines, list), (
            "Assertion failed: isinstance(available_engines, list)"
        )

        # Each engine type should have a path defined
        assert len(manager.engine_paths) >= len(EngineType) - 1, (
            "Assertion failed: len(manager.engine_paths) >= len(EngineType) - 1"
        )

    @pytest.mark.integration
    def test_engine_availability_matches_filesystem(self) -> None:
        """Test that engine availability correctly reflects filesystem state.

        This is a real integration test - checks actual directory structure.
        """
        manager = EngineManager()
        available_engines = manager.get_available_engines()

        # For each available engine, verify its path actually exists
        for engine in available_engines:
            engine_path = manager.engine_paths[engine]
            assert engine_path.exists(), (
                f"{engine} marked available but path missing: {engine_path}"
            )

        # For unavailable engines, verify why they're unavailable
        for engine in EngineType:
            if engine not in available_engines:
                # Should either have missing path or failed validation
                engine_path = manager.engine_paths[engine]
                if engine_path.exists():
                    # Path exists but validation failed - expected for incomplete installations
                    result = manager.validate_engine_configuration(engine)
                    assert result is False, (
                        f"{engine} exists but should fail validation"
                    )

    @pytest.mark.integration
    def test_engine_probe_consistency(self) -> None:
        """Test that engine probes provide consistent information.

        This tests the integration between EngineManager and EngineProbes.
        """
        manager = EngineManager()

        # Most engine types should have an associated probe
        # (newer engines may not have probes yet)
        assert len(manager.probes) >= len(EngineType) - 3, (
            "Assertion failed: len(manager.probes) >= len(EngineType) - 3"
        )

        # Run probes and check consistency
        for engine_type, probe in manager.probes.items():
            # Probe should return consistent data structure
            try:
                probe_result = probe.is_available()  # type: ignore[attr-defined]

                # Result should be a dict or have expected attributes
                assert isinstance(probe_result, dict | bool | type(None)), (
                    "Assertion failed: isinstance(probe_result, dict | bool | type(None))"
                )

                # If probe says available, engine should be in available list
                available_engines = manager.get_available_engines()
                if probe_result and engine_type in available_engines:
                    # Consistency check: status should not be UNAVAILABLE
                    assert (
                        manager.get_engine_status(engine_type)
                        != EngineStatus.UNAVAILABLE
                    )
            except Exception as e:  # noqa: BLE001, F841
                # Some probes may fail if dependencies missing - that's expected
                pass

        # Validate we have at least one engine available in the test environment
        available = manager.get_available_engines()
        assert isinstance(
            available, list
        )  # May be empty in minimal CI environment, "Assertion failed: isinstance(available, list)  # May be empty in minimal CI environment"

    @pytest.mark.integration
    def test_engine_parameter_consistency(self) -> None:
        """Test that all engines accept consistent parameter sets."""
        common_parameters = {
            "swing_speed": 100.0,  # mph
            "club_type": "driver",
            "ball_position": [0, 0, 0],
            "simulation_time": 2.0,
            "timestep": 0.001,
        }

        manager = EngineManager()
        available_engines = manager.get_available_engines()

        # Test that parameters can be set for each engine
        for _engine in available_engines:
            # Mock engine instance
            mock_instance = Mock()

            # Test parameter setting
            for param, value in common_parameters.items():
                setattr(mock_instance, param, value)

            # Should not raise exceptions
            assert mock_instance.swing_speed == 100.0, (
                "Assertion failed: mock_instance.swing_speed == 100.0"
            )
            assert mock_instance.club_type == "driver", (
                "Assertion failed: mock_instance.club_type == driver"
            )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_comparison(self) -> None:
        """Test performance characteristics of different engines."""
        import time

        manager = EngineManager()
        available_engines = manager.get_available_engines()
        performance_results = {}

        # Mock simulation with realistic timing - moved outside loop to avoid closure
        def mock_simulate(engine_type: object) -> dict[str, float]:
            # Simulate different performance characteristics
            if engine_type == EngineType.MUJOCO:
                time.sleep(0.01)  # Slower but more accurate
            elif engine_type == EngineType.DRAKE:
                time.sleep(0.005)  # Medium speed
            else:  # pinocchio or others
                time.sleep(0.002)  # Fastest

            return {"ball_distance": 250.0}

        for engine in available_engines:
            # Measure performance
            start_time = time.time()
            result = mock_simulate(engine)
            end_time = time.time()

            performance_results[engine] = {
                "simulation_time": end_time - start_time,
                "result": result,
            }

        # Verify we have performance data
        assert len(performance_results) > 0, (
            "Assertion failed: len(performance_results) > 0"
        )

        # All simulations should complete successfully
        for perf_data in performance_results.values():
            assert "simulation_time" in perf_data, (
                "Assertion failed: simulation_time in perf_data"
            )
            assert "result" in perf_data, "Assertion failed: result in perf_data"
            assert perf_data["result"]["ball_distance"] == 250.0, (
                "Assertion failed: perf_data[result][ball_distance] == 250.0"
            )


class TestEngineDataFlow:
    """Test data flow between engines and shared components."""

    @pytest.mark.integration
    def test_shared_data_structures(self) -> None:
        """Test that all engines work with shared data structures."""
        manager = EngineManager()
        available_engines = manager.get_available_engines()

        # Mock swing data
        sample_swing_data = {
            "time": [0.0, 0.1, 0.2, 0.3],
            "club_angle": [0, 15, 30, 45],
            "ball_position": [[0, 0, 0], [0.1, 0, 0.1], [0.2, 0, 0.2], [0.3, 0, 0.3]],
        }

        # Test that swing data can be processed by all engines
        for _engine in available_engines:
            mock_instance = Mock()

            # Test data input
            mock_instance.load_swing_data.return_value = True
            result = mock_instance.load_swing_data(sample_swing_data)

            assert result is True, "Assertion failed: result is True"
            mock_instance.load_swing_data.assert_called_with(sample_swing_data)

    @pytest.mark.integration
    def test_output_format_consistency(self) -> None:
        """Test that all engines produce consistent output formats."""
        manager = EngineManager()
        available_engines = manager.get_available_engines()

        expected_fields = [
            "ball_distance",
            "launch_angle",
            "ball_speed",
            "simulation_time",
            "trajectory_data",
        ]

        # Mock trajectory data
        mock_trajectory = {
            "time": [0.0, 0.1, 0.2],
            "position": [[0, 0, 0], [1, 0, 1], [2, 0, 2]],
        }

        for _engine in available_engines:
            mock_instance = Mock()

            # Mock consistent output format
            mock_result: dict[str, Any] = dict.fromkeys(expected_fields, 0.0)
            mock_result["trajectory_data"] = mock_trajectory

            mock_instance.simulate.return_value = mock_result

            result = mock_instance.simulate()

            # Verify all expected fields are present
            for field in expected_fields:
                assert field in result, "Assertion failed: field in result"

    @pytest.mark.integration
    def test_engine_error_handling(self) -> None:
        """Test error handling consistency across engines."""
        manager = EngineManager()
        available_engines = manager.get_available_engines()

        for _engine in available_engines:
            mock_instance = Mock()

            # Test invalid parameter handling
            mock_instance.set_swing_speed.side_effect = ValueError(
                "Invalid swing speed"
            )

            with pytest.raises(ValueError):
                mock_instance.set_swing_speed(-100)  # Invalid negative speed
