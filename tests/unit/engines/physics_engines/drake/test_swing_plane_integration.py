"""Tests for swing_plane_integration.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.swing_plane_integration import DrakeSwingPlaneAnalyzer


class DummyMetrics:
    def __init__(self):
        self.steepness_deg = 45.0
        self.rmse = 0.05
        self.normal_vector = np.array([0, 0, 1])
        self.point_on_plane = np.array([0, 0, 0])
        self.direction_deg = 0.0
        self.max_deviation = 0.1


@pytest.fixture
def dummy_metrics():
    return DummyMetrics()


class TestDrakeSwingPlaneAnalyzer:
    @patch("src.engines.physics_engines.drake.python.swing_plane_integration.SwingPlaneAnalyzer")
    def test_analyze_trajectory(self, mock_analyzer_cls, dummy_metrics):
        analyzer_mock = mock_analyzer_cls.return_value
        analyzer_mock.analyze.return_value = dummy_metrics
        
        analyzer = DrakeSwingPlaneAnalyzer()
        positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        
        metrics = analyzer.analyze_trajectory(positions)
        assert metrics == dummy_metrics
        analyzer_mock.analyze.assert_called_once()
        np.testing.assert_array_equal(analyzer_mock.analyze.call_args[0][0], positions)

    def test_analyze_trajectory_invalid_shape(self):
        analyzer = DrakeSwingPlaneAnalyzer()
        with pytest.raises(ValueError, match="Positions must be \\(N, 3\\) array"):
            analyzer.analyze_trajectory(np.array([[0, 0], [1, 1], [2, 2]]))
            
        with pytest.raises(ValueError, match="At least 3 positions required"):
            analyzer.analyze_trajectory(np.array([[0, 0, 0], [1, 1, 1]]))

    @patch.object(DrakeSwingPlaneAnalyzer, "analyze_trajectory")
    def test_analyze_from_drake_context(self, mock_analyze, dummy_metrics):
        mock_analyze.return_value = dummy_metrics
        analyzer = DrakeSwingPlaneAnalyzer()
        
        plant_mock = MagicMock()
        context_mock = MagicMock()
        
        # Mock sample_times path
        context_mock.sample_times.return_value = [0.0, 0.5, 1.0]
        context_mock.value.return_value = "val"
        
        plant_context_mock = MagicMock()
        plant_mock.GetMyContextFromRoot.return_value = plant_context_mock
        
        pose_mock = MagicMock()
        pose_mock.translation.return_value = np.array([1, 2, 3])
        plant_mock.EvalBodyPoseInWorld.return_value = pose_mock
        
        metrics = analyzer.analyze_from_drake_context(context_mock, plant_mock, club_body_index=1, num_samples=3)
        assert metrics == dummy_metrics
        assert mock_analyze.call_count == 1
        
        positions = mock_analyze.call_args[0][0]
        assert len(positions) == 3
        np.testing.assert_array_equal(positions[0], np.array([1, 2, 3]))

    def test_integrate_with_optimization(self, dummy_metrics):
        analyzer = DrakeSwingPlaneAnalyzer()
        analyzer.analyzer.analyze = MagicMock(return_value=dummy_metrics)
        
        optimizer_mock = MagicMock()
        
        analyzer.integrate_with_optimization(optimizer_mock, swing_plane_constraint_weight=2.0)
        optimizer_mock.add_objective.assert_called_once()
        
        # Test the cost function
        cost_func = optimizer_mock.add_objective.call_args[1]["cost_function"]
        trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        cost = cost_func(trajectory)
        assert cost == dummy_metrics.rmse ** 2

    @patch("builtins.open")
    def test_export_for_analysis(self, mock_open, dummy_metrics):
        analyzer = DrakeSwingPlaneAnalyzer()
        trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        
        analyzer.export_for_analysis(dummy_metrics, trajectory, "dummy/path.json")
        mock_open.assert_called_once()
