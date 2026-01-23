"""Tests for Swing Plane Visualization.

Guideline L1 implementation tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.reference_frames import SwingPlaneFrame
from src.shared.python.swing_plane_visualization import (
    SwingPlaneVisualizer,
    compute_trajectory_deviations,
    create_deviation_colormap,
    create_fsp_visualization,
    create_instantaneous_plane_visualization,
    generate_plane_vertices,
)


class TestPlaneVertices:
    """Tests for plane vertex generation."""

    def test_vertices_form_square(self) -> None:
        """Generated vertices should form a square plane."""
        origin = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        in_plane_x = np.array([1.0, 0.0, 0.0])
        in_plane_y = np.array([0.0, 1.0, 0.0])
        half_size = 1.0

        vertices = generate_plane_vertices(
            origin, normal, in_plane_x, in_plane_y, half_size
        )

        assert vertices.shape == (4, 3)

        # All vertices should be at z=0 for horizontal plane
        np.testing.assert_allclose(vertices[:, 2], 0.0, atol=1e-10)

        # Vertices should be at corners of [-1, 1] x [-1, 1] square
        for v in vertices:
            assert abs(v[0]) == pytest.approx(1.0)
            assert abs(v[1]) == pytest.approx(1.0)

    def test_vertices_centered_on_origin(self) -> None:
        """Vertices should be centered around the origin."""
        origin = np.array([5.0, 3.0, 1.0])
        normal = np.array([0.0, 0.0, 1.0])
        in_plane_x = np.array([1.0, 0.0, 0.0])
        in_plane_y = np.array([0.0, 1.0, 0.0])

        vertices = generate_plane_vertices(
            origin, normal, in_plane_x, in_plane_y, half_size=1.0
        )

        # Center of vertices should equal origin
        center = np.mean(vertices, axis=0)
        np.testing.assert_allclose(center, origin, atol=1e-10)


class TestInstantaneousPlaneVisualization:
    """Tests for instantaneous swing plane visualization."""

    def test_creates_valid_visualization(self) -> None:
        """Should create a valid visualization object."""
        velocity = np.array([10.0, 0.0, 0.0])
        grip = np.array([0.0, 0.0, 1.0])
        clubhead = np.array([1.0, 0.0, 0.0])

        vis = create_instantaneous_plane_visualization(velocity, grip, clubhead)

        assert vis is not None
        assert vis.vertices.shape == (4, 3)
        assert vis.plane_frame is not None
        assert vis.is_fsp is False
        assert len(vis.normal_arrow_start) == 3
        assert len(vis.normal_arrow_end) == 3

    def test_normal_arrow_points_outward(self) -> None:
        """Normal arrow should extend from plane center outward."""
        velocity = np.array([10.0, 0.0, 0.0])
        grip = np.array([0.0, 0.0, 1.0])
        clubhead = np.array([0.0, 0.0, 0.0])

        vis = create_instantaneous_plane_visualization(velocity, grip, clubhead)

        # Arrow end should be different from start
        arrow_vec = vis.normal_arrow_end - vis.normal_arrow_start
        assert np.linalg.norm(arrow_vec) > 0.1


class TestFSPVisualization:
    """Tests for Functional Swing Plane visualization."""

    @pytest.fixture
    def planar_trajectory(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Create a planar trajectory for testing."""
        t = np.linspace(0, 1, 50)
        trajectory = np.column_stack(
            [
                t,
                np.zeros_like(t),
                np.zeros_like(t),
            ]
        )
        impact_time = 0.5
        return trajectory, t, impact_time

    def test_creates_valid_fsp_visualization(
        self, planar_trajectory: tuple[np.ndarray, np.ndarray, float]
    ) -> None:
        """Should create a valid FSP visualization."""
        trajectory, timestamps, impact_time = planar_trajectory

        vis = create_fsp_visualization(
            trajectory, timestamps, impact_time, window_ms=500.0
        )

        assert vis is not None
        assert vis.is_fsp is True
        assert vis.plane_frame is not None
        assert "FSP" in vis.label

    def test_fsp_label_includes_rmse(
        self, planar_trajectory: tuple[np.ndarray, np.ndarray, float]
    ) -> None:
        """FSP label should include RMSE value."""
        trajectory, timestamps, impact_time = planar_trajectory

        vis = create_fsp_visualization(trajectory, timestamps, impact_time)

        assert "RMSE" in vis.label


class TestTrajectoryDeviations:
    """Tests for trajectory deviation computation."""

    def test_on_plane_trajectory_has_zero_deviation(self) -> None:
        """Trajectory on the plane should have zero deviation."""
        # Horizontal plane at z=0
        plane_frame = SwingPlaneFrame(
            origin=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            in_plane_x=np.array([1.0, 0.0, 0.0]),
            in_plane_y=np.array([0.0, 1.0, 0.0]),
            grip_axis=np.array([1.0, 0.0, 0.0]),
        )

        # Trajectory in the XY plane at z=0
        trajectory = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )

        deviations = compute_trajectory_deviations(trajectory, plane_frame)

        np.testing.assert_allclose(deviations, 0.0, atol=1e-10)

    def test_above_plane_has_positive_deviation(self) -> None:
        """Points above the plane should have positive deviation."""
        plane_frame = SwingPlaneFrame(
            origin=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            in_plane_x=np.array([1.0, 0.0, 0.0]),
            in_plane_y=np.array([0.0, 1.0, 0.0]),
            grip_axis=np.array([1.0, 0.0, 0.0]),
        )

        # Points at z=0.5 (above plane)
        trajectory = np.array(
            [
                [0.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
            ]
        )

        deviations = compute_trajectory_deviations(trajectory, plane_frame)

        np.testing.assert_allclose(deviations, 0.5, atol=1e-10)


class TestDeviationColormap:
    """Tests for deviation-based coloring."""

    def test_zero_deviation_is_green(self) -> None:
        """Zero deviation should give green color."""
        deviations = np.array([0.0])
        colors = create_deviation_colormap(deviations)

        # Green = [0, 1, 0]
        np.testing.assert_allclose(colors[0], [0.0, 1.0, 0.0], atol=1e-10)

    def test_positive_deviation_trends_red(self) -> None:
        """Positive deviation should trend toward red."""
        deviations = np.array([0.1])  # At max_deviation default of 0.1
        colors = create_deviation_colormap(deviations, max_deviation=0.1)

        # At full positive, should be red [1, 0, 0]
        np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=1e-10)

    def test_negative_deviation_trends_blue(self) -> None:
        """Negative deviation should trend toward blue."""
        deviations = np.array([-0.1])  # At -max_deviation
        colors = create_deviation_colormap(deviations, max_deviation=0.1)

        # At full negative, should be blue [0, 0, 1]
        np.testing.assert_allclose(colors[0], [0.0, 0.0, 1.0], atol=1e-10)


class TestSwingPlaneVisualizer:
    """Tests for SwingPlaneVisualizer class."""

    def test_update_instantaneous_plane(self) -> None:
        """Should update instantaneous plane visualization."""
        viz = SwingPlaneVisualizer()

        result = viz.update_instantaneous_plane(
            clubhead_velocity=np.array([10.0, 0.0, 0.0]),
            grip_position=np.array([0.0, 0.0, 1.0]),
            clubhead_position=np.array([1.0, 0.0, 0.0]),
        )

        assert result is not None
        assert viz.current_scene.instantaneous_plane is not None

    def test_record_trajectory_and_compute_fsp(self) -> None:
        """Should record trajectory and compute FSP."""
        viz = SwingPlaneVisualizer()

        # Record trajectory points
        for i in range(20):
            t = i * 0.05
            pos = np.array([t, 0.0, 0.0])
            viz.record_trajectory_point(pos, t)

        # Compute FSP
        fsp = viz.compute_fsp(impact_time=0.5, window_ms=500.0)

        assert fsp is not None
        assert viz.current_scene.fsp is not None
        assert viz.current_scene.plane_metrics is not None

    def test_get_trajectory_visualization(self) -> None:
        """Should get trajectory visualization."""
        viz = SwingPlaneVisualizer()

        # Record points
        for i in range(10):
            viz.record_trajectory_point(np.array([float(i), 0.0, 0.0]), float(i))

        traj = viz.get_trajectory_visualization()

        assert traj is not None
        assert len(traj.points) == 10
        assert len(traj.timestamps) == 10

    def test_export_scene_json(self, tmp_path: Path) -> None:
        """Should export scene to JSON file."""
        viz = SwingPlaneVisualizer()

        # Create some data
        viz.update_instantaneous_plane(
            np.array([10.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        for i in range(10):
            viz.record_trajectory_point(np.array([float(i), 0.0, 0.0]), float(i))

        viz.compute_fsp(impact_time=5.0)
        viz.get_trajectory_visualization()

        # Export
        output_file = tmp_path / "scene.json"
        viz.export_scene_json(output_file)

        assert output_file.exists()

        # Verify JSON structure
        import json

        with open(output_file) as f:
            data = json.load(f)

        assert "instantaneous_plane" in data
        assert "fsp" in data
        assert "trajectory" in data

    def test_reset_clears_state(self) -> None:
        """Reset should clear all state."""
        viz = SwingPlaneVisualizer()

        # Add some data
        viz.record_trajectory_point(np.array([1.0, 0.0, 0.0]), 0.0)
        viz.update_instantaneous_plane(
            np.array([10.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        # Reset
        viz.reset()

        assert len(viz.trajectory_history) == 0
        assert len(viz.timestamp_history) == 0
        assert viz.current_scene.instantaneous_plane is None
