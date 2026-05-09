"""Tests for Contact-Based Grip Model.

Guideline K2 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.physics.grip_contact_model import (
    ContactPoint,
    ContactState,
    GripContactExporter,
    GripContactModel,
    GripParameters,
    PressureVisualizationData,
    check_friction_cone,
    classify_contact_state,
    compute_center_of_pressure,
    compute_pressure_visualization,
    create_mujoco_grip_contacts,
    decompose_contact_force,
)


class TestGripContactExporter:
    """Tests for contact data export functionality (Issue #757)."""

    @pytest.fixture
    def model_with_data(self) -> GripContactModel:
        """Create model with sample contact data."""
        model = GripContactModel()

        # Add several timesteps of data
        for t in range(5):
            positions = np.array([[0.0, 0.0, t * 0.001], [0.01, 0.0, t * 0.001]])
            normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
            forces = np.array([[0.0, 0.0, 50.0 + t * 10], [0.0, 0.0, 50.0 + t * 10]])
            velocities = np.zeros((2, 3))
            body_names = ["left_hand", "left_hand"]

            model.update_from_mujoco(
                positions, normals, forces, velocities, body_names, timestamp=t * 0.1
            )

        return model

    def test_capture_timestep(self, model_with_data: GripContactModel) -> None:
        """Exporter should capture timestep data."""
        exporter = GripContactExporter(model_with_data)
        timestep = exporter.capture_timestep()

        assert timestep is not None
        assert timestep.num_contacts == 2
        assert timestep.total_normal_force > 0

    def test_grip_contact_model_export_to_dict(
        self, model_with_data: GripContactModel
    ) -> None:
        """Should export all captured timesteps as dict."""
        exporter = GripContactExporter(model_with_data)

        # Capture current state
        exporter.capture_timestep()

        data = exporter.export_to_dict()

        assert "metadata" in data
        assert "timesteps" in data
        assert len(data["timesteps"]) == 1
        assert "total_normal_force" in data["timesteps"][0]

    def test_export_to_csv_data(self, model_with_data: GripContactModel) -> None:
        """Should export as flat dictionaries for CSV."""
        exporter = GripContactExporter(model_with_data)
        exporter.capture_timestep()

        csv_data = exporter.export_to_csv_data()

        assert len(csv_data) == 1
        assert "timestamp" in csv_data[0]
        assert "cop_x" in csv_data[0]
        assert "cop_y" in csv_data[0]
        assert "cop_z" in csv_data[0]

    def test_grip_contact_model_summary_statistics(
        self, model_with_data: GripContactModel
    ) -> None:
        """Should compute summary statistics."""
        exporter = GripContactExporter(model_with_data)

        # Capture multiple timesteps by re-updating model
        for t in range(3):
            positions = np.array([[0.0, 0.0, 0.0]])
            normals = np.array([[0.0, 0.0, 1.0]])
            forces = np.array([[0.0, 0.0, 50.0 + t * 10]])
            velocities = np.zeros((1, 3))

            model_with_data.update_from_mujoco(
                positions, normals, forces, velocities, ["hand"], timestamp=t * 0.1
            )
            exporter.capture_timestep()

        summary = exporter.get_summary_statistics()

        assert "force_mean" in summary
        assert "force_max" in summary
        assert "num_timesteps" in summary
        assert summary["num_timesteps"] == 3

    def test_reset_clears_data(self, model_with_data: GripContactModel) -> None:
        """Reset should clear captured timesteps."""
        exporter = GripContactExporter(model_with_data)
        exporter.capture_timestep()

        assert len(exporter.timesteps) > 0

        exporter.reset()

        assert len(exporter.timesteps) == 0
