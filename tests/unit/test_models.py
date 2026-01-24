"""Unit tests for C3D Viewer data models."""

import numpy as np

from src.shared.python.path_utils import get_simscape_model_path, setup_import_paths

# Setup import path using centralized utility
setup_import_paths(additional_paths=[get_simscape_model_path()])

from apps.core.models import AnalogData, C3DDataModel, MarkerData  # noqa: E402


class TestModels:
    """Tests for data model integrity and methods."""

    def test_marker_data_initialization(self):
        """Test MarkerData creation and defaults."""
        pos = np.zeros((10, 3))
        res = np.zeros((10,))
        marker = MarkerData(name="TEST", position=pos, residuals=res)

        assert marker.name == "TEST"
        assert np.array_equal(marker.position, pos)
        assert np.array_equal(marker.residuals, res)

    def test_marker_data_optional_residuals(self):
        """Test MarkerData without residuals."""
        pos = np.zeros((10, 3))
        marker = MarkerData(name="TEST", position=pos)

        assert marker.residuals is None

    def test_analog_data_initialization(self):
        """Test AnalogData creation and defaults."""
        vals = np.zeros((100,))
        analog = AnalogData(name="EMG1", values=vals, unit="V")

        assert analog.name == "EMG1"
        assert np.array_equal(analog.values, vals)
        assert analog.unit == "V"

    def test_analog_data_default_unit(self):
        """Test AnalogData default unit."""
        vals = np.zeros((100,))
        analog = AnalogData(name="EMG1", values=vals)

        assert analog.unit == ""

    def test_c3d_data_model_methods(self):
        """Test C3DDataModel helper methods."""
        # Setup
        marker_a = MarkerData("HEAD", np.zeros((10, 3)))
        marker_b = MarkerData("TOE", np.zeros((10, 3)))
        markers = {"HEAD": marker_a, "TOE": marker_b}

        analog_1 = AnalogData("Force", np.zeros((100,)))
        analog = {"Force": analog_1}

        model = C3DDataModel(
            filepath="test.c3d",
            markers=markers,
            analog=analog,
            point_rate=100.0,
            analog_rate=1000.0,
        )

        # Verify names
        m_names = model.marker_names()
        assert len(m_names) == 2
        assert "HEAD" in m_names
        assert "TOE" in m_names

        a_names = model.analog_names()
        assert len(a_names) == 1
        assert "Force" in a_names

    def test_c3d_data_model_defaults(self):
        """Test C3DDataModel default fields."""
        model = C3DDataModel(filepath="empty.c3d")

        assert model.markers == {}
        assert model.analog == {}
        assert model.point_rate == 0.0
        assert model.analog_rate == 0.0
        assert model.point_time is None
        assert model.analog_time is None
        assert model.metadata == {}
