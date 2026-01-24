"""Integration tests for C3D workflow: Ingest -> Analysis -> GUI.

TEST-004: Added @pytest.mark.integration markers for test categorization.
"""

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.shared.python.engine_availability import PYQT6_AVAILABLE
from src.shared.python.gui_utils import get_qapp
from src.shared.python.path_utils import get_simscape_model_path

# Mark all tests in this file as integration tests
pytestmark: list[pytest.MarkDecorator] = [pytest.mark.integration]

# Add source path for imports using centralized path utility
SRC_PATH = get_simscape_model_path("3D_Golf_Model")
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import C3D modules if PyQt6 is available
C3D_IMPORTS_AVAILABLE = False
if PYQT6_AVAILABLE:
    try:
        from apps.c3d_viewer import C3DDataModel, C3DViewerMainWindow
        from c3d_reader import C3DDataReader

        C3D_IMPORTS_AVAILABLE = True
    except (ImportError, OSError):
        pass

if not C3D_IMPORTS_AVAILABLE:
    C3DDataModel = None  # type: ignore[misc, assignment]
    C3DViewerMainWindow = None  # type: ignore[misc, assignment]
    C3DDataReader = None  # type: ignore[misc, assignment]
    pytestmark = [
        pytest.mark.integration,
        pytest.mark.skip(reason="PyQt6 GUI libraries or C3D modules not available"),
    ]


@pytest.fixture
def mock_c3d_file(tmp_path):
    """Create a dummy file path."""
    f = tmp_path / "test.c3d"
    f.touch()
    return f


@pytest.fixture
def mock_ezc3d():
    """Mock ezc3d module behavior."""
    with patch("c3d_reader.ezc3d") as mock:
        # Construct a fake C3D structure
        # Points: (4, Nmarkers, Nframes)
        # Analogs: (1, Nchannels, Nframes) for 1 subframe per frame simple case
        frames = 10
        c3d_struct = {
            "parameters": {
                "POINT": {
                    "LABELS": {"value": ["Marker1", "Marker2"]},
                    "FRAMES": {"value": [frames]},
                    "RATE": {"value": [100.0]},
                    "UNITS": {"value": ["mm"]},
                },
                "ANALOG": {
                    "LABELS": {"value": ["Analog1"]},
                    "RATE": {"value": [1000.0]},
                    "UNITS": {"value": ["V"]},
                },
                "EVENT": {
                    "LABELS": {"value": ["Heel Strike"]},
                    "TIMES": {"value": [0.05]},
                },
            },
            "data": {
                "points": np.zeros((4, 2, frames)),
                "analogs": np.zeros((1, 1, frames)),
            },
        }
        # Populate trajectory for Marker1 X (0..9)
        c3d_struct["data"]["points"][0, 0, :] = np.arange(frames, dtype=float)  # type: ignore[index]
        # Populate analog values (constant 5.0)
        c3d_struct["data"]["analogs"][0, 0, :] = 5.0  # type: ignore[index]

        mock.c3d.return_value = c3d_struct
        yield mock


def test_reader_ingestion(mock_c3d_file, mock_ezc3d):
    """Test C3D reading and dataframe conversion."""
    reader = C3DDataReader(mock_c3d_file)
    meta = reader.get_metadata()

    assert meta.frame_count == 10
    assert meta.frame_rate == 100.0
    assert meta.marker_labels == ["Marker1", "Marker2"]
    assert meta.analog_units == ["V"]
    assert meta.events[0].label == "Heel Strike"

    df = reader.points_dataframe(include_time=True)
    assert len(df) == 20  # 2 markers * 10 frames
    assert "time" in df.columns
    assert "residual" in df.columns

    # Check values
    m1 = df[df["marker"] == "Marker1"]
    assert m1.iloc[1]["x"] == 1.0  # Frame 1 is 1.0


def test_unit_conversion(mock_c3d_file, mock_ezc3d):
    """Test unit scaling logic (mm -> m)."""
    reader = C3DDataReader(mock_c3d_file)

    # Request meters
    df_m = reader.points_dataframe(target_units="m")
    m1_data = df_m[df_m["marker"] == "Marker1"]["x"].values
    # Original 1.0 mm should be 0.001 m
    np.testing.assert_almost_equal(m1_data[1], 0.001)


def test_export_workflow(mock_c3d_file, mock_ezc3d, tmp_path):
    """Test export functionality."""
    reader = C3DDataReader(mock_c3d_file)
    out_csv = tmp_path / "output.csv"

    reader.export_points(out_csv, target_units="m")
    assert out_csv.exists()

    # Read back
    df = pd.read_csv(out_csv)
    assert len(df) == 20
    assert "x" in df.columns


@pytest.fixture(scope="session")
def qapp():
    """Manage a single QApplication instance for the test session."""

    app = get_qapp()
    yield app


def test_gui_load_logic(qapp, mock_c3d_file, mock_ezc3d):
    """Test GUI loading logic using the refactored path."""
    try:
        window = C3DViewerMainWindow()
    except Exception as e:
        pytest.skip(f"GUI initialization failed (headless environment?): {e}")

    # Load the model directly using the service function
    from apps.services.c3d_loader import load_c3d_file

    model = load_c3d_file(str(mock_c3d_file))

    # Verify C3DDataModel population
    assert isinstance(model, C3DDataModel)
    assert "Marker1" in model.markers
    assert "Analog1" in model.analog
    assert model.metadata["Units (POINT)"] == "mm"

    # Verify UI population
    window.model = model
    window._populate_ui_with_model()

    # Verify UI state through tabs
    assert window.marker_plot_tab.list_markers.count() == 2
    assert window.analog_plot_tab.list_analog.count() == 1

    # Verify plotting logic doesn't crash
    window.marker_plot_tab.list_markers.setCurrentRow(0)  # Select Marker1
    # Note: _populate_ui_with_model calls update_ui, which might need manual trigger in test if signals aren't connected
    # But C3DViewerMainWindow._populate_ui_with_model calls update_ui -> which should update tabs
    # Let's explicitly call the tab update method to be sure
    window.marker_plot_tab.update_plot()

    # Clean up
    window.close()
