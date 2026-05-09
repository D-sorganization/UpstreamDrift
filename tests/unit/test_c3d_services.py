"""Unit tests for C3D services."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.shared.python.engine_core.engine_availability import (
    PYQT6_AVAILABLE,
    PYTEST_QT_AVAILABLE,
)

# Now we can import apps.*
try:
    from apps.core.models import C3DDataModel
    from apps.services.analysis import compute_marker_statistics
    from apps.services.c3d_loader import load_c3d_file

    # We also need to patch the full path correctly in tests
    LOADER_PATH = "apps.services.c3d_loader"
    THREAD_PATH = "apps.services.loader_thread"
    C3D_APPS_AVAILABLE = True
except ImportError:
    # Skip module if apps cannot be imported
    C3D_APPS_AVAILABLE = False

if not C3D_APPS_AVAILABLE:
    pytest.skip("Could not import C3D apps", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test Analysis Service
# ---------------------------------------------------------------------------


def test_compute_marker_statistics_basic() -> None:
    """Test basic stats computation."""
    t = np.array([0.0, 1.0, 2.0])
    pos = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=float)
    # dist: 0->1 = 1.0, 1->3 = 2.0
    # speed: 1.0/1.0=1.0, 2.0/1.0=2.0

    stats = compute_marker_statistics(t, pos)

    assert stats["path_length"] == pytest.approx(3.0)
    assert stats["max_speed"] == pytest.approx(2.0)
    assert stats["mean_speed"] == pytest.approx(1.5)


def test_compute_marker_statistics_empty() -> None:
    """Test stats with empty or single point."""
    t = np.array([0.0])
    pos = np.array([[0, 0, 0]])
    stats = compute_marker_statistics(t, pos)
    assert (
        np.isnan(stats["path_length"]) or stats["path_length"] == 0.0
    )  # Implementation detail: check code
    assert np.isnan(stats["max_speed"])


def test_compute_marker_statistics_nan_handling() -> None:
    """Test handling of NaN values."""
    t = np.array([0.0, 1.0])
    pos = np.array([[0, 0, 0], [np.nan, 0, 0]])
    stats = compute_marker_statistics(t, pos)
    # nansum should handle it? or propagate nan depending on numpy version/func
    # Our implementation uses nansum, but diff with nan results in nan.
    # nansum([nan]) -> 0.0
    assert stats["path_length"] == 0.0


# ---------------------------------------------------------------------------
# Test Loader Service
# ---------------------------------------------------------------------------


@patch("apps.services.c3d_loader.C3DDataReader")
@patch("os.path.exists")
def test_load_c3d_file_success(mock_exists, mock_reader_cls) -> None:
    """Test successful loading via service."""
    mock_exists.return_value = True
    mock_reader = mock_reader_cls.return_value

    # Mock metadata
    mock_meta = MagicMock()
    mock_meta.marker_labels = ["M1"]
    mock_meta.analog_labels = ["A1"]
    mock_meta.analog_units = ["V"]
    mock_meta.frame_rate = 100.0
    mock_meta.analog_rate = 1000.0
    mock_meta.frame_count = 10
    mock_meta.marker_count = 1
    mock_meta.events = []
    mock_reader.get_metadata.return_value = mock_meta

    # Mock dataframes
    import pandas as pd

    mock_reader.points_dataframe.return_value = pd.DataFrame(
        {
            "time": np.linspace(0, 0.1, 10),
            "marker": ["M1"] * 10,
            "x": np.zeros(10),
            "y": np.zeros(10),
            "z": np.zeros(10),
            "residual": np.zeros(10),
        }
    )

    mock_reader.analog_dataframe.return_value = pd.DataFrame(
        {
            "time": np.linspace(0, 0.1, 100),
            "channel": ["A1"] * 100,
            "value": np.ones(100),
        }
    )

    model = load_c3d_file("/fake/path.c3d")

    assert isinstance(model, C3DDataModel)
    assert model.filepath == "/fake/path.c3d"
    assert "M1" in model.markers
    assert "A1" in model.analog


@patch("os.path.exists")
def test_load_c3d_file_not_found(mock_exists) -> None:
    """Test file not found error."""
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        load_c3d_file("/nonexistent.c3d")


# ---------------------------------------------------------------------------
# Test Loader Thread
# ---------------------------------------------------------------------------
