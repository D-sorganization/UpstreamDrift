"""test_c3d_reader_fixed.py module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Adjust import path if needed based on where pytest is running from
from upstream_drift_tools.lab.bio.c3d_reader import C3DDataReader


class TestC3DDataReader:
    @pytest.fixture
    def mock_ezc3d(self):
        with patch("upstream_drift_tools.lab.bio.c3d_reader.ezc3d") as mock:
            yield mock

    @pytest.fixture
    def sample_c3d_data(self):
        # Create a mock C3D structure matching ezc3d output
        return {
            "parameters": {
                "POINT": {
                    "LABELS": {"value": ["Marker1", "Marker2"]},
                    "FRAMES": {"value": [100]},
                    "RATE": {"value": [120.0]},
                    "UNITS": {"value": ["mm"]},
                },
                "ANALOG": {
                    "LABELS": {"value": ["Analog1", "Analog2"]},
                    "UNITS": {"value": ["V", "V"]},
                    "RATE": {"value": [1200.0]},
                },
                "EVENT": {
                    "LABELS": {"value": ["Start", "End"]},
                    "TIMES": {"value": [[0.0, 10.0], [0.5, 9.5]]},
                },
            },
            "data": {
                "points": np.random.rand(4, 2, 100),  # 4x2x100 array
                "analogs": np.random.rand(1, 2, 100),  # 1x2x100 array
            },
        }

    def test_initialization(self):
        reader = C3DDataReader("test.c3d")
        assert reader.file_path == Path("test.c3d")

    def test_metadata_extraction(self, mock_ezc3d, sample_c3d_data):
        mock_ezc3d.c3d.return_value = sample_c3d_data

        # Mock file existence
        with patch("pathlib.Path.exists", return_value=True):
            reader = C3DDataReader("test.c3d")
            metadata = reader.get_metadata()

            assert metadata.frame_count == 100
            assert metadata.frame_rate == 120.0
            assert metadata.marker_labels == ["Marker1", "Marker2"]
            assert len(metadata.events) == 2
            assert abs(metadata.duration - 100 / 120.0) < 1e-6

    def test_points_dataframe(self, mock_ezc3d, sample_c3d_data):
        mock_ezc3d.c3d.return_value = sample_c3d_data

        with patch("pathlib.Path.exists", return_value=True):
            reader = C3DDataReader("test.c3d")
            df = reader.points_dataframe()

            assert not df.empty
            assert "x" in df.columns
            assert "y" in df.columns
            assert "z" in df.columns
            assert "marker" in df.columns
            assert set(df["marker"].unique()) == {"Marker1", "Marker2"}

    def test_analog_dataframe(self, mock_ezc3d, sample_c3d_data):
        mock_ezc3d.c3d.return_value = sample_c3d_data

        with patch("pathlib.Path.exists", return_value=True):
            reader = C3DDataReader("test.c3d")
            df = reader.analog_dataframe()

            assert not df.empty
            assert "channel" in df.columns
            assert "value" in df.columns
            assert set(df["channel"].unique()) == {"Analog1", "Analog2"}
