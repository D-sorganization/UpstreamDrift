from typing import Any

"""test_c3d_reader_fixed.py module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Adjust import path if needed based on where pytest is running from
from sidekick.lab.bio import c3d_reader as c3d_reader_module
from sidekick.lab.bio.c3d_reader import C3DDataReader


class TestC3DDataReader:
    @pytest.fixture
    def mock_ezc3d(self) -> Any:
        with patch.object(c3d_reader_module, "ezc3d") as mock:
            yield mock

    @pytest.fixture
    def valid_c3d_path(self, tmp_path: Path) -> Path:
        c3d_path = tmp_path / "test.c3d"
        c3d_path.write_bytes(b"\x02\x50")
        return c3d_path

    @pytest.fixture
    def sample_c3d_data(self) -> Any:
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

    def test_initialization(self) -> Any:
        reader = C3DDataReader("test.c3d")
        assert reader.file_path == Path("test.c3d")

    def test_metadata_extraction(
        self, mock_ezc3d: Any, sample_c3d_data: Any, valid_c3d_path: Path
    ) -> Any:
        mock_ezc3d.c3d.return_value = sample_c3d_data

        reader = C3DDataReader(valid_c3d_path)
        metadata = reader.get_metadata()

        assert metadata.frame_count == 100
        assert metadata.frame_rate == 120.0
        assert metadata.marker_labels == ["Marker1", "Marker2"]
        assert len(metadata.events) == 2
        assert abs(metadata.duration - 100 / 120.0) < 1e-6

    def test_points_dataframe(
        self, mock_ezc3d: Any, sample_c3d_data: Any, valid_c3d_path: Path
    ) -> Any:
        mock_ezc3d.c3d.return_value = sample_c3d_data

        reader = C3DDataReader(valid_c3d_path)
        df = reader.points_dataframe()

        assert not df.empty
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns
        assert "marker" in df.columns
        assert set(df["marker"].unique()) == {"Marker1", "Marker2"}

    def test_analog_dataframe(
        self, mock_ezc3d: Any, sample_c3d_data: Any, valid_c3d_path: Path
    ) -> Any:
        mock_ezc3d.c3d.return_value = sample_c3d_data

        reader = C3DDataReader(valid_c3d_path)
        df = reader.analog_dataframe()

        assert not df.empty
        assert "channel" in df.columns
        assert "value" in df.columns
        assert set(df["channel"].unique()) == {"Analog1", "Analog2"}

    def test_invalid_c3d_header(self, tmp_path: Path) -> Any:
        invalid_path = tmp_path / "invalid.c3d"
        invalid_path.write_bytes(b"\x02\x00")
        reader = C3DDataReader(invalid_path)
        with (
            patch.object(c3d_reader_module, "ezc3d", object()),
            pytest.raises(ValueError, match="Not a valid C3D file"),
        ):
            reader.get_metadata()
