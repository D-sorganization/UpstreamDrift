"""Tests for C3D force plate parsing functionality.

Implements Guideline E5: Ground Reaction Forces.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Setup import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_PATH = os.path.join(
    PROJECT_ROOT, "engines/Simscape_Multibody_Models/3D_Golf_Model/python/src"
)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from c3d_reader import C3DDataReader, C3DMetadata  # noqa: E402


class TestForcePlateChannelDetection:
    """Tests for force plate channel detection logic."""

    @pytest.fixture
    def mock_reader(self) -> C3DDataReader:
        """Create a reader with mocked methods."""
        reader = C3DDataReader("test_capture.c3d")
        return reader

    def test_standard_channel_naming(self, mock_reader: C3DDataReader) -> None:
        """Test detection of standard Fx1, Fy1 naming convention."""
        # Mock metadata with standard force plate labels
        mock_meta = C3DMetadata(
            marker_labels=["Marker1"],
            frame_count=100,
            frame_rate=100.0,
            units="mm",
            analog_labels=[
                "Fx1",
                "Fy1",
                "Fz1",
                "Mx1",
                "My1",
                "Mz1",
                "Fx2",
                "Fy2",
                "Fz2",
                "Mx2",
                "My2",
                "Mz2",
            ],
            analog_units=["N"] * 12,
            analog_rate=1000.0,
            events=[],
        )
        mock_reader.get_metadata = MagicMock(return_value=mock_meta)

        channels = mock_reader.get_force_plate_channels()

        assert 1 in channels
        assert 2 in channels
        assert channels[1]["fx"] == "Fx1"
        assert channels[1]["fy"] == "Fy1"
        assert channels[1]["fz"] == "Fz1"
        assert channels[1]["mx"] == "Mx1"
        assert channels[1]["my"] == "My1"
        assert channels[1]["mz"] == "Mz1"
        assert channels[2]["fx"] == "Fx2"

    def test_prefixed_channel_naming(self, mock_reader: C3DDataReader) -> None:
        """Test detection of Force.Fx1 prefixed naming."""
        mock_meta = C3DMetadata(
            marker_labels=[],
            frame_count=100,
            frame_rate=100.0,
            units="mm",
            analog_labels=[
                "Force.Fx1",
                "Force.Fy1",
                "Force.Fz1",
                "Force.Mx1",
                "Force.My1",
                "Force.Mz1",
            ],
            analog_units=["N"] * 6,
            analog_rate=1000.0,
            events=[],
        )
        mock_reader.get_metadata = MagicMock(return_value=mock_meta)

        channels = mock_reader.get_force_plate_channels()

        assert 1 in channels
        assert len(channels) == 1
        assert channels[1]["fx"] == "Force.Fx1"

    def test_prefix_style_naming(self, mock_reader: C3DDataReader) -> None:
        """Test detection of FP1_Fx prefix style."""
        mock_meta = C3DMetadata(
            marker_labels=[],
            frame_count=100,
            frame_rate=100.0,
            units="mm",
            analog_labels=[
                "FP1Fx",
                "FP1Fy",
                "FP1Fz",
                "FP1Mx",
                "FP1My",
                "FP1Mz",
            ],
            analog_units=["N"] * 6,
            analog_rate=1000.0,
            events=[],
        )
        mock_reader.get_metadata = MagicMock(return_value=mock_meta)

        channels = mock_reader.get_force_plate_channels()

        assert 1 in channels
        assert channels[1]["fx"] == "FP1Fx"

    def test_no_force_plate_channels(self, mock_reader: C3DDataReader) -> None:
        """Test handling when no force plate channels are present."""
        mock_meta = C3DMetadata(
            marker_labels=["Marker1"],
            frame_count=100,
            frame_rate=100.0,
            units="mm",
            analog_labels=["EMG1", "EMG2", "Accelerometer"],
            analog_units=["V", "V", "g"],
            analog_rate=1000.0,
            events=[],
        )
        mock_reader.get_metadata = MagicMock(return_value=mock_meta)

        channels = mock_reader.get_force_plate_channels()

        assert len(channels) == 0


class TestForcePlateDataframe:
    """Tests for force plate dataframe extraction."""

    @pytest.fixture
    def mock_reader_with_data(self) -> C3DDataReader:
        """Create a reader with mocked force plate data."""
        reader = C3DDataReader("test_capture.c3d")

        # Mock metadata
        mock_meta = C3DMetadata(
            marker_labels=["Marker1"],
            frame_count=10,
            frame_rate=100.0,
            units="mm",
            analog_labels=["Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1"],
            analog_units=["N", "N", "N", "Nm", "Nm", "Nm"],
            analog_rate=1000.0,
            events=[],
        )
        reader.get_metadata = MagicMock(return_value=mock_meta)

        # Mock analog dataframe with synthetic force plate data
        n_samples = 100
        sample_indices = np.arange(n_samples)

        # Create synthetic GRF data
        # Fx1: lateral force oscillating
        fx1 = 10.0 * np.sin(2 * np.pi * sample_indices / n_samples)
        # Fy1: AP force oscillating
        fy1 = 20.0 * np.sin(2 * np.pi * sample_indices / n_samples + 0.5)
        # Fz1: vertical force - 800N body weight
        fz1 = 800.0 + 100.0 * np.sin(2 * np.pi * sample_indices / n_samples)
        # Moments
        mx1 = 5.0 * np.sin(2 * np.pi * sample_indices / n_samples)
        my1 = 10.0 * np.sin(2 * np.pi * sample_indices / n_samples)
        mz1 = 2.0 * np.sin(2 * np.pi * sample_indices / n_samples)

        analog_df = pd.DataFrame(
            {
                "sample": np.repeat(sample_indices, 6),
                "channel": np.tile(
                    ["Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1"], n_samples
                ),
                "value": np.concatenate(
                    [np.column_stack([fx1, fy1, fz1, mx1, my1, mz1]).flatten()]
                ),
            }
        )

        reader.analog_dataframe = MagicMock(return_value=analog_df)

        return reader

    def test_force_plate_dataframe_basic(
        self, mock_reader_with_data: C3DDataReader
    ) -> None:
        """Test basic force plate dataframe extraction."""
        df = mock_reader_with_data.force_plate_dataframe(include_time=False)

        assert "sample" in df.columns
        assert "plate" in df.columns
        assert "fx" in df.columns
        assert "fy" in df.columns
        assert "fz" in df.columns
        assert "mx" in df.columns
        assert "my" in df.columns
        assert "mz" in df.columns
        assert len(df) == 100  # 100 samples

    def test_force_plate_dataframe_with_time(
        self, mock_reader_with_data: C3DDataReader
    ) -> None:
        """Test force plate dataframe includes time column."""
        df = mock_reader_with_data.force_plate_dataframe(include_time=True)

        assert "time" in df.columns
        # With 1000 Hz sample rate, sample 0 should be at t=0
        assert df.iloc[0]["time"] == 0.0
        # Sample 10 should be at t=0.01s
        assert abs(df.iloc[10]["time"] - 0.01) < 1e-9

    def test_cop_computation(self, mock_reader_with_data: C3DDataReader) -> None:
        """Test center of pressure computation."""
        df = mock_reader_with_data.force_plate_dataframe(compute_cop=True)

        assert "cop_x" in df.columns
        assert "cop_y" in df.columns
        assert "cop_z" in df.columns

        # COP should be finite for non-zero Fz
        assert np.all(np.isfinite(df["cop_x"]))
        assert np.all(np.isfinite(df["cop_y"]))
        assert np.all(np.isfinite(df["cop_z"]))

    def test_cop_nan_when_no_contact(
        self, mock_reader_with_data: C3DDataReader
    ) -> None:
        """Test COP is NaN when vertical force is too small."""
        reader = mock_reader_with_data

        # Override analog data with near-zero Fz
        n_samples = 10
        analog_df = pd.DataFrame(
            {
                "sample": np.repeat(np.arange(n_samples), 6),
                "channel": np.tile(
                    ["Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1"], n_samples
                ),
                "value": np.tile([10.0, 20.0, 5.0, 1.0, 2.0, 0.5], n_samples),
                # Fz=5N is below 10N threshold
            }
        )
        reader.analog_dataframe = MagicMock(return_value=analog_df)

        df = reader.force_plate_dataframe(compute_cop=True)

        # All COP values should be NaN due to low Fz
        assert np.all(np.isnan(df["cop_x"]))
        assert np.all(np.isnan(df["cop_y"]))

    def test_specific_plate_extraction(
        self, mock_reader_with_data: C3DDataReader
    ) -> None:
        """Test extracting data for a specific plate number."""
        df = mock_reader_with_data.force_plate_dataframe(plate_number=1)

        assert all(df["plate"] == 1)

    def test_invalid_plate_number_raises(
        self, mock_reader_with_data: C3DDataReader
    ) -> None:
        """Test that requesting an invalid plate raises ValueError."""
        with pytest.raises(ValueError, match="Force plate 99 not found"):
            mock_reader_with_data.force_plate_dataframe(plate_number=99)

    def test_empty_when_no_force_plates(self) -> None:
        """Test graceful handling when no force plates detected."""
        reader = C3DDataReader("test.c3d")

        mock_meta = C3DMetadata(
            marker_labels=["M1"],
            frame_count=10,
            frame_rate=100.0,
            units="mm",
            analog_labels=["EMG1"],
            analog_units=["V"],
            analog_rate=1000.0,
            events=[],
        )
        reader.get_metadata = MagicMock(return_value=mock_meta)

        with patch.object(reader, "get_force_plate_channels", return_value={}):
            df = reader.force_plate_dataframe()

        assert len(df) == 0
        assert "fx" in df.columns  # Schema preserved


class TestForcePlateCount:
    """Tests for force plate count method."""

    def test_force_plate_count(self) -> None:
        """Test counting detected force plates."""
        reader = C3DDataReader("test.c3d")

        mock_meta = C3DMetadata(
            marker_labels=[],
            frame_count=10,
            frame_rate=100.0,
            units="mm",
            analog_labels=[
                "Fx1",
                "Fy1",
                "Fz1",
                "Mx1",
                "My1",
                "Mz1",
                "Fx2",
                "Fy2",
                "Fz2",
                "Mx2",
                "My2",
                "Mz2",
            ],
            analog_units=["N"] * 12,
            analog_rate=1000.0,
            events=[],
        )
        reader.get_metadata = MagicMock(return_value=mock_meta)

        assert reader.get_force_plate_count() == 2
