"""Extended C3D reader tests: malformed inputs, edge cases, analog edges (#1062).

Design by Contract
------------------
- C3DEvent rejects empty labels
- C3DDataReader rejects empty file paths
- C3DMetadata validates analog_units/labels length parity
- Export rejects unsupported formats
- Analog-only files produce valid empty marker DataFrames
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sidekick.lab.bio.c3d_reader import (
    C3DDataReader,
    C3DEvent,
    C3DMetadata,
)

# ---------------------------------------------------------------------------
# C3DEvent validation
# ---------------------------------------------------------------------------


class TestC3DEventValidation:
    """Tests for C3DEvent data class validation."""

    def test_valid_event(self) -> None:
        """A labeled event with positive time is valid."""
        event = C3DEvent(label="HeelStrike", time=1.23)
        assert event.label == "HeelStrike"
        assert event.time == pytest.approx(1.23)

    def test_empty_label_raises(self) -> None:
        """Empty label must raise ValueError."""
        with pytest.raises(ValueError, match="label"):
            C3DEvent(label="", time=0.0)

    def test_negative_time_allowed(self) -> None:
        """Negative time (pre-trigger) is allowed per C3D spec."""
        event = C3DEvent(label="PreTrigger", time=-0.5)
        assert event.time == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# C3DMetadata validation
# ---------------------------------------------------------------------------


class TestC3DMetadataValidation:
    """Tests for C3DMetadata constraints."""

    def test_analog_length_mismatch_raises(self) -> None:
        """analog_units and analog_labels must have same length."""
        with pytest.raises(ValueError, match="same length"):
            C3DMetadata(
                marker_labels=["M1"],
                frame_count=100,
                frame_rate=120.0,
                units="mm",
                analog_labels=["Ch1", "Ch2"],
                analog_units=["V"],  # mismatch!
                analog_rate=1200.0,
                events=[],
            )

    def test_duration_zero_rate(self) -> None:
        """Duration is 0 when frame_rate is 0."""
        meta = C3DMetadata(
            marker_labels=["M1"],
            frame_count=100,
            frame_rate=0.0,
            units="mm",
            analog_labels=[],
            analog_units=[],
            analog_rate=None,
            events=[],
        )
        assert meta.duration == 0.0

    def test_duration_correct(self) -> None:
        """Duration = frame_count / frame_rate."""
        meta = C3DMetadata(
            marker_labels=["M1"],
            frame_count=240,
            frame_rate=120.0,
            units="mm",
            analog_labels=[],
            analog_units=[],
            analog_rate=None,
            events=[],
        )
        assert meta.duration == pytest.approx(2.0)

    def test_marker_count(self) -> None:
        meta = C3DMetadata(
            marker_labels=["M1", "M2", "M3"],
            frame_count=10,
            frame_rate=100.0,
            units="mm",
            analog_labels=[],
            analog_units=[],
            analog_rate=None,
            events=[],
        )
        assert meta.marker_count == 3

    def test_analog_count(self) -> None:
        meta = C3DMetadata(
            marker_labels=[],
            frame_count=10,
            frame_rate=100.0,
            units="mm",
            analog_labels=["Ch1", "Ch2"],
            analog_units=["V", "N"],
            analog_rate=1200.0,
            events=[],
        )
        assert meta.analog_count == 2


# ---------------------------------------------------------------------------
# C3DDataReader contract tests
# ---------------------------------------------------------------------------


class TestC3DDataReaderContracts:
    """DbC tests for C3DDataReader.__init__."""

    def test_empty_path_raises(self) -> None:
        """Empty file path must raise a precondition error."""
        with pytest.raises((ValueError, Exception)):
            C3DDataReader("")

    def test_valid_path_stores(self) -> None:
        reader = C3DDataReader("sample.c3d")
        assert reader.file_path == Path("sample.c3d")


# ---------------------------------------------------------------------------
# Analog edge cases
# ---------------------------------------------------------------------------


class TestAnalogEdgeCases:
    """Tests for analog channel edge cases."""

    @pytest.fixture()
    def mock_ezc3d(self):
        with patch("sidekick.lab.bio.c3d_reader.ezc3d") as mock:
            yield mock

    @pytest.fixture()
    def c3d_no_analog(self):
        """C3D data with markers but no analog channels."""
        return {
            "parameters": {
                "POINT": {
                    "LABELS": {"value": ["M1"]},
                    "FRAMES": {"value": [50]},
                    "RATE": {"value": [100.0]},
                    "UNITS": {"value": ["mm"]},
                },
                "ANALOG": {
                    "LABELS": {"value": []},
                    "UNITS": {"value": []},
                    "RATE": {"value": [0.0]},
                },
                "EVENT": {
                    "LABELS": {"value": []},
                    "TIMES": {"value": [[], []]},
                },
            },
            "data": {
                "points": np.random.rand(4, 1, 50),
                "analogs": np.empty((1, 0, 0)),
            },
        }

    @pytest.fixture()
    def c3d_many_analog(self):
        """C3D data with many analog channels (high analog ratio)."""
        n_analog = 32
        n_frames = 100
        analog_ratio = 10  # 10x oversampled
        return {
            "parameters": {
                "POINT": {
                    "LABELS": {"value": ["M1"]},
                    "FRAMES": {"value": [n_frames]},
                    "RATE": {"value": [100.0]},
                    "UNITS": {"value": ["mm"]},
                },
                "ANALOG": {
                    "LABELS": {"value": [f"Ch{i}" for i in range(n_analog)]},
                    "UNITS": {"value": ["V"] * n_analog},
                    "RATE": {"value": [1000.0]},
                },
                "EVENT": {
                    "LABELS": {"value": []},
                    "TIMES": {"value": [[], []]},
                },
            },
            "data": {
                "points": np.random.rand(4, 1, n_frames),
                "analogs": np.random.rand(1, n_analog, n_frames * analog_ratio),
            },
        }

    def test_no_analog_metadata(self, mock_ezc3d, c3d_no_analog) -> None:
        """File with no analog channels reports 0 analog count."""
        mock_ezc3d.c3d.return_value = c3d_no_analog
        with patch("pathlib.Path.exists", return_value=True):
            reader = C3DDataReader("test.c3d")
            meta = reader.get_metadata()
            assert meta.analog_count == 0

    def test_high_analog_count(self, mock_ezc3d, c3d_many_analog) -> None:
        """File with 32 analog channels at 10x oversampling."""
        mock_ezc3d.c3d.return_value = c3d_many_analog
        with patch("pathlib.Path.exists", return_value=True):
            reader = C3DDataReader("test.c3d")
            meta = reader.get_metadata()
            assert meta.analog_count == 32
            assert meta.analog_rate == 1000.0

    def test_analog_dataframe_many_channels(self, mock_ezc3d, c3d_many_analog) -> None:
        """Analog DataFrame includes all channels."""
        mock_ezc3d.c3d.return_value = c3d_many_analog
        with patch("pathlib.Path.exists", return_value=True):
            reader = C3DDataReader("test.c3d")
            df = reader.analog_dataframe()
            unique_channels = set(df["channel"].unique())
            assert len(unique_channels) == 32
