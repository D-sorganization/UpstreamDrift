"""Tests for security vulnerabilities in C3D data reader."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.c3d_reader import C3DDataReader


def _stub_reader_with_points_mock(
    marker_labels: tuple[str, ...] = ("Marker1",),
) -> C3DDataReader:
    """Create a stubbed reader with synthetic point data."""
    frame_count = 1
    point_rate = 100.0
    points = np.zeros((4, len(marker_labels), frame_count))

    reader = C3DDataReader(Path("synthetic"))
    reader._c3d_data = {
        "data": {"points": points, "analogs": np.zeros((1, 0, frame_count))},
        "parameters": {
            "POINT": {
                "LABELS": {"value": list(marker_labels)},
                "FRAMES": {"value": [frame_count]},
                "RATE": {"value": [point_rate]},
                "UNITS": {"value": ["m"]},
            },
        },
    }
    return reader


def test_csv_injection_prevention_default(tmp_path: Path) -> None:
    """
    Test that CSV injection is prevented by sanitizing output by default.
    """
    malicious_label = "=SUM(1+1)"
    reader = _stub_reader_with_points_mock(marker_labels=(malicious_label,))

    export_path = tmp_path / "sanitized.csv"
    reader.export_points(export_path)

    assert export_path.exists()

    # Read back with pandas to verify value
    df = pd.read_csv(export_path)

    # Verify the marker column value is escaped
    # The 'marker' column should be the second column (index 1) or named 'marker'
    assert "marker" in df.columns
    actual_label = df["marker"].iloc[0]

    expected_escaped = f"'{malicious_label}"
    assert actual_label == expected_escaped
    assert str(actual_label).startswith("'")


def test_csv_injection_allowed_when_requested(tmp_path: Path) -> None:
    """
    Test that CSV injection is allowed (raw output) when sanitize=False.
    """
    malicious_label = "=SUM(1+1)"
    reader = _stub_reader_with_points_mock(marker_labels=(malicious_label,))

    export_path = tmp_path / "raw.csv"
    reader.export_points(export_path, sanitize=False)

    assert export_path.exists()

    # Read back with pandas to verify value
    df = pd.read_csv(export_path)

    assert "marker" in df.columns
    actual_label = df["marker"].iloc[0]

    # Should be the raw malicious label
    assert actual_label == malicious_label
    assert not str(actual_label).startswith("'")
