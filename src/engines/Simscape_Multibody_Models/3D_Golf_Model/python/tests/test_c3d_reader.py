"""Tests for C3D data loading utilities."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from src.c3d_reader import C3DDataReader, C3DEvent, load_tour_average_reader
from src.shared.python.engine_availability import skip_if_unavailable

# Skip tests if ezc3d is not available (e.g., Python 3.9)
pytestmark = skip_if_unavailable("ezc3d")

EXPECTED_MARKER_COUNT = 38
EXPECTED_FRAME_COUNT = 654
EXPECTED_FRAME_RATE_HZ = 360.0
EXPECTED_POINT_UNITS = "m"
EXPECTED_ANALOG_COUNT = 0


def _tour_average_reader() -> C3DDataReader:
    """Create a C3DDataReader for the tour average test file."""
    repository_root = Path(__file__).resolve().parents[2]
    return load_tour_average_reader(repository_root)


def _stub_reader_with_points(
    *,
    frame_count: int = 2,
    marker_labels: tuple[str, ...] = ("Marker1", "Marker2"),
    point_units: str = "m",
    point_rate: int = 100,
    analog_array: npt.NDArray[np.floating[Any]] | None = None,
    analog_parameters: dict[str, Any] | None = None,
) -> C3DDataReader:
    """Create a stubbed reader with synthetic point data for isolated testing."""

    points = np.zeros((4, len(marker_labels), frame_count))
    analogs = (
        analog_array if analog_array is not None else np.zeros((1, 0, frame_count))
    )
    reader = C3DDataReader(Path("synthetic"))
    reader._c3d_data = {
        "data": {"points": points, "analogs": analogs},
        "parameters": {
            "POINT": {
                "LABELS": {"value": list(marker_labels)},
                "FRAMES": {"value": [frame_count]},
                "RATE": {"value": [point_rate]},
                "UNITS": {"value": [point_units]},
            },
            **({"ANALOG": analog_parameters} if analog_parameters else {}),
        },
    }
    return reader


def test_metadata_matches_expected_capture() -> None:
    """Test that metadata extraction matches expected values from the test file."""
    reader = _tour_average_reader()
    metadata = reader.get_metadata()

    assert metadata.marker_count == EXPECTED_MARKER_COUNT
    assert metadata.frame_count == EXPECTED_FRAME_COUNT
    assert metadata.frame_rate == pytest.approx(EXPECTED_FRAME_RATE_HZ)
    assert metadata.units == EXPECTED_POINT_UNITS
    assert metadata.analog_count == EXPECTED_ANALOG_COUNT
    assert metadata.duration == pytest.approx(
        EXPECTED_FRAME_COUNT / EXPECTED_FRAME_RATE_HZ
    )
    assert metadata.marker_labels[1] == "WaistLeft"
    assert metadata.events == []


def test_points_dataframe_shape_and_columns() -> None:
    """Test that points dataframe has correct shape and columns."""
    reader = _tour_average_reader()
    dataframe = reader.points_dataframe()

    expected_rows = EXPECTED_FRAME_COUNT * EXPECTED_MARKER_COUNT
    assert dataframe.shape[0] == expected_rows
    assert dataframe.shape[1] in (6, 7)  # With or without the optional time column
    expected_columns = {"frame", "marker", "x", "y", "z", "residual"}
    assert expected_columns.issubset(set(dataframe.columns))


def test_marker_time_series_is_ordered_and_numeric() -> None:
    """Test that marker time series data is properly ordered and numeric."""
    reader = _tour_average_reader()
    dataframe = reader.points_dataframe()
    waist_left_frames = dataframe[dataframe["marker"] == "WaistLeft"].reset_index(
        drop=True
    )

    assert waist_left_frames["frame"].is_monotonic_increasing
    assert waist_left_frames[["x", "y", "z"]].apply(np.isfinite).all().all()
    if "time" in waist_left_frames.columns:
        assert waist_left_frames["time"].is_monotonic_increasing
        assert waist_left_frames.loc[1, "time"] - waist_left_frames.loc[
            0, "time"
        ] == pytest.approx(1 / EXPECTED_FRAME_RATE_HZ)


def test_marker_subset_and_unit_conversion() -> None:
    """Test marker filtering and unit conversion functionality."""
    reader = _tour_average_reader()
    dataframe_meters = reader.points_dataframe(
        markers=["WaistLeft"], include_time=False
    )
    dataframe_mm = reader.points_dataframe(
        markers=["WaistLeft"], include_time=False, target_units="mm"
    )

    assert set(dataframe_meters["marker"].unique()) == {"WaistLeft"}
    assert set(dataframe_mm["marker"].unique()) == {"WaistLeft"}

    finite_m = dataframe_meters[["x", "y", "z"]].stack().dropna().iloc[0]
    finite_mm = dataframe_mm[["x", "y", "z"]].stack().dropna().iloc[0]
    assert finite_mm == pytest.approx(finite_m * 1000.0)


def test_residual_filtering_sets_noisy_points_to_nan() -> None:
    """Test that residual filtering correctly sets noisy points to NaN."""
    reader = _tour_average_reader()
    dataframe = reader.points_dataframe(residual_nan_threshold=0.5)

    assert dataframe[["x", "y", "z"]].isna().all().all()


def test_points_dataframe_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Ensure missing capture files raise a clear FileNotFoundError."""

    reader = C3DDataReader(tmp_path / "does_not_exist.c3d")

    with pytest.raises(FileNotFoundError):
        reader.points_dataframe()


def test_get_point_parameters_missing_section_raises_value_error() -> None:
    """Missing POINT parameters should fail fast with a descriptive error."""

    reader = C3DDataReader(Path("incomplete"))
    reader._c3d_data = {
        "data": {"points": np.zeros((4, 1, 1)), "analogs": np.zeros((1, 0, 1))},
        "parameters": {},
    }

    with pytest.raises(ValueError):
        reader.get_metadata()


def test_analog_dataframe_handles_missing_channels_gracefully() -> None:
    """Test that analog dataframe handles missing channels without errors."""
    reader = _tour_average_reader()
    analog_df = reader.analog_dataframe()

    assert analog_df.empty
    assert list(analog_df.columns) == ["sample", "time", "channel", "value"]


def test_analog_dataframe_orders_samples_and_channels() -> None:
    """Test that analog dataframe correctly orders samples and channels."""
    reader = C3DDataReader(Path("dummy"))
    analog_array = np.array(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            [
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
        ]
    )
    points = np.zeros((4, 1, 3))
    reader._c3d_data = {
        "data": {"points": points, "analogs": analog_array},
        "parameters": {
            "POINT": {
                "LABELS": {"value": ["Marker1"]},
                "FRAMES": {"value": [3]},
                "RATE": {"value": [120]},
                "UNITS": {"value": ["m"]},
            },
            "ANALOG": {
                "LABELS": {"value": ["A1", "A2"]},
                "RATE": {"value": [240]},
            },
        },
    }

    analog_df = reader.analog_dataframe()

    assert list(analog_df.columns) == ["sample", "time", "channel", "value"]
    assert analog_df["sample"].is_monotonic_increasing
    assert analog_df[analog_df["sample"] == 1]["time"].iat[0] == pytest.approx(1 / 240)

    pivoted = analog_df.pivot(index="sample", columns="channel", values="value")
    expected = np.array(
        [
            [1.0, 4.0],
            [7.0, 10.0],
            [2.0, 5.0],
            [8.0, 11.0],
            [3.0, 6.0],
            [9.0, 12.0],
        ]
    )
    np.testing.assert_allclose(pivoted.to_numpy(), expected)


def test_points_export_supports_multiple_formats(tmp_path: Path) -> None:
    """Test that points can be exported to CSV, JSON, and NPZ formats."""
    reader = _tour_average_reader()
    export_dir = tmp_path / "exports"

    csv_path = reader.export_points(export_dir / "points.csv", markers=["WaistLeft"])
    json_path = reader.export_points(export_dir / "points.json", include_time=False)
    npz_path = reader.export_points(
        export_dir / "points_archive.npz", residual_nan_threshold=0.5
    )

    assert csv_path.exists()
    csv_file_size = csv_path.stat().st_size
    assert csv_file_size > 0

    assert json_path.exists()
    assert json_path.stat().st_size > 0

    assert npz_path.exists()
    with np.load(npz_path) as archive:
        assert set(archive.files) >= {"frame", "marker", "x", "y", "z", "residual"}


def test_analog_export_writes_empty_structure(tmp_path: Path) -> None:
    """Test that analog export creates proper file structure even with empty data."""
    reader = _tour_average_reader()
    path = reader.export_analog(tmp_path / "analog.csv")

    assert path.exists()
    contents = path.read_text(encoding="utf-8").strip().splitlines()
    assert contents[0].split(",") == ["sample", "time", "channel", "value"]


def test_export_points_requires_inferable_format(tmp_path: Path) -> None:
    """Exporting without a known extension should raise a ValueError."""

    reader = _stub_reader_with_points()

    with pytest.raises(ValueError):
        reader.export_points(tmp_path / "points")


def test_unit_scale_rejects_unsupported_units() -> None:
    """Unsupported unit conversions should raise ValueError rather than mis-scale."""

    with pytest.raises(ValueError):
        C3DDataReader._unit_scale("cm", "m")


def test_get_events_parses_event_times() -> None:
    """Event parsing should produce trimmed labels with finite times only."""

    reader = _stub_reader_with_points()
    reader._c3d_data["parameters"]["EVENT"] = {
        "LABELS": {"value": [" Foot Strike", "Follow Through "]},
        "TIMES": {"value": [[0.0, 1.2], [0.5, 1.7]]},
    }

    events = reader._get_events()

    assert events == [
        C3DEvent(label="Foot Strike", time=0.5),
        C3DEvent(label="Follow Through", time=1.7),
    ]


def test_analog_dataframe_fills_default_labels_without_rate() -> None:
    """Analog channels without labels should receive stable defaults."""

    analog_array = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ]
    )
    reader = _stub_reader_with_points(frame_count=3, marker_labels=("M1",))
    reader._c3d_data["data"]["analogs"] = analog_array

    analog_df = reader.analog_dataframe()

    assert list(analog_df.columns) == ["sample", "channel", "value"]
    assert set(analog_df["channel"].unique()) == {"Analog_1", "Analog_2"}
    assert analog_df.iloc[0]["value"] == pytest.approx(1.0)


def test_points_dataframe_sorts_by_time_and_marker() -> None:
    """Points DataFrame should be ordered by time then marker name when requested."""

    reader = _stub_reader_with_points(
        marker_labels=("B", "A"),
        frame_count=3,
        point_rate=50,
    )

    dataframe = reader.points_dataframe()

    assert dataframe["time"].is_monotonic_increasing
    grouped = dataframe.groupby("time")["marker"].apply(list)
    assert all(markers == sorted(markers) for markers in grouped)


def test_export_points_rejects_unknown_format(tmp_path: Path) -> None:
    """Export should fail fast when provided an unsupported file format string."""

    reader = _stub_reader_with_points()

    with pytest.raises(ValueError):
        reader.export_points(tmp_path / "points.out", file_format="txt")

    assert not (tmp_path / "points.out").exists()


def test_get_events_handles_missing_times() -> None:
    """Event labels without time data should return an empty list."""

    reader = _stub_reader_with_points()
    reader._c3d_data["parameters"]["EVENT"] = {"LABELS": {"value": ["A", "B"]}}

    assert reader._get_events() == []


def test_analog_export_writes_time_when_rate_available(tmp_path: Path) -> None:
    """Analog export should include the time column when a rate is defined."""

    analog_array = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    analog_parameters = {"LABELS": {"value": ["Ch1", "Ch2"]}, "RATE": {"value": [200]}}
    reader = _stub_reader_with_points(
        analog_array=analog_array, analog_parameters=analog_parameters, frame_count=2
    )

    path = reader.export_analog(tmp_path / "analog.csv")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == ["sample", "time", "channel", "value"]
    assert lines[1].startswith("0,0.0,Ch1,1.0")
    assert lines[2].split(",")[1] == "0.0"
    assert lines[3].split(",")[1] == "0.005"


def test_points_dataframe_handles_zero_frame_rate() -> None:
    """Test that zero frame rate handles gracefully (omits time column)."""
    reader = _stub_reader_with_points(point_rate=0)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        dataframe = reader.points_dataframe(include_time=True)

        # Ensure no runtime warnings (division by zero)
        runtime_warnings = [w for w in record if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0

    assert "time" not in dataframe.columns
