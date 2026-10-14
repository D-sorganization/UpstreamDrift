"""Characterization tests for ``C3DDataReader.force_plate_dataframe``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from sidekick.lab.bio import c3d_reader as c3d_reader_module
from sidekick.lab.bio.c3d_reader import C3DDataReader

ANALOG_RATE = 1000.0
FRAME_COUNT = 4
PLATE_LABELS = [
    f"{kind}{axis}{plate}"
    for plate in (1, 2)
    for kind in ("F", "M")
    for axis in ("x", "y", "z")
]


def _c3d_data(labels: list[str]) -> dict[str, Any]:
    analogs = np.arange(len(labels) * FRAME_COUNT, dtype=float).reshape(
        1, len(labels), FRAME_COUNT
    )
    # Make Fz1 large enough (>10 N) for valid COP on plate 1, small on plate 2.
    if "Fz1" in labels:
        analogs[0, labels.index("Fz1"), :] = [100.0, 100.0, 5.0, 100.0]
    return {
        "parameters": {
            "POINT": {
                "LABELS": {"value": []},
                "FRAMES": {"value": [FRAME_COUNT]},
                "RATE": {"value": [100.0]},
                "UNITS": {"value": ["mm"]},
            },
            "ANALOG": {
                "LABELS": {"value": labels},
                "UNITS": {"value": ["N"] * len(labels)},
                "RATE": {"value": [ANALOG_RATE]},
            },
        },
        "data": {"points": np.zeros((4, 0, FRAME_COUNT)), "analogs": analogs},
    }


@pytest.fixture
def make_reader(tmp_path: Path) -> Any:
    path = tmp_path / "plates.c3d"
    path.write_bytes(b"\x02\x50")

    def _make(labels: list[str]) -> Any:
        with patch.object(c3d_reader_module, "ezc3d") as mock:
            mock.c3d.return_value = _c3d_data(labels)
            return C3DDataReader(path)

    return _make


def test_all_plates_with_time_and_cop(make_reader: Any) -> None:
    labels = PLATE_LABELS
    reader = make_reader(labels)
    with patch.object(c3d_reader_module, "ezc3d") as mock:
        mock.c3d.return_value = _c3d_data(labels)
        df = reader.force_plate_dataframe()
    assert list(df.columns) == [
        "sample",
        "time",
        "plate",
        "fx",
        "fy",
        "fz",
        "mx",
        "my",
        "mz",
        "cop_x",
        "cop_y",
        "cop_z",
    ]
    assert len(df) == 2 * FRAME_COUNT
    assert df["plate"].tolist() == [1] * FRAME_COUNT + [2] * FRAME_COUNT
    np.testing.assert_allclose(
        df["time"].to_numpy()[:FRAME_COUNT],
        np.arange(FRAME_COUNT) / ANALOG_RATE,
    )
    plate1 = df[df["plate"] == 1]
    # Fz1 = [100, 100, 5, 100]: third sample is below the 10 N contact threshold
    assert plate1["cop_x"].isna().tolist() == [False, False, True, False]
    assert plate1["cop_z"].dropna().eq(0.0).all()
    fz = plate1["fz"].to_numpy()
    valid = np.abs(fz) > 10.0
    np.testing.assert_allclose(
        plate1["cop_x"].to_numpy()[valid],
        (-plate1["my"].to_numpy() / fz)[valid],
    )
    np.testing.assert_allclose(
        plate1["cop_y"].to_numpy()[valid],
        (plate1["mx"].to_numpy() / fz)[valid],
    )


def test_plate_selection_ground_height_no_time_no_cop(make_reader: Any) -> None:
    reader = make_reader(PLATE_LABELS)
    with patch.object(c3d_reader_module, "ezc3d") as mock:
        mock.c3d.return_value = _c3d_data(PLATE_LABELS)
        df = reader.force_plate_dataframe(
            plate_number=2, include_time=False, compute_cop=False
        )
        with_height = reader.force_plate_dataframe(plate_number=1, ground_height=0.25)
    assert list(df.columns) == [
        "sample",
        "plate",
        "fx",
        "fy",
        "fz",
        "mx",
        "my",
        "mz",
    ]
    assert df["plate"].unique().tolist() == [2]
    assert len(df) == FRAME_COUNT
    assert with_height["cop_z"].dropna().eq(0.25).all()


def test_unknown_plate_raises(make_reader: Any) -> None:
    reader = make_reader(PLATE_LABELS)
    with patch.object(c3d_reader_module, "ezc3d") as mock:
        mock.c3d.return_value = _c3d_data(PLATE_LABELS)
        with pytest.raises(ValueError, match="Force plate 9 not found"):
            reader.force_plate_dataframe(plate_number=9)


def test_no_plate_channels_returns_empty_frame(make_reader: Any) -> None:
    labels = ["Analog1", "Analog2"]
    reader = make_reader(labels)
    with patch.object(c3d_reader_module, "ezc3d") as mock:
        mock.c3d.return_value = _c3d_data(labels)
        df = reader.force_plate_dataframe(include_time=False, compute_cop=False)
    assert df.empty
    assert list(df.columns) == [
        "sample",
        "plate",
        "fx",
        "fy",
        "fz",
        "mx",
        "my",
        "mz",
    ]


def test_incomplete_plate_is_skipped(make_reader: Any) -> None:
    labels = ["Fx1", "Fy1", "Fz1", "Mx1", "My1"]  # Mz1 missing
    reader = make_reader(labels)
    with patch.object(c3d_reader_module, "ezc3d") as mock:
        mock.c3d.return_value = _c3d_data(labels)
        df = reader.force_plate_dataframe()
    assert df.empty
    assert "time" in df.columns
    assert "cop_x" in df.columns
