"""Unit tests for ``C3DDataReader`` exercising every public method.

Uses ``unittest.mock.patch`` to substitute a synthetic ezc3d-shaped dict for
``ezc3d.c3d``, avoiding the need for a real C3D file in unit tests. A real
file (``data/C3D_TA_Driver.c3d``) is exercised in a single sanity test.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sidekick.lab.bio import _c3d_io as io_mod
from sidekick.lab.bio._c3d_marker_set import MarkerSet
from sidekick.lab.bio.c3d_reader import C3DDataReader

from ._synthetic import _synthetic_c3d_dict

REAL_C3D = Path(__file__).resolve().parents[5] / "data" / "C3D_TA_Driver.c3d"


def _force_plate_dict(n_frames: int = 4, n_subframes: int = 1) -> dict:
    """Synthetic C3D dict with one fully-channeled force plate."""
    labels = ["Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1"]
    n_analog = len(labels)
    analog = np.zeros((n_subframes, n_analog, n_frames), dtype=float)
    # set fz on plate 1 above the 10 N threshold for half the samples
    fz_idx = labels.index("Fz1")
    mx_idx = labels.index("Mx1")
    my_idx = labels.index("My1")
    analog[:, fz_idx, n_frames // 2 :] = 100.0
    analog[:, mx_idx, :] = 5.0
    analog[:, my_idx, :] = -10.0
    return _synthetic_c3d_dict(
        n_frames=n_frames,
        n_markers=2,
        marker_names=["A", "B"],
        n_analog=n_analog,
        analog_labels=labels,
        analog_units=[""] * n_analog,
        analog_rate=1000.0,
        analog_subframes=n_subframes,
        analog_data=analog,
    )


def _patched_reader(synthetic: dict, file_path: Path) -> C3DDataReader:
    """Build a reader whose ``_load`` returns the supplied synthetic dict."""
    file_path.write_bytes(b"\x00")
    reader = C3DDataReader(file_path)
    # Pre-populate cached load to skip ezc3d.c3d.
    reader._c3d_data = synthetic  # noqa: SLF001
    return reader


# ----- get_metadata + caching -----------------------------------------------


def test_get_metadata_caches(tmp_path: Path) -> None:
    reader = _patched_reader(_synthetic_c3d_dict(), tmp_path / "x.c3d")
    md1 = reader.get_metadata()
    md2 = reader.get_metadata()
    assert md1 is md2


# ----- points_dataframe -----------------------------------------------------


def test_points_dataframe_default(tmp_path: Path) -> None:
    reader = _patched_reader(
        _synthetic_c3d_dict(n_frames=3, n_markers=2, marker_names=["A", "B"]),
        tmp_path / "x.c3d",
    )
    df = reader.points_dataframe()
    assert {"frame", "marker", "x", "y", "z", "residual", "time"}.issubset(df.columns)


def test_points_dataframe_filters_and_units(tmp_path: Path) -> None:
    # Use small native-m coordinates so x1000 mm conversion stays under 10 m.
    pts = np.zeros((4, 3, 2), dtype=float)
    for m in range(3):
        pts[0, m, :] = 0.001 * (m + 1)
        pts[1, m, :] = 0.001
        pts[2, m, :] = 0.001
    reader = _patched_reader(
        _synthetic_c3d_dict(
            n_frames=2, n_markers=3, marker_names=["A", "B", "C"], point_data=pts
        ),
        tmp_path / "x.c3d",
    )
    df = reader.points_dataframe(
        include_time=False,
        markers=["A"],
        residual_nan_threshold=0.5,
        target_units="mm",
    )
    assert set(df["marker"].unique()) == {"A"}
    assert "time" not in df.columns


# ----- analog_dataframe -----------------------------------------------------


def test_analog_dataframe(tmp_path: Path) -> None:
    reader = _patched_reader(
        _synthetic_c3d_dict(n_frames=2, n_markers=1, marker_names=["A"], n_analog=2),
        tmp_path / "x.c3d",
    )
    df = reader.analog_dataframe(include_time=True)
    assert "time" in df.columns
    assert df.shape[0] > 0


# ----- export_points / export_analog ----------------------------------------


@pytest.mark.parametrize("ext", ["csv", "json", "npz"])
def test_export_points_formats(tmp_path: Path, ext: str) -> None:
    reader = _patched_reader(
        _synthetic_c3d_dict(n_frames=2, n_markers=1, marker_names=["A"]),
        tmp_path / "x.c3d",
    )
    out = tmp_path / f"points.{ext}"
    result = reader.export_points(out)
    assert result.exists()
    if ext == "csv":
        text = out.read_text()
        assert "marker" in text
    elif ext == "json":
        payload = json.loads(out.read_text())
        assert "metadata" in payload
    else:
        arr = np.load(out, allow_pickle=False)
        assert "marker" in arr.files


def test_export_points_explicit_format(tmp_path: Path) -> None:
    reader = _patched_reader(
        _synthetic_c3d_dict(n_frames=1, n_markers=1, marker_names=["A"]),
        tmp_path / "x.c3d",
    )
    out = tmp_path / "explicit.bin"
    reader.export_points(out, file_format="json")
    payload = json.loads(out.read_text())
    assert "data" in payload


def test_export_analog_csv(tmp_path: Path) -> None:
    reader = _patched_reader(
        _synthetic_c3d_dict(
            n_frames=2,
            n_markers=1,
            marker_names=["A"],
            n_analog=1,
            analog_labels=["A1"],
        ),
        tmp_path / "x.c3d",
    )
    out = tmp_path / "analog.csv"
    reader.export_analog(out)
    assert out.exists()


def test_export_analog_with_kwargs(tmp_path: Path) -> None:
    reader = _patched_reader(
        _synthetic_c3d_dict(
            n_frames=2,
            n_markers=1,
            marker_names=["A"],
            n_analog=1,
            analog_labels=["A1"],
        ),
        tmp_path / "x.c3d",
    )
    out = tmp_path / "analog.json"
    reader.export_analog(out, include_time=False, file_format="json")
    payload = json.loads(out.read_text())
    assert "data" in payload


# ----- force-plate APIs -----------------------------------------------------


def test_get_force_plate_count_zero(tmp_path: Path) -> None:
    reader = _patched_reader(
        _synthetic_c3d_dict(n_frames=2, n_markers=1, marker_names=["A"]),
        tmp_path / "x.c3d",
    )
    assert reader.get_force_plate_count() == 0


def test_force_plate_dataframe_synthetic(tmp_path: Path) -> None:
    reader = _patched_reader(_force_plate_dict(n_frames=4), tmp_path / "x.c3d")
    assert reader.get_force_plate_count() == 1
    plate_channels = reader.get_force_plate_channels()
    assert 1 in plate_channels
    df = reader.force_plate_dataframe()
    assert "fx" in df.columns and "cop_x" in df.columns
    # The synthetic data has 4 frames * 1 subframe = 4 samples
    assert df.shape[0] == 4


def test_force_plate_dataframe_filter_and_no_cop(tmp_path: Path) -> None:
    reader = _patched_reader(_force_plate_dict(n_frames=4), tmp_path / "x.c3d")
    df = reader.force_plate_dataframe(
        plate_number=1,
        include_time=False,
        compute_cop=False,
        ground_height=0.1,
    )
    assert "time" not in df.columns
    assert "cop_x" not in df.columns
    assert df["plate"].unique().tolist() == [1]


def test_force_plate_columns_static() -> None:
    cols = C3DDataReader._force_plate_columns(True, True)
    assert cols[0] == "sample"
    assert "cop_x" in cols


# ----- _load goes through ezc3d on first call -------------------------------


def test_load_goes_through_ezc3d(tmp_path: Path) -> None:
    fake = _synthetic_c3d_dict()
    file_path = tmp_path / "x.c3d"
    file_path.write_bytes(b"\x00")
    reader = C3DDataReader(file_path)
    with patch.object(io_mod.ezc3d, "c3d", return_value=fake) as m:
        md = reader.get_metadata()
    m.assert_called_once()
    assert md.marker_count == 3


# ----- real-file sanity (skip if absent) ------------------------------------


def test_real_tour_average_driver_reader_sanity() -> None:
    if not REAL_C3D.exists():
        pytest.skip(f"missing fixture: {REAL_C3D}")

    reader = C3DDataReader(REAL_C3D)
    metadata = reader.get_metadata()
    points = reader.points_dataframe()

    assert metadata.marker_count == 38
    assert metadata.frame_count == 654
    assert metadata.frame_rate == pytest.approx(360.0)
    assert metadata.units == "m"
    assert metadata.duration == pytest.approx(654 / 360.0)
    assert metadata.analog_count == 0
    assert metadata.force_plates == ()
    assert metadata.events == []
    assert metadata.marker_set is MarkerSet.GOLF_TOUR_AVERAGE_BODY
    assert {"WaistLeft", "BackTop", "HeadTop", "LWristTop", "RAnkleOut"}.issubset(
        set(metadata.marker_labels)
    )
    assert points.shape == (38 * 654, 7)
