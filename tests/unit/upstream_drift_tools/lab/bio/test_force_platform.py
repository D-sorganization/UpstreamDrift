"""Tests for the FORCE_PLATFORM parameter group parser and calibration path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from upstream_drift_tools.lab.bio._c3d_analog import (
    build_force_plate_dataframe_from_calibration,
)
from upstream_drift_tools.lab.bio._c3d_io import build_metadata
from upstream_drift_tools.lab.bio._c3d_models import ForcePlateCalibration

from ._synthetic import _synthetic_c3d_dict

pytestmark = pytest.mark.unit


def _square_plate_corners(
    centre: tuple[float, float, float], half_side_mm: float = 250.0
) -> np.ndarray:
    """Build a 3x4 corners array (mm) with corners ordered +x+y, -x+y, -x-y, +x-y."""
    cx, cy, cz = centre
    h = half_side_mm
    # ezc3d returns corners as (3, 4) per plate.
    return np.array(
        [
            [cx + h, cx - h, cx - h, cx + h],
            [cy + h, cy + h, cy - h, cy - h],
            [cz, cz, cz, cz],
        ],
        dtype=float,
    )


def test_build_metadata_parses_force_platform_type2() -> None:
    """A type-2 plate with CAL_MATRIX should be exposed via metadata.force_plates."""
    n_frames = 5
    subframes = 10
    n_analog = 6
    cal_matrix = np.eye(6) * 2.0  # voltages -> forces, gain 2.0
    fp = {
        "USED": np.array([1]),
        "TYPE": np.array([2]),
        "CORNERS": _square_plate_corners((500.0, 400.0, 0.0))[:, :, np.newaxis],
        "ORIGIN": np.array([[0.0], [0.0], [-30.0]]),  # mm, sensor 30 mm below top
        "CAL_MATRIX": cal_matrix[:, :, np.newaxis],
        "CHANNEL": np.arange(1, n_analog + 1).reshape(n_analog, 1),
    }
    raw = _synthetic_c3d_dict(
        n_frames=n_frames,
        n_analog=n_analog,
        analog_labels=[f"FP1_{c}" for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")],
        analog_subframes=subframes,
        force_platform=fp,
    )

    metadata = build_metadata(raw, Path("synthetic.c3d"))
    assert len(metadata.force_plates) == 1
    plate = metadata.force_plates[0]
    assert plate.plate_type == 2
    assert plate.corners.shape == (4, 3)
    # Corners are returned in metres.
    assert np.allclose(plate.corners[0], [0.75, 0.65, 0.0])
    assert np.allclose(plate.origin, [0.0, 0.0, -0.030])
    assert plate.cal_matrix is not None
    assert plate.cal_matrix.shape == (6, 6)
    assert plate.channel_indices == (0, 6)


def test_build_metadata_type1_no_cal_matrix() -> None:
    """Type-1 plates do not require a calibration matrix."""
    fp = {
        "USED": np.array([1]),
        "TYPE": np.array([1]),
        "CORNERS": _square_plate_corners((0.0, 0.0, 0.0))[:, :, np.newaxis],
        "ORIGIN": np.array([[0.0], [0.0], [0.0]]),
    }
    raw = _synthetic_c3d_dict(
        n_frames=2,
        n_analog=6,
        analog_labels=["Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1"],
        force_platform=fp,
    )
    metadata = build_metadata(raw, Path("synthetic.c3d"))
    assert len(metadata.force_plates) == 1
    assert metadata.force_plates[0].cal_matrix is None
    assert metadata.force_plates[0].plate_type == 1


def test_build_metadata_type4_full_cal_matrix() -> None:
    """Type-4 plates carry an 8-channel calibration matrix (6x8)."""
    cal_matrix = np.zeros((6, 8))
    cal_matrix[:6, :6] = np.eye(6)
    fp = {
        "USED": np.array([1]),
        "TYPE": np.array([4]),
        "CORNERS": _square_plate_corners((0.0, 0.0, 0.0))[:, :, np.newaxis],
        "ORIGIN": np.array([[0.0], [0.0], [-25.0]]),
        "CAL_MATRIX": cal_matrix[:, :, np.newaxis],
        "CHANNEL": np.arange(1, 9).reshape(8, 1),
    }
    raw = _synthetic_c3d_dict(
        n_frames=2,
        n_analog=8,
        analog_labels=[f"FP_{i}" for i in range(8)],
        force_platform=fp,
    )
    metadata = build_metadata(raw, Path("synthetic.c3d"))
    assert len(metadata.force_plates) == 1
    plate = metadata.force_plates[0]
    assert plate.plate_type == 4
    assert plate.cal_matrix is not None
    assert plate.cal_matrix.shape == (6, 8)
    assert plate.channel_indices == (0, 8)


def test_build_metadata_used_zero_returns_empty() -> None:
    """USED == 0 yields no force plates so the analog-regex fallback runs."""
    fp = {
        "USED": np.array([0]),
        "TYPE": np.array([], dtype=int),
        "CORNERS": np.zeros((3, 4, 0)),
        "ORIGIN": np.zeros((3, 0)),
    }
    raw = _synthetic_c3d_dict(
        n_frames=2,
        n_analog=6,
        analog_labels=["Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1"],
        force_platform=fp,
    )
    metadata = build_metadata(raw, Path("synthetic.c3d"))
    assert metadata.force_plates == ()


def test_build_metadata_no_force_platform_group() -> None:
    """Files without a FORCE_PLATFORM group yield an empty tuple."""
    raw = _synthetic_c3d_dict(n_frames=2, n_analog=0)
    metadata = build_metadata(raw, Path("synthetic.c3d"))
    assert metadata.force_plates == ()


def test_force_plate_dataframe_applies_cal_matrix() -> None:
    """A type-2 plate with a 2x gain matrix doubles the raw voltages."""
    n_frames = 4
    subframes = 5
    n_analog = 6
    # Build raw voltages such that channel i has a constant value (i+1).
    analog = np.zeros((subframes, n_analog, n_frames))
    for i in range(n_analog):
        analog[:, i, :] = i + 1
    cal_matrix = np.eye(6) * 2.0

    plate = ForcePlateCalibration(
        corners=np.array(
            [
                [0.25, 0.25, 0.0],
                [-0.25, 0.25, 0.0],
                [-0.25, -0.25, 0.0],
                [0.25, -0.25, 0.0],
            ]
        ),
        origin=np.array([0.0, 0.0, -0.030]),
        cal_matrix=cal_matrix,
        plate_type=2,
        channel_indices=(0, 6),
    )

    df = build_force_plate_dataframe_from_calibration(
        analog_array=analog,
        calibrations=(plate,),
        analog_rate=1000.0,
        file_name="synthetic.c3d",
        plate_number=None,
        include_time=True,
        compute_cop=True,
        ground_height=0.0,
    )

    expected_rows = n_frames * subframes
    assert len(df) == expected_rows
    # Channel i had value i+1, gain 2 -> 2*(i+1).
    assert np.allclose(df["fx"], 2.0)
    assert np.allclose(df["fy"], 4.0)
    assert np.allclose(df["fz"], 6.0)
    assert np.allclose(df["mx"], 8.0)
    assert np.allclose(df["my"], 10.0)
    assert np.allclose(df["mz"], 12.0)
    # CoP should be finite (fz=6 > threshold) and within plate bounds.
    assert np.all(np.isfinite(df["cop_x"]))
    assert np.all(np.isfinite(df["cop_y"]))


def test_force_plate_dataframe_type1_passthrough() -> None:
    """Type-1 plates return raw channel values directly without cal_matrix."""
    n_frames = 2
    subframes = 3
    analog = np.zeros((subframes, 6, n_frames))
    analog[:, 2, :] = 750.0  # fz constant
    plate = ForcePlateCalibration(
        corners=np.array(
            [
                [0.25, 0.25, 0.0],
                [-0.25, 0.25, 0.0],
                [-0.25, -0.25, 0.0],
                [0.25, -0.25, 0.0],
            ]
        ),
        origin=np.array([0.0, 0.0, 0.0]),
        cal_matrix=None,
        plate_type=1,
        channel_indices=(0, 6),
    )
    df = build_force_plate_dataframe_from_calibration(
        analog_array=analog,
        calibrations=(plate,),
        analog_rate=None,
        file_name="synthetic.c3d",
        plate_number=None,
        include_time=False,
        compute_cop=False,
        ground_height=0.0,
    )
    assert np.allclose(df["fz"], 750.0)
    assert "cop_x" not in df.columns


def test_force_plate_dataframe_cop_in_lab_frame() -> None:
    """CoP is rotated to lab frame using corners (plate centred at lab (1, 2, 0))."""
    n_frames = 1
    subframes = 1
    analog = np.zeros((subframes, 6, n_frames))
    analog[:, 2, :] = 1000.0  # fz = 1000 N
    # mx, my zero -> cop_local = (0, 0, 0). With z0=0, lab CoP = plate centre.
    centre = np.array([1.0, 2.0, 0.0])
    half = 0.25
    corners = np.array(
        [
            centre + [half, half, 0.0],
            centre + [-half, half, 0.0],
            centre + [-half, -half, 0.0],
            centre + [half, -half, 0.0],
        ]
    )
    plate = ForcePlateCalibration(
        corners=corners,
        origin=np.array([0.0, 0.0, 0.0]),
        cal_matrix=None,
        plate_type=1,
        channel_indices=(0, 6),
    )
    df = build_force_plate_dataframe_from_calibration(
        analog_array=analog,
        calibrations=(plate,),
        analog_rate=1000.0,
        file_name="synthetic.c3d",
        plate_number=None,
        include_time=False,
        compute_cop=True,
        ground_height=0.0,
    )
    assert np.allclose(df["cop_x"].iloc[0], 1.0)
    assert np.allclose(df["cop_y"].iloc[0], 2.0)
    assert np.allclose(df["cop_z"].iloc[0], 0.0)


def test_force_plate_dataframe_select_specific_plate() -> None:
    """Selecting a specific plate filters to that plate only."""
    plate1 = ForcePlateCalibration(
        corners=np.array(
            [
                [0.25, 0.25, 0.0],
                [-0.25, 0.25, 0.0],
                [-0.25, -0.25, 0.0],
                [0.25, -0.25, 0.0],
            ]
        ),
        origin=np.array([0.0, 0.0, 0.0]),
        cal_matrix=None,
        plate_type=1,
        channel_indices=(0, 6),
    )
    plate2 = ForcePlateCalibration(
        corners=plate1.corners + np.array([1.0, 0.0, 0.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        cal_matrix=None,
        plate_type=1,
        channel_indices=(6, 12),
    )
    analog = np.zeros((1, 12, 1))
    df = build_force_plate_dataframe_from_calibration(
        analog_array=analog,
        calibrations=(plate1, plate2),
        analog_rate=None,
        file_name="synthetic.c3d",
        plate_number=2,
        include_time=False,
        compute_cop=False,
        ground_height=0.0,
    )
    assert set(df["plate"].unique()) == {2}


def test_force_plate_calibration_validates_shapes() -> None:
    """Bad shapes raise immediately at construction."""
    with pytest.raises(ValueError, match="corners"):
        ForcePlateCalibration(
            corners=np.zeros((3, 3)),
            origin=np.zeros(3),
            cal_matrix=None,
            plate_type=1,
            channel_indices=(0, 6),
        )
    with pytest.raises(ValueError, match="origin"):
        ForcePlateCalibration(
            corners=np.zeros((4, 3)),
            origin=np.zeros(2),
            cal_matrix=None,
            plate_type=1,
            channel_indices=(0, 6),
        )
    with pytest.raises(ValueError, match="plate_type"):
        ForcePlateCalibration(
            corners=np.zeros((4, 3)),
            origin=np.zeros(3),
            cal_matrix=None,
            plate_type=99,
            channel_indices=(0, 6),
        )
    with pytest.raises(ValueError, match="channel_indices"):
        ForcePlateCalibration(
            corners=np.zeros((4, 3)),
            origin=np.zeros(3),
            cal_matrix=None,
            plate_type=1,
            channel_indices=(5, 2),
        )
