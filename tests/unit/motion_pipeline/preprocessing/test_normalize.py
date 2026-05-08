"""Unit tests for motion_pipeline.preprocessing.normalize."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    KeypointSequence,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.preprocessing.normalize import (
    UnitSystem,
    UpAxis,
    convert_units,
    normalize_coordinates,
)

from ._local_fixtures import make_keypoint_sequence, make_marker_trajectory


def test_normalize_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_coordinates("not-a-seq")  # type: ignore[arg-type]


def test_convert_units_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        convert_units("not-a-seq")  # type: ignore[arg-type]


def test_normalize_y_to_z_swaps_axes() -> None:
    """Rotation Y_UP -> Z_UP: a point with y=1 should end up at z=1."""
    traj = make_marker_trajectory(num_frames=2)
    # Replace markers with a known point: (0, 1, 0)
    from src.shared.python.motion_pipeline.contracts import Marker, MarkerFrame

    f = MarkerFrame(
        timestamp=0.0,
        markers={"M": Marker(name="M", x=0.0, y=1.0, z=0.0)},
        frame_index=0,
    )
    traj_known = MarkerTrajectory(id="t", frames=[f])
    out = normalize_coordinates(
        traj_known, target_up=UpAxis.Z_UP, source_up=UpAxis.Y_UP, center_origin=False
    )
    m = out.frames[0].markers["M"]
    assert m.x == pytest.approx(0.0, abs=1e-9)
    assert m.z == pytest.approx(1.0, abs=1e-9)


def test_normalize_same_axis_is_identity() -> None:
    traj = make_marker_trajectory(num_frames=2)
    out = normalize_coordinates(
        traj, target_up=UpAxis.Y_UP, source_up=UpAxis.Y_UP, center_origin=False
    )
    for orig, new in zip(traj.frames, out.frames, strict=False):
        for name in orig.markers:
            assert orig.markers[name].x == pytest.approx(new.markers[name].x)
            assert orig.markers[name].y == pytest.approx(new.markers[name].y)
            assert orig.markers[name].z == pytest.approx(new.markers[name].z)


def test_convert_units_mm_to_m_scales_by_1e_minus_3() -> None:
    from src.shared.python.motion_pipeline.contracts import Marker, MarkerFrame

    f = MarkerFrame(
        timestamp=0.0,
        markers={"M": Marker(name="M", x=1000.0, y=2000.0, z=500.0)},
        frame_index=0,
    )
    traj = MarkerTrajectory(id="t", frames=[f])
    out = convert_units(
        traj, target_unit=UnitSystem.METERS, source_unit=UnitSystem.MILLIMETERS
    )
    m = out.frames[0].markers["M"]
    assert m.x == pytest.approx(1.0)
    assert m.y == pytest.approx(2.0)
    assert m.z == pytest.approx(0.5)


def test_convert_units_inches_round_trip() -> None:
    from src.shared.python.motion_pipeline.contracts import Marker, MarkerFrame

    f = MarkerFrame(
        timestamp=0.0,
        markers={"M": Marker(name="M", x=1.0, y=1.0, z=1.0)},
        frame_index=0,
    )
    traj = MarkerTrajectory(id="t", frames=[f])
    inch = convert_units(
        traj, target_unit=UnitSystem.INCHES, source_unit=UnitSystem.METERS
    )
    back = convert_units(
        inch, target_unit=UnitSystem.METERS, source_unit=UnitSystem.INCHES
    )
    m = back.frames[0].markers["M"]
    assert m.x == pytest.approx(1.0, rel=1e-3)


def test_normalize_center_origin_zeros_first_frame_centroid() -> None:
    traj = make_marker_trajectory(num_frames=5, marker_names=("A", "B", "C"), seed=1)
    out = normalize_coordinates(
        traj, target_up=UpAxis.Y_UP, source_up=UpAxis.Y_UP, center_origin=True
    )
    first = out.frames[0]
    cx = float(np.mean([m.x for m in first.markers.values()]))
    cy = float(np.mean([m.y for m in first.markers.values()]))
    cz = float(np.mean([m.z for m in first.markers.values()]))
    assert cx == pytest.approx(0.0, abs=1e-9)
    assert cy == pytest.approx(0.0, abs=1e-9)
    assert cz == pytest.approx(0.0, abs=1e-9)


def test_normalize_keypoints_basic() -> None:
    seq = make_keypoint_sequence(num_frames=3, num_kp=2)
    out = normalize_coordinates(seq, target_up=UpAxis.Y_UP, source_up=UpAxis.Y_UP)
    assert isinstance(out, KeypointSequence)
    assert out.num_frames == seq.num_frames
    assert out.metadata.get("normalized") is True


def test_convert_units_keypoints_metadata() -> None:
    seq = make_keypoint_sequence(num_frames=3, num_kp=1)
    out = convert_units(
        seq, target_unit=UnitSystem.METERS, source_unit=UnitSystem.METERS
    )
    assert out.metadata.get("units_converted") is True


def test_get_unit_scale_meters_to_mm() -> None:
    from src.shared.python.motion_pipeline.preprocessing.normalize import (
        _get_unit_scale,
    )

    assert _get_unit_scale(UnitSystem.METERS, UnitSystem.MILLIMETERS) == pytest.approx(
        1000.0
    )
    assert _get_unit_scale(UnitSystem.METERS, UnitSystem.METERS) == pytest.approx(1.0)


def test_get_up_axis_transform_identity_for_same() -> None:
    from src.shared.python.motion_pipeline.preprocessing.normalize import (
        _get_up_axis_transform,
    )

    np.testing.assert_allclose(
        _get_up_axis_transform(UpAxis.Y_UP, UpAxis.Y_UP), np.eye(3)
    )


def test_get_up_axis_transform_z_to_y_inverse() -> None:
    from src.shared.python.motion_pipeline.preprocessing.normalize import (
        _get_up_axis_transform,
    )

    fwd = _get_up_axis_transform(UpAxis.Y_UP, UpAxis.Z_UP)
    back = _get_up_axis_transform(UpAxis.Z_UP, UpAxis.Y_UP)
    np.testing.assert_allclose(fwd @ back, np.eye(3), atol=1e-9)
