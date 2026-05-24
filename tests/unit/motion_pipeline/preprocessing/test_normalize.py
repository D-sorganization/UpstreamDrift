"""Unit tests for motion_pipeline.preprocessing.normalize."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    Marker,
    MarkerFrame,
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


def test_normalize_empty_keypoint_sequence_returns_unchanged() -> None:
    seq = KeypointSequence.model_construct(id="empty", frames=[])
    out = normalize_coordinates(seq, target_up=UpAxis.Y_UP, source_up=UpAxis.Y_UP)
    assert out is seq
    assert len(out.frames) == 0


def test_normalize_empty_marker_trajectory_returns_unchanged() -> None:
    traj = MarkerTrajectory.model_construct(id="empty", frames=[])
    out = normalize_coordinates(traj, target_up=UpAxis.Y_UP, source_up=UpAxis.Y_UP)
    assert out is traj
    assert len(out.frames) == 0


def test_convert_units_empty_keypoint_sequence_returns_unchanged() -> None:
    seq = KeypointSequence.model_construct(id="empty", frames=[])
    out = convert_units(
        seq, target_unit=UnitSystem.METERS, source_unit=UnitSystem.METERS
    )
    assert out is seq
    assert len(out.frames) == 0


def test_convert_units_empty_marker_trajectory_returns_unchanged() -> None:
    traj = MarkerTrajectory.model_construct(id="empty", frames=[])
    out = convert_units(
        traj, target_unit=UnitSystem.METERS, source_unit=UnitSystem.METERS
    )
    assert out is traj
    assert len(out.frames) == 0


def test_detect_up_axis_keypoints() -> None:
    from src.shared.python.motion_pipeline.preprocessing.normalize import (
        _detect_up_axis_keypoints,
    )

    # 1. Y_UP detection (y has largest variance)
    frames_y = [
        KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(name="kp", x=1.0, y=10.0, z=1.0, confidence=1.0),
                Keypoint(name="kp", x=1.1, y=0.0, z=1.1, confidence=1.0),
            ],
            schema_name="custom",
            frame_index=0,
        )
    ]
    assert _detect_up_axis_keypoints(frames_y) == UpAxis.Y_UP

    # 2. Z_UP detection (z has largest variance)
    frames_z = [
        KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(name="kp", x=1.0, y=1.0, z=10.0, confidence=1.0),
                Keypoint(name="kp", x=1.1, y=1.1, z=0.0, confidence=1.0),
            ],
            schema_name="custom",
            frame_index=0,
        )
    ]
    assert _detect_up_axis_keypoints(frames_z) == UpAxis.Z_UP

    # 3. X_UP detection (x has largest variance)
    frames_x = [
        KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(name="kp", x=10.0, y=1.0, z=1.0, confidence=1.0),
                Keypoint(name="kp", x=0.0, y=1.1, z=1.1, confidence=1.0),
            ],
            schema_name="custom",
            frame_index=0,
        )
    ]
    assert _detect_up_axis_keypoints(frames_x) == UpAxis.X_UP

    # 4. Empty frames fallback
    assert _detect_up_axis_keypoints([]) == UpAxis.Y_UP


def test_detect_up_axis_markers() -> None:
    from src.shared.python.motion_pipeline.preprocessing.normalize import (
        _detect_up_axis_markers,
    )

    # 1. Y_UP detection
    frames_y = [
        MarkerFrame(
            timestamp=0.0,
            markers={
                "M1": Marker(name="M1", x=1.0, y=10.0, z=1.0),
                "M2": Marker(name="M2", x=1.1, y=0.0, z=1.1),
            },
            frame_index=0,
        )
    ]
    assert _detect_up_axis_markers(frames_y) == UpAxis.Y_UP

    # 2. Z_UP detection
    frames_z = [
        MarkerFrame(
            timestamp=0.0,
            markers={
                "M1": Marker(name="M1", x=1.0, y=1.0, z=10.0),
                "M2": Marker(name="M2", x=1.1, y=1.1, z=0.0),
            },
            frame_index=0,
        )
    ]
    assert _detect_up_axis_markers(frames_z) == UpAxis.Z_UP

    # 3. X_UP detection
    frames_x = [
        MarkerFrame(
            timestamp=0.0,
            markers={
                "M1": Marker(name="M1", x=10.0, y=1.0, z=1.0),
                "M2": Marker(name="M2", x=0.0, y=1.1, z=1.1),
            },
            frame_index=0,
        )
    ]
    assert _detect_up_axis_markers(frames_x) == UpAxis.X_UP

    # 4. Empty frames fallback
    assert _detect_up_axis_markers([]) == UpAxis.Y_UP


def test_detect_unit_keypoints() -> None:
    from src.shared.python.motion_pipeline.preprocessing.normalize import (
        _detect_unit_keypoints,
    )

    # 1. Millimeters (extent > 100)
    frames_mm = [
        KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(name="kp", x=0.0, y=1000.0, z=0.0, confidence=1.0),
                Keypoint(name="kp", x=0.0, y=0.0, z=0.0, confidence=1.0),
            ],
            schema_name="custom",
            frame_index=0,
        )
    ]
    assert _detect_unit_keypoints(frames_mm) == UnitSystem.MILLIMETERS

    # 2. Centimeters (10 < extent <= 100)
    frames_cm = [
        KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(name="kp", x=0.0, y=50.0, z=0.0, confidence=1.0),
                Keypoint(name="kp", x=0.0, y=0.0, z=0.0, confidence=1.0),
            ],
            schema_name="custom",
            frame_index=0,
        )
    ]
    assert _detect_unit_keypoints(frames_cm) == UnitSystem.CENTIMETERS

    # 3. Meters (extent <= 10)
    frames_m = [
        KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(name="kp", x=0.0, y=1.8, z=0.0, confidence=1.0),
                Keypoint(name="kp", x=0.0, y=0.0, z=0.0, confidence=1.0),
            ],
            schema_name="custom",
            frame_index=0,
        )
    ]
    assert _detect_unit_keypoints(frames_m) == UnitSystem.METERS

    # 4. Empty frames fallback
    assert _detect_unit_keypoints([]) == UnitSystem.METERS


def test_detect_unit_markers() -> None:
    from src.shared.python.motion_pipeline.preprocessing.normalize import (
        _detect_unit_markers,
    )

    # 1. Millimeters (extent > 100)
    frames_mm = [
        MarkerFrame(
            timestamp=0.0,
            markers={
                "M1": Marker(name="M1", x=0.0, y=1000.0, z=0.0),
                "M2": Marker(name="M2", x=0.0, y=0.0, z=0.0),
            },
            frame_index=0,
        )
    ]
    assert _detect_unit_markers(frames_mm) == UnitSystem.MILLIMETERS

    # 2. Centimeters (10 < extent <= 100)
    frames_cm = [
        MarkerFrame(
            timestamp=0.0,
            markers={
                "M1": Marker(name="M1", x=0.0, y=50.0, z=0.0),
                "M2": Marker(name="M2", x=0.0, y=0.0, z=0.0),
            },
            frame_index=0,
        )
    ]
    assert _detect_unit_markers(frames_cm) == UnitSystem.CENTIMETERS

    # 3. Meters (extent <= 10)
    frames_m = [
        MarkerFrame(
            timestamp=0.0,
            markers={
                "M1": Marker(name="M1", x=0.0, y=1.8, z=0.0),
                "M2": Marker(name="M2", x=0.0, y=0.0, z=0.0),
            },
            frame_index=0,
        )
    ]
    assert _detect_unit_markers(frames_m) == UnitSystem.METERS

    # 4. Empty frames fallback
    assert _detect_unit_markers([]) == UnitSystem.METERS


def test_normalize_auto_detection_and_centering_keypoints() -> None:
    # y-up, in meters, shifted centroid
    seq = KeypointSequence(
        id="seq",
        frames=[
            KeypointFrame(
                timestamp=0.0,
                keypoints=[
                    Keypoint(name="kp1", x=2.0, y=10.0, z=2.0, confidence=1.0),
                    Keypoint(name="kp2", x=4.0, y=0.0, z=4.0, confidence=1.0),
                ],
                schema_name="custom",
                frame_index=0,
            )
        ],
    )
    # y-up should be detected since y variance (25) is larger than x/z variance (1)
    # center_origin = True should center the coordinates (centroid at x=3.0, y=5.0, z=3.0)
    out = normalize_coordinates(
        seq, target_up=UpAxis.Y_UP, source_up=None, center_origin=True
    )
    assert out.metadata["source_up"] == UpAxis.Y_UP.value

    # Centroid should be 0,0,0
    kp1 = out.frames[0].keypoints[0]
    kp2 = out.frames[0].keypoints[1]
    assert kp1.x == pytest.approx(-1.0)
    assert kp1.y == pytest.approx(5.0)
    assert kp1.z == pytest.approx(-1.0)
    assert kp2.x == pytest.approx(1.0)
    assert kp2.y == pytest.approx(-5.0)
    assert kp2.z == pytest.approx(1.0)


def test_normalize_auto_detection_markers() -> None:
    traj = MarkerTrajectory(
        id="traj",
        frames=[
            MarkerFrame(
                timestamp=0.0,
                markers={
                    "M1": Marker(name="M1", x=2.0, y=10.0, z=2.0),
                    "M2": Marker(name="M2", x=4.0, y=0.0, z=4.0),
                },
                frame_index=0,
            )
        ],
    )
    # Auto detect Y_UP and center it
    out = normalize_coordinates(
        traj, target_up=UpAxis.Y_UP, source_up=None, center_origin=True
    )
    assert out.metadata["source_up"] == UpAxis.Y_UP.value
    m1 = out.frames[0].markers["M1"]
    m2 = out.frames[0].markers["M2"]
    assert m1.x == pytest.approx(-1.0)
    assert m1.y == pytest.approx(5.0)
    assert m1.z == pytest.approx(-1.0)
    assert m2.x == pytest.approx(1.0)
    assert m2.y == pytest.approx(-5.0)
    assert m2.z == pytest.approx(1.0)


def test_convert_units_auto_detection_keypoints() -> None:
    # mm data (extent y is 1000)
    seq = KeypointSequence(
        id="seq",
        frames=[
            KeypointFrame(
                timestamp=0.0,
                keypoints=[
                    Keypoint(name="kp1", x=0.0, y=1000.0, z=0.0, confidence=1.0),
                    Keypoint(name="kp2", x=0.0, y=0.0, z=0.0, confidence=1.0),
                ],
                schema_name="custom",
                frame_index=0,
            )
        ],
    )
    # Convert to meters with auto-detected mm
    out = convert_units(seq, target_unit=UnitSystem.METERS, source_unit=None)
    assert out.metadata["source_unit"] == UnitSystem.MILLIMETERS.value
    assert out.frames[0].keypoints[0].y == pytest.approx(1.0)


def test_convert_units_auto_detection_markers() -> None:
    # mm data
    traj = MarkerTrajectory(
        id="traj",
        frames=[
            MarkerFrame(
                timestamp=0.0,
                markers={
                    "M1": Marker(name="M1", x=0.0, y=1000.0, z=0.0),
                    "M2": Marker(name="M2", x=0.0, y=0.0, z=0.0),
                },
                frame_index=0,
            )
        ],
    )
    out = convert_units(traj, target_unit=UnitSystem.METERS, source_unit=None)
    assert out.metadata["source_unit"] == UnitSystem.MILLIMETERS.value
    assert out.frames[0].markers["M1"].y == pytest.approx(1.0)


def test_center_functions_empty_returns() -> None:
    from src.shared.python.motion_pipeline.preprocessing.normalize import (
        _center_keypoints_origin,
        _center_markers_origin,
    )

    assert _center_keypoints_origin([]) == []
    assert _center_markers_origin([]) == []


def test_normalize_keypoints_without_z() -> None:
    seq = KeypointSequence(
        id="seq2d",
        frames=[
            KeypointFrame(
                timestamp=0.0,
                keypoints=[
                    Keypoint(name="kp1", x=1.0, y=2.0, z=None, confidence=1.0),
                ],
                schema_name="custom",
                frame_index=0,
            )
        ],
    )
    # Convert Y_UP to Z_UP. Since Z is None, it should remain None.
    out = normalize_coordinates(
        seq, target_up=UpAxis.Z_UP, source_up=UpAxis.Y_UP, center_origin=False
    )
    assert out.frames[0].keypoints[0].z is None

    # Check convert units preserves None Z
    out_units = convert_units(
        seq, target_unit=UnitSystem.MILLIMETERS, source_unit=UnitSystem.METERS
    )
    assert out_units.frames[0].keypoints[0].z is None

    # Check center origin preserves None Z
    out_centered = normalize_coordinates(
        seq, target_up=UpAxis.Y_UP, source_up=UpAxis.Y_UP, center_origin=True
    )
    assert out_centered.frames[0].keypoints[0].z is None
