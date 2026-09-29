"""Parity tests for the vectorized normalize.py (issue #8925).

Each test compares the current, vectorized implementation against a
private reference reimplementation of the *old* per-marker/per-keypoint
Python-loop logic that normalize.py used before vectorization. Keeping
the reference inline means this suite stays permanent even though the
original loop-based production code was deleted.
"""

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
from src.shared.python.motion_pipeline.preprocessing import normalize as norm
from src.shared.python.motion_pipeline.preprocessing.normalize import (
    UnitSystem,
    UpAxis,
    convert_units,
    normalize_coordinates,
)

from ._local_fixtures import make_keypoint_sequence, make_marker_trajectory

pytestmark = pytest.mark.unit

RTOL = 1e-12
ATOL = 1e-12


# ---------------------------------------------------------------------------
# Reference (pre-vectorization) implementations
# ---------------------------------------------------------------------------


def _ref_normalize_markers(
    traj: MarkerTrajectory,
    target_up: UpAxis,
    source_up: UpAxis | None,
    center_origin: bool,
) -> MarkerTrajectory:
    if source_up is None:
        source_up = norm._detect_up_axis_markers(traj.frames)
    transform = norm._get_up_axis_transform(source_up, target_up)

    new_frames = []
    for frame in traj.frames:
        new_markers = {}
        for name, marker in frame.markers.items():
            pos = np.array([marker.x, marker.y, marker.z])
            new_pos = transform @ pos
            new_markers[name] = Marker(
                name=name,
                x=new_pos[0],
                y=new_pos[1],
                z=new_pos[2],
                residual=marker.residual,
                occluded=marker.occluded,
            )
        new_frames.append(
            MarkerFrame(
                timestamp=frame.timestamp,
                markers=new_markers,
                frame_index=frame.frame_index,
            )
        )

    if center_origin:
        new_frames = _ref_center_markers_origin(new_frames)

    return MarkerTrajectory(
        id=traj.id,
        frames=new_frames,
        calibration=traj.calibration,
        subject_id=traj.subject_id,
        metadata={
            **traj.metadata,
            "normalized": True,
            "source_up": source_up.value,
            "target_up": target_up.value,
            "centered": center_origin,
        },
    )


def _ref_center_markers_origin(frames: list[MarkerFrame]) -> list[MarkerFrame]:
    if not frames:
        return frames
    first_frame = frames[0]
    all_markers = list(first_frame.markers.values())
    centroid_x = np.mean([m.x for m in all_markers])
    centroid_y = np.mean([m.y for m in all_markers])
    centroid_z = np.mean([m.z for m in all_markers])

    new_frames = []
    for frame in frames:
        new_markers = {}
        for name, marker in frame.markers.items():
            new_markers[name] = Marker(
                name=name,
                x=marker.x - centroid_x,
                y=marker.y - centroid_y,
                z=marker.z - centroid_z,
                residual=marker.residual,
                occluded=marker.occluded,
            )
        new_frames.append(
            MarkerFrame(
                timestamp=frame.timestamp,
                markers=new_markers,
                frame_index=frame.frame_index,
            )
        )
    return new_frames


def _ref_normalize_keypoints(
    seq: KeypointSequence,
    target_up: UpAxis,
    source_up: UpAxis | None,
    center_origin: bool,
) -> KeypointSequence:
    if source_up is None:
        source_up = norm._detect_up_axis_keypoints(seq.frames)
    transform = norm._get_up_axis_transform(source_up, target_up)

    new_frames = []
    for frame in seq.frames:
        new_keypoints = []
        for kp in frame.keypoints:
            pos = np.array([kp.x, kp.y, kp.z if kp.z is not None else 0.0])
            new_pos = transform @ pos
            new_kp = Keypoint(
                x=new_pos[0],
                y=new_pos[1],
                z=new_pos[2] if kp.z is not None else None,
                confidence=kp.confidence,
                name=kp.name,
            )
            new_keypoints.append(new_kp)
        new_frames.append(
            KeypointFrame(
                timestamp=frame.timestamp,
                keypoints=new_keypoints,
                schema_name=frame.schema_name,
                frame_index=frame.frame_index,
            )
        )

    if center_origin:
        new_frames = _ref_center_keypoints_origin(new_frames)

    return KeypointSequence(
        id=seq.id,
        frames=new_frames,
        calibration=seq.calibration,
        metadata={
            **seq.metadata,
            "normalized": True,
            "source_up": source_up.value,
            "target_up": target_up.value,
            "centered": center_origin,
        },
    )


def _ref_center_keypoints_origin(frames: list[KeypointFrame]) -> list[KeypointFrame]:
    if not frames:
        return frames
    first_frame = frames[0]
    centroid_x = np.mean([kp.x for kp in first_frame.keypoints])
    centroid_y = np.mean([kp.y for kp in first_frame.keypoints])
    centroid_z = np.mean([kp.z for kp in first_frame.keypoints if kp.z is not None])

    new_frames = []
    for frame in frames:
        new_keypoints = []
        for kp in frame.keypoints:
            new_kp = Keypoint(
                x=kp.x - centroid_x,
                y=kp.y - centroid_y,
                z=(kp.z - centroid_z) if kp.z is not None else None,
                confidence=kp.confidence,
                name=kp.name,
            )
            new_keypoints.append(new_kp)
        new_frames.append(
            KeypointFrame(
                timestamp=frame.timestamp,
                keypoints=new_keypoints,
                schema_name=frame.schema_name,
                frame_index=frame.frame_index,
            )
        )
    return new_frames


def _ref_convert_keypoint_units(
    seq: KeypointSequence,
    target_unit: UnitSystem,
    source_unit: UnitSystem | None,
) -> KeypointSequence:
    if source_unit is None:
        source_unit = norm._detect_unit_keypoints(seq.frames)
    scale = norm._get_unit_scale(source_unit, target_unit)

    new_frames = []
    for frame in seq.frames:
        new_keypoints = []
        for kp in frame.keypoints:
            new_kp = Keypoint(
                x=kp.x * scale,
                y=kp.y * scale,
                z=kp.z * scale if kp.z is not None else None,
                confidence=kp.confidence,
                name=kp.name,
            )
            new_keypoints.append(new_kp)
        new_frames.append(
            KeypointFrame(
                timestamp=frame.timestamp,
                keypoints=new_keypoints,
                schema_name=frame.schema_name,
                frame_index=frame.frame_index,
            )
        )
    return KeypointSequence(
        id=seq.id,
        frames=new_frames,
        calibration=seq.calibration,
        metadata={
            **seq.metadata,
            "units_converted": True,
            "source_unit": source_unit.value,
            "target_unit": target_unit.value,
        },
    )


def _ref_convert_marker_units(
    traj: MarkerTrajectory,
    target_unit: UnitSystem,
    source_unit: UnitSystem | None,
) -> MarkerTrajectory:
    if source_unit is None:
        source_unit = norm._detect_unit_markers(traj.frames)
    scale = norm._get_unit_scale(source_unit, target_unit)

    new_frames = []
    for frame in traj.frames:
        new_markers = {}
        for name, marker in frame.markers.items():
            new_markers[name] = Marker(
                name=name,
                x=marker.x * scale,
                y=marker.y * scale,
                z=marker.z * scale,
                residual=marker.residual,
                occluded=marker.occluded,
            )
        new_frames.append(
            MarkerFrame(
                timestamp=frame.timestamp,
                markers=new_markers,
                frame_index=frame.frame_index,
            )
        )
    return MarkerTrajectory(
        id=traj.id,
        frames=new_frames,
        calibration=traj.calibration,
        subject_id=traj.subject_id,
        metadata={
            **traj.metadata,
            "units_converted": True,
            "source_unit": source_unit.value,
            "target_unit": target_unit.value,
        },
    )


def _ref_detect_up_axis_keypoints(frames: list[KeypointFrame]) -> UpAxis:
    if not frames:
        return UpAxis.Y_UP
    all_x = [kp.x for f in frames for kp in f.keypoints]
    all_y = [kp.y for f in frames for kp in f.keypoints]
    all_z = [kp.z for f in frames for kp in f.keypoints if kp.z is not None]
    var_x = np.var(all_x)
    var_y = np.var(all_y)
    var_z = np.var(all_z) if all_z else 0
    if var_y > var_x and var_y > var_z:
        return UpAxis.Y_UP
    if var_z > var_x and var_z > var_y:
        return UpAxis.Z_UP
    return UpAxis.X_UP


def _ref_detect_up_axis_markers(frames: list[MarkerFrame]) -> UpAxis:
    if not frames:
        return UpAxis.Y_UP
    all_x = [m.x for f in frames for m in f.markers.values()]
    all_y = [m.y for f in frames for m in f.markers.values()]
    all_z = [m.z for f in frames for m in f.markers.values()]
    var_x = np.var(all_x)
    var_y = np.var(all_y)
    var_z = np.var(all_z)
    if var_y > var_x and var_y > var_z:
        return UpAxis.Y_UP
    if var_z > var_x and var_z > var_y:
        return UpAxis.Z_UP
    return UpAxis.X_UP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keypoint_sequence_with_missing_z(
    num_frames: int = 120, num_kp: int = 24, seed: int = 3
) -> KeypointSequence:
    """Same-index keypoints consistently 2D or 3D across frames (like a
    real per-landmark schema), covering the z-is-None threading path.
    """
    rng = np.random.default_rng(seed)
    has_z = [j % 3 != 0 for j in range(num_kp)]  # every 3rd keypoint is 2D-only
    frames = []
    for i in range(num_frames):
        kps = [
            Keypoint(
                x=float(rng.normal()),
                y=float(rng.normal()),
                z=float(rng.normal()) if has_z[j] else None,
                confidence=1.0,
                name=f"kp_{j}",
            )
            for j in range(num_kp)
        ]
        frames.append(
            KeypointFrame(
                timestamp=i / 30.0, keypoints=kps, schema_name="custom", frame_index=i
            )
        )
    return KeypointSequence(id="seq_missing_z", frames=frames)


def _assert_marker_traj_allclose(a: MarkerTrajectory, b: MarkerTrajectory) -> None:
    assert len(a.frames) == len(b.frames)
    for fa, fb in zip(a.frames, b.frames, strict=True):
        assert set(fa.markers) == set(fb.markers)
        for name in fa.markers:
            ma, mb = fa.markers[name], fb.markers[name]
            np.testing.assert_allclose(ma.x, mb.x, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(ma.y, mb.y, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(ma.z, mb.z, rtol=RTOL, atol=ATOL)
            assert ma.residual == mb.residual
            assert ma.occluded == mb.occluded


def _assert_keypoint_seq_allclose(a: KeypointSequence, b: KeypointSequence) -> None:
    assert len(a.frames) == len(b.frames)
    for fa, fb in zip(a.frames, b.frames, strict=True):
        assert len(fa.keypoints) == len(fb.keypoints)
        for ka, kb in zip(fa.keypoints, fb.keypoints, strict=True):
            np.testing.assert_allclose(ka.x, kb.x, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(ka.y, kb.y, rtol=RTOL, atol=ATOL)
            assert (ka.z is None) == (kb.z is None)
            if ka.z is not None:
                np.testing.assert_allclose(ka.z, kb.z, rtol=RTOL, atol=ATOL)
            assert ka.confidence == kb.confidence
            assert ka.name == kb.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("center_origin", [True, False])
@pytest.mark.parametrize(
    ("source_up", "target_up"),
    [
        (UpAxis.Y_UP, UpAxis.Z_UP),
        (UpAxis.Z_UP, UpAxis.X_UP),
        (UpAxis.X_UP, UpAxis.Y_UP),
        (UpAxis.Y_UP, UpAxis.Y_UP),
    ],
)
def test_normalize_markers_parity(source_up, target_up, center_origin) -> None:
    traj = make_marker_trajectory(
        num_frames=180, marker_names=[f"M{i}" for i in range(40)]
    )
    got = normalize_coordinates(
        traj, target_up=target_up, source_up=source_up, center_origin=center_origin
    )
    want = _ref_normalize_markers(
        traj, target_up=target_up, source_up=source_up, center_origin=center_origin
    )
    _assert_marker_traj_allclose(got, want)
    assert got.metadata == want.metadata


@pytest.mark.parametrize("center_origin", [True, False])
@pytest.mark.parametrize(
    ("source_up", "target_up"),
    [
        (UpAxis.Y_UP, UpAxis.Z_UP),
        (UpAxis.Z_UP, UpAxis.X_UP),
        (UpAxis.X_UP, UpAxis.Y_UP),
    ],
)
def test_normalize_keypoints_parity_full_3d(
    source_up, target_up, center_origin
) -> None:
    seq = make_keypoint_sequence(num_frames=150, num_kp=32, seed=7)
    got = normalize_coordinates(
        seq, target_up=target_up, source_up=source_up, center_origin=center_origin
    )
    want = _ref_normalize_keypoints(
        seq, target_up=target_up, source_up=source_up, center_origin=center_origin
    )
    _assert_keypoint_seq_allclose(got, want)
    assert got.metadata == want.metadata


@pytest.mark.parametrize("center_origin", [True, False])
def test_normalize_keypoints_parity_mixed_z(center_origin) -> None:
    seq = _make_keypoint_sequence_with_missing_z(num_frames=100, num_kp=24)
    got = normalize_coordinates(
        seq, target_up=UpAxis.Z_UP, source_up=UpAxis.Y_UP, center_origin=center_origin
    )
    want = _ref_normalize_keypoints(
        seq, target_up=UpAxis.Z_UP, source_up=UpAxis.Y_UP, center_origin=center_origin
    )
    _assert_keypoint_seq_allclose(got, want)


def test_convert_marker_units_parity() -> None:
    traj = make_marker_trajectory(
        num_frames=200, marker_names=[f"M{i}" for i in range(30)]
    )
    for target in (UnitSystem.MILLIMETERS, UnitSystem.CENTIMETERS, UnitSystem.INCHES):
        got = convert_units(traj, target_unit=target, source_unit=UnitSystem.METERS)
        want = _ref_convert_marker_units(
            traj, target_unit=target, source_unit=UnitSystem.METERS
        )
        _assert_marker_traj_allclose(got, want)
        assert got.metadata == want.metadata


def test_convert_keypoint_units_parity_mixed_z() -> None:
    seq = _make_keypoint_sequence_with_missing_z(num_frames=90, num_kp=18)
    for target in (UnitSystem.MILLIMETERS, UnitSystem.CENTIMETERS, UnitSystem.INCHES):
        got = convert_units(seq, target_unit=target, source_unit=UnitSystem.METERS)
        want = _ref_convert_keypoint_units(
            seq, target_unit=target, source_unit=UnitSystem.METERS
        )
        _assert_keypoint_seq_allclose(got, want)


def test_detect_up_axis_markers_parity() -> None:
    traj = make_marker_trajectory(
        num_frames=250, marker_names=[f"M{i}" for i in range(50)]
    )
    assert norm._detect_up_axis_markers(traj.frames) == _ref_detect_up_axis_markers(
        traj.frames
    )


def test_detect_up_axis_keypoints_parity_mixed_z() -> None:
    seq = _make_keypoint_sequence_with_missing_z(num_frames=150, num_kp=27)
    assert norm._detect_up_axis_keypoints(seq.frames) == _ref_detect_up_axis_keypoints(
        seq.frames
    )


def test_detect_up_axis_keypoints_parity_no_z_at_all() -> None:
    frames = []
    rng = np.random.default_rng(11)
    for i in range(50):
        kps = [
            Keypoint(x=float(rng.normal()), y=float(rng.normal()), z=None, name="kp")
            for _ in range(5)
        ]
        frames.append(
            KeypointFrame(
                timestamp=i / 30.0, keypoints=kps, schema_name="custom", frame_index=i
            )
        )
    seq = KeypointSequence(id="seq_no_z", frames=frames)
    assert norm._detect_up_axis_keypoints(seq.frames) == _ref_detect_up_axis_keypoints(
        seq.frames
    )


def test_center_keypoints_origin_parity() -> None:
    seq = _make_keypoint_sequence_with_missing_z(num_frames=60, num_kp=15)
    got = norm._center_keypoints_origin(seq.frames)
    want = _ref_center_keypoints_origin(seq.frames)
    _assert_keypoint_seq_allclose(
        KeypointSequence(id="a", frames=got), KeypointSequence(id="b", frames=want)
    )


def test_center_markers_origin_parity() -> None:
    traj = make_marker_trajectory(
        num_frames=80, marker_names=[f"M{i}" for i in range(12)]
    )
    got = norm._center_markers_origin(traj.frames)
    want = _ref_center_markers_origin(traj.frames)
    _assert_marker_traj_allclose(
        MarkerTrajectory(id="a", frames=got),
        MarkerTrajectory(id="b", frames=want),
    )


def test_normalize_keypoints_first_frame_all_z_none_produces_nan_centroid() -> None:
    """Edge case preserved from the original: if frame 0 has no 3D
    keypoints, the z-centroid is nan (matching np.mean([]) in the old
    implementation), and centering then nans out every z value.
    """
    frames = [
        KeypointFrame(
            timestamp=0.0,
            keypoints=[Keypoint(x=1.0, y=2.0, z=None, name="kp")],
            schema_name="custom",
            frame_index=0,
        ),
        KeypointFrame(
            timestamp=1.0 / 30.0,
            keypoints=[Keypoint(x=1.0, y=2.0, z=3.0, name="kp")],
            schema_name="custom",
            frame_index=1,
        ),
    ]
    seq = KeypointSequence(id="seq_edge", frames=frames)
    with pytest.warns(RuntimeWarning):
        out = normalize_coordinates(
            seq, target_up=UpAxis.Y_UP, source_up=UpAxis.Y_UP, center_origin=True
        )
    assert out.frames[0].keypoints[0].z is None
    assert np.isnan(out.frames[1].keypoints[0].z)
