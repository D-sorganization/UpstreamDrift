"""Tests for force/torque series alignment with video frames and trace import (FTO-24, #11309).

Authorities and conventions:
- Space: canonical_z_up_to_adr0041_world maps canonical Z-up to ADR-0041 camera world (x_world = x_can, y_world = z_can, z_world = -y_can).
- Time: ReferenceRegistration.scene_time and TimeMapping.scene_to_reference.
- Vectors: forces transform as polar vectors (F_world = R_total @ F_can);
  torques transform as axial vectors / pseudovectors (tau_world = det(R_total) * (R_total @ tau_can)).
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import uuid

import numpy as np
import pytest

from src.motion_capture.reference.force_alignment import (
    force_frame_for_video,
    load_trace_forces,
    series_to_viewport_payload_wrench,
    write_trace_forces,
)
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    ReferenceTransform,
    TimeMapping,
)
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    ForceTorqueSeries,
    OverlayWrench,
    WrenchKind,
)

pytestmark = pytest.mark.unit


def _synthetic_registration(
    *,
    offset_s: float = 0.0,
    rate_scale: float = 1.0,
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    mirror_lateral: bool = False,
) -> ReferenceRegistration:
    """Create a valid synthetic ReferenceRegistration."""
    return ReferenceRegistration(
        reference_id=str(uuid.uuid4()),
        calibration_id="synthetic_calib_001",
        transform=ReferenceTransform(
            rotation=rotation,
            translation_m=translation_m,
            scale=scale,
            is_calibrated=True,
        ),
        time_mapping=TimeMapping(offset_s=offset_s, rate_scale=rate_scale),
        is_calibrated=True,
        mirror_lateral=mirror_lateral,
    )


def test_time_mapping_alignment() -> None:
    """Identity registration and affine time map (offset -0.5s, rate 1):

    time_mapping.scene_to_reference maps scene_t to ref_t.
    With offset_s = -0.5, scene_to_reference(1.0) = 1.0 - (-0.5) = 1.5 s.
    At video t = 1.0, the wrench comes from series t = 1.5.
    """
    reg = _synthetic_registration(offset_s=-0.5, rate_scale=1.0)

    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:lead_foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 500.0),
        source="sim",
    )
    f1 = ForceTorqueFrame(time_s=1.0, engine="mujoco", wrenches=(w1,))

    w2 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:lead_foot",
        body="foot",
        point_m=(0.1, 0.2, 0.3),
        force_n=(10.0, 20.0, 800.0),
        source="sim",
    )
    f2 = ForceTorqueFrame(time_s=1.5, engine="mujoco", wrenches=(w2,))

    series = ForceTorqueSeries(frames=(f1, f2), engine="mujoco")

    # Sample at video_time_s = 1.0 -> ref_t = 1.5 -> matches f2
    result = force_frame_for_video(series, video_time_s=1.0, registration=reg)
    assert result is not None
    assert result.time_s == 1.0
    assert result.world_frame == "adr0041_world"
    assert len(result.wrenches) == 1
    w_res = result.wrenches[0]
    assert w_res.label == "contact:lead_foot"
    # Point in f2 was (0.1, 0.2, 0.3) in canonical z-up
    # Under canonical_z_up_to_adr0041_world: (x, z, -y) = (0.1, 0.3, -0.2)
    assert np.allclose(w_res.point_m, (0.1, 0.3, -0.2))


def test_canonical_z_up_force_maps_to_adr0041_y_up() -> None:
    """A canonical Z-up +z force (0, 0, 100) maps to ADR-0041 +y (0, 100, 0).

    In canonical z-up:
      X = forward, Y = left, Z = up.
    In ADR-0041 world:
      X = forward, Y = up, Z = right.
    Therefore, +z canonical maps to +y world.
    """
    reg = _synthetic_registration()

    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(1.0, 2.0, 3.0),
        force_n=(0.0, 0.0, 100.0),
        source="sim",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(w,))
    series = ForceTorqueSeries(frames=(frame,), engine="mujoco")

    res = force_frame_for_video(series, video_time_s=0.0, registration=reg)
    assert res is not None
    assert res.world_frame == "adr0041_world"
    w_out = res.wrenches[0]

    # Point: (1, 2, 3) -> (1, 3, -2)
    assert np.allclose(w_out.point_m, (1.0, 3.0, -2.0))
    # Force: (0, 0, 100) -> (0, 100, 0)
    assert w_out.force_n is not None
    assert np.allclose(w_out.force_n, (0.0, 100.0, 0.0))


def test_registration_rotation_translation_scale() -> None:
    """A 90 deg rotation about Y in scene rotates the force; magnitude is unchanged;

    point is translated and scaled.
    """
    # 90 deg rotation about scene Y (Y is up in ADR-0041):
    # R_y(90 deg): x' = z, y' = y, z' = -x
    rot_90_y = (
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    )
    reg = _synthetic_registration(
        rotation=rot_90_y,
        translation_m=(10.0, 20.0, 30.0),
        scale=2.0,
    )

    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(1.0, 0.0, 0.0),  # In canonical z-up, forward +x
        force_n=(50.0, 0.0, 0.0),  # Forward +x force of 50 N
        source="sim",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(w,))
    series = ForceTorqueSeries(frames=(frame,), engine="mujoco")

    res = force_frame_for_video(series, video_time_s=0.0, registration=reg)
    assert res is not None
    w_out = res.wrenches[0]

    # Force magnitude must be strictly preserved (never scaled): 50 N
    assert w_out.force_n is not None
    mag = np.linalg.norm(w_out.force_n)
    assert np.isclose(mag, 50.0)

    # In canonical z-up to adr0041: (1, 0, 0) -> (1, 0, 0).
    # Then rotated by R_y(90 deg): (1, 0, 0) -> (0, 0, -1).
    assert np.allclose(w_out.force_n, (0.0, 0.0, -50.0))

    # Point: canonical (1, 0, 0) -> adr0041 (1, 0, 0).
    # apply: scale * R @ p + t = 2.0 * (0, 0, -1) + (10, 20, 30) = (10, 20, 28)
    assert np.allclose(w_out.point_m, (10.0, 20.0, 28.0))

    # Scale metadata is recorded in frame metadata
    assert res.metadata.get("registration_scale") == 2.0


def test_mirrored_registration_polar_force_and_axial_torque() -> None:
    r"""Worked numeric example for mirrored registration (det R = -1).

    In canonical z-up:
      X = forward (toward target)
      Y = lateral (golfer's left)
      Z = vertical (up)

    Mirroring reflects the lateral axis: Y_can -> -Y_can.
    Transformation from canonical to ADR-0041 camera world:
      x_adr = x_can
      y_adr = z_can
      z_adr = -y_can
    With lateral mirror: y_can_mirrored = -y_can, so:
      z_adr = -(-y_can) = y_can.
    Total coordinate matrix:
      A = [[1, 0, 0],
           [0, 0, 1],
           [0, 1, 0]]
      det(A) = 1*(0 - 1) = -1.

    Polar vector (Force) transforms via A:
      F_can = (10.0, 20.0, 30.0) N
      F_world = A @ F_can = (10.0, 30.0, 20.0) N.

    Axial vector (Torque / pseudovector) transforms via det(A) * (A @ tau):
      tau = r x F. Under parity reflection, r -> r' and F -> F', so
      (r' x F') gains an extra parity flip compared to a simple matrix multiply.
      tau_can = (10.0, 20.0, 30.0) N*m
      tau_world = det(A) * (A @ tau_can)
                = -1 * (10.0, 30.0, 20.0)
                = (-10.0, -30.0, -20.0) N*m.

    Point transformation:
      p_can = (1.0, 2.0, 3.0) m
      p_world = (1.0, 3.0, 2.0) m.
    """
    reg = _synthetic_registration(mirror_lateral=True)

    w = OverlayWrench(
        kind=WrenchKind.EXTERNAL,
        label="external:test",
        body="torso",
        point_m=(1.0, 2.0, 3.0),
        force_n=(10.0, 20.0, 30.0),
        torque_nm=(10.0, 20.0, 30.0),
        source="sim",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(w,))
    series = ForceTorqueSeries(frames=(frame,), engine="mujoco")

    res = force_frame_for_video(series, video_time_s=0.0, registration=reg)
    assert res is not None
    w_out = res.wrenches[0]

    # Point: (1.0, 3.0, 2.0)
    assert np.allclose(w_out.point_m, (1.0, 3.0, 2.0))
    # Polar vector (Force): (10.0, 30.0, 20.0)
    assert w_out.force_n is not None
    assert np.allclose(w_out.force_n, (10.0, 30.0, 20.0))
    # Axial vector (Torque): (-10.0, -30.0, -20.0) -- extra sign flip from det(A) = -1
    assert w_out.torque_nm is not None
    assert np.allclose(w_out.torque_nm, (-10.0, -30.0, -20.0))


def test_gap_exceeding_max_gap_returns_none() -> None:
    """When video timestamp maps to a reference time with no series frame within max_gap_s,

    returns None.
    """
    reg = _synthetic_registration()

    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        source="sim",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(w,))
    series = ForceTorqueSeries(frames=(frame,), engine="mujoco")

    # video_time_s = 1.0 is 1.0 s away from frame at 0.0 s > max_gap_s (0.1)
    res = force_frame_for_video(
        series, video_time_s=1.0, registration=reg, max_gap_s=0.1
    )
    assert res is None


def test_trace_hdf5_force_group_roundtrip() -> None:
    """Trace HDF5 file with force_torque_series group roundtrips identically."""
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:lead",
        body="lead_foot",
        point_m=(0.1, 0.2, 0.3),
        force_n=(10.0, 20.0, 30.0),
        source="sim",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:wrist",
        body="wrist",
        point_m=(0.4, 0.5, 0.6),
        torque_nm=(5.0, 6.0, 7.0),
        source="sim",
    )
    f0 = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(w1,))
    f1 = ForceTorqueFrame(time_s=0.05, engine="mujoco", wrenches=(w1, w2))
    series = ForceTorqueSeries(frames=(f0, f1), engine="mujoco")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        path = Path(tmp.name)

    try:
        write_trace_forces(path, series)
        loaded = load_trace_forces(path)
        assert loaded is not None
        assert loaded.engine == series.engine
        assert len(loaded.frames) == len(series.frames)
        for f_orig, f_load in zip(series.frames, loaded.frames, strict=True):
            assert math.isclose(f_orig.time_s, f_load.time_s)
            assert f_orig.engine == f_load.engine
            assert len(f_orig.wrenches) == len(f_load.wrenches)
            for w_o, w_l in zip(f_orig.wrenches, f_load.wrenches, strict=True):
                assert w_o.kind == w_l.kind
                assert w_o.label == w_l.label
                assert w_o.body == w_l.body
                assert np.allclose(w_o.point_m, w_l.point_m)
                if w_o.force_n is not None:
                    assert w_l.force_n is not None
                    assert np.allclose(w_o.force_n, w_l.force_n)
                else:
                    assert w_l.force_n is None
                if w_o.torque_nm is not None:
                    assert w_l.torque_nm is not None
                    assert np.allclose(w_o.torque_nm, w_l.torque_nm)
                else:
                    assert w_l.torque_nm is None
    finally:
        path.unlink(missing_ok=True)


def test_trace_without_declared_wrench_point_gives_no_fabricated_wrench() -> None:
    """Trace without declared wrench point in meta gives None (never fabricates a point)."""
    import h5py

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        path = Path(tmp.name)

    try:
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = "2.0.0"
            f.create_dataset("t", data=np.array([0.0, 0.01]))
            f.create_dataset("q", data=np.zeros((2, 4)))
            f.create_dataset("v", data=np.zeros((2, 4)))
            f.create_dataset("wrench", data=np.ones((2, 6)))  # Wrench without point!
            # meta does NOT contain wrench_point
        loaded = load_trace_forces(path)
        assert loaded is None
    finally:
        path.unlink(missing_ok=True)


def test_series_to_viewport_payload_wrench() -> None:
    """series_to_viewport_payload_wrench produces (T, 6) sum of contact and external wrenches

    about the world origin.
    """
    # Frame 0: Contact at (1, 0, 0) with force (0, 0, 100).
    # Torque about origin = r x F = (1, 0, 0) x (0, 0, 100) = (0, -100, 0).
    w_cont = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(1.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        torque_nm=(0.0, 0.0, 10.0),
        source="sim",
    )
    # Total torque about origin = w_cont.torque + r x F = (0, 0, 10) + (0, -100, 0) = (0, -100, 10).
    # Joint actuator should be IGNORED (only contact/external are summed):
    w_act = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:wrist",
        body="wrist",
        point_m=(0.0, 0.0, 1.0),
        torque_nm=(50.0, 0.0, 0.0),
        source="sim",
    )
    f0 = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(w_cont, w_act))

    # Frame 1: No external/contact wrenches
    f1 = ForceTorqueFrame(time_s=0.1, engine="mujoco", wrenches=(w_act,))

    series = ForceTorqueSeries(frames=(f0, f1), engine="mujoco")
    payload = series_to_viewport_payload_wrench(series)

    assert payload.shape == (2, 6)
    # Frame 0: force (0, 0, 100), torque (0, -100, 10)
    assert np.allclose(payload[0], [0.0, 0.0, 100.0, 0.0, -100.0, 10.0])
    # Frame 1: all zeros
    assert np.allclose(payload[1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
