"""Tests for force/torque overlay alignment with video frames (FTO-24, #11309)."""

from __future__ import annotations

import json
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
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.series import ForceTorqueSeries
from src.shared.python.simulation_backends.protocol import Trace
from src.shared.python.simulation_backends.trace_io import write_trace

pytestmark = pytest.mark.unit


def synthetic_registration(
    *,
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
    offset_s: float = 0.0,
    rate_scale: float = 1.0,
    mirror_lateral: bool = False,
    max_gap_s: float = 0.25,
) -> ReferenceRegistration:
    """Create a synthetic ReferenceRegistration fixture."""
    return ReferenceRegistration(
        reference_id=str(uuid.uuid4()),
        calibration_id="synthetic_rig_01",
        transform=ReferenceTransform(
            rotation=rotation,
            translation_m=translation_m,
            scale=scale,
        ),
        time_mapping=TimeMapping(
            offset_s=offset_s,
            rate_scale=rate_scale,
        ),
        mirror_lateral=mirror_lateral,
        max_gap_s=max_gap_s,
    )


def synthetic_force_series() -> ForceTorqueSeries:
    """Create a synthetic ForceTorqueSeries with distinct frames."""
    f0 = ForceTorqueFrame(
        time_s=1.0,
        engine="synthetic",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.EXTERNAL,
                label="external:club",
                body="club",
                point_m=(0.1, 0.2, 0.3),
                force_n=(10.0, 20.0, 30.0),
                torque_nm=(1.0, 2.0, 3.0),
                source="synthetic",
            ),
        ),
        world_frame="world_Zup",
    )
    f1 = ForceTorqueFrame(
        time_s=1.5,
        engine="synthetic",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.EXTERNAL,
                label="external:club",
                body="club",
                point_m=(0.2, 0.4, 0.6),
                force_n=(50.0, 0.0, 100.0),
                torque_nm=(5.0, 10.0, 15.0),
                source="synthetic",
            ),
        ),
        world_frame="world_Zup",
    )
    f2 = ForceTorqueFrame(
        time_s=2.0,
        engine="synthetic",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.EXTERNAL,
                label="external:club",
                body="club",
                point_m=(0.3, 0.6, 0.9),
                force_n=(100.0, 0.0, 200.0),
                torque_nm=(10.0, 20.0, 30.0),
                source="synthetic",
            ),
        ),
        world_frame="world_Zup",
    )
    return ForceTorqueSeries(engine="synthetic", frames=(f0, f1, f2))


def test_time_mapping_aligns_video_to_series_time() -> None:
    """Identity registration and affine time map: video t=1.0 samples series t=1.5."""
    # offset_s = -0.5 with rate_scale = 1.0 yields:
    # scene_to_reference(1.0) = (1.0 - (-0.5)) / 1.0 = 1.5
    reg = synthetic_registration(offset_s=-0.5, rate_scale=1.0)
    series = synthetic_force_series()

    frame = force_frame_for_video(series, video_time_s=1.0, registration=reg)
    assert frame is not None
    assert frame.time_s == pytest.approx(1.0)
    assert frame.world_frame == "adr0041_world"
    assert frame.metadata.get("registration_scale") == pytest.approx(1.0)

    # Frame values correspond to series frame at t = 1.5
    w = frame.wrenches[0]
    assert w.label == "external:club"
    # Canonical point (0.2, 0.4, 0.6) mapped via canonical_z_up_to_adr0041_world:
    # x_adr = 0.2, y_adr = 0.6, z_adr = -0.4
    np.testing.assert_allclose(w.point_m, (0.2, 0.6, -0.4), atol=1e-9)


def test_z_up_force_maps_to_adr0041_y_up() -> None:
    """A canonical Z-up +z force vector maps to ADR-0041 +y vector."""
    reg = synthetic_registration()
    f0 = ForceTorqueFrame(
        time_s=0.0,
        engine="synthetic",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.CONTACT,
                label="contact:ground",
                body="lead_foot",
                point_m=(0.0, 0.0, 0.0),
                force_n=(0.0, 0.0, 500.0),  # +z in Z-up canonical
                torque_nm=None,
                source="synthetic",
            ),
        ),
        world_frame="world_Zup",
    )
    series = ForceTorqueSeries(engine="synthetic", frames=(f0,))

    frame = force_frame_for_video(series, video_time_s=0.0, registration=reg)
    assert frame is not None
    assert frame.world_frame == "adr0041_world"
    w = frame.wrenches[0]
    # In ADR-0041, y is up: canonical (0, 0, 500) -> (0, 500, 0)
    np.testing.assert_allclose(w.force_n, (0.0, 500.0, 0.0), atol=1e-9)


def test_rotation_scale_and_translation() -> None:
    """A registration rotation of 90 deg about y rotates force; scale/trans apply to points only."""
    # 90 deg rotation around scene Y:
    # x' = z, y' = y, z' = -x
    rot_y90 = (
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    )
    reg = synthetic_registration(
        rotation=rot_y90,
        translation_m=(10.0, 20.0, 30.0),
        scale=2.0,
    )
    f0 = ForceTorqueFrame(
        time_s=0.0,
        engine="synthetic",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.EXTERNAL,
                label="external:test",
                body="club",
                point_m=(1.0, 0.0, 0.0),  # Canonical point
                force_n=(100.0, 0.0, 0.0),  # Canonical force
                torque_nm=(0.0, 50.0, 0.0),  # Canonical torque along Y
                source="synthetic",
            ),
        ),
        world_frame="world_Zup",
    )
    series = ForceTorqueSeries(engine="synthetic", frames=(f0,))
    frame = force_frame_for_video(series, video_time_s=0.0, registration=reg)
    assert frame is not None
    w = frame.wrenches[0]

    # Force:
    # canonical (100, 0, 0) -> adr0041 (100, 0, 0)
    # rot_y90 @ (100, 0, 0) = (0, 0, -100)
    # Magnitude unchanged = 100.0
    np.testing.assert_allclose(w.force_n, (0.0, 0.0, -100.0), atol=1e-9)
    assert np.linalg.norm(w.force_n) == pytest.approx(100.0)

    # Point:
    # canonical (1, 0, 0) -> adr0041 (1, 0, 0)
    # scale * rot_y90 @ (1, 0, 0) + trans = 2.0 * (0, 0, -1) + (10, 20, 30) = (10, 20, 28)
    np.testing.assert_allclose(w.point_m, (10.0, 20.0, 28.0), atol=1e-9)

    # Torque:
    # canonical (0, 50, 0) -> adr0041: x=0, y=0, z=-50 -> (0, 0, -50)
    # rot_y90 @ (0, 0, -50) = (-50, 0, 0)
    np.testing.assert_allclose(w.torque_nm, (-50.0, 0.0, 0.0), atol=1e-9)
    assert np.linalg.norm(w.torque_nm) == pytest.approx(50.0)
    assert frame.metadata.get("registration_scale") == pytest.approx(2.0)


def test_mirrored_registration_polar_force_and_axial_torque() -> None:
    r"""Verify lateral mirroring transforms forces as polar vectors and torques as axial vectors.

    Worked Numeric Example:
    -----------------------
    Canonical coordinates (Z-up):
      p_can = (0.0, 1.0, 0.0)
      F_can = (1.0, 0.0, 0.0)
      tau_can = p_can x F_can = (0.0, 0.0, -1.0)
    Under lateral mirror (mirror_lateral=True, reflection across Y in canonical):
      1. Point:
         p_m = (0.0, -1.0, 0.0)
         Canonical to ADR-0041: x = 0, y = z = 0, z = -y = 1
         p_scene = (0.0, 0.0, 1.0)
      2. Force (polar vector):
         F_m = M_can @ F_can = (1.0, 0.0, 0.0)
         Canonical to ADR-0041: x = 1, y = 0, z = 0
         F_scene = (1.0, 0.0, 0.0)
      3. Torque (axial vector / pseudovector):
         Gains an additional sign flip from det(M) = -1:
         tau_m = (-1) * (M_can @ tau_can) = (-1) * (0.0, 0.0, -1.0) = (0.0, 0.0, 1.0)
         Canonical to ADR-0041: x = 0, y = z = 1, z = -y = 0
         tau_scene = (0.0, 1.0, 0.0)
      Check physical consistency in scene frame:
         p_scene x F_scene = (0.0, 0.0, 1.0) x (1.0, 0.0, 0.0) = (0.0, 1.0, 0.0) == tau_scene!
    """
    reg = synthetic_registration(mirror_lateral=True)
    f0 = ForceTorqueFrame(
        time_s=0.0,
        engine="synthetic",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.EXTERNAL,
                label="external:test",
                body="club",
                point_m=(0.0, 1.0, 0.0),
                force_n=(1.0, 0.0, 0.0),
                torque_nm=(0.0, 0.0, -1.0),
                source="synthetic",
            ),
        ),
        world_frame="world_Zup",
    )
    series = ForceTorqueSeries(engine="synthetic", frames=(f0,))
    frame = force_frame_for_video(series, video_time_s=0.0, registration=reg)
    assert frame is not None
    w = frame.wrenches[0]

    np.testing.assert_allclose(w.point_m, (0.0, 0.0, 1.0), atol=1e-9)
    np.testing.assert_allclose(w.force_n, (1.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(w.torque_nm, (0.0, 1.0, 0.0), atol=1e-9)

    # Physical cross product check
    expected_cross = np.cross(w.point_m, w.force_n)
    np.testing.assert_allclose(w.torque_nm, expected_cross, atol=1e-9)


def test_beyond_max_gap_returns_none() -> None:
    """When video timestamp maps beyond max_gap_s, returns None."""
    reg = synthetic_registration(max_gap_s=0.1)
    f0 = ForceTorqueFrame(time_s=1.0, engine="synthetic", world_frame="world_Zup")
    f1 = ForceTorqueFrame(time_s=2.0, engine="synthetic", world_frame="world_Zup")
    series = ForceTorqueSeries(engine="synthetic", frames=(f0, f1))

    # Requested at t = 1.5 s: gap is 1.0 s, exceeding max_gap_s=0.1
    frame = force_frame_for_video(series, video_time_s=1.5, registration=reg)
    assert frame is None

    # Outside series range
    frame_outside = force_frame_for_video(series, video_time_s=0.5, registration=reg)
    assert frame_outside is None


def test_trace_hdf5_force_group_roundtrip() -> None:
    """Trace HDF5 with force_torque_series group roundtrips losslessly."""
    series = synthetic_force_series()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_trace.h5"

        # Create dummy trace and write it
        trace = Trace(
            t=np.array([1.0, 1.5, 2.0]),
            q=np.zeros((3, 2)),
            v=np.zeros((3, 2)),
            backend="synthetic",
        )
        write_trace(trace, path)

        # Write force series into the trace
        write_trace_forces(path, series)

        # Load back
        loaded = load_trace_forces(path)
        assert loaded is not None
        assert loaded.engine == series.engine
        assert len(loaded.frames) == len(series.frames)
        for orig_f, loaded_f in zip(series.frames, loaded.frames, strict=True):
            assert loaded_f.time_s == pytest.approx(orig_f.time_s)
            assert len(loaded_f.wrenches) == len(orig_f.wrenches)
            for ow, lw in zip(orig_f.wrenches, loaded_f.wrenches, strict=True):
                assert lw.label == ow.label
                assert lw.kind == ow.kind
                np.testing.assert_allclose(lw.point_m, ow.point_m)
                if ow.force_n is not None:
                    np.testing.assert_allclose(lw.force_n, ow.force_n)
                if ow.torque_nm is not None:
                    np.testing.assert_allclose(lw.torque_nm, ow.torque_nm)


def test_trace_without_declared_wrench_point_gives_no_fabricated_wrench() -> None:
    """A trace with wrench array but without declared point in meta gives None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace_no_point.h5"
        trace = Trace(
            t=np.array([0.0, 0.1, 0.2]),
            q=np.zeros((3, 2)),
            v=np.zeros((3, 2)),
            wrench=np.ones((3, 6)),  # 6-axis wrench but no point declared
            backend="mujoco",
        )
        write_trace(trace, path)

        loaded = load_trace_forces(path)
        assert loaded is None


def test_trace_with_declared_wrench_point_imports_external_wrench() -> None:
    """A trace with wrench array and declared point in meta imports as EXTERNAL wrench."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace_with_point.h5"
        trace = Trace(
            t=np.array([0.0, 0.1]),
            q=np.zeros((2, 2)),
            v=np.zeros((2, 2)),
            wrench=np.array(
                [
                    [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
                    [15.0, 25.0, 35.0, 1.5, 2.5, 3.5],
                ]
            ),
            meta={"root_point_m": json.dumps([0.0, 0.0, 0.85])},
            backend="ode",
        )
        write_trace(trace, path)

        loaded = load_trace_forces(path)
        assert loaded is not None
        assert loaded.engine == "ode"
        assert len(loaded.frames) == 2
        f0 = loaded.frames[0]
        assert len(f0.wrenches) == 1
        w0 = f0.wrenches[0]
        assert w0.kind == WrenchKind.EXTERNAL
        assert w0.label == "external:root"
        np.testing.assert_allclose(w0.point_m, (0.0, 0.0, 0.85))
        np.testing.assert_allclose(w0.force_n, (10.0, 20.0, 30.0))
        np.testing.assert_allclose(w0.torque_nm, (1.0, 2.0, 3.0))


def test_series_to_viewport_payload_wrench() -> None:
    """Sum of external and contact wrenches about the world origin."""
    f0 = ForceTorqueFrame(
        time_s=0.0,
        engine="synthetic",
        wrenches=(
            # Force 100 N along Z applied at (0, 2, 0). Moment arm r x F = (0, 2, 0) x (0, 0, 100) = (200, 0, 0)
            OverlayWrench(
                kind=WrenchKind.CONTACT,
                label="contact:ground",
                body="foot",
                point_m=(0.0, 2.0, 0.0),
                force_n=(0.0, 0.0, 100.0),
                torque_nm=(10.0, 0.0, 0.0),  # Pure torque 10 N*m along X
                source="synthetic",
            ),
            # Internal joint actuator: MUST be ignored
            OverlayWrench(
                kind=WrenchKind.JOINT_ACTUATOR,
                label="actuator:shoulder",
                body="arm",
                point_m=(0.0, 0.0, 1.5),
                force_n=(0.0, 0.0, 999.0),
                torque_nm=(999.0, 0.0, 0.0),
                source="synthetic",
            ),
        ),
        world_frame="world_Zup",
    )
    series = ForceTorqueSeries(engine="synthetic", frames=(f0,))
    payload = series_to_viewport_payload_wrench(series)

    assert payload.shape == (1, 6)
    # Total force: (0, 0, 100)
    # Total torque about origin: tau_point + r x F = (10, 0, 0) + (200, 0, 0) = (210, 0, 0)
    np.testing.assert_allclose(
        payload[0], [0.0, 0.0, 100.0, 210.0, 0.0, 0.0], atol=1e-9
    )
