"""Unit tests for ForceTorqueSeries interpolation, dict and npz persistence."""

from __future__ import annotations

import io
import pytest

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    ForceTorqueSeries,
    OverlayWrench,
    WrenchKind,
)

pytestmark = pytest.mark.unit


def test_series_validation_increasing_times():
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10.0),
        source="test",
    )
    f1 = ForceTorqueFrame(time_s=1.0, engine="engineA", wrenches=(w,))
    f2 = ForceTorqueFrame(time_s=0.5, engine="engineA", wrenches=(w,))  # decreasing!
    with pytest.raises(ValueError, match="strictly increasing"):
        ForceTorqueSeries(frames=(f1, f2), engine="engineA")


def test_series_validation_single_engine():
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10.0),
        source="test",
    )
    f1 = ForceTorqueFrame(time_s=0.0, engine="engineA", wrenches=(w,))
    f2 = ForceTorqueFrame(time_s=1.0, engine="engineB", wrenches=(w,))
    with pytest.raises(ValueError, match="engine mismatch"):
        ForceTorqueSeries(frames=(f1, f2), engine="engineA")


def test_series_frame_at_interpolation():
    w0_a = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10.0),
        torque_nm=(1.0, 0.0, 0.0),
        source="test",
    )
    w0_b = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:wrist",
        body="hand",
        point_m=(0.0, 0.1, 0.0),
        torque_nm=(2.0, 0.0, 0.0),  # torque only
        source="test",
    )
    al0 = AxialLoadFrame(time_s=0.0, values_n={"shaft": 10.0}, source="test")
    f0 = ForceTorqueFrame(
        time_s=0.0,
        engine="mujoco",
        wrenches=(w0_a, w0_b),
        axial_loads=al0,
    )

    w1_a = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(2.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 30.0),
        torque_nm=None,  # torque missing in f1!
        source="test",
    )
    w1_c = OverlayWrench(
        kind=WrenchKind.EXTERNAL,
        label="ext:push",  # only in f1!
        body="torso",
        point_m=(0.0, 0.0, 1.0),
        force_n=(5.0, 0.0, 0.0),
        source="test",
    )
    al1 = AxialLoadFrame(time_s=1.0, values_n={"shaft": 30.0}, source="test")
    f1 = ForceTorqueFrame(
        time_s=1.0,
        engine="mujoco",
        wrenches=(w1_a, w1_c),
        axial_loads=al1,
    )

    series = ForceTorqueSeries(frames=(f0, f1), engine="mujoco")

    # 1. Exact time returns exact frame
    assert series.frame_at(0.0, max_gap_s=1.5) == f0
    assert series.frame_at(1.0, max_gap_s=1.5) == f1

    # 2. Out of range returns None
    assert series.frame_at(-0.1, max_gap_s=1.5) is None
    assert series.frame_at(1.1, max_gap_s=1.5) is None

    # 3. Gap larger than max_gap_s returns None
    assert series.frame_at(0.5, max_gap_s=0.8) is None

    # 4. Midpoint interpolation (t=0.5, alpha=0.5)
    f_mid = series.frame_at(0.5, max_gap_s=1.5)
    assert f_mid is not None
    assert f_mid.time_s == 0.5
    assert f_mid.engine == "mujoco"

    # Only contact:foot was present in both neighbours!
    # joint:wrist and ext:push were in only one neighbour, so omitted!
    assert len(f_mid.wrenches) == 1
    w_interp = f_mid.wrenches[0]
    assert w_interp.label == "contact:foot"
    assert w_interp.point_m == (1.0, 0.0, 0.0)
    # Force was present in both: average (0.0, 0.0, 20.0)
    assert w_interp.force_n == (0.0, 0.0, 20.0)
    # Torque was present in f0 but missing in f1: becomes None!
    assert w_interp.torque_nm is None

    # Axial loads interpolated: (10 + 30)/2 = 20.0
    assert f_mid.axial_loads is not None
    assert f_mid.axial_loads.values_n["shaft"] == 20.0


def test_series_to_dict_and_from_dict():
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10.0),
        source="test",
    )
    f1 = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w,))
    f2 = ForceTorqueFrame(time_s=0.1, engine="test", wrenches=(w,))
    series = ForceTorqueSeries(frames=(f1, f2), engine="test")

    d = series.to_dict()
    assert d["engine"] == "test"
    assert len(d["frames"]) == 2

    restored = ForceTorqueSeries.from_dict(d)
    assert restored.engine == series.engine
    assert len(restored.frames) == 2
    assert restored.frames[0].time_s == 0.0
    assert restored.frames[1].time_s == 0.1


def test_series_npz_round_trip_preserves_masks():
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10.0),
        torque_nm=None,  # missing torque
        source="test",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:wrist",
        body="hand",
        point_m=(0.1, 0.2, 0.3),
        force_n=None,  # missing force
        torque_nm=(1.0, 2.0, 3.0),
        source="test",
    )
    f1 = ForceTorqueFrame(time_s=0.0, engine="npz_engine", wrenches=(w1, w2))
    f2 = ForceTorqueFrame(time_s=0.5, engine="npz_engine", wrenches=(w1, w2))
    series = ForceTorqueSeries(frames=(f1, f2), engine="npz_engine")

    buf = io.BytesIO()
    series.to_npz(buf)
    buf.seek(0)

    restored = ForceTorqueSeries.from_npz(buf)
    assert restored.engine == "npz_engine"
    assert len(restored.frames) == 2
    assert restored.frames[0].wrenches[0].torque_nm is None
    assert restored.frames[0].wrenches[1].force_n is None
    assert restored.frames[0].wrenches[0].force_n == (0.0, 0.0, 10.0)
    assert restored.frames[0].wrenches[1].torque_nm == (1.0, 2.0, 3.0)


def test_series_empty_and_disjoint_interpolation():
    empty_series = ForceTorqueSeries(frames=(), engine="empty_engine")
    assert empty_series.frame_at(0.0, max_gap_s=1.0) is None

    # Wrenches that share a label but have disjoint halves (w0 has only force, w1 has only torque)
    # Neither half is present in both, so both become None -> wrench is omitted!
    w0 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:disjoint",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(1.0, 0.0, 0.0),
        torque_nm=None,
        source="test",
    )
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:disjoint",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=None,
        torque_nm=(0.0, 1.0, 0.0),
        source="test",
    )
    # Asymmetric axial load keys: f0 has shaft_a, f1 has shaft_b
    al0 = AxialLoadFrame(time_s=0.0, values_n={"shaft_a": 10.0}, source="test")
    al1 = AxialLoadFrame(time_s=1.0, values_n={"shaft_b": 20.0}, source="test")

    f0 = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w0,), axial_loads=al0)
    f1 = ForceTorqueFrame(time_s=1.0, engine="test", wrenches=(w1,), axial_loads=al1)
    s = ForceTorqueSeries(frames=(f0, f1), engine="test")

    f_mid = s.frame_at(0.5, max_gap_s=1.5)
    assert f_mid is not None
    # Wrench was omitted because neither half was present in both neighbours
    assert len(f_mid.wrenches) == 0
    # Asymmetric keys both become None in interpolated frame
    assert f_mid.axial_loads is not None
    assert f_mid.axial_loads.values_n["shaft_a"] is None
    assert f_mid.axial_loads.values_n["shaft_b"] is None
