"""Unit tests for ForceTorqueSeries interpolation, storage, and round-trips."""

from __future__ import annotations

import io
import math
import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    ForceTorqueSeries,
    OverlayWrench,
    WrenchKind,
)

pytestmark = pytest.mark.unit


def _create_sample_series() -> ForceTorqueSeries:
    w0_both = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:hip_r",
        body="femur_r",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 10.0, 0.0),
        torque_nm=(5.0, 0.0, 0.0),
        source="engine",
    )
    w0_force = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:heel_r",
        body="calcn_r",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        torque_nm=None,
        source="engine",
    )
    axial0 = AxialLoadFrame(time_s=0.0, values_n={"femur_r": -50.0}, source="engine")
    f0 = ForceTorqueFrame(
        time_s=0.0,
        engine="mujoco",
        wrenches=(w0_both, w0_force),
        axial_loads=axial0,
    )

    w1_both = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:hip_r",
        body="femur_r",
        point_m=(0.0, 0.0, 0.2),
        force_n=(0.0, 30.0, 0.0),
        torque_nm=(15.0, 0.0, 0.0),
        source="engine",
    )
    w1_force = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:heel_r",
        body="calcn_r",
        point_m=(0.0, 0.0, 0.0),
        force_n=None,  # now force unavailable
        torque_nm=(0.0, 1.0, 0.0),  # torque newly available
        source="engine",
    )
    w1_extra = OverlayWrench(
        kind=WrenchKind.GRIP,
        label="grip:lead",
        body="hand",
        point_m=(0.0, 0.0, 1.0),
        force_n=(10.0, 0.0, 0.0),
        torque_nm=None,
        source="engine",
    )
    axial1 = AxialLoadFrame(time_s=1.0, values_n={"femur_r": -150.0}, source="engine")
    f1 = ForceTorqueFrame(
        time_s=1.0,
        engine="mujoco",
        wrenches=(w1_both, w1_force, w1_extra),
        axial_loads=axial1,
    )

    return ForceTorqueSeries(engine="mujoco", frames=(f0, f1))


def test_series_validation():
    f0 = ForceTorqueFrame(time_s=1.0, engine="mujoco", wrenches=())
    f1_bad_time = ForceTorqueFrame(time_s=0.5, engine="mujoco", wrenches=())
    f1_bad_engine = ForceTorqueFrame(time_s=2.0, engine="drake", wrenches=())

    with pytest.raises(ValueError, match="strictly increasing"):
        ForceTorqueSeries(engine="mujoco", frames=(f0, f1_bad_time))

    with pytest.raises(ValueError, match="engine mismatch"):
        ForceTorqueSeries(engine="mujoco", frames=(f0, f1_bad_engine))

    with pytest.raises(ValueError, match="engine mismatch"):
        ForceTorqueSeries(engine="pinocchio", frames=(f0,))


def test_series_frame_at_exact_and_bounds():
    series = _create_sample_series()

    # Exact time returns that frame
    f0_ret = series.frame_at(0.0, max_gap_s=1.5)
    assert f0_ret is not None
    assert f0_ret.time_s == 0.0
    assert len(f0_ret.wrenches) == 2

    f1_ret = series.frame_at(1.0, max_gap_s=1.5)
    assert f1_ret is not None
    assert f1_ret.time_s == 1.0
    assert len(f1_ret.wrenches) == 3

    # Out of bounds returns None
    assert series.frame_at(-0.1, max_gap_s=1.5) is None
    assert series.frame_at(1.1, max_gap_s=1.5) is None

    # Gap exceeded returns None
    assert series.frame_at(0.5, max_gap_s=0.5) is None


def test_series_frame_at_interpolation():
    series = _create_sample_series()

    mid = series.frame_at(0.5, max_gap_s=1.5)
    assert mid is not None
    assert mid.time_s == 0.5
    assert mid.engine == "mujoco"

    # w_both present in both with both halves: average point, force, torque
    hip = next(w for w in mid.wrenches if w.label == "joint:hip_r")
    assert hip.point_m == pytest.approx((0.0, 0.0, 0.1))
    assert hip.force_n == pytest.approx((0.0, 20.0, 0.0))
    assert hip.torque_nm == pytest.approx((10.0, 0.0, 0.0))

    # w_force had force in f0 and torque in f1: both halves None at midpoint -> entire wrench omitted
    heel = [w for w in mid.wrenches if w.label == "contact:heel_r"]
    assert len(heel) == 0

    # w1_extra present only in f1: omitted
    grip = [w for w in mid.wrenches if w.label == "grip:lead"]
    assert len(grip) == 0

    # Axial loads interpolated
    assert mid.axial_loads is not None
    assert mid.axial_loads.values_n["femur_r"] == pytest.approx(-100.0)


def test_series_npz_roundtrip():
    series = _create_sample_series()

    buf = io.BytesIO()
    series.to_npz(buf)
    buf.seek(0)

    restored = ForceTorqueSeries.from_npz(buf)
    assert restored.engine == series.engine
    assert len(restored.frames) == len(series.frames)

    f0 = restored.frames[0]
    assert f0.time_s == 0.0
    w0_hip = next(w for w in f0.wrenches if w.label == "joint:hip_r")
    assert w0_hip.force_n == (0.0, 10.0, 0.0)
    assert w0_hip.torque_nm == (5.0, 0.0, 0.0)

    w0_heel = next(w for w in f0.wrenches if w.label == "contact:heel_r")
    assert w0_heel.force_n == (0.0, 0.0, 100.0)
    assert w0_heel.torque_nm is None  # verified mask preserved None, not zero!

    f1 = restored.frames[1]
    assert f1.time_s == 1.0
    w1_heel = next(w for w in f1.wrenches if w.label == "contact:heel_r")
    assert w1_heel.force_n is None
    assert w1_heel.torque_nm == (0.0, 1.0, 0.0)


def test_series_dict_roundtrip():
    series = _create_sample_series()
    d = series.to_dict()
    assert d["schema_version"] == "force-torque-series-v1"
    assert d["engine"] == "mujoco"
    assert len(d["frames"]) == 2

    restored = ForceTorqueSeries.from_dict(d)
    assert restored.engine == series.engine
    assert len(restored.frames) == 2
    assert restored.frames[0].time_s == 0.0
    assert restored.frames[1].time_s == 1.0


def test_series_more_edge_cases():
    # Empty engine raises ValueError
    with pytest.raises(ValueError, match="engine must be a non-empty string"):
        ForceTorqueSeries(engine="")

    # Invalid frame type raises TypeError
    with pytest.raises(TypeError, match="All frames must be ForceTorqueFrame"):
        ForceTorqueSeries(engine="test", frames=("not_a_frame",))  # type: ignore[arg-type]

    # Empty series frame_at returns None
    empty_series = ForceTorqueSeries(engine="test")
    assert empty_series.frame_at(0.0) is None

    # Series with label mismatch in kind or body
    w0 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="item:1",
        body="body_a",
        point_m=(0.0, 0.0, 0.0),
        force_n=(1.0, 0.0, 0.0),
        source="test",
    )
    w1_mismatch_body = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="item:1",
        body="body_b",  # body differs
        point_m=(0.0, 0.0, 0.0),
        force_n=(2.0, 0.0, 0.0),
        source="test",
    )
    f0 = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w0,))
    f1 = ForceTorqueFrame(time_s=1.0, engine="test", wrenches=(w1_mismatch_body,))
    s_mismatch = ForceTorqueSeries(engine="test", frames=(f0, f1))
    mid = s_mismatch.frame_at(0.5, max_gap_s=1.5)
    assert mid is not None
    assert len(mid.wrenches) == 0  # mismatched body skipped
