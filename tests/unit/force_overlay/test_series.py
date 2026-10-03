"""Unit tests for ForceTorqueSeries (ADR-0052, #11286)."""

from __future__ import annotations

import io
from pathlib import Path
import pytest

from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.series import ForceTorqueSeries

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _make_sample_series() -> ForceTorqueSeries:
    """Create a sample 3-frame series for testing."""
    w1_0 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),
        torque_nm=(1.0, 0.0, 0.0),
        source="engine",
    )
    w2_0 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:ankle",
        body="shank",
        point_m=(0.0, 0.0, 0.1),
        force_n=None,
        torque_nm=(0.0, 10.0, 0.0),
        source="engine",
    )
    ax_0 = AxialLoadFrame(
        time_s=0.0, values_n={"shank": 100.0, "thigh": -50.0}, source="engine"
    )
    f0 = ForceTorqueFrame(
        time_s=0.0, engine="mujoco", wrenches=(w1_0, w2_0), axial_loads=ax_0
    )

    w1_1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.1, 0.0, 0.0),
        force_n=(0.0, 0.0, 200.0),
        torque_nm=(2.0, 0.0, 0.0),
        source="engine",
    )
    w2_1 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:ankle",
        body="shank",
        point_m=(0.1, 0.0, 0.1),
        force_n=None,
        torque_nm=(0.0, 20.0, 0.0),
        source="engine",
    )
    ax_1 = AxialLoadFrame(
        time_s=0.05, values_n={"shank": 200.0, "thigh": -100.0}, source="engine"
    )
    f1 = ForceTorqueFrame(
        time_s=0.05, engine="mujoco", wrenches=(w1_1, w2_1), axial_loads=ax_1
    )

    w1_2 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:foot",
        body="foot",
        point_m=(0.2, 0.0, 0.0),
        force_n=(0.0, 0.0, 300.0),
        torque_nm=(3.0, 0.0, 0.0),
        source="engine",
    )
    # Note: w2 omitted in frame 2
    f2 = ForceTorqueFrame(
        time_s=0.10, engine="mujoco", wrenches=(w1_2,), axial_loads=None
    )

    return ForceTorqueSeries(frames=(f0, f1, f2))


def test_series_validation() -> None:
    """Series validates strictly increasing times and single engine."""
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:1",
        body="b",
        point_m=(0.0, 0.0, 0.0),
        force_n=(1.0, 0.0, 0.0),
        source="s",
    )
    f0 = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(w,))
    f1_bad_time = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(w,))
    f1_decreasing = ForceTorqueFrame(time_s=-0.1, engine="mujoco", wrenches=(w,))
    f1_diff_engine = ForceTorqueFrame(time_s=0.1, engine="drake", wrenches=(w,))

    # Non-increasing time
    with pytest.raises(ValueError, match="strictly increasing"):
        ForceTorqueSeries(frames=(f0, f1_bad_time))
    with pytest.raises(ValueError, match="strictly increasing"):
        ForceTorqueSeries(frames=(f0, f1_decreasing))

    # Multiple engines
    with pytest.raises(ValueError, match="single engine|same engine"):
        ForceTorqueSeries(frames=(f0, f1_diff_engine))


def test_series_properties_and_access() -> None:
    """Series exposes engine, times_s, len, indexing, iteration."""
    series = _make_sample_series()
    assert series.engine == "mujoco"
    assert series.times_s == (0.0, 0.05, 0.10)
    assert len(series) == 3
    assert series[0].time_s == 0.0
    assert series[2].time_s == 0.10
    assert [f.time_s for f in series] == [0.0, 0.05, 0.10]


def test_series_frame_at_exact() -> None:
    """Exact time query returns the exact frame."""
    series = _make_sample_series()
    assert series.frame_at(0.0) is series[0]
    assert series.frame_at(0.05) is series[1]
    assert series.frame_at(0.10) is series[2]


def test_series_frame_at_out_of_range_or_gap() -> None:
    """Out of range or gap > max_gap_s returns None."""
    series = _make_sample_series()
    assert series.frame_at(-0.01) is None
    assert series.frame_at(0.15) is None

    # Between 0.0 and 0.05 is 0.05 gap. If max_gap_s is 0.02, returns None
    assert series.frame_at(0.025, max_gap_s=0.02) is None


def test_series_frame_at_interpolation() -> None:
    """Interpolate between neighbours: points, halves, labels, axial loads."""
    series = _make_sample_series()
    # At t = 0.025, alpha = 0.5 between frame 0 (t=0.0) and frame 1 (t=0.05)
    f_interp = series.frame_at(0.025)
    assert f_interp is not None
    assert f_interp.time_s == pytest.approx(0.025)
    assert f_interp.engine == "mujoco"

    # Wrench 1 (contact:foot) was present in both
    w1_map = {w.label: w for w in f_interp.wrenches}
    assert "contact:foot" in w1_map
    w1 = w1_map["contact:foot"]
    assert w1.point_m == pytest.approx((0.05, 0.0, 0.0))
    assert w1.force_n == pytest.approx((0.0, 0.0, 150.0))
    assert w1.torque_nm == pytest.approx((1.5, 0.0, 0.0))

    # Wrench 2 (joint:ankle) was torque-only in both
    assert "joint:ankle" in w1_map
    w2 = w1_map["joint:ankle"]
    assert w2.point_m == pytest.approx((0.05, 0.0, 0.1))
    assert w2.force_n is None
    assert w2.torque_nm == pytest.approx((0.0, 15.0, 0.0))

    # Axial loads present in both neighbours
    assert f_interp.axial_loads is not None
    assert f_interp.axial_loads.values_n["shank"] == pytest.approx(150.0)
    assert f_interp.axial_loads.values_n["thigh"] == pytest.approx(-75.0)

    # Now interpolate between frame 1 (t=0.05) and frame 2 (t=0.10)
    # Wrench 2 is in frame 1 but NOT in frame 2 -> omitted in interpolated frame!
    # Frame 2 has axial_loads=None -> interpolated axial_loads must be None!
    f_interp_1_2 = series.frame_at(0.075)
    assert f_interp_1_2 is not None
    assert len(f_interp_1_2.wrenches) == 1
    assert f_interp_1_2.wrenches[0].label == "contact:foot"
    assert f_interp_1_2.axial_loads is None


def test_series_frame_at_half_present_in_only_one_neighbour() -> None:
    """A half present in only one neighbour becomes None in the interpolated wrench."""
    # Frame 0: force only
    w0 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:test",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(10.0, 0.0, 0.0),
        torque_nm=(5.0, 0.0, 0.0),
        source="engine",
    )
    f0 = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(w0,))

    # Frame 1: force present, but torque is None
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:test",
        body="foot",
        point_m=(1.0, 0.0, 0.0),
        force_n=(20.0, 0.0, 0.0),
        torque_nm=None,
        source="engine",
    )
    f1 = ForceTorqueFrame(time_s=1.0, engine="mujoco", wrenches=(w1,))

    series = ForceTorqueSeries(frames=(f0, f1))
    f_interp = series.frame_at(0.5, max_gap_s=2.0)
    assert f_interp is not None
    w_interp = f_interp.wrenches[0]
    assert w_interp.point_m == pytest.approx((0.5, 0.0, 0.0))
    assert w_interp.force_n == pytest.approx((15.0, 0.0, 0.0))
    # Torque was present only in frame 0 -> becomes None!
    assert w_interp.torque_nm is None


def test_series_npz_roundtrip(tmp_path: Path) -> None:
    """Series to_npz and from_npz round-trip with allow_pickle=False and boolean masks."""
    series = _make_sample_series()
    npz_path = tmp_path / "test_series.npz"

    # Save to path
    series.to_npz(npz_path)

    # Load from path
    loaded = ForceTorqueSeries.from_npz(npz_path)
    assert len(loaded) == len(series)
    assert loaded.engine == series.engine
    assert loaded.times_s == series.times_s

    for orig_f, loaded_f in zip(series, loaded, strict=True):
        assert loaded_f.time_s == orig_f.time_s
        assert loaded_f.engine == orig_f.engine
        assert loaded_f.world_frame == orig_f.world_frame
        assert loaded_f.units == orig_f.units
        assert len(loaded_f.wrenches) == len(orig_f.wrenches)
        for orig_w, loaded_w in zip(orig_f.wrenches, loaded_f.wrenches, strict=True):
            assert loaded_w.kind == orig_w.kind
            assert loaded_w.label == orig_w.label
            assert loaded_w.body == orig_w.body
            assert loaded_w.point_m == pytest.approx(orig_w.point_m)
            if orig_w.force_n is None:
                assert loaded_w.force_n is None
            else:
                assert loaded_w.force_n == pytest.approx(orig_w.force_n)
            if orig_w.torque_nm is None:
                assert loaded_w.torque_nm is None
            else:
                assert loaded_w.torque_nm == pytest.approx(orig_w.torque_nm)
            assert loaded_w.source == orig_w.source

        if orig_f.axial_loads is None:
            assert loaded_f.axial_loads is None
        else:
            assert loaded_f.axial_loads is not None
            assert loaded_f.axial_loads.time_s == orig_f.axial_loads.time_s
            assert loaded_f.axial_loads.values_n == orig_f.axial_loads.values_n
            assert loaded_f.axial_loads.source == orig_f.axial_loads.source

    # Also test stream / BytesIO I/O
    buf = io.BytesIO()
    series.to_npz(buf)
    buf.seek(0)
    loaded_buf = ForceTorqueSeries.from_npz(buf)
    assert len(loaded_buf) == len(series)


def test_series_dict_roundtrip() -> None:
    """Series to_dict and from_dict roundtrip."""
    series = _make_sample_series()
    d = series.to_dict()
    assert d["schema_version"] == "force-torque-series-v1"
    assert d["engine"] == "mujoco"
    assert len(d["frames"]) == 3

    loaded = ForceTorqueSeries.from_dict(d)
    assert len(loaded) == 3
    assert loaded.engine == "mujoco"
    assert loaded.times_s == series.times_s
