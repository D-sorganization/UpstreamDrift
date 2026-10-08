"""Synthetic analytic tests for impact-parameter extraction (GCV-15)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.impact_parameters import (
    ClubheadSeries,
    TargetFrame,
    ToolsDeliveryGateway,
    ToolsDeliveryUnavailableError,
    extract_impact_parameters,
    load_tools_delivery_gateway,
)

pytestmark = pytest.mark.unit

N = 41
IDX = 30
DT = 0.002
SPEED = 45.0


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _dir(frame, aoa, path, sign):
    """World unit vector for elevation/heading in the target frame."""
    a, p = math.radians(aoa), math.radians(path)
    comps = (math.cos(a) * math.cos(p), -sign * math.cos(a) * math.sin(p), math.sin(a))
    return comps[0] * frame.x_t + comps[1] * frame.y_t + comps[2] * frame.z_t


def make_scene(aoa=-5.0, path=3.0, face=1.0, loft=12.0, hand="right", rot=0.0):
    base = TargetFrame(handedness=hand)
    frame = TargetFrame(
        target_dir=tuple(_rot_z(rot) @ base.x_t),
        ball_m=tuple(_rot_z(rot) @ np.array([0.1, 0.2, 0.02])),
        handedness=hand,
    )
    sign = frame.lateral_sign
    v = SPEED * _dir(frame, aoa, path, sign)
    n = _dir(frame, loft, face, sign)
    accel = 1000.0 * frame.z_t  # vertical acceleration -> low point after impact
    t = np.arange(N) * DT
    dtau = (t - t[IDX])[:, None]
    ball = np.asarray(frame.ball_m)
    pos = ball + v * dtau + 0.5 * accel * dtau**2
    vel = v + accel * dtau
    series = ClubheadSeries(
        times_s=t,
        face_center_m=pos,
        velocity_mps=vel,
        face_normal=np.tile(n, (N, 1)),
        toe_axis=np.tile(np.cross(frame.z_t, n), (N, 1)),
    )
    return series, frame


def _tools_missing(_name: str):
    raise ImportError("no tools")


def test_square_level_all_zero():
    series, frame = make_scene(aoa=0, path=0, face=0, loft=0)
    r = extract_impact_parameters(series, frame, impact_index=IDX, use_tools=False)
    for name in (
        "attack_angle_deg",
        "club_path_deg",
        "face_angle_deg",
        "face_to_path_deg",
        "dynamic_loft_deg",
        "spin_loft_deg",
    ):
        assert getattr(r, name) == pytest.approx(0.0, abs=1e-9), name
    assert r.clubhead_speed_mps == pytest.approx(SPEED)
    assert r.clubhead_speed_mph == pytest.approx(SPEED * 2.23694, rel=1e-5)


def test_known_aoa_path_face_and_spin_loft():
    series, frame = make_scene(aoa=-5, path=3, face=1, loft=12)
    r = extract_impact_parameters(series, frame, impact_index=IDX, use_tools=False)
    assert r.attack_angle_deg == pytest.approx(-5.0)
    assert r.club_path_deg == pytest.approx(3.0)
    assert r.face_angle_deg == pytest.approx(1.0)
    assert r.face_to_path_deg == pytest.approx(-2.0)
    assert r.dynamic_loft_deg == pytest.approx(12.0)
    a, p, f, lo = (math.radians(x) for x in (-5, 3, 1, 12))
    vh = np.array([math.cos(a) * math.cos(p), math.cos(a) * math.sin(p), math.sin(a)])
    nh = np.array(
        [math.cos(lo) * math.cos(f), math.cos(lo) * math.sin(f), math.sin(lo)]
    )
    assert r.spin_loft_deg == pytest.approx(math.degrees(math.acos(vh @ nh)))
    assert r.frame["handedness"] == "right"
    assert r.frame["frame_id"].startswith("ud_world")


@pytest.mark.parametrize("rot", [0.7, -2.1, math.pi])
def test_rotation_invariance(rot):
    ref = extract_impact_parameters(*make_scene(), impact_index=IDX, use_tools=False)
    got = extract_impact_parameters(
        *make_scene(rot=rot), impact_index=IDX, use_tools=False
    )
    for name in (
        "attack_angle_deg",
        "club_path_deg",
        "face_angle_deg",
        "face_to_path_deg",
        "dynamic_loft_deg",
        "spin_loft_deg",
        "swing_direction_deg",
        "swing_plane_angle_deg",
        "low_point_ahead_of_ball_m",
        "low_point_height_m",
    ):
        assert getattr(got, name) == pytest.approx(getattr(ref, name), abs=1e-7), name


def test_left_handed_mirror_flips_lateral_signs_only():
    rh = extract_impact_parameters(
        *make_scene(hand="right"), impact_index=IDX, use_tools=False
    )
    lh = extract_impact_parameters(
        *make_scene(hand="left"), impact_index=IDX, use_tools=False
    )
    # Same intent in each golfer's own convention -> same values.
    assert lh.club_path_deg == pytest.approx(rh.club_path_deg)
    assert lh.face_to_path_deg == pytest.approx(rh.face_to_path_deg)
    # Same world scene analysed as LH flips lateral signs only.
    series, _ = make_scene(hand="right")
    frame_rh = make_scene(hand="right")[1]
    frame_lh = TargetFrame(
        target_dir=frame_rh.target_dir,
        ball_m=frame_rh.ball_m,
        handedness="left",
    )
    flipped = extract_impact_parameters(
        series, frame_lh, impact_index=IDX, use_tools=False
    )
    assert flipped.club_path_deg == pytest.approx(-rh.club_path_deg)
    assert flipped.face_angle_deg == pytest.approx(-rh.face_angle_deg)
    assert flipped.face_to_path_deg == pytest.approx(-rh.face_to_path_deg)
    assert flipped.attack_angle_deg == pytest.approx(rh.attack_angle_deg)
    assert flipped.dynamic_loft_deg == pytest.approx(rh.dynamic_loft_deg)
    assert flipped.spin_loft_deg == pytest.approx(rh.spin_loft_deg)
    assert flipped.low_point_ahead_of_ball_m == pytest.approx(
        rh.low_point_ahead_of_ball_m
    )


def test_low_point_swing_plane_and_direction_analytic():
    series, frame = make_scene(aoa=-5, path=3)
    r = extract_impact_parameters(series, frame, impact_index=IDX, use_tools=False)
    v = SPEED * _dir(frame, -5, 3, 1.0)
    vz = float(v @ frame.z_t)
    t_low = -vz / 1000.0  # seconds after impact
    x_low = float(v @ frame.x_t) * t_low
    # sampled minimum: nearest sample to the analytic minimum
    k = round(t_low / DT)
    dtk = k * DT
    expected = float(v @ frame.x_t) * dtk
    assert r.low_point_ahead_of_ball_m == pytest.approx(expected, abs=1e-9)
    assert abs(r.low_point_ahead_of_ball_m - x_low) < SPEED * DT
    assert r.low_point_ahead_of_ball_m > 0
    assert r.swing_direction_deg == pytest.approx(3.0, abs=1e-6)
    assert r.swing_plane_angle_deg == pytest.approx(90.0, abs=1e-6)


def test_zero_speed_raises():
    series, frame = make_scene()
    zero = ClubheadSeries(
        times_s=series.times_s,
        face_center_m=series.face_center_m,
        velocity_mps=np.zeros((N, 3)),
        face_normal=series.face_normal,
    )
    with pytest.raises(ValueError, match="below"):
        extract_impact_parameters(zero, frame, impact_index=IDX)


def test_unobservable_face_is_unavailable_never_invented():
    series, frame = make_scene()
    blind = ClubheadSeries(
        times_s=series.times_s,
        face_center_m=series.face_center_m,
        velocity_mps=series.velocity_mps,
        face_unobservable_reason="mocap club axial rotation unobservable",
    )
    r = extract_impact_parameters(blind, frame, impact_index=IDX)
    for name in (
        "face_angle_deg",
        "face_to_path_deg",
        "dynamic_loft_deg",
        "spin_loft_deg",
    ):
        assert getattr(r, name) is None
        assert "unobservable" in r.unavailable[name]
    assert r.club_path_deg == pytest.approx(3.0)
    assert r.tools is None


def test_series_requires_reason_without_face():
    with pytest.raises(ValueError, match="face_unobservable_reason"):
        ClubheadSeries(
            times_s=np.arange(3.0),
            face_center_m=np.zeros((3, 3)),
            velocity_mps=np.ones((3, 3)),
        )


def test_impact_index_resolution():
    series, frame = make_scene()
    last = extract_impact_parameters(
        series, frame, contact_index=IDX + 1, use_tools=False
    )
    assert last.impact_index == IDX
    assert last.impact_time_source == "last_pre_contact"
    with pytest.raises(ValueError):
        extract_impact_parameters(series, frame, contact_index=0)
    with pytest.raises(ValueError):
        extract_impact_parameters(series, frame, impact_index=N)
    with pytest.raises(ValueError):
        extract_impact_parameters(series, frame, impact_index=1, contact_index=2)
    # peak-speed detection on a series that accelerates to the last sample
    t = np.arange(N) * DT
    pos = np.zeros((N, 3))
    pos[:, 0] = 0.5 * 500.0 * t**2
    vel = np.zeros((N, 3))
    vel[:, 0] = 500.0 * t
    accel = ClubheadSeries(
        times_s=t,
        face_center_m=pos,
        velocity_mps=vel,
        face_unobservable_reason="test",
    )
    frame_x = TargetFrame(target_dir=(1.0, 0.0, 0.0))
    got = extract_impact_parameters(accel, frame_x, min_speed_mps=0.1)
    assert got.impact_time_source == "peak_clubhead_speed"
    assert got.impact_index >= N - 3


def test_toe_high_smash_and_collinear_plane():
    series, frame = make_scene(aoa=0, path=0, face=0, loft=0)
    n = series.face_normal[IDX]
    toe = series.toe_axis[IDX]
    high = np.cross(toe, n)
    contact = series.face_center_m[IDX] + 0.01 * toe + 0.004 * high
    r = extract_impact_parameters(
        series,
        frame,
        impact_index=IDX,
        use_tools=False,
        ball_contact_m=contact,
        ball_speed_mps=66.0,
        impact_model_status="uncalibrated",
    )
    assert r.toe_mm == pytest.approx(10.0)
    assert r.high_mm == pytest.approx(4.0)
    assert r.smash_factor == pytest.approx(66.0 / SPEED)
    assert "uncalibrated" in r.smash_factor_label
    bare = extract_impact_parameters(series, frame, impact_index=IDX, use_tools=False)
    assert bare.toe_mm is None and "toe_mm" in bare.unavailable
    assert bare.smash_factor is None and "smash_factor" in bare.unavailable
    # straight constant-velocity path -> plane undefined
    t = series.times_s
    line = ClubheadSeries(
        times_s=t,
        face_center_m=np.outer(t, [SPEED, 0, 0]),
        velocity_mps=np.tile([SPEED, 0, 0], (N, 1)),
        face_normal=series.face_normal,
    )
    r2 = extract_impact_parameters(
        line, TargetFrame(target_dir=(1.0, 0.0, 0.0)), impact_index=IDX, use_tools=False
    )
    assert r2.swing_plane_angle_deg is None
    assert "collinear" in r2.unavailable["swing_plane_angle_deg"]
    with pytest.raises(ValueError):
        extract_impact_parameters(
            series,
            frame,
            impact_index=IDX,
            use_tools=False,
            ball_contact_m=[0, 0],
        )


def test_target_frame_validation_and_report():
    with pytest.raises(ValueError):
        TargetFrame(target_dir=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        TargetFrame(handedness="ambi")
    with pytest.raises(ValueError):
        TargetFrame(target_dir=(0.0, -2.0, 0.0))
    series, frame = make_scene()
    rep = extract_impact_parameters(series, frame, impact_index=IDX).to_report()
    assert rep["frame"]["target_dir"] == [0.0, -1.0, 0.0]
    with pytest.raises(TypeError):
        extract_impact_parameters(None, frame)  # type: ignore[arg-type]


# ---- Tools gateway -------------------------------------------------------


def test_gateway_fails_closed_when_tools_missing():
    with pytest.raises(ToolsDeliveryUnavailableError):
        load_tools_delivery_gateway(_tools_missing)
    series, frame = make_scene()
    import src.shared.python.impact_parameters.extract as ex

    orig = ex.load_tools_delivery_gateway
    ex.load_tools_delivery_gateway = lambda: load_tools_delivery_gateway(_tools_missing)
    try:
        r = extract_impact_parameters(series, frame, impact_index=IDX)
    finally:
        ex.load_tools_delivery_gateway = orig
    assert r.tools is None
    assert "fail closed" in r.unavailable["tools_estimates"]
    assert r.club_path_deg == pytest.approx(3.0)  # UD definitions still computed


def test_gateway_rejects_incompatible_facade():
    with pytest.raises(Exception, match="missing required export"):
        ToolsDeliveryGateway(SimpleNamespace())
    with pytest.raises(TypeError):
        ToolsDeliveryGateway(None)
    with pytest.raises(TypeError):
        load_tools_delivery_gateway(importer=None)  # type: ignore[arg-type]


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize(
    "angles",
    [(-5, 3, 1, 12), (4, -6, -2, 9), (0, 0, 0, 0), (-12, 8, 5, 25)],
)
def test_tools_agree_with_ud_definitions_within_0p01_deg(hand, angles):
    pytest.importorskip("shared.python.swing_sim.impact")
    aoa, path, face, loft = angles
    series, frame = make_scene(aoa, path, face, loft, hand=hand, rot=0.9)
    r = extract_impact_parameters(series, frame, impact_index=IDX)
    assert r.tools is not None
    assert r.tools_max_deviation_deg is not None
    assert r.tools_max_deviation_deg < 0.01
    assert r.tools.spin_loft_3d_deg == pytest.approx(r.spin_loft_deg, abs=0.01)
    assert r.tools.club_path_deg == pytest.approx(path, abs=0.01)
    assert r.tools.face_to_path_deg == pytest.approx(face - path, abs=0.01)
    # D-plane tilt is a Tools model estimate, sign: + = fade side
    assert r.tools.spin_axis_tilt_deg is not None or r.tools.dplane_status != "defined"


# ---- Contract edges ------------------------------------------------------


def _kw(series, **over):
    base = {
        "times_s": series.times_s,
        "face_center_m": series.face_center_m,
        "velocity_mps": series.velocity_mps,
        "face_normal": series.face_normal,
        "toe_axis": series.toe_axis,
    }
    base.update(over)
    return base


def test_series_validation_errors():
    s, _ = make_scene()
    with pytest.raises(ValueError, match="at least 2"):
        ClubheadSeries(**_kw(s, times_s=np.array([0.0])))
    with pytest.raises(ValueError, match="increasing"):
        ClubheadSeries(**_kw(s, times_s=np.zeros(N)))
    with pytest.raises(ValueError, match="shape"):
        ClubheadSeries(**_kw(s, velocity_mps=np.zeros((N, 2))))
    with pytest.raises(ValueError, match="rows"):
        ClubheadSeries(**_kw(s, velocity_mps=np.zeros((N - 1, 3))))
    with pytest.raises(ValueError, match="finite"):
        ClubheadSeries(**_kw(s, velocity_mps=np.full((N, 3), np.nan)))
    with pytest.raises(ValueError, match="required"):
        ClubheadSeries(**_kw(s, velocity_mps=None))
    with pytest.raises(ValueError, match="nonzero"):
        ClubheadSeries(**_kw(s, face_normal=np.zeros((N, 3))))
    assert len(s) == N


def test_frame_validation_errors():
    with pytest.raises(ValueError, match="shape"):
        TargetFrame(ball_m=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        TargetFrame(ball_m=(0.0, float("nan"), 0.0))
    with pytest.raises(ValueError, match="up must"):
        TargetFrame(up=(0.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="ground_height"):
        TargetFrame(ground_height_m=float("inf"))


def test_degenerate_geometry_is_unavailable():
    s, frame = make_scene(aoa=0, path=0, face=0, loft=0)
    up = frame.z_t
    vertical_v = np.tile(SPEED * up, (N, 1))
    vertical_n = np.tile(up, (N, 1))
    r = extract_impact_parameters(
        ClubheadSeries(**_kw(s, velocity_mps=vertical_v, face_normal=vertical_n)),
        frame,
        impact_index=IDX,
        use_tools=False,
    )
    assert r.club_path_deg is None and "club_path_deg" in r.unavailable
    assert r.face_angle_deg is None and r.face_to_path_deg is None
    assert r.attack_angle_deg == pytest.approx(90.0)
    assert r.dynamic_loft_deg == pytest.approx(90.0)
    # in-plane travel with no horizontal part
    assert "swing_direction_deg" in r.unavailable


def test_plane_needs_three_samples_and_toe_cases():
    s, frame = make_scene()
    short = ClubheadSeries(
        times_s=np.array([0.0, 1.0]),
        face_center_m=np.zeros((2, 3)),
        velocity_mps=np.array([[SPEED, 0, 0], [SPEED, 0, 0]]),
        face_unobservable_reason="x",
    )
    r = extract_impact_parameters(
        short, TargetFrame(target_dir=(1.0, 0.0, 0.0)), impact_index=0
    )
    assert "fewer than 3" in r.unavailable["swing_plane_angle_deg"]
    no_toe = ClubheadSeries(**_kw(s, toe_axis=None))
    r = extract_impact_parameters(
        no_toe, frame, impact_index=IDX, use_tools=False, ball_contact_m=[0, 0, 0]
    )
    assert "unobservable" in r.unavailable["toe_mm"]
    parallel = ClubheadSeries(**_kw(s, toe_axis=s.face_normal))
    r = extract_impact_parameters(
        parallel, frame, impact_index=IDX, use_tools=False, ball_contact_m=[0, 0, 0]
    )
    assert "parallel" in r.unavailable["high_mm"]


def test_option_validation():
    s, frame = make_scene()
    with pytest.raises(ValueError, match="min_speed"):
        extract_impact_parameters(s, frame, impact_index=IDX, min_speed_mps=0)
    with pytest.raises(ValueError, match="ball_speed"):
        extract_impact_parameters(
            s,
            frame,
            impact_index=IDX,
            use_tools=False,
            ball_speed_mps=-1.0,
            impact_model_status="x",
        )


def test_tools_gateway_zero_speed_and_injected_gateway():
    pytest.importorskip("shared.python.swing_sim.impact")
    gw = load_tools_delivery_gateway()
    _, frame = make_scene()
    with pytest.raises(ValueError, match="speed"):
        gw.estimate([0, 0, 0], [1, 0, 0], frame)
    s, _ = make_scene()
    r = extract_impact_parameters(s, frame, impact_index=IDX, tools_gateway=gw)
    assert r.tools is not None
    assert r.to_report()["tools"]["dplane_status"]
