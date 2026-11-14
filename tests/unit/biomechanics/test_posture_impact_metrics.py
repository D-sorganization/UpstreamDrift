"""Tests for posture, translation, impact, and GRF moment metrics (GM-5 #14428)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.biomechanics.posture_impact_metrics import (
    ChuEventFrames,
    compute_segment_translations,
    compute_trunk_posture_events,
    evaluate_pivoting_moment,
    join_impact_metrics,
)
from src.shared.python.contracts import ContractViolationError

pytestmark = pytest.mark.unit

TARGET = np.array([1.0, 0.0, 0.0])
BALL = np.array([0.0, 1.0, 0.0])
EVENTS = ChuEventFrames(
    address=0, top=10, mid_downswing=15, impact=20, follow_through=30
)


def test_known_trunk_angles_reproduce_exact_degrees() -> None:
    axes = np.zeros((35, 3))

    def make_axis(tilt_deg: float, bend_deg: float) -> np.ndarray:
        t, b = math.tan(math.radians(tilt_deg)), math.tan(math.radians(bend_deg))
        vec = np.array([b, t, 1.0])
        return vec / np.linalg.norm(vec)

    axes[0] = make_axis(30.0, 0.0)
    axes[10] = make_axis(15.0, -10.0)
    axes[15] = make_axis(25.0, 5.0)
    axes[20] = make_axis(20.0, 25.0)
    axes[30] = make_axis(-5.0, 10.0)

    rep = compute_trunk_posture_events(axes, EVENTS, TARGET, BALL)
    assert rep.lab_frame == "X_target_Y_ball_Z_up"
    assert rep.address.forward_tilt_deg == pytest.approx(30.0, abs=1e-5)
    assert rep.address.lateral_bend_deg == pytest.approx(0.0, abs=1e-5)
    assert rep.top_of_backswing.forward_tilt_deg == pytest.approx(15.0, abs=1e-5)
    assert rep.top_of_backswing.lateral_bend_deg == pytest.approx(-10.0, abs=1e-5)
    assert rep.mid_downswing.forward_tilt_deg == pytest.approx(25.0, abs=1e-5)
    assert rep.mid_downswing.lateral_bend_deg == pytest.approx(5.0, abs=1e-5)
    assert rep.impact.forward_tilt_deg == pytest.approx(20.0, abs=1e-5)
    assert rep.impact.lateral_bend_deg == pytest.approx(25.0, abs=1e-5)
    assert rep.follow_through.forward_tilt_deg == pytest.approx(-5.0, abs=1e-5)
    assert rep.follow_through.lateral_bend_deg == pytest.approx(10.0, abs=1e-5)


def test_known_displacement_trajectories_reproduce_sway_thrust_lift_and_velocities() -> (
    None
):
    dt = 0.01
    t = np.arange(0.0, 0.40, dt)
    omega = 2.0 * math.pi
    sway_raw = 0.05 * np.sin(omega * t)
    thrust_raw = 0.08 * t
    lift_raw = -0.02 * (t**2)
    pos = np.column_stack([sway_raw + 1.0, thrust_raw + 2.0, lift_raw + 0.9])

    kin = compute_segment_translations(pos, 0, dt, "pelvis", TARGET, BALL)
    assert kin.sway_m[0] == pytest.approx(0.0, abs=1e-9)
    assert kin.thrust_m[0] == pytest.approx(0.0, abs=1e-9)
    assert kin.lift_m[0] == pytest.approx(0.0, abs=1e-9)

    idx = 10
    assert kin.sway_m[idx] == pytest.approx(sway_raw[idx], abs=1e-9)
    assert kin.thrust_m[idx] == pytest.approx(thrust_raw[idx], abs=1e-9)
    assert kin.lift_m[idx] == pytest.approx(lift_raw[idx], abs=1e-9)
    assert kin.sway_velocity_mps[idx] == pytest.approx(
        0.05 * omega * math.cos(omega * t[idx]), rel=1e-3
    )
    assert kin.thrust_velocity_mps[idx] == pytest.approx(0.08, rel=1e-3)
    assert kin.lift_velocity_mps[idx] == pytest.approx(-0.04 * t[idx], rel=1e-3)


def test_join_impact_metrics_record_validation() -> None:
    dt, n = 0.01, 35
    axes = np.zeros((n, 3))
    axes[:] = [0.0, math.sin(math.radians(20.0)), math.cos(math.radians(20.0))]
    trunk = compute_trunk_posture_events(axes, EVENTS, TARGET, BALL)

    p_pos = np.zeros((n, 3))
    p_pos[:, 0], p_pos[:, 1], p_pos[:, 2] = (
        np.linspace(0, 0.2, n),
        np.linspace(0, 0.1, n),
        np.linspace(0.9, 0.85, n),
    )
    pelvis = compute_segment_translations(p_pos, 0, dt, "pelvis", TARGET, BALL)

    t_pos = np.zeros((n, 3))
    t_pos[:, 0], t_pos[:, 1], t_pos[:, 2] = (
        np.linspace(0, 0.15, n),
        np.linspace(0, 0.05, n),
        np.linspace(1.4, 1.38, n),
    )
    thorax = compute_segment_translations(t_pos, 0, dt, "thorax", TARGET, BALL)

    sumry = join_impact_metrics(
        48.5, EVENTS.impact, pelvis, thorax, trunk, impact_time_s=0.20
    )
    assert sumry.club_speed_mps == 48.5 and sumry.impact_time_s == 0.20
    assert sumry.trunk_forward_tilt_deg == pytest.approx(20.0, abs=1e-5)
    assert sumry.pelvis_sway_m == pytest.approx(p_pos[20, 0] - p_pos[0, 0], abs=1e-9)
    assert sumry.thorax_sway_m == pytest.approx(t_pos[20, 0] - t_pos[0, 0], abs=1e-9)

    with pytest.raises(ContractViolationError):
        join_impact_metrics(-5.0, EVENTS.impact, pelvis, thorax, trunk)


def test_han_2019_pivoting_moment_provenance_contract() -> None:
    mod = evaluate_pivoting_moment(24.5, "modelled", foot="trail")
    assert (
        mod.source == "modelled"
        and mod.pivoting_moment_nm == 24.5
        and mod.axis == "vertical"
    )

    meas = evaluate_pivoting_moment(
        31.2, "measured_force_plate", plate_data_path="plate_01.c3d"
    )
    assert (
        meas.source == "measured_force_plate" and meas.plate_data_path == "plate_01.c3d"
    )

    with pytest.raises(ContractViolationError):
        evaluate_pivoting_moment(31.2, "measured_force_plate", plate_data_path=None)
    with pytest.raises(ContractViolationError):
        evaluate_pivoting_moment(31.2, "measured_force_plate", plate_data_path="   ")
    with pytest.raises(ContractViolationError):
        evaluate_pivoting_moment(31.2, "unknown_source")
