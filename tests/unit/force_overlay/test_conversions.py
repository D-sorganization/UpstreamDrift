"""Unit tests for shared force conversions (ADR-0052, #11287)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import (
    AxialLoadFrame,
    axial_force_from_proximal_reaction,
)
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.conversions import (
    SegmentAxis,
    axial_loads_from_reactions,
    frame_with_axial_loads,
    joint_torque_wrench,
    move_wrench_point,
    world_wrench_from_local,
)

pytestmark = pytest.mark.unit


def test_joint_torque_wrench_revolute():
    # Revolute joint about +z with tau = 10 N*m
    w = joint_torque_wrench(
        label="joint:knee_z",
        body="tibia",
        tau_nm=10.0,
        axis_world=(0.0, 0.0, 1.0),
        anchor_world=(0.1, 0.2, 0.5),
        source="test_engine",
    )
    assert w.kind == WrenchKind.JOINT_ACTUATOR
    assert w.label == "joint:knee_z"
    assert w.body == "tibia"
    assert w.point_m == (0.1, 0.2, 0.5)
    assert w.force_n is None
    assert w.torque_nm == (0.0, 0.0, 10.0)
    assert w.source == "test_engine"


def test_joint_torque_wrench_gimbal_and_validation():
    # 3-axis gimbal summing moments
    taus = [10.0, -5.0, 2.0]
    axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    w = joint_torque_wrench(
        label="joint:hip_gimbal",
        body="femur",
        tau_nm=taus,
        axis_world=axes,
        anchor_world=(0.0, 0.0, 0.0),
        source="engine",
    )
    assert w.torque_nm == (10.0, -5.0, 2.0)
    assert w.force_n is None

    # Non-unit axis raises ValueError
    with pytest.raises(ValueError, match="unit norm"):
        joint_torque_wrench(
            label="joint:bad_axis",
            body="body",
            tau_nm=5.0,
            axis_world=(0.0, 0.0, 2.0),
            anchor_world=(0.0, 0.0, 0.0),
            source="engine",
        )

    # Mismatched lengths
    with pytest.raises(ValueError, match="dimension mismatch"):
        joint_torque_wrench(
            label="joint:mismatch",
            body="body",
            tau_nm=[1.0, 2.0],
            axis_world=[(1.0, 0.0, 0.0)],
            anchor_world=(0.0, 0.0, 0.0),
            source="engine",
        )


def test_world_wrench_from_local():
    # 90 deg rotation about z: maps local +x to world +y, local +y to world -x
    R_z90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Force only: local +x becomes world +y
    w_force = world_wrench_from_local(
        label="contact:tip",
        body="hand",
        kind=WrenchKind.CONTACT,
        force_local=(10.0, 0.0, 0.0),
        torque_local=None,
        rotation_world_from_local=R_z90,
        point_world=(1.0, 2.0, 3.0),
        source="engine",
    )
    assert w_force.point_m == (1.0, 2.0, 3.0)
    assert w_force.force_n is not None
    assert w_force.force_n == pytest.approx((0.0, 10.0, 0.0), abs=1e-9)
    assert w_force.torque_nm is None

    # Torque only: local +y becomes world -x
    w_torque = world_wrench_from_local(
        label="joint:tau",
        body="hand",
        kind=WrenchKind.JOINT_ACTUATOR,
        force_local=None,
        torque_local=(0.0, 5.0, 0.0),
        rotation_world_from_local=R_z90,
        point_world=(0.0, 0.0, 0.0),
        source="engine",
    )
    assert w_torque.force_n is None
    assert w_torque.torque_nm is not None
    assert w_torque.torque_nm == pytest.approx((-5.0, 0.0, 0.0), abs=1e-9)

    # Both halves
    w_both = world_wrench_from_local(
        label="grip:lead",
        body="hand",
        kind=WrenchKind.GRIP,
        force_local=(10.0, 0.0, 0.0),
        torque_local=(0.0, 5.0, 0.0),
        rotation_world_from_local=R_z90,
        point_world=(0.0, 0.0, 0.0),
        source="engine",
    )
    assert w_both.force_n == pytest.approx((0.0, 10.0, 0.0), abs=1e-9)
    assert w_both.torque_nm == pytest.approx((-5.0, 0.0, 0.0), abs=1e-9)

    # Both halves None raises ValueError
    with pytest.raises(ValueError, match="At least one"):
        world_wrench_from_local(
            label="grip:empty",
            body="hand",
            kind=WrenchKind.GRIP,
            force_local=None,
            torque_local=None,
            rotation_world_from_local=R_z90,
            point_world=(0.0, 0.0, 0.0),
            source="engine",
        )

    # Non-orthonormal matrix raises ValueError
    bad_R = np.eye(3) * 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        world_wrench_from_local(
            label="grip:bad_r",
            body="hand",
            kind=WrenchKind.GRIP,
            force_local=(1.0, 0.0, 0.0),
            torque_local=None,
            rotation_world_from_local=bad_R,
            point_world=(0.0, 0.0, 0.0),
            source="engine",
        )


def test_move_wrench_point():
    # Known force and torque moved by r: tau_B = tau_A + (p_A - p_B) x F
    # p_A = (0, 0, 0), p_B = (0, 1, 0) -> r = (0, -1, 0)
    # F = (0, 0, 10), tau_A = (1, 2, 3)
    # r x F = (-10, 0, 0)
    # tau_B = (1 - 10, 2 + 0, 3 + 0) = (-9, 2, 3)
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10.0),
        torque_nm=(1.0, 2.0, 3.0),
        source="engine",
    )
    moved = move_wrench_point(w, (0.0, 1.0, 0.0))
    assert moved.point_m == (0.0, 1.0, 0.0)
    assert moved.force_n == (0.0, 0.0, 10.0)
    assert moved.torque_nm == pytest.approx((-9.0, 2.0, 3.0), abs=1e-9)

    # Torque-only moved to new point: torque_nm becomes None because force is unavailable!
    w_torque_only = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:hip",
        body="femur",
        point_m=(0.0, 0.0, 0.0),
        force_n=None,
        torque_nm=(0.0, 5.0, 0.0),
        source="engine",
    )
    with pytest.raises(ValueError, match="Cannot move torque-only wrench.*both halves"):
        move_wrench_point(w_torque_only, (0.0, 1.0, 0.0))

    # Force-only moved to new point: torque_nm is unknown (None), so result remains force-only!
    w_force_only = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:heel",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10.0),
        torque_nm=None,
        source="engine",
    )
    moved_force_only = move_wrench_point(w_force_only, (0.0, 1.0, 0.0))
    assert moved_force_only.point_m == (0.0, 1.0, 0.0)
    assert moved_force_only.force_n == (0.0, 0.0, 10.0)
    assert moved_force_only.torque_nm is None


def test_segment_axis_validation():
    # Valid SegmentAxis
    ax = SegmentAxis(
        segment="femur_r",
        joint_label="joint:hip_r",
        proximal_m=(0.0, 0.0, 0.5),
        distal_m=(0.0, 0.0, 0.1),
    )
    assert ax.segment == "femur_r"
    assert ax.joint_label == "joint:hip_r"

    # Coincident endpoints raise ValueError
    with pytest.raises(ValueError, match="coincident"):
        SegmentAxis(
            segment="femur_r",
            joint_label="joint:hip_r",
            proximal_m=(0.0, 0.0, 0.5),
            distal_m=(0.0, 0.0, 0.5),
        )


def test_axial_loads_from_reactions():
    # Hanging segment (proximal at (0,0), distal at (0,-1)) with reaction (0, 10, 0)
    # tension is positive: F_parent_on_segment = (0, 10, 0)
    # - (0, 10, 0) dot (0, -1, 0) = +10
    w_hanging = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="joint:shoulder",
        body="upper_arm",
        point_m=(0.0, 0.0, 1.5),
        force_n=(0.0, 0.0, 10.0),
        torque_nm=None,
        source="engine",
    )
    w_pushing = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="joint:leg",
        body="tibia",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 10.0),
        torque_nm=None,
        source="engine",
    )
    frame = ForceTorqueFrame(
        time_s=1.0,
        engine="mujoco",
        wrenches=(w_hanging, w_pushing),
    )

    axis_arm = SegmentAxis(
        "upper_arm", "joint:shoulder", (0.0, 0.0, 1.5), (0.0, 0.0, 0.5)
    )
    axis_leg = SegmentAxis("tibia", "joint:leg", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    axis_missing = SegmentAxis(
        "forearm", "joint:elbow", (0.0, 0.0, 0.5), (0.0, 0.0, 0.0)
    )

    axial = axial_loads_from_reactions(
        frame=frame,
        axes=[axis_arm, axis_leg, axis_missing],
        source="axial_solver",
    )
    assert isinstance(axial, AxialLoadFrame)
    assert axial.time_s == 1.0
    assert axial.values_n["upper_arm"] == pytest.approx(10.0)  # tension positive
    assert axial.values_n["tibia"] == pytest.approx(-10.0)  # compression negative
    assert axial.values_n["forearm"] is None  # missing reaction is None

    # frame_with_axial_loads adds axial loads to frame
    frame_with_ax = frame_with_axial_loads(
        frame=frame,
        axes=[axis_arm, axis_leg],
        source="axial_solver",
    )
    assert frame_with_ax.axial_loads is not None
    assert frame_with_ax.axial_loads.values_n["upper_arm"] == pytest.approx(10.0)
    assert frame_with_ax.axial_loads.values_n["tibia"] == pytest.approx(-10.0)


def test_conversions_edge_cases():
    # R wrong shape
    with pytest.raises(ValueError, match="shape"):
        world_wrench_from_local(
            label="test:label",
            body="body",
            kind=WrenchKind.CONTACT,
            force_local=(1.0, 0.0, 0.0),
            torque_local=None,
            rotation_world_from_local=np.eye(2),
            point_world=(0.0, 0.0, 0.0),
            source="test",
        )

    # move_wrench_point when point is identical
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(1.0, 2.0, 3.0),
        force_n=(0.0, 0.0, 10.0),
        torque_nm=None,
        source="test",
    )
    same = move_wrench_point(w, (1.0, 2.0, 3.0))
    assert same is w

    # SegmentAxis empty segment or empty joint_label
    with pytest.raises(ValueError, match="segment must be a non-empty string"):
        SegmentAxis("", "joint:test", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="joint_label must be a non-empty string"):
        SegmentAxis("segment", "", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
