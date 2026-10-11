"""Tests for force_overlay shared conversions (ADR-0052, #11287)."""

from __future__ import annotations

import math
from typing import Any
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

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_joint_torque_wrench_revolute() -> None:
    """A revolute joint about +z with tau = 10 N*m gives torque (0, 0, 10) at anchor, force None."""
    w = joint_torque_wrench(
        label="actuator:knee_z",
        body="femur",
        tau_nm=10.0,
        axis_world=(0.0, 0.0, 1.0),
        anchor_world=(0.1, 0.2, 0.3),
        source="test",
    )
    assert isinstance(w, OverlayWrench)
    assert w.kind == WrenchKind.JOINT_ACTUATOR
    assert w.label == "actuator:knee_z"
    assert w.body == "femur"
    assert w.point_m == (0.1, 0.2, 0.3)
    assert w.force_n is None
    assert w.torque_nm == (0.0, 0.0, 10.0)
    assert w.source == "test"


def test_joint_torque_wrench_gimbal_sum() -> None:
    """A 3-axis gimbal sums its moments across multiple axes."""
    tau_nm = [5.0, -3.0, 2.0]
    axes = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    w = joint_torque_wrench(
        label="actuator:shoulder_gimbal",
        body="humerus",
        tau_nm=tau_nm,
        axis_world=axes,
        anchor_world=(0.0, 0.0, 1.5),
        source="test",
    )
    assert w.force_n is None
    assert w.torque_nm == (5.0, -3.0, 2.0)


def test_joint_torque_wrench_non_unit_axis_raises() -> None:
    """Non-unit axis (norm != 1.0 within 1e-6) raises ValueError."""
    with pytest.raises(ValueError, match="unit norm"):
        joint_torque_wrench(
            label="actuator:bad_axis",
            body="torso",
            tau_nm=1.0,
            axis_world=(1.0, 1.0, 0.0),  # norm = sqrt(2) != 1.0
            anchor_world=(0.0, 0.0, 0.0),
            source="test",
        )


def test_joint_torque_wrench_mismatched_dimensions_raises() -> None:
    """Mismatched axis and torque lengths raise ValueError."""
    with pytest.raises(ValueError):
        joint_torque_wrench(
            label="actuator:mismatch",
            body="torso",
            tau_nm=[1.0, 2.0],
            axis_world=[(1.0, 0.0, 0.0)],
            anchor_world=(0.0, 0.0, 0.0),
            source="test",
        )


def test_world_wrench_from_local_rotation() -> None:
    """A 90-degree rotation about z maps local +x to world +y."""
    # 90 deg rotation about z: [0, -1, 0; 1, 0, 0; 0, 0, 1]
    r_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    w = world_wrench_from_local(
        label="contact:foot",
        body="foot",
        kind=WrenchKind.CONTACT,
        force_local=(100.0, 0.0, 0.0),
        torque_local=(0.0, 0.0, 10.0),
        rotation_world_from_local=r_z_90,
        point_world=(0.2, 0.3, 0.0),
        source="test",
    )
    assert w.point_m == (0.2, 0.3, 0.0)
    assert w.force_n is not None
    assert w.torque_nm is not None
    assert pytest.approx(w.force_n[0], abs=1e-9) == 0.0
    assert pytest.approx(w.force_n[1], abs=1e-9) == 100.0
    assert pytest.approx(w.force_n[2], abs=1e-9) == 0.0
    assert pytest.approx(w.torque_nm[0], abs=1e-9) == 0.0
    assert pytest.approx(w.torque_nm[1], abs=1e-9) == 0.0
    assert pytest.approx(w.torque_nm[2], abs=1e-9) == 10.0


def test_world_wrench_from_local_optional_halves() -> None:
    """Torque-only input stays torque-only; force-only stays force-only."""
    r_identity = np.eye(3, dtype=float)

    # Torque only
    w_t = world_wrench_from_local(
        label="actuator:joint",
        body="arm",
        kind=WrenchKind.JOINT_ACTUATOR,
        force_local=None,
        torque_local=(0.0, 5.0, 0.0),
        rotation_world_from_local=r_identity,
        point_world=(0.0, 0.0, 0.0),
        source="test",
    )
    assert w_t.force_n is None
    assert w_t.torque_nm == (0.0, 5.0, 0.0)

    # Force only
    w_f = world_wrench_from_local(
        label="external:push",
        body="arm",
        kind=WrenchKind.EXTERNAL,
        force_local=(10.0, 0.0, 0.0),
        torque_local=None,
        rotation_world_from_local=r_identity,
        point_world=(0.0, 0.0, 0.0),
        source="test",
    )
    assert w_f.force_n == (10.0, 0.0, 0.0)
    assert w_f.torque_nm is None


def test_world_wrench_from_local_invalid_rotation_raises() -> None:
    """Non-orthonormal matrix or det != 1 raises ValueError."""
    # Scaled matrix: not orthonormal
    r_scaled = 2.0 * np.eye(3, dtype=float)
    with pytest.raises(ValueError, match="orthonormal"):
        world_wrench_from_local(
            label="contact:invalid",
            body="foot",
            kind=WrenchKind.CONTACT,
            force_local=(1.0, 0.0, 0.0),
            torque_local=None,
            rotation_world_from_local=r_scaled,
            point_world=(0.0, 0.0, 0.0),
            source="test",
        )

    # Reflection matrix: det = -1
    r_reflect = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="determinant"):
        world_wrench_from_local(
            label="contact:reflect",
            body="foot",
            kind=WrenchKind.CONTACT,
            force_local=(1.0, 0.0, 0.0),
            torque_local=None,
            rotation_world_from_local=r_reflect,
            point_world=(0.0, 0.0, 0.0),
            source="test",
        )


def test_move_wrench_point_with_force_and_torque() -> None:
    """A known force moved by r gives tau_B = tau_A + (p_A - p_B) x F."""
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 100.0),  # +z force
        torque_nm=(10.0, 0.0, 0.0),  # +x torque
        source="test",
    )
    # Move to new point p_B = (0, 1, 0)
    # p_A - p_B = (0, -1, 0)
    # r x F = (0, -1, 0) x (0, 0, 100) = (-100, 0, 0)
    # tau_B = (10, 0, 0) + (-100, 0, 0) = (-90, 0, 0)
    w_moved = move_wrench_point(w, (0.0, 1.0, 0.0))
    assert w_moved.point_m == (0.0, 1.0, 0.0)
    assert w_moved.force_n == (0.0, 0.0, 100.0)
    assert w_moved.torque_nm is not None
    assert pytest.approx(w_moved.torque_nm[0]) == -90.0
    assert pytest.approx(w_moved.torque_nm[1]) == 0.0
    assert pytest.approx(w_moved.torque_nm[2]) == 0.0


def test_move_wrench_point_torque_only_yields_none_torque_raises_when_moved() -> None:
    """A torque-only wrench (force_n is None) moved to a different point cannot determine torque and raises."""
    w = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="actuator:motor",
        body="wheel",
        point_m=(0.0, 0.0, 0.0),
        force_n=None,
        torque_nm=(0.0, 0.0, 10.0),
        source="test",
    )
    # Same point returns wrench unchanged
    assert move_wrench_point(w, (0.0, 0.0, 0.0)) == w

    # Different point raises ValueError
    with pytest.raises(
        ValueError,
        match="Cannot move wrench to a new application point when force_n is None",
    ):
        move_wrench_point(w, (1.0, 2.0, 3.0))


def test_synthetic_two_link_chain_axial_load_agreement() -> None:
    """Agreement with the synthetic two-link chain in tests/unit/body_part_viz/test_axial_loads.py."""
    # Link 1: proximal (0, 0, 2), distal (0, 0, 1)
    # Link 2: proximal (0, 0, 1), distal (0, 0, 0)
    p1, d1 = (0.0, 0.0, 2.0), (0.0, 0.0, 1.0)
    p2, d2 = (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)

    # Reaction forces: link 1 pulled up with 100 N, link 2 pulled up with 40 N
    f1 = (0.0, 0.0, 100.0)
    f2 = (0.0, 0.0, 40.0)

    expected_1 = axial_force_from_proximal_reaction(f1, p1, d1)
    expected_2 = axial_force_from_proximal_reaction(f2, p2, d2)
    assert expected_1 == 100.0
    assert expected_2 == 40.0

    frame = ForceTorqueFrame(
        time_s=1.0,
        engine="synthetic_chain",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label="joint_reaction:link1_proximal",
                body="link1",
                point_m=p1,
                force_n=f1,
                torque_nm=None,
                source="synthetic",
            ),
            OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label="joint_reaction:link2_proximal",
                body="link2",
                point_m=p2,
                force_n=f2,
                torque_nm=None,
                source="synthetic",
            ),
        ),
    )
    axes = [
        SegmentAxis("link1", "joint_reaction:link1_proximal", p1, d1),
        SegmentAxis("link2", "joint_reaction:link2_proximal", p2, d2),
    ]

    axial_frame = axial_loads_from_reactions(frame, axes, source="synthetic_evaluator")
    assert pytest.approx(axial_frame.values_n["link1"]) == expected_1
    assert pytest.approx(axial_frame.values_n["link2"]) == expected_2


def test_move_wrench_point_force_only_yields_none_torque() -> None:
    """A force-only wrench moved to a new point retains torque_nm is None."""
    w = OverlayWrench(
        kind=WrenchKind.EXTERNAL,
        label="external:push",
        body="box",
        point_m=(0.0, 0.0, 0.0),
        force_n=(10.0, 0.0, 0.0),
        torque_nm=None,
        source="test",
    )
    w_moved = move_wrench_point(w, (0.0, 1.0, 0.0))
    assert w_moved.point_m == (0.0, 1.0, 0.0)
    assert w_moved.force_n == (10.0, 0.0, 0.0)
    assert w_moved.torque_nm is None


def test_segment_axis_validation() -> None:
    """SegmentAxis validates inputs and rejects coincident endpoints."""
    axis = SegmentAxis(
        segment="femur",
        joint_label="joint_reaction:hip",
        proximal_m=(0.0, 0.0, 1.0),
        distal_m=(0.0, 0.0, 0.5),
    )
    assert axis.segment == "femur"
    assert axis.joint_label == "joint_reaction:hip"

    # Coincident endpoints raise ValueError
    with pytest.raises(ValueError, match="coincident"):
        SegmentAxis(
            segment="femur",
            joint_label="joint_reaction:hip",
            proximal_m=(1.0, 2.0, 3.0),
            distal_m=(1.0, 2.0, 3.0),
        )


def test_axial_loads_hanging_segment_tension() -> None:
    """Hanging segment: reaction matches axial_force_from_proximal_reaction (tension positive)."""
    proximal = (0.0, 0.0, 1.0)
    distal = (0.0, 0.0, 0.0)  # axis is down (-z)
    # Hanging load: parent pulls up (+z) with 50 N
    reaction_force = (0.0, 0.0, 50.0)

    # Direct calculation check:
    expected_tension = axial_force_from_proximal_reaction(
        reaction_force, proximal, distal
    )
    assert expected_tension > 0.0  # tension is positive

    frame = ForceTorqueFrame(
        time_s=0.0,
        engine="test_engine",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label="joint_reaction:hip",
                body="femur",
                point_m=proximal,
                force_n=reaction_force,
                torque_nm=None,
                source="test",
            ),
        ),
    )
    axes = [
        SegmentAxis(
            segment="femur",
            joint_label="joint_reaction:hip",
            proximal_m=proximal,
            distal_m=distal,
        )
    ]

    axial_frame = axial_loads_from_reactions(frame, axes, source="test_provider")
    assert isinstance(axial_frame, AxialLoadFrame)
    assert axial_frame.time_s == 0.0
    assert axial_frame.source == "test_provider"
    assert pytest.approx(axial_frame.values_n["femur"]) == expected_tension


def test_axial_loads_pushing_segment_compression() -> None:
    """Pushing segment: compression is negative."""
    proximal = (0.0, 0.0, 0.0)
    distal = (0.0, 0.0, 1.0)  # axis is up (+z)
    # Compression: reaction force pushes upward along proximal->distal
    reaction_force = (0.0, 0.0, 50.0)

    expected = axial_force_from_proximal_reaction(reaction_force, proximal, distal)
    assert expected < 0.0  # compression is negative

    frame = ForceTorqueFrame(
        time_s=0.5,
        engine="test_engine",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label="joint_reaction:base",
                body="column",
                point_m=proximal,
                force_n=reaction_force,
                torque_nm=None,
                source="test",
            ),
        ),
    )
    axes = [
        SegmentAxis(
            segment="column",
            joint_label="joint_reaction:base",
            proximal_m=proximal,
            distal_m=distal,
        )
    ]
    axial_frame = axial_loads_from_reactions(frame, axes, source="test")
    assert pytest.approx(axial_frame.values_n["column"]) == expected


def test_axial_loads_missing_reaction_gives_none() -> None:
    """Missing reaction or force-less reaction gives None for that segment; others compute."""
    proximal_1 = (0.0, 0.0, 1.0)
    distal_1 = (0.0, 0.0, 0.0)
    proximal_2 = (1.0, 0.0, 1.0)
    distal_2 = (1.0, 0.0, 0.0)

    frame = ForceTorqueFrame(
        time_s=0.1,
        engine="test_engine",
        wrenches=(
            # reaction for segment 1 with force
            OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label="joint_reaction:seg1",
                body="seg1",
                point_m=proximal_1,
                force_n=(0.0, 0.0, 20.0),
                torque_nm=None,
                source="test",
            ),
            # reaction for segment 2 torque-only (force is None)
            OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label="joint_reaction:seg2",
                body="seg2",
                point_m=proximal_2,
                force_n=None,
                torque_nm=(1.0, 0.0, 0.0),
                source="test",
            ),
        ),
    )
    axes = [
        SegmentAxis("seg1", "joint_reaction:seg1", proximal_1, distal_1),
        SegmentAxis("seg2", "joint_reaction:seg2", proximal_2, distal_2),
        SegmentAxis(
            "seg3", "joint_reaction:seg3", proximal_1, distal_1
        ),  # missing label
    ]

    axial_frame = axial_loads_from_reactions(frame, axes, source="test")
    assert axial_frame.values_n["seg1"] is not None
    assert axial_frame.values_n["seg2"] is None
    assert axial_frame.values_n["seg3"] is None


def test_frame_with_axial_loads() -> None:
    """frame_with_axial_loads returns a copy of the frame with axial_loads populated."""
    proximal = (0.0, 0.0, 1.0)
    distal = (0.0, 0.0, 0.0)
    frame = ForceTorqueFrame(
        time_s=0.2,
        engine="test_engine",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.JOINT_REACTION,
                label="joint_reaction:link1",
                body="link1",
                point_m=proximal,
                force_n=(0.0, 0.0, 10.0),
                torque_nm=None,
                source="test",
            ),
        ),
    )
    axes = [SegmentAxis("link1", "joint_reaction:link1", proximal, distal)]

    new_frame = frame_with_axial_loads(frame, axes, source="axial_converter")
    assert new_frame.axial_loads is not None
    assert new_frame.axial_loads.values_n["link1"] is not None
    assert new_frame.time_s == frame.time_s
    assert new_frame.engine == frame.engine
    assert new_frame.wrenches == frame.wrenches


def test_joint_torque_wrench_additional_validation() -> None:
    """Validate all error paths in joint_torque_wrench."""
    # non-finite axis_world
    with pytest.raises(ValueError, match="finite"):
        joint_torque_wrench("act:1", "b", 1.0, (1.0, float("nan"), 0.0), (0, 0, 0), "s")

    # non-finite scalar tau_nm
    with pytest.raises(ValueError, match="finite"):
        joint_torque_wrench("act:1", "b", float("inf"), (1.0, 0.0, 0.0), (0, 0, 0), "s")

    # bad shape axis_world for scalar
    with pytest.raises(ValueError, match="shape"):
        joint_torque_wrench("act:1", "b", 1.0, [1.0, 0.0, 0.0, 0.0], (0, 0, 0), "s")

    # 2D tau_nm
    with pytest.raises(ValueError, match="1-dimensional"):
        joint_torque_wrench("act:1", "b", [[1.0]], [[1.0, 0.0, 0.0]], (0, 0, 0), "s")  # type: ignore[arg-type,list-item]

    # non-finite tau_nm sequence
    with pytest.raises(ValueError, match="finite"):
        joint_torque_wrench(
            "act:1", "b", [float("nan")], [[1.0, 0.0, 0.0]], (0, 0, 0), "s"
        )

    # multi-axis non-unit axis
    with pytest.raises(ValueError, match="unit norm"):
        joint_torque_wrench("act:1", "b", [1.0], [[1.0, 1.0, 0.0]], (0, 0, 0), "s")

    # unsupported tau_nm type
    with pytest.raises(TypeError, match="float or Sequence"):
        joint_torque_wrench("act:1", "b", "bad_tau", (1.0, 0.0, 0.0), (0, 0, 0), "s")  # type: ignore[arg-type]


def test_world_wrench_from_local_additional_validation() -> None:
    """Validate error paths in world_wrench_from_local."""
    # bad shape R
    with pytest.raises(ValueError, match="shape"):
        world_wrench_from_local(
            "l",
            "b",
            WrenchKind.CONTACT,
            (1, 0, 0),
            None,
            [[1, 0], [0, 1]],
            (0, 0, 0),
            "s",
        )

    # non-finite R
    with pytest.raises(ValueError, match="finite"):
        r_inf = np.eye(3)
        r_inf[0, 0] = float("inf")
        world_wrench_from_local(
            "l", "b", WrenchKind.CONTACT, (1, 0, 0), None, r_inf, (0, 0, 0), "s"
        )

    # both force and torque None
    with pytest.raises(ValueError, match="At least one"):
        world_wrench_from_local(
            "l", "b", WrenchKind.CONTACT, None, None, np.eye(3), (0, 0, 0), "s"
        )


def test_segment_axis_string_validation() -> None:
    """SegmentAxis validates segment and joint_label string properties."""
    with pytest.raises(ValueError, match="segment"):
        SegmentAxis("", "j", (0, 0, 0), (0, 0, 1))

    with pytest.raises(ValueError, match="segment"):
        SegmentAxis("   ", "j", (0, 0, 0), (0, 0, 1))

    with pytest.raises(ValueError, match="joint_label"):
        SegmentAxis("s", "", (0, 0, 0), (0, 0, 1))

    with pytest.raises(ValueError, match="joint_label"):
        SegmentAxis("s", "   ", (0, 0, 0), (0, 0, 1))
