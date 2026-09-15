"""Tests for marker kinematics and pose IK on the MuJoCo full-body model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python import full_body_markers as module
from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching.contact_law import GroundPlane

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
GROUND = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)


@pytest.fixture(scope="module")
def kinematics() -> module.FullBodyMarkerKinematics:
    spec_bytes = SPEC.read_bytes()
    spec = json.loads(spec_bytes)
    attachments = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    attachments["RKneeOut"] = ("femur_r", (0.0, -0.4, 0.06))  # synthetic leg marker
    attachments["LToeOut"] = ("calcn_l", (0.16, 0.0, -0.04))
    attachments["LKneeOut"] = ("tibia_l", (0.0, -0.05, 0.06))
    attachments["LAnkleOut"] = ("talus_l", (0.0, 0.0, 0.05))
    return module.FullBodyMarkerKinematics(
        NativeMujocoFullBodyModel(spec_bytes), attachments
    )


def test_body_markers_use_the_specification_body_frame(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    """An offset at the knee joint in the spec femur frame lands on the knee."""
    spec = json.loads(SPEC.read_text())
    knee = next(j for j in spec["joints"] if j["name"] == "knee_r")
    offset = tuple(np.array(knee["parent_to_base"])[:3, 3])
    probe = module.FullBodyMarkerKinematics(
        kinematics.adapter, {"probe": ("femur_r", offset)}
    )
    q = np.zeros(len(probe.coordinate_order))
    q[2] = 1.0
    world = probe.marker_positions(q)[0]
    anchor = probe.data.xanchor[probe.model.joint("knee_angle_r").id]
    np.testing.assert_allclose(world, anchor, atol=1e-9)
    rotation, translation = probe.body_poses(q, ["femur_r"])["femur_r"]
    np.testing.assert_allclose(
        rotation @ np.array(offset) + translation, anchor, atol=1e-9
    )
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_marker_positions_follow_attached_bodies(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    q = np.zeros(len(kinematics.coordinate_order))
    base = kinematics.marker_positions(q)
    q[kinematics.coordinate_order.index("TranslationInputZ")] = 0.5
    moved = kinematics.marker_positions(q)
    shifts = moved - base  # one rigid translation of 0.5 m along the slide axis
    np.testing.assert_allclose(np.linalg.norm(shifts, axis=1), 0.5, atol=1e-12)
    np.testing.assert_allclose(shifts - shifts[0], 0.0, atol=1e-12)
    assert set(kinematics.labels) >= {"RKneeOut", "LToeOut", "WaistLeft"}
    with pytest.raises(ValueError):
        kinematics.marker_positions(q[:-1])


def test_pose_ik_recovers_markers_and_respects_ground_and_locks(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    rng = np.random.default_rng(3)
    n = len(kinematics.coordinate_order)
    q_true = np.zeros(n)
    q_true[:3] = [0.3, -0.2, 1.2]
    q_true[6:] = rng.uniform(-0.25, 0.25, n - 6)
    targets = kinematics.marker_positions(q_true)
    valid = np.ones(len(targets), dtype=bool)
    valid[4] = False
    q_start = q_true + rng.normal(0.0, 0.05, n)
    fit = kinematics.solve_pose(
        targets, valid, q_start, ground=GROUND, closure_weight=0.0, prior_weight=1e-6
    )
    assert fit.marker_rms_m < 1e-4
    assert fit.iterations >= 1 and len(fit.per_marker_m) == int(valid.sum())
    # Ground penalty: a pose whose feet are below the plane is lifted.
    low = q_true.copy()
    low[2] = -0.2
    lifted = kinematics.solve_pose(
        kinematics.marker_positions(low),
        np.ones(len(targets), bool),
        low,
        ground=GROUND,
        closure_weight=0.0,
        prior_weight=1e-6,
        ground_weight=1e4,
    )
    assert lifted.lowest_sphere_height_m > -5e-3
    assert (
        kinematics.sphere_heights(low, GROUND)["heel_l"] < lifted.lowest_sphere_height_m
    )
    # Locks pin coordinates exactly.
    pinned = kinematics.solve_pose(
        targets,
        valid,
        q_start,
        ground=GROUND,
        closure_weight=0.0,
        locked={"knee_angle_r": 0.1},
    )
    assert pinned.q[kinematics.coordinate_order.index("knee_angle_r")] == 0.1
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets, np.zeros(len(targets), bool), q_start, ground=GROUND
        )
    # Flat feet: with no markers the legs bend until all four spheres touch.
    standing = kinematics.solve_pose(
        targets,
        np.zeros(len(targets), bool),
        q_start,
        ground=GROUND,
        closure_weight=0.0,
        flat_feet=True,
        ground_weight=1e4,
    )
    heights = kinematics.sphere_heights(standing.q, GROUND)
    assert max(abs(h) for h in heights.values()) < 1e-3
    one_foot = kinematics.solve_pose(
        targets,
        np.zeros(len(targets), bool),
        q_start,
        ground=GROUND,
        closure_weight=0.0,
        flat_feet=("heel_l", "forefoot_l"),
        ground_weight=1e4,
    )
    heights = kinematics.sphere_heights(one_foot.q, GROUND)
    assert abs(heights["heel_l"]) < 1e-3 and abs(heights["forefoot_l"]) < 1e-3
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets, valid, q_start, ground=GROUND, flat_feet=("nope",)
        )
    balanced = kinematics.solve_pose(
        targets,
        np.zeros(len(targets), bool),
        standing.q,
        ground=GROUND,
        closure_weight=0.0,
        flat_feet=True,
        ground_weight=1e4,
        balance_weight=1e4,
    )
    assert kinematics.support_offset(balanced.q, GROUND) < 1e-3
    assert (
        max(abs(h) for h in kinematics.sphere_heights(balanced.q, GROUND).values())
        < 1e-3
    )
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets, valid, q_start, ground=GROUND, locked={"nope": 0.0}
        )
    bounded = kinematics.solve_pose(
        targets,
        valid,
        q_start,
        ground=GROUND,
        closure_weight=0.0,
        bounds={"knee_angle_r": (-0.05, 0.05)},
    )
    assert abs(bounded.q[kinematics.coordinate_order.index("knee_angle_r")]) <= 0.05
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets, valid, q_start, ground=GROUND, bounds={"knee_angle_r": (1.0, 0.0)}
        )


def test_trajectory_solver_warm_starts(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    n = len(kinematics.coordinate_order)
    q_a = np.zeros(n)
    q_a[2] = 1.0
    q_b = q_a.copy()
    q_b[kinematics.coordinate_order.index("knee_angle_l")] = 0.3
    targets = np.stack(
        [kinematics.marker_positions(q_a), kinematics.marker_positions(q_b)]
    )
    valid = np.ones(targets.shape[:2], dtype=bool)
    q, fits = kinematics.solve_trajectory(
        targets, valid, q_a, ground=GROUND, closure_weight=0.0, prior_weight=1e-6
    )
    assert q.shape == (2, n) and all(f.marker_rms_m < 1e-4 for f in fits)
    assert abs(q[1, kinematics.coordinate_order.index("knee_angle_l")] - 0.3) < 5e-3
    with pytest.raises(ValueError):
        kinematics.solve_trajectory(targets[0], valid[0], q_a, ground=GROUND)
    # Restarts from a poor start never make a frame worse than the plain solve
    # (one frame, so both runs share the same start pose).
    far = q_a.copy()
    far[6:] += 1.0
    plain = kinematics.solve_trajectory(
        targets[:1], valid[:1], far, ground=GROUND, closure_weight=0.0, iterations=5
    )[1][0]
    restarted = kinematics.solve_trajectory(
        targets[:1],
        valid[:1],
        far,
        ground=GROUND,
        closure_weight=0.0,
        iterations=5,
        restarts=3,
        restart_threshold_m=1e-4,
    )[1][0]
    assert restarted.marker_rms_m <= plain.marker_rms_m + 1e-12
    with pytest.raises(ValueError):
        kinematics.solve_trajectory(targets, valid, q_a, ground=GROUND, restarts=-1)


def test_planted_stance_spheres_do_not_slide(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    """Anchored spheres keep their ground point across frames; pins alone do not."""
    n = len(kinematics.coordinate_order)
    q_a = np.zeros(n)
    q_a[2] = 1.0
    q_b = q_a.copy()
    q_b[0] += 0.05  # the whole body shifted 5 cm: markers say slide
    targets = np.stack(
        [kinematics.marker_positions(q_a), kinematics.marker_positions(q_b)]
    )
    valid = np.ones(targets.shape[:2], dtype=bool)
    stance = [("heel_l", "forefoot_l"), ("heel_l", "forefoot_l")]
    free, _ = kinematics.solve_trajectory(
        targets,
        valid,
        q_a,
        ground=GROUND,
        closure_weight=0.0,
        prior_weight=1e-6,
        flat_feet_per_frame=stance,
        ground_weight=1e4,
    )
    planted, fits = kinematics.solve_trajectory(
        targets,
        valid,
        q_a,
        ground=GROUND,
        closure_weight=0.0,
        prior_weight=1e-6,
        flat_feet_per_frame=stance,
        plant_stance=True,
        ground_weight=1e4,
    )
    moved_free = kinematics.sphere_ground_points(free[1], GROUND)["heel_l"]
    start = kinematics.sphere_ground_points(free[0], GROUND)["heel_l"]
    moved_planted = kinematics.sphere_ground_points(planted[1], GROUND)["heel_l"]
    assert np.linalg.norm(moved_free - start) > 0.03
    assert np.linalg.norm(moved_planted - start) < 2e-3
    # A prior trajectory replaces the warm start.
    prior = np.stack([q_a, q_a])
    pulled, _ = kinematics.solve_trajectory(
        targets,
        valid,
        q_a,
        ground=GROUND,
        closure_weight=0.0,
        prior_weight=1e3,
        prior_trajectory=prior,
    )
    assert np.abs(pulled[1] - q_a).max() < 1e-2
    with pytest.raises(ValueError):
        kinematics.solve_trajectory(
            targets, valid, q_a, ground=GROUND, plant_stance=True
        )
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets[0], valid[0], q_a, ground=GROUND, anchors={"nope": (0, 0, 0)}
        )


def test_marker_weights_drop_a_marker_from_the_fit(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    rng = np.random.default_rng(5)
    n = len(kinematics.coordinate_order)
    q_true = np.zeros(n)
    q_true[:3] = [0.3, -0.2, 1.2]
    q_true[6:] = rng.uniform(-0.25, 0.25, n - 6)
    targets = kinematics.marker_positions(q_true)
    valid = np.ones(len(targets), dtype=bool)
    spoiled = kinematics.labels[0]
    targets[0] += [0.3, 0.0, 0.0]  # one marker is wrong by 30 cm
    q_start = q_true + rng.normal(0.0, 0.05, n)
    fit = kinematics.solve_pose(
        targets,
        valid,
        q_start,
        ground=GROUND,
        closure_weight=0.0,
        prior_weight=1e-6,
        marker_weights={spoiled: 0.0},
    )
    assert fit.per_marker_m[spoiled] == pytest.approx(0.3, abs=1e-3)
    others = [e for label, e in fit.per_marker_m.items() if label != spoiled]
    assert max(others) < 1e-3
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets, valid, q_start, ground=GROUND, marker_weights={spoiled: -1.0}
        )
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets, valid, q_start, ground=GROUND, marker_weights={"nope": 1.0}
        )


def _rotation_xyz(x: float, y: float, z: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = (
        np.cos(x),
        np.sin(x),
        np.cos(y),
        np.sin(y),
        np.cos(z),
        np.sin(z),
    )
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rx @ ry @ rz


def test_continuous_branches_keep_the_pose_and_remove_jumps() -> None:
    order = ["r0", "r1", "r2", "r3", "r4", "r5", "ax", "ay", "az", "e"]
    frames = np.zeros((3, len(order)))
    frames[0, 6:9] = [0.2, 0.4, -0.3]
    frames[1, 6:9] = [0.2 + np.pi, np.pi - 0.45, -0.3 + np.pi]  # other branch
    frames[2, 6:9] = [0.3, 0.5, -0.2]
    frames[:, 9] = [0.1, 0.1 + 2 * np.pi, 0.2]  # a 2 pi wrap
    fixed = module.continuous_branches(frames, order, gimbals=[("ax", "ay", "az")])
    for k in range(3):
        np.testing.assert_allclose(
            _rotation_xyz(*fixed[k, 6:9]), _rotation_xyz(*frames[k, 6:9]), atol=1e-12
        )
    assert np.abs(np.diff(fixed[:, 6:], axis=0)).max() < 0.5
    np.testing.assert_allclose(fixed[:, :6], frames[:, :6])
    with pytest.raises(ValueError):
        module.continuous_branches(frames, order, gimbals=[("ax", "ay", "nope")])
    with pytest.raises(ValueError):
        module.continuous_branches(frames[:, :4], order)


def test_prior_weights_hold_named_coordinates(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    rng = np.random.default_rng(7)
    n = len(kinematics.coordinate_order)
    q_true = np.zeros(n)
    q_true[:3] = [0.3, -0.2, 1.2]
    q_true[6:] = rng.uniform(-0.25, 0.25, n - 6)
    targets = kinematics.marker_positions(q_true)
    valid = np.ones(len(targets), dtype=bool)
    q_start = q_true.copy()
    held = "knee_angle_l"  # observable through the left ankle and toe markers
    index = kinematics.coordinate_order.index(held)
    q_start[index] += 0.3
    free = kinematics.solve_pose(
        targets, valid, q_start, ground=GROUND, closure_weight=0.0, prior_weight=1e-6
    )
    stiff = kinematics.solve_pose(
        targets,
        valid,
        q_start,
        ground=GROUND,
        closure_weight=0.0,
        prior_weight=1e-6,
        prior_weights={held: 1e6},
    )
    assert abs(free.q[index] - q_true[index]) < 1e-3
    assert abs(stiff.q[index] - q_start[index]) < 1e-3
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets, valid, q_start, ground=GROUND, prior_weights={"nope": 1.0}
        )
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets, valid, q_start, ground=GROUND, prior_weights={held: -1.0}
        )


def test_axis_targets_turn_a_frame_axis_toward_a_direction(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    n = len(kinematics.coordinate_order)
    q0 = np.zeros(n)
    q0[2] = 1.2
    targets = kinematics.marker_positions(q0)
    valid = np.ones(len(targets), dtype=bool)
    frame = "Hub"
    site = kinematics.model.site(kinematics.adapter.metadata["frame_sites"][frame]).id
    kinematics._set(q0)
    before = kinematics.data.site_xmat[site].reshape(3, 3)[:, 0].copy()
    want = np.array([0.0, 0.0, 1.0])
    assert before @ want < 0.5
    fit = kinematics.solve_pose(
        targets,
        valid,
        q0,
        ground=GROUND,
        closure_weight=0.0,
        prior_weight=1e-6,
        axis_targets={frame: ((1.0, 0.0, 0.0), want, 1e3)},  # outweighs the markers
    )
    kinematics._set(fit.q)
    after = kinematics.data.site_xmat[site].reshape(3, 3)[:, 0]
    assert after @ want > 0.99
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets,
            valid,
            q0,
            ground=GROUND,
            axis_targets={"nope": ((1, 0, 0), want, 1)},
        )
    with pytest.raises(ValueError):
        kinematics.solve_pose(
            targets,
            valid,
            q0,
            ground=GROUND,
            axis_targets={frame: ((0, 0, 0), want, 1)},
        )


def test_axis_targets_per_frame_are_validated_and_applied(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    n = len(kinematics.coordinate_order)
    q0 = np.zeros(n)
    q0[2] = 1.2
    targets = np.stack([kinematics.marker_positions(q0)] * 2)
    valid = np.ones(targets.shape[:2], dtype=bool)
    frame = "Hub"
    site = kinematics.model.site(kinematics.adapter.metadata["frame_sites"][frame]).id
    want = np.array([0.0, 0.0, 1.0])
    per_frame = [None, {frame: ((1.0, 0.0, 0.0), want, 1e3)}]
    q, _ = kinematics.solve_trajectory(
        targets,
        valid,
        q0,
        ground=GROUND,
        closure_weight=0.0,
        prior_weight=1e-6,
        axis_targets_per_frame=per_frame,
    )
    kinematics._set(q[0])
    first = kinematics.data.site_xmat[site].reshape(3, 3)[:, 0] @ want
    kinematics._set(q[1])
    second = kinematics.data.site_xmat[site].reshape(3, 3)[:, 0] @ want
    assert second > 0.99 and first < 0.5
    with pytest.raises(ValueError):
        kinematics.solve_trajectory(
            targets, valid, q0, ground=GROUND, axis_targets_per_frame=[None]
        )


def test_closure_rotation_weight_can_release_the_grip_orientation(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    n = len(kinematics.coordinate_order)
    q0 = np.zeros(n)
    q0[2] = 1.2
    kinematics._set(q0)
    rows: list = []
    jacs: list = []
    kinematics._append_closure(rows, jacs, 1e2)
    assert [r.shape for r in rows] == [(3,), (3,)]  # positions and orientations
    rows, jacs = [], []
    kinematics._append_closure(rows, jacs, 1e2, 0.0)
    assert [r.shape for r in rows] == [(3,)]  # positions only: the roll is free
    rows, jacs = [], []
    kinematics._append_closure(rows, jacs, 0.0, 0.0)
    assert rows == []
    with pytest.raises(ValueError):
        kinematics._append_closure([], [], 1e2, -1.0)


def test_com_target_rows_pull_the_centre_of_mass(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    kin = kinematics
    spheres = list(kin._spheres)
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    q0 = np.zeros(len(kin.coordinate_order))
    before = kin.com_plane_position(q0, ground)
    goal = before + np.array([0.03, -0.02])
    fit = kin.solve_pose(
        np.full((len(kin.labels), 3), np.nan),
        np.zeros(len(kin.labels), dtype=bool),
        q0,
        ground=ground,
        flat_feet=True,
        prior_weight=1e-3,
        com_target=(goal, 500.0),
        iterations=80,
    )
    after = kin.com_plane_position(fit.q, ground)
    assert np.linalg.norm(after - goal) < np.linalg.norm(before - goal) * 0.5
    assert np.linalg.norm(after - before) > 0.01
    with pytest.raises(ValueError):
        kin.solve_pose(
            np.full((len(kin.labels), 3), np.nan),
            np.zeros(len(kin.labels), dtype=bool),
            q0,
            ground=ground,
            flat_feet=True,
            com_target=(goal, -1.0),
        )
    traj, _ = kin.solve_trajectory(
        np.full((2, len(kin.labels), 3), np.nan),
        np.zeros((2, len(kin.labels)), dtype=bool),
        q0,
        ground=ground,
        flat_feet_per_frame=[spheres, spheres],
        prior_weight=1e-3,
        com_targets_per_frame=[None, (goal, 50.0)],
        iterations=20,
    )
    assert traj.shape[0] == 2
    with pytest.raises(ValueError):
        kin.solve_trajectory(
            np.full((2, len(kin.labels), 3), np.nan),
            np.zeros((2, len(kin.labels)), dtype=bool),
            q0,
            ground=ground,
            com_targets_per_frame=[None],
        )


def test_locked_per_frame_pins_named_coordinates(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    kin = kinematics
    spheres = list(kin._spheres)
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    q0 = np.zeros(len(kin.coordinate_order))
    traj, _ = kin.solve_trajectory(
        np.full((2, len(kin.labels), 3), np.nan),
        np.zeros((2, len(kin.labels)), dtype=bool),
        q0,
        ground=ground,
        flat_feet_per_frame=[spheres, spheres],
        prior_weight=1e-3,
        locked_per_frame=[None, {"TranslationInputX": 0.05, "HipInputZ": 0.1}],
        iterations=10,
    )
    x = kin.coordinate_order.index("TranslationInputX")
    z = kin.coordinate_order.index("HipInputZ")
    assert traj[1, x] == 0.05 and traj[1, z] == 0.1
    with pytest.raises(ValueError):
        kin.solve_trajectory(
            np.full((2, len(kin.labels), 3), np.nan),
            np.zeros((2, len(kin.labels)), dtype=bool),
            q0,
            ground=ground,
            locked_per_frame=[None],
        )


def test_delegating_properties_satisfy_lod(
    kinematics: module.FullBodyMarkerKinematics,
) -> None:
    """Delegating properties expose spheres, closure sites and marker offsets without reach-through."""
    kin = kinematics
    assert kin.nq == len(kin.coordinate_order)
    assert kin.nq > 0

    assert isinstance(kin.sphere_names, tuple)
    assert len(kin.sphere_names) > 0
    assert "heel_r" in kin.sphere_names

    assert isinstance(kin.closure_sites, tuple)
    assert len(kin.closure_sites) == 2

    offsets = kin.marker_bodies_and_offsets
    assert isinstance(offsets, dict)
    assert set(offsets.keys()) == set(kin.labels)
    for body, offset in offsets.values():
        assert isinstance(body, str)
        assert isinstance(offset, np.ndarray)
        assert offset.shape == (3,)
