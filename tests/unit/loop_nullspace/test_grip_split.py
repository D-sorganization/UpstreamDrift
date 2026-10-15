"""Unit tests for NSX-16 grip-frame hand-split coordinates and bare-club net wrench (#12561)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.grip_wrench import SPLIT_METHODS, SPLITS
from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    GripFrame,
)
from src.shared.python.loop_nullspace.fixtures import (
    planar_golfer_snapshot,
    spatial_two_arm_club_snapshot,
)
from src.shared.python.loop_nullspace.grip_split import (
    SPLIT_COORDINATE_NAMES,
    grasp_map,
    hand_wrenches,
    net_club_wrench,
    split_basis,
    split_coordinates,
)

pytestmark = pytest.mark.unit


def _canonical_grip_setup(
    half_sep_m: float = 0.04,
) -> tuple[GripFrame, np.ndarray, np.ndarray]:
    grip_frame = GripFrame(
        rotation=np.eye(3, dtype=np.float64),
        origin=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    r_l = np.array([-half_sep_m, 0.0, 0.0], dtype=np.float64)
    r_r = np.array([half_sep_m, 0.0, 0.0], dtype=np.float64)
    return grip_frame, r_l, r_r


def test_split_method_includes_loop_nullspace_family() -> None:
    assert "loop_nullspace_family" in SPLIT_METHODS
    assert "loop_nullspace_family" in SPLITS


def test_split_coordinate_names() -> None:
    assert SPLIT_COORDINATE_NAMES == (
        "axial_pull_squeeze_n",
        "shear_v_with_moment_n",
        "shear_w_with_moment_n",
        "differential_twist_nm",
        "differential_bend_v_nm",
        "differential_bend_w_nm",
    )


def test_grasp_map_and_split_basis_exact_kernel() -> None:
    grip_frame, r_l, r_r = _canonical_grip_setup(0.04)
    G = grasp_map(grip_frame, r_l, r_r)
    K = split_basis(grip_frame, r_l, r_r)

    assert G.shape == (6, 12)
    assert K.shape == (12, 6)
    assert np.linalg.matrix_rank(G) == 6
    assert np.linalg.matrix_rank(K) == 6
    np.testing.assert_allclose(G @ K, np.zeros((6, 6)), atol=1e-12)


def test_hand_wrenches_and_split_coordinates_round_trip() -> None:
    grip_frame, r_l, r_r = _canonical_grip_setup(0.04)
    G = grasp_map(grip_frame, r_l, r_r)
    w_net = np.array([120.0, -40.0, 80.0, 6.0, -18.0, 24.0], dtype=np.float64)
    s_in = np.array([35.0, -15.0, 10.0, 2.0, -3.0, 4.0], dtype=np.float64)

    w_l, w_r = hand_wrenches(
        w_net, s_in, grip_frame=grip_frame, r_L_world_m=r_l, r_R_world_m=r_r
    )
    assert w_l.shape == (6,)
    assert w_r.shape == (6,)

    # Reconstruct net wrench via grasp map G
    w_reconstructed = G @ np.concatenate([w_l, w_r])
    np.testing.assert_allclose(w_reconstructed, w_net, atol=1e-12)

    # Exact inverse s -> (W_L, W_R) -> s
    s_out = split_coordinates(
        w_l, w_r, grip_frame=grip_frame, r_L_world_m=r_l, r_R_world_m=r_r
    )
    np.testing.assert_allclose(s_out, s_in, atol=1e-12)


def test_net_wrench_invariance_across_arbitrary_splits() -> None:
    rng = np.random.default_rng(12561)
    # Use a general rotated grip frame and arbitrary midpoint
    theta = 0.42
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    origin = np.array([0.25, -0.15, 1.10], dtype=np.float64)
    grip_frame = GripFrame(rotation=rot, origin=origin)
    r_l = origin - 0.05 * rot[:, 0] + 0.01 * rot[:, 1]
    r_r = origin + 0.05 * rot[:, 0] - 0.01 * rot[:, 1]

    G = grasp_map(grip_frame, r_l, r_r)
    K = split_basis(grip_frame, r_l, r_r)
    np.testing.assert_allclose(G @ K, np.zeros((6, 6)), atol=1e-12)

    w_net = rng.normal(scale=50.0, size=6)
    for _ in range(10):
        s_vec = rng.normal(scale=25.0, size=6)
        w_l, w_r = hand_wrenches(
            w_net, s_vec, grip_frame=grip_frame, r_L_world_m=r_l, r_R_world_m=r_r
        )
        # Net force F_L + F_R and net grasp-map wrench G [W_L; W_R] are invariant across s
        np.testing.assert_allclose(w_l[:3] + w_r[:3], w_net[:3], atol=1e-12)
        np.testing.assert_allclose(G @ np.concatenate([w_l, w_r]), w_net, atol=1e-12)
        s_rec = split_coordinates(
            w_l, w_r, grip_frame=grip_frame, r_L_world_m=r_l, r_R_world_m=r_r
        )
        np.testing.assert_allclose(s_rec, s_vec, atol=1e-12)


def test_split_coordinates_origin_shift_invariance() -> None:
    grip_frame, r_l, r_r = _canonical_grip_setup(0.04)
    w_net = np.array([80.0, 20.0, -30.0, 4.0, -5.0, 9.0], dtype=np.float64)
    s_vec = np.array([12.0, -8.0, 6.0, 1.5, -2.5, 0.75], dtype=np.float64)

    w_l, w_r = hand_wrenches(
        w_net, s_vec, grip_frame=grip_frame, r_L_world_m=r_l, r_R_world_m=r_r
    )
    shift = np.array([1.2, -0.7, 0.4], dtype=np.float64)
    shifted_frame = GripFrame(
        rotation=grip_frame.rotation, origin=grip_frame.origin + shift
    )
    s_shifted = split_coordinates(
        w_l,
        w_r,
        grip_frame=shifted_frame,
        r_L_world_m=r_l + shift,
        r_R_world_m=r_r + shift,
    )
    np.testing.assert_allclose(s_shifted, s_vec, atol=1e-12)


def test_net_club_wrench_invariant_record() -> None:
    snapshot, _ = spatial_two_arm_club_snapshot(floating_base=False)
    rec = net_club_wrench(
        snapshot,
        club_com_accel_world_m_s2=np.array([10.0, -5.0, 2.0]),
        club_ang_vel_world_rad_s=np.array([1.0, 2.0, -3.0]),
        club_ang_accel_world_rad_s2=np.array([20.0, -10.0, 15.0]),
    )
    assert rec.allocation_dependence is AllocationDependence.INVARIANT
    assert "C-FRAME-GRIP" in rec.conventions
    assert rec.hypotheses == ()
    assert rec.values.shape == (6,)
    assert np.all(np.isfinite(rec.values))

    # Verify exact Newton-Euler equation against snapshot.bare_club
    mass = snapshot.bare_club.mass_kg
    com = snapshot.bare_club.com_m
    inertia = snapshot.bare_club.inertia_com_kg_m2
    g = np.array([0.0, 0.0, -9.81])
    a_com = np.array([10.0, -5.0, 2.0])
    omega = np.array([1.0, 2.0, -3.0])
    alpha = np.array([20.0, -10.0, 15.0])
    f_expected = mass * (a_com - g)
    m_expected = (
        inertia @ alpha
        + np.cross(omega, inertia @ omega)
        + np.cross(com - snapshot.grip_origin, f_expected)
    )
    np.testing.assert_allclose(rec.values[:3], f_expected, atol=1e-12)
    np.testing.assert_allclose(rec.values[3:], m_expected, atol=1e-12)


def test_net_club_wrench_default_kinematics_from_snapshot() -> None:
    snapshot, _ = planar_golfer_snapshot(actuation="full")
    rec = net_club_wrench(snapshot)
    assert rec.allocation_dependence is AllocationDependence.INVARIANT
    assert "C-FRAME-GRIP" in rec.conventions
    assert rec.values.shape == (6,)
    assert np.all(np.isfinite(rec.values))
