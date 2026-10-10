"""Shared OpenSim ``BushingForce`` law (issue #11739, OSV-7 phase 2).

The law is checked against OpenSim's own ``BushingForce`` records on a tiny
two-free-body model (skipped when ``opensim`` is missing), and by its
analytic properties everywhere.
"""

from __future__ import annotations

import math
from importlib import import_module

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.grip_contact.bushing_law import (
    BushingState,
    body_xyz_angles,
    body_xyz_n_matrix,
    bushing_wrench,
)
from src.shared.python.grip_contact.parameters import BushingParameters

pytestmark = pytest.mark.unit

K_T = (1.0e6, 9.0e5, 8.0e5)
K_R = (1600.0, 1500.0, 1400.0)
C_T = (554.0, 209.0, 212.0)
C_R = (0.81, 28.6, 27.3)


def _params(damped: bool = True) -> BushingParameters:
    zero = (0.0, 0.0, 0.0)
    return BushingParameters(K_T, K_R, C_T if damped else zero, C_R if damped else zero)


def _state(rotvec, pos, vel=(0, 0, 0), omega=(0, 0, 0)) -> BushingState:
    return BushingState(
        Rotation.from_rotvec(rotvec).as_matrix(),
        np.asarray(pos, float),
        np.asarray(vel, float),
        np.asarray(omega, float),
    )


def test_angles_are_body_fixed_xyz() -> None:
    angles = np.array([0.3, -0.4, 0.7])
    rot = Rotation.from_euler("XYZ", angles).as_matrix()
    np.testing.assert_allclose(body_xyz_angles(rot), angles, atol=1e-14)


def test_angles_refuse_the_gimbal_lock() -> None:
    rot = Rotation.from_euler("XYZ", [0.1, math.pi / 2, 0.2]).as_matrix()
    with pytest.raises(ValueError, match="singular"):
        body_xyz_angles(rot)
    with pytest.raises(ValueError, match="singular"):
        body_xyz_n_matrix(np.array([0.0, math.pi / 2, 0.0]))


def test_n_matrix_maps_body_angular_velocity_to_angle_rates() -> None:
    angles = np.array([0.2, -0.3, 0.5])
    rates = np.array([0.7, -1.1, 0.4])
    eps = 1e-7
    r0 = Rotation.from_euler("XYZ", angles - 0.5 * eps * rates).as_matrix()
    r1 = Rotation.from_euler("XYZ", angles + 0.5 * eps * rates).as_matrix()
    rot = Rotation.from_euler("XYZ", angles).as_matrix()
    omega_world = Rotation.from_matrix(r1 @ r0.T).as_rotvec() / eps
    w_body = rot.T @ omega_world
    np.testing.assert_allclose(body_xyz_n_matrix(angles) @ w_body, rates, rtol=1e-6)


def test_static_translation_obeys_hookes_law_per_axis() -> None:
    hand = _state([0.2, -0.1, 0.3], [0.1, 0.2, 0.3])
    delta = np.array([1e-3, -2e-3, 0.5e-3])
    club = BushingState(
        hand.rotation, hand.position_m + hand.rotation @ delta, *[np.zeros(3)] * 2
    )
    wrench = bushing_wrench(_params(), hand, club)
    np.testing.assert_allclose(wrench.force_n, -hand.rotation @ (np.array(K_T) * delta))
    np.testing.assert_allclose(wrench.moment_nm, 0.0, atol=1e-12)


def test_small_rotation_gives_k_r_theta_moment() -> None:
    hand = _state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    theta = np.array([1e-6, -2e-6, 3e-6])
    club = _state(theta, [0.0, 0.0, 0.0])
    wrench = bushing_wrench(_params(), hand, club)
    np.testing.assert_allclose(wrench.moment_nm, -np.array(K_R) * theta, rtol=1e-5)


def test_coincident_frames_moving_together_are_force_free() -> None:
    """Coincident frames with the same spatial velocity: no deflection rate."""
    rot = Rotation.from_rotvec([0.3, 0.2, -0.4]).as_matrix()
    omega, v0 = np.array([3.0, -2.0, 5.0]), np.array([1.0, 2.0, -1.0])
    hand = BushingState(rot, np.ones(3), v0, omega)
    club = BushingState(rot, np.ones(3), v0, omega)
    wrench = bushing_wrench(_params(), hand, club)
    np.testing.assert_allclose(wrench.force_n, 0.0, atol=1e-12)
    np.testing.assert_allclose(wrench.moment_nm, 0.0, atol=1e-12)


def test_offset_frame_on_a_spinning_body_has_zero_translation_rate() -> None:
    """delta_dot is the frame-1 rate: a point fixed in frame 1 has none."""
    rot = Rotation.from_rotvec([0.1, -0.5, 0.2]).as_matrix()
    omega, v0 = np.array([4.0, 1.0, -3.0]), np.array([0.5, 0.0, 2.0])
    arm = rot @ np.array([0.0, 0.0, 1e-3])
    hand = BushingState(rot, np.zeros(3), v0, omega)
    club = BushingState(rot, arm, v0 + np.cross(omega, arm), omega)
    wrench = bushing_wrench(_params(), hand, club)
    np.testing.assert_allclose(wrench.translation_rate_m_s, 0.0, atol=1e-12)
    np.testing.assert_allclose(wrench.angle_rates_rad_s, 0.0, atol=1e-12)


def test_preconditions() -> None:
    with pytest.raises(TypeError):
        bushing_wrench("k", _state([0, 0, 0], [0, 0, 0]), _state([0, 0, 0], [0, 0, 0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="orthonormal"):
        BushingState(2.0 * np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="finite 3-vector"):
        BushingState(np.eye(3), np.array([np.nan, 0, 0]), np.zeros(3), np.zeros(3))


# --------------------------------------------------------------- OpenSim


def _osim_transform(osim, rot: np.ndarray, pos: np.ndarray):  # noqa: ANN001, ANN202
    euler = Rotation.from_matrix(rot).as_euler("XYZ")
    r = osim.Rotation()
    r.setRotationToBodyFixedXYZ(osim.Vec3(*(float(a) for a in euler)))
    return osim.Transform(r, osim.Vec3(*(float(p) for p in pos)))


def _frame_state(osim, frame, state) -> BushingState:  # noqa: ANN001
    x = frame.getTransformInGround(state)
    v = frame.getVelocityInGround(state)
    rot = np.array([[x.R().get(i, j) for j in range(3)] for i in range(3)])
    return BushingState(
        rot,
        np.array([x.p().get(i) for i in range(3)]),
        np.array([v.get(1).get(i) for i in range(3)]),
        np.array([v.get(0).get(i) for i in range(3)]),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_opensim_bushing_force_records(seed: int) -> None:
    """Deflection of a few mm/deg with random rates: OpenSim records to 1e-9."""
    try:
        osim = import_module("opensim")
    except ImportError:
        pytest.skip("opensim is not installed")
    osim.Logger.setLevelString("Warn")
    rng = np.random.default_rng(seed)
    model = osim.Model()
    model.setGravity(osim.Vec3(0.0, 0.0, 0.0))
    hand = osim.Body("hand", 1.0, osim.Vec3(0.0), osim.Inertia(0.1))
    club = osim.Body(
        "club", 0.3, osim.Vec3(0.01, 0.02, 0.3), osim.Inertia(0.01, 0.02, 0.03)
    )
    for body in (hand, club):
        model.addBody(body)
        model.addJoint(osim.FreeJoint(f"j_{body.getName()}", model.getGround(), body))
    off1 = (
        Rotation.from_rotvec(rng.normal(size=3) * 0.4).as_matrix(),
        rng.normal(size=3) * 0.05,
    )
    off2 = (
        Rotation.from_rotvec(rng.normal(size=3) * 0.4).as_matrix(),
        rng.normal(size=3) * 0.05,
    )
    f1 = osim.PhysicalOffsetFrame("f1", hand, _osim_transform(osim, *off1))
    f2 = osim.PhysicalOffsetFrame("f2", club, _osim_transform(osim, *off2))
    hand.addComponent(f1)
    club.addComponent(f2)
    force = osim.BushingForce(
        "bushing",
        f1,
        f2,
        osim.Vec3(*K_T),
        osim.Vec3(*K_R),
        osim.Vec3(*C_T),
        osim.Vec3(*C_R),
    )
    model.addForce(force)
    state = model.initSystem()
    coords = model.getCoordinateSet()
    for i in range(6):
        coords.get(i).setValue(state, float(rng.normal() * 0.5), False)
    model.realizeVelocity(state)
    r1 = _frame_state(osim, f1, state)
    deflect = Rotation.from_rotvec(rng.normal(size=3) * 0.02).as_matrix()
    r2 = r1.rotation @ deflect
    p2 = r1.position_m + r1.rotation @ (rng.normal(size=3) * 2e-3)
    rot_club = r2 @ off2[0].T
    pos_club = p2 - rot_club @ off2[1]
    euler = Rotation.from_matrix(rot_club).as_euler("XYZ")
    for k, value in enumerate([*euler, *pos_club]):
        coords.get(6 + k).setValue(state, float(value), False)
    for i in range(12):
        coords.get(i).setSpeedValue(state, float(rng.normal() * 0.5))
    model.realizeDynamics(state)
    s1, s2 = _frame_state(osim, f1, state), _frame_state(osim, f2, state)
    wrench = bushing_wrench(_params(), s1, s2)
    rec = force.getRecordValues(state)
    values = np.array([rec.get(i) for i in range(rec.size())])
    x = club.getTransformInGround(state)
    origin = np.array([x.p().get(i) for i in range(3)])
    # Records: force and torque on frame2's body about that body's origin.
    np.testing.assert_allclose(wrench.force_n, values[6:9], rtol=1e-9, atol=1e-9)
    moment_at_origin = wrench.moment_nm + np.cross(
        s2.position_m - origin, wrench.force_n
    )
    np.testing.assert_allclose(moment_at_origin, values[9:12], rtol=1e-9, atol=1e-9)
