"""Pinocchio bushing grip on the same input as OpenSim (issue #11739, OSV-7).

* Hand frames: Pinocchio's own forward kinematics of the full-body weld model
  (``FullBodyPinocchioModel``; ``forwardKinematics`` with the prescribed
  coordinates and speeds, ``getVelocity`` of the club joint), carrying both
  hand bushing frames as in the OpenSim model.
* Club: a one-body Pinocchio model on a ``JointModelFreeFlyer`` (spec mass,
  centre of mass, inertia and gravity).  The two bushing wrenches come from
  the shared OpenSim ``BushingForce`` law
  (:mod:`src.shared.python.grip_contact.bushing_law`); their sum, moved to the
  club body origin and expressed in the body frame, is the free-flyer joint
  torque, and ``aba`` gives the club acceleration.
* Integrator: SciPy ``solve_ivp`` DOP853 (explicit, error controlled) on the
  state ``[q (7), v (6)]``; the quaternion is normalised at every evaluation.
"""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from src.engines.physics_engines.pinocchio.python.native_model import (
    FullBodyPinocchioModel,
)
from src.shared.python.grip_contact import (
    BushingWrench,
    ClubDynamics,
    CoordinateSpline,
    CoordinateSwing,
    GripInterface,
    RigidBodyState,
    bushing_wrench,
    hand_frame_states,
)
from src.shared.python.grip_contact.bushing_law import cross3
from src.shared.python.grip_contact.parity import BushingProbe, GripKineticsSeries

ENGINE = "pinocchio"
DEFAULT_RTOL = 1.0e-9
DEFAULT_ATOL = 1.0e-12
DEFAULT_MAX_STEP_S = 2.5e-4
SIDES = ("L", "R")


def _pin() -> Any:
    return import_module("pinocchio")


class WeldClubKinematics:
    """Pinocchio forward kinematics of the weld model's club (spec club frame)."""

    def __init__(self, spec: dict[str, Any], names: list[str]) -> None:
        self._pin = _pin()
        self.adapter = FullBodyPinocchioModel(spec)
        self.model = self.adapter.model
        self.data = self.model.createData()
        self.names = list(names)
        self._joint, self._body_pose = self.adapter.body_placement(
            spec["closure"]["body_b"]
        )

    def state(self, q: np.ndarray, qdot: np.ndarray) -> RigidBodyState:
        """Pose and spatial velocity of the spec club frame for ``(q, qdot)``."""
        pin = self._pin
        config = self.adapter.configuration(dict(zip(self.names, q, strict=True)))
        rates = self.adapter.velocity(dict(zip(self.names, qdot, strict=True)))
        pin.forwardKinematics(self.model, self.data, config, rates)
        joint_pose = self.data.oMi[self._joint]
        local = self.data.v[self._joint]  # joint-frame spatial velocity
        rot_j = np.array(joint_pose.rotation)
        omega = rot_j @ np.array(local.angular)
        v_joint = rot_j @ np.array(local.linear)
        body = RigidBodyState(rot_j, np.array(joint_pose.translation), v_joint, omega)
        frame = body.frame(np.array(self._body_pose.homogeneous))
        return RigidBodyState(
            frame.rotation, frame.position_m, frame.velocity_m_s, frame.omega_rad_s
        )


class ClubOnBushings:
    """Free-flyer club driven by the shared bushing law through ``aba``."""

    def __init__(
        self,
        spec: dict[str, Any],
        names: list[str],
        interface: GripInterface,
    ) -> None:
        pin = _pin()
        self._pin = pin
        self.interface = interface
        self.kinematics = WeldClubKinematics(spec, names)
        club = ClubDynamics.from_spec(spec)
        model = pin.Model()
        model.gravity.linear[:] = np.asarray(spec["gravity_m_s2"], float)
        joint = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "club")
        model.appendBodyToJoint(
            joint,
            pin.Inertia(club.mass_kg, np.asarray(club.com_m), club.inertia_com_kg_m2),
            pin.SE3.Identity(),
        )
        self.model, self.data = model, model.createData()
        self._offsets = {s: interface.frame(s).matrix() for s in SIDES}

    @staticmethod
    def club_state(y: np.ndarray) -> RigidBodyState:
        """World state of the club body frame from ``[q (7), v (6)]``."""
        quat = y[3:7] / np.linalg.norm(y[3:7])
        x, yq, z, w = quat
        rot = np.array(
            [
                [1 - 2 * (yq * yq + z * z), 2 * (x * yq - z * w), 2 * (x * z + yq * w)],
                [2 * (x * yq + z * w), 1 - 2 * (x * x + z * z), 2 * (yq * z - x * w)],
                [2 * (x * z - yq * w), 2 * (yq * z + x * w), 1 - 2 * (x * x + yq * yq)],
            ]
        )
        return RigidBodyState(rot, y[:3].copy(), rot @ y[7:10], rot @ y[10:13])

    def wrenches(
        self, t: float, y: np.ndarray, hand_source: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, BushingWrench]]:
        """Hand frames, club frames and bushing wrenches at ``(t, y)``."""
        q, qdot = hand_source(t)
        hands = hand_frame_states(self.kinematics.state(q, qdot), self.interface)
        club = self.club_state(y)
        frames = {s: club.frame(self._offsets[s]) for s in SIDES}
        law = {
            s: bushing_wrench(self.interface.bushing, hands[s], frames[s])
            for s in SIDES
        }
        return hands, frames, law

    def joint_torque(
        self, club: RigidBodyState, frames: dict[str, Any], law: dict[str, Any]
    ) -> np.ndarray:
        """Free-flyer generalised force: total wrench at the body origin, local."""
        force = np.zeros(3)
        moment = np.zeros(3)
        for s in SIDES:
            force += law[s].force_n
            arm = frames[s].position_m - club.position_m
            moment += law[s].moment_nm + cross3(arm, law[s].force_n)
        rot_t = club.rotation.T
        return np.concatenate([rot_t @ force, rot_t @ moment])

    def derivatives(self, t: float, y: np.ndarray, hand_source: Any) -> np.ndarray:
        """``d/dt [q, v]`` with ``aba`` for the club acceleration."""
        _, frames, law = self.wrenches(t, y, hand_source)
        club = self.club_state(y)
        tau = self.joint_torque(club, frames, law)
        quat = y[3:7] / np.linalg.norm(y[3:7])
        q = np.concatenate([y[:3], quat])
        v = y[7:13]
        accel = np.array(self._pin.aba(self.model, self.data, q, v, tau))
        omega = v[3:]
        qv, qw = quat[:3], quat[3]
        quat_dot = 0.5 * np.concatenate(
            [qw * omega + cross3(qv, omega), [-float(qv @ omega)]]
        )
        return np.concatenate([club.rotation @ v[:3], quat_dot, accel])


def _initial_state(weld: RigidBodyState, pin: Any) -> np.ndarray:
    quat = pin.Quaternion(weld.rotation)
    return np.concatenate(
        [
            weld.position_m,
            [quat.x, quat.y, quat.z, quat.w],
            np.zeros(6),
        ]
    )


def simulate_grip_bushing(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    interface: GripInterface | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_step_s: float = DEFAULT_MAX_STEP_S,
    t_end_s: float | None = None,
) -> GripKineticsSeries:
    """Integrate the free club on two bushings over the prescribed swing.

    Raises:
        RuntimeError: if the integrator fails.
    """
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    sim = ClubOnBushings(spec, list(swing.names), interface)
    spline = CoordinateSpline(swing.time_s, swing.q)
    times = swing.time_s
    if t_end_s is not None:
        times = times[times <= t_end_s + 1e-12]
    pin = sim._pin  # noqa: SLF001
    y0 = _initial_state(sim.kinematics.state(*spline.evaluate(float(times[0]))), pin)
    sol = solve_ivp(
        lambda t, y: sim.derivatives(t, y, spline.evaluate),
        (float(times[0]), float(times[-1])),
        y0,
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
        max_step=max_step_s,
    )
    if not sol.success:
        raise RuntimeError(f"pinocchio club integration failed: {sol.message}")
    wrench: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    hand_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    club_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    club_rot = []
    for k, t in enumerate(times):
        y = sol.y[:, k]
        hands, frames, law = sim.wrenches(float(t), y, spline.evaluate)
        for s in SIDES:
            wrench[s][0].append(law[s].force_n)
            wrench[s][1].append(law[s].moment_nm)
            hand_pose[s][0].append(hands[s].rotation)
            hand_pose[s][1].append(hands[s].position_m)
            club_pose[s][0].append(frames[s].rotation)
            club_pose[s][1].append(frames[s].position_m)
        club_rot.append(sim.club_state(y).rotation)
    arrays = {
        name: {s: (np.array(d[s][0]), np.array(d[s][1])) for s in SIDES}
        for name, d in (("w", wrench), ("h", hand_pose), ("c", club_pose))
    }
    return GripKineticsSeries.from_frames(
        ENGINE,
        times,
        arrays["w"],
        arrays["h"],
        arrays["c"],
        np.array(club_rot),
        metadata={
            "integrator": "SciPy solve_ivp DOP853, error controlled",
            "rtol": rtol,
            "atol": atol,
            "max_step_s": max_step_s,
            "rhs_evaluations": int(sol.nfev),
            "force_law": "shared OpenSim BushingForce law, Pinocchio aba",
            "kinematics": "Pinocchio full-body weld model forward kinematics",
        },
    )


def probe_bushing_forces(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    translation_hand_m: np.ndarray,
    interface: GripInterface | None = None,
) -> BushingProbe:
    """Bushing forces for the club displaced by ``translation_hand_m``.

    The engine total is ``m (a_com - g)`` from Pinocchio's ``aba`` with the
    free-flyer joint torque, so it passes through Pinocchio's dynamics.
    """
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    sim = ClubOnBushings(spec, list(swing.names), interface)
    pin = sim._pin  # noqa: SLF001
    q0 = np.asarray(swing.q[0], float)
    still = np.zeros_like(q0)
    weld0 = sim.kinematics.state(q0, still)
    hand_rot = weld0.rotation @ np.asarray(interface.left.rotation, float)
    y = _initial_state(weld0, pin)
    y[:3] += hand_rot @ np.asarray(translation_hand_m, float)
    _, _, law = sim.wrenches(0.0, y, lambda _t: (q0, still))
    ydot = sim.derivatives(0.0, y, lambda _t: (q0, still))
    club = ClubDynamics.from_spec(spec)
    rot = sim.club_state(y).rotation
    # v = 0: the body-origin acceleration is R a_lin; add alpha x r for the COM.
    a_origin = rot @ ydot[7:10]
    alpha = rot @ ydot[10:13]
    a_com = a_origin + cross3(alpha, rot @ np.asarray(club.com_m))
    gravity = np.asarray(spec["gravity_m_s2"], float)
    return BushingProbe(
        force_n={s: law[s].force_n for s in SIDES},
        hand_rotation=hand_rot,
        engine_total_force_n=club.mass_kg * (a_com - gravity),
    )
