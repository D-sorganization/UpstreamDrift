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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from src.engines.physics_engines.pinocchio.python.native_model import (
    FullBodyPinocchioModel,
)
from src.shared.python.grip_contact import (
    BushingState,
    ClubDynamics,
    CoordinateSpline,
    CoordinateSwing,
    GripInterface,
    RigidBodyState,
    bushing_wrench,
    hand_frame_states,
)
from src.shared.python.grip_contact.bushing_law import cross3
from src.shared.python.grip_contact.capability_inventory import (
    load_grip_capability_inventory,
)
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.grip_force_limit import finite_limit_n
from src.shared.python.grip_contact.hand_drift import (
    HandDrift,
    hand_relative_drift,
)
from src.shared.python.grip_contact.parity import (
    BushingProbe,
    GripKineticsSeries,
    QuantityError,
    _compute_quantity_error,
)
from src.shared.python.grip_contact.topology_contact import (
    HAND_HAND_IDENTIFIABILITY_NOTE,
    PairTraceRecorder,
    TopologyHandHandWrench,
    build_topology_model,
    topology_hand_hand_wrenches,
    topology_side_wrench,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.motion_matching.full_body_v2 import separate_v2_hands
from src.shared.python.motion_matching.grip_contract import GripSpec

ENGINE = "pinocchio"
DEFAULT_RTOL = 1.0e-9
DEFAULT_ATOL = 1.0e-12
DEFAULT_MAX_STEP_S = 2.5e-4
SIDES = ("L", "R")
#: Distinct ``(hand_source, t)`` hand-frame evaluations kept (Radau stages + 1).
HAND_CACHE_SIZE = 8


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
        self._q_idx = np.array(
            [self.adapter._coordinates[n] for n in self.names],
            dtype=int,  # noqa: SLF001
        )
        self._v_idx = np.array(
            [self.adapter._velocity_indices[n] for n in self.names],
            dtype=int,  # noqa: SLF001
        )
        self._config = np.zeros(self.model.nq, dtype=float)
        self._rates = np.zeros(self.model.nv, dtype=float)
        self._joint, self._body_pose = self.adapter.body_placement(
            spec["closure"]["body_b"]
        )
        self._body_hom = np.array(self._body_pose.homogeneous, dtype=float)

    def state(self, q: np.ndarray, qdot: np.ndarray) -> RigidBodyState:
        """Pose and spatial velocity of the spec club frame for ``(q, qdot)``."""
        pin = self._pin
        self._config[self._q_idx] = q
        self._rates[self._v_idx] = qdot
        pin.forwardKinematics(self.model, self.data, self._config, self._rates)
        joint_pose = self.data.oMi[self._joint]
        local = self.data.v[self._joint]  # joint-frame spatial velocity
        rot_j = np.array(joint_pose.rotation)
        omega = rot_j @ np.array(local.angular)
        v_joint = rot_j @ np.array(local.linear)
        arm = rot_j @ self._body_hom[:3, 3]
        return RigidBodyState(
            rot_j @ self._body_hom[:3, :3],
            np.array(joint_pose.translation) + arm,
            v_joint + cross3(omega, arm),
            omega,
        )


def _resolve_grip_force_limits(grip_pose: GripPose) -> dict[str, float | None]:
    inventory = load_grip_capability_inventory()
    force_limits: dict[str, float | None] = {}
    for hand in Hand:
        limit = grip_pose.grip_force_limit_n(hand)
        inventory.require_grip_force_limit_enforceable(ENGINE, "weld_bushing", limit)
        side = grip_pose.side(hand)
        force_limits[side] = finite_limit_n(limit)
    return force_limits


def _build_free_club_pin_model(
    pin: Any, spec: dict[str, Any], club_dyn: ClubDynamics
) -> tuple[Any, Any]:
    club_model = pin.Model()
    club_model.gravity.linear[:] = np.asarray(spec["gravity_m_s2"], float)
    c_joint = club_model.addJoint(
        0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "club"
    )
    club_model.appendBodyToJoint(
        c_joint,
        pin.Inertia(
            club_dyn.mass_kg, np.asarray(club_dyn.com_m), club_dyn.inertia_com_kg_m2
        ),
        pin.SE3.Identity(),
    )
    return club_model, club_model.createData()


class ClubOnBushings:
    """Free-flyer club driven by the shared bushing law through ``aba``."""

    def __init__(
        self,
        spec: dict[str, Any],
        names: list[str],
        interface: GripInterface,
        *,
        grip_pose: GripPose = DEFAULT_GRIP_POSE,
    ) -> None:
        pin = _pin()
        self._pin = pin
        self.interface = interface
        self.grip_pose = grip_pose
        self._force_limits = _resolve_grip_force_limits(grip_pose)
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
        self._hand_cache: dict[tuple[Any, float], dict[str, Any]] = {}

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

    def hand_frames(self, t: float, hand_source: Any) -> dict[str, Any]:
        """Hand bushing frames at ``t``, memoised for recent times.

        Precondition: ``hand_source`` is a pure function of ``t`` (a spline or
        a fixed pose).  Radau re-evaluates the right-hand side at the same
        stage times across Newton iterations and its finite-difference
        Jacobian, so the weld forward kinematics is computed once per
        distinct ``(hand_source, t)``; the returned frames are bit-identical
        to a fresh evaluation.
        """
        key = (hand_source, t)
        hands = self._hand_cache.get(key)
        if hands is None:
            q, qdot = hand_source(t)
            hands = hand_frame_states(self.kinematics.state(q, qdot), self.interface)
            if len(self._hand_cache) >= HAND_CACHE_SIZE:
                del self._hand_cache[next(iter(self._hand_cache))]
            self._hand_cache[key] = hands
        return dict(hands)

    def contact_law(
        self, hands: dict[str, Any], frames: dict[str, Any]
    ) -> dict[str, Any]:
        """Per-side wrench on the club frame: the shared bushing law."""
        return {
            s: bushing_wrench(
                self.interface.bushing,
                hands[s],
                frames[s],
                force_limit_n=self._force_limits.get(s),
            )
            for s in SIDES
        }

    def evaluate(
        self, t: float, y: np.ndarray, hand_source: Any
    ) -> tuple[dict[str, Any], RigidBodyState, dict[str, Any], dict[str, Any]]:
        """Hand frames, club body state, club frames and law at ``(t, y)``."""
        hands = self.hand_frames(t, hand_source)
        club = self.club_state(y)
        frames = {s: club.frame(self._offsets[s]) for s in SIDES}
        return hands, club, frames, self.contact_law(hands, frames)

    def wrenches(
        self, t: float, y: np.ndarray, hand_source: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Hand frames, club frames and law wrenches at ``(t, y)``."""
        hands, _, frames, law = self.evaluate(t, y, hand_source)
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

    def free_flyer_aba_derivatives(
        self, club: RigidBodyState, y: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        """State derivative d/dt [q, v] using Pinocchio aba for a free-flyer club."""
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

    def derivatives(self, t: float, y: np.ndarray, hand_source: Any) -> np.ndarray:
        """``d/dt [q, v]`` with ``aba`` for the club acceleration."""
        _, club, frames, law = self.evaluate(t, y, hand_source)
        tau = self.joint_torque(club, frames, law)
        return self.free_flyer_aba_derivatives(club, y, tau)


def _initial_state(weld: RigidBodyState, pin: Any) -> np.ndarray:
    quat = pin.Quaternion(weld.rotation)
    return np.concatenate(
        [
            weld.position_m,
            [quat.x, quat.y, quat.z, quat.w],
            np.zeros(6),
        ]
    )


def hold_state(sim: Any, q: np.ndarray) -> tuple[Any, np.ndarray]:
    """Hand source at rest and the club state at the weld pose for ``q``."""
    still = np.zeros_like(q)
    weld = sim.kinematics.state(q, still)
    return (lambda _t: (q, still)), _initial_state(weld, sim._pin)  # noqa: SLF001


def initial_swing_state(
    sim: Any,
    swing: CoordinateSwing,
    times: np.ndarray,
    hold: bool,
) -> tuple[Any, np.ndarray]:
    """Return kinematics source and initial state vector for a swing."""
    if hold:
        return hold_state(sim, np.asarray(swing.q[0], float))
    spline = CoordinateSpline(swing.time_s, swing.q)
    source = spline.evaluate
    weld0 = sim.kinematics.state(*source(float(times[0])))
    y0 = _initial_state(weld0, sim._pin)  # noqa: SLF001
    y0[7:10] = weld0.rotation.T @ weld0.velocity_m_s
    y0[10:13] = weld0.rotation.T @ weld0.omega_rad_s
    return source, y0


def integrate_free_flyer_radau(
    sim: Any,
    source: Any,
    times: np.ndarray,
    y0: np.ndarray,
    tol: Any,
) -> Any:
    """Solve free-flyer club IVP using SciPy Radau integrator."""
    return solve_ivp(
        lambda t, y: sim.derivatives(t, y, source),
        (float(times[0]), float(times[-1])),
        y0,
        method="Radau",
        t_eval=times,
        rtol=tol.rtol,
        atol=tol.atol,
        max_step=tol.max_step_s,
    )


def simulate_grip_bushing(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    interface: GripInterface | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_step_s: float = DEFAULT_MAX_STEP_S,
    t_end_s: float | None = None,
    *,
    grip_pose: GripPose = DEFAULT_GRIP_POSE,
) -> GripKineticsSeries:
    """Integrate the free club on two bushings over the prescribed swing.

    Raises:
        RuntimeError: if the integrator fails.
    """
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    sim = ClubOnBushings(spec, list(swing.names), interface, grip_pose=grip_pose)
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
    *,
    grip_pose: GripPose = DEFAULT_GRIP_POSE,
) -> BushingProbe:
    """Bushing forces for the club displaced by ``translation_hand_m``.

    The engine total is ``m (a_com - g)`` from Pinocchio's ``aba`` with the
    free-flyer joint torque, so it passes through Pinocchio's dynamics.
    """
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    sim = ClubOnBushings(spec, list(swing.names), interface, grip_pose=grip_pose)
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


class SeparatedHandsKinematics:
    """Pinocchio forward kinematics of separated LHandBody and RHandStandoff chains (#13309)."""

    def __init__(self, spec: dict[str, Any], names: list[str]) -> None:
        self._pin = _pin()
        self.adapter = FullBodyPinocchioModel(spec)
        self.model = self.adapter.model
        self.data = self.model.createData()
        self.names = list(names)
        self._q_idx = np.array(
            [self.adapter._coordinates[n] for n in self.names],
            dtype=int,  # noqa: SLF001
        )
        self._v_idx = np.array(
            [self.adapter._velocity_indices[n] for n in self.names],
            dtype=int,  # noqa: SLF001
        )
        self._config = np.zeros(self.model.nq, dtype=float)
        self._rates = np.zeros(self.model.nv, dtype=float)
        gb = spec["grip_bushing"]
        self._joints: dict[str, int] = {}
        self._hand_in_joint: dict[str, np.ndarray] = {}
        for side in SIDES:
            entry = gb[side]
            j_idx, b_pose = self.adapter.body_placement(entry["hand_body"])
            self._joints[side] = j_idx
            self._hand_in_joint[side] = np.array(
                b_pose.homogeneous, dtype=float
            ) @ np.asarray(entry["hand_frame"], dtype=float)
        j_l, b_l = self.adapter.body_placement(gb["L"]["hand_body"])
        self._left_joint = j_l
        self._left_body_hom = np.array(b_l.homogeneous, dtype=float)

    def _run_fk(self, q: np.ndarray, qdot: np.ndarray) -> None:
        self._config[self._q_idx] = q
        self._rates[self._v_idx] = qdot
        self._pin.forwardKinematics(self.model, self.data, self._config, self._rates)

    def left_body_state(self, q: np.ndarray, qdot: np.ndarray) -> RigidBodyState:
        """Pose and spatial velocity of ``LHandBody`` (coincident with welded club frame at q=0)."""
        self._run_fk(q, qdot)
        jp = self.data.oMi[self._left_joint]
        lv = self.data.v[self._left_joint]
        rot = np.array(jp.rotation)
        body = RigidBodyState(
            rot,
            np.array(jp.translation),
            rot @ np.array(lv.linear),
            rot @ np.array(lv.angular),
        )
        f = body.frame(self._left_body_hom)
        return RigidBodyState(f.rotation, f.position_m, f.velocity_m_s, f.omega_rad_s)

    def left_hand_state(self, q: np.ndarray, qdot: np.ndarray) -> BushingState:
        """Direct world ``BushingState`` of the lead hand frame on ``LHandBody``."""
        self._run_fk(q, qdot)
        j_idx = self._joints["L"]
        jp = self.data.oMi[j_idx]
        lv = self.data.v[j_idx]
        rot = np.array(jp.rotation)
        off = self._hand_in_joint["L"]
        omega = rot @ np.array(lv.angular)
        arm = rot @ off[:3, 3]
        return BushingState(
            rot @ off[:3, :3],
            np.array(jp.translation) + arm,
            rot @ np.array(lv.linear) + cross3(omega, arm),
            omega,
        )

    def hand_states(self, q: np.ndarray, qdot: np.ndarray) -> dict[str, BushingState]:
        """Independent left-wrist (``LHandBody``) and right-wrist (``RHandStandoff``) states."""
        self._run_fk(q, qdot)
        out: dict[str, BushingState] = {}
        for side in SIDES:
            j_idx = self._joints[side]
            jp = self.data.oMi[j_idx]
            lv = self.data.v[j_idx]
            rot = np.array(jp.rotation)
            off = self._hand_in_joint[side]
            omega = rot @ np.array(lv.angular)
            arm = rot @ off[:3, 3]
            out[side] = BushingState(
                rot @ off[:3, :3],
                np.array(jp.translation) + arm,
                rot @ np.array(lv.linear) + cross3(omega, arm),
                omega,
            )
        return out


@dataclass(frozen=True)
class SeparatedGripRun:
    """Result of ``simulate_separated_v2_grip`` with hand-on-club and hand-hand traces (#13309)."""

    series: GripKineticsSeries
    club_com_local_m: np.ndarray
    pair_normal_force_n: dict[str, np.ndarray]
    pair_tangential_force_n: dict[str, np.ndarray]
    pair_contact_point_drift_m: dict[str, np.ndarray]
    pair_penetration_m: dict[str, np.ndarray]
    hand_hand_wrenches: tuple[TopologyHandHandWrench, ...]
    relative_hand_drift: HandDrift
    max_action_reaction_error_n: float

    def net_moment_com_nm(self, target: GripKineticsSeries | None = None) -> np.ndarray:
        """Total closure moment about the bare club COM in world coordinates (N*m)."""
        s = self.series if target is None else target
        r_com = np.einsum("nij,j->ni", s.club_rotation, self.club_com_local_m)
        return (
            s.torque_on_club_nm["L"]
            + s.torque_on_club_nm["R"]
            + np.cross(s.grip_point_m["L"] - r_com, s.force_on_club_n["L"])
            + np.cross(s.grip_point_m["R"] - r_com, s.force_on_club_n["R"])
        )

    def net_moment_com_error(self, reference: GripKineticsSeries) -> QuantityError:
        """Parity error of net closure moment about the bare club COM against ``reference``."""
        return _compute_quantity_error(
            self.net_moment_com_nm(self.series),
            self.net_moment_com_nm(reference),
            "net_moment_com",
        )


def _build_se3_hand_source(
    kin: SeparatedHandsKinematics,
    spline: CoordinateSpline,
    times: np.ndarray,
) -> Any:
    """Build SE(3)-consistent separated-hand evaluator preserving exact sample states."""
    packed = np.empty((times.size, 24), dtype=float)
    eye3 = np.eye(3)
    for k, t_val in enumerate(times):
        qk, qdk = spline.evaluate(float(t_val))
        hs = kin.hand_states(qk, qdk)
        rl, pl = hs["L"].rotation, hs["L"].position_m
        rr, pr = hs["R"].rotation, hs["R"].position_m
        packed[k, :9] = rl.ravel()
        packed[k, 9:12] = pl
        packed[k, 12:21] = (rl.T @ rr).ravel()
        packed[k, 21:24] = rl.T @ (pr - pl)
    sp24 = CoordinateSpline(times, packed)

    def evaluate_hands(t: float) -> dict[str, BushingState]:
        val, dot = sp24.evaluate(t)
        rl = val[:9].reshape(3, 3)
        rl = 0.5 * rl @ (3.0 * eye3 - rl.T @ rl)
        rl_dot = dot[:9].reshape(3, 3)
        skew_l = rl_dot @ rl.T
        wl = np.array(
            [
                0.5 * (skew_l[2, 1] - skew_l[1, 2]),
                0.5 * (skew_l[0, 2] - skew_l[2, 0]),
                0.5 * (skew_l[1, 0] - skew_l[0, 1]),
            ]
        )
        pl = val[9:12]
        vl = dot[9:12]
        r_rel = val[12:21].reshape(3, 3)
        r_rel = 0.5 * r_rel @ (3.0 * eye3 - r_rel.T @ r_rel)
        r_rel_dot = dot[12:21].reshape(3, 3)
        p_rel = val[21:24]
        p_rel_dot = dot[21:24]
        skew_rel = r_rel_dot @ r_rel.T
        w_rel_local = np.array(
            [
                0.5 * (skew_rel[2, 1] - skew_rel[1, 2]),
                0.5 * (skew_rel[0, 2] - skew_rel[2, 0]),
                0.5 * (skew_rel[1, 0] - skew_rel[0, 1]),
            ]
        )
        rr = rl @ r_rel
        wr = wl + rl @ w_rel_local
        arm = rl @ p_rel
        pr = pl + arm
        vr = vl + cross3(wl, arm) + rl @ p_rel_dot
        return {
            "L": BushingState(rl, pl, vl, wl),
            "R": BushingState(rr, pr, vr, wr),
        }

    return evaluate_hands


def _evaluate_grip_bushing_interaction(
    y: np.ndarray,
    hands: dict[str, Any],
    offsets: dict[str, Any],
    bushing: Any,
    force_limits: Mapping[str, float | None],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    club = ClubOnBushings.club_state(y)
    frames = {s: club.frame(offsets[s]) for s in SIDES}
    law = {
        s: bushing_wrench(
            bushing,
            hands[s],
            frames[s],
            force_limit_n=force_limits.get(s),
        )
        for s in SIDES
    }
    return club, frames, law


def simulate_separated_v2_grip_styles(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    interface: GripInterface | None = None,
    styles: Sequence[GripStyle] = (
        GripStyle.OVERLAP,
        GripStyle.INTERLOCK,
        GripStyle.TEN_FINGER,
    ),
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_step_s: float = DEFAULT_MAX_STEP_S,
    *,
    grip_pose: GripPose = DEFAULT_GRIP_POSE,
) -> dict[GripStyle, SeparatedGripRun]:
    """Integrate the separated-v2 club once and collect per-style contact traces without cross-style merging."""
    style_tuple = tuple(GripStyle(s) for s in styles)
    if not style_tuple:
        raise ValueError("styles must contain at least one GripStyle")
    force_limits = _resolve_grip_force_limits(grip_pose)
    spec = json.loads(spec_bytes)
    if not spec.get("hand_separation", {}).get("enabled"):
        spec = separate_v2_hands(spec)
    interface = interface or GripInterface.from_spec(spec)
    kin = SeparatedHandsKinematics(spec, list(swing.names))
    club_dyn = ClubDynamics.from_spec(spec)
    pin = _pin()
    club_model, club_data = _build_free_club_pin_model(pin, spec, club_dyn)
    offsets = {s: interface.frame(s).matrix() for s in SIDES}
    spline = CoordinateSpline(swing.time_s, swing.q)
    times = swing.time_s
    hand_eval = _build_se3_hand_source(kin, spline, times)
    y0 = _initial_state(kin.left_body_state(*spline.evaluate(float(times[0]))), pin)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        hands = hand_eval(t)
        club, frames, law = _evaluate_grip_bushing_interaction(
            y, hands, offsets, interface.bushing, force_limits
        )
        force = law["L"].force_n + law["R"].force_n
        moment = (
            law["L"].moment_nm
            + cross3(frames["L"].position_m - club.position_m, law["L"].force_n)
            + law["R"].moment_nm
            + cross3(frames["R"].position_m - club.position_m, law["R"].force_n)
        )
        rot_t = club.rotation.T
        tau = np.concatenate([rot_t @ force, rot_t @ moment])
        quat = y[3:7] / np.linalg.norm(y[3:7])
        q_c = np.concatenate([y[:3], quat])
        v_c = y[7:13]
        accel = np.array(pin.aba(club_model, club_data, q_c, v_c, tau))
        omega = v_c[3:]
        qv, qw = quat[:3], quat[3]
        quat_dot = 0.5 * np.concatenate(
            [qw * omega + cross3(qv, omega), [-float(qv @ omega)]]
        )
        return np.concatenate([club.rotation @ v_c[:3], quat_dot, accel])

    sol = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        y0,
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
        max_step=max_step_s,
    )
    if not sol.success:
        raise RuntimeError(f"separated v2 club integration failed: {sol.message}")
    return {
        st: _collect_separated_run(
            sol.y,
            times,
            hand_eval,
            interface,
            club_dyn,
            st,
            int(sol.nfev),
            force_limits=force_limits,
        )
        for st in style_tuple
    }


def simulate_separated_v2_grip(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    interface: GripInterface | None = None,
    style: GripStyle = GripStyle.OVERLAP,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_step_s: float = DEFAULT_MAX_STEP_S,
    *,
    grip_pose: GripPose = DEFAULT_GRIP_POSE,
) -> SeparatedGripRun:
    """Simulate separated-hand ``full-body-v2`` grip retention and physical hand-hand contact (#13309).

    Evaluates ``LHandBody`` from the left-wrist chain and ``RHandStandoff`` from
    the right-wrist chain, retains each hand on the free bare club via the shared
    retention law, and evaluates physical hand-hand pairs with ``mode='separated_v2'``.
    """
    target_style = GripStyle(style)
    return simulate_separated_v2_grip_styles(
        spec_bytes,
        swing,
        interface=interface,
        styles=(target_style,),
        rtol=rtol,
        atol=atol,
        max_step_s=max_step_s,
        grip_pose=grip_pose,
    )[target_style]


def _collect_separated_run(
    sol_y: np.ndarray,
    times: np.ndarray,
    hand_eval: Any,
    interface: GripInterface,
    club_dyn: ClubDynamics,
    style: GripStyle,
    nfev: int,
    force_limits: Mapping[str, float | None] | None = None,
) -> SeparatedGripRun:
    force_limits = force_limits or {}
    offsets = {s: interface.frame(s).matrix() for s in SIDES}
    topo = contact_topology(GripSpec(style=style))
    topo_model = build_topology_model(interface, topology=topo)
    recorder = PairTraceRecorder(topo_model, mode="separated_v2")
    wrench: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    hand_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    club_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    club_rot = []
    hh_list: list[TopologyHandHandWrench] = []
    max_ar_err = 0.0
    for k, t in enumerate(times):
        y = sol_y[:, k]
        hands = hand_eval(float(t))
        club, frames, law = _evaluate_grip_bushing_interaction(
            y, hands, offsets, interface.bushing, force_limits
        )
        topo_law = {
            s: topology_side_wrench(topo_model, s, hands[s], frames[s]) for s in SIDES
        }
        hh = topology_hand_hand_wrenches(
            topo_model, hands["L"], hands["R"], mode="separated_v2"
        )
        recorder.record(topo_law, hands, frames, hand_hand=hh)
        hh_list.append(hh)
        ar_err = float(np.linalg.norm(hh.force_on_lead_n + hh.force_on_trail_n))
        max_ar_err = max(max_ar_err, ar_err)
        for s in SIDES:
            wrench[s][0].append(law[s].force_n)
            wrench[s][1].append(law[s].moment_nm)
            hand_pose[s][0].append(hands[s].rotation)
            hand_pose[s][1].append(hands[s].position_m)
            club_pose[s][0].append(frames[s].rotation)
            club_pose[s][1].append(frames[s].position_m)
        club_rot.append(club.rotation)
    arrays = {
        name: {s: (np.array(d[s][0]), np.array(d[s][1])) for s in SIDES}
        for name, d in (("w", wrench), ("h", hand_pose), ("c", club_pose))
    }
    rec = recorder.result()
    pen_m = recorder.pair_penetration_m()
    drift = hand_relative_drift(arrays["h"])
    series = GripKineticsSeries.from_frames(
        ENGINE,
        times,
        arrays["w"],
        arrays["h"],
        arrays["c"],
        np.array(club_rot),
        metadata={
            "mode": "separated_v2",
            "grip_style": style.value,
            "rhs_evaluations": nfev,
            "identifiability_note": HAND_HAND_IDENTIFIABILITY_NOTE,
            "pair_penetration_m": pen_m,
            "relative_hand_pose": drift.summary(),
            **rec,
        },
    )
    return SeparatedGripRun(
        series=series,
        club_com_local_m=np.asarray(club_dyn.com_m, dtype=float),
        pair_normal_force_n=rec["pair_normal_force_n"],
        pair_tangential_force_n=rec["pair_tangential_force_n"],
        pair_contact_point_drift_m=rec["pair_contact_point_drift_m"],
        pair_penetration_m=pen_m,
        hand_hand_wrenches=tuple(hh_list),
        relative_hand_drift=drift,
        max_action_reaction_error_n=max_ar_err,
    )
