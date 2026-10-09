"""MuJoCo bushing grip on the same input as OpenSim (issue #11739, OSV-7 phase 2).

Two MuJoCo models are used:

* the full-body weld MJCF (``export_full_body_mjcf``) as MuJoCo's own forward
  kinematics: for the prescribed coordinates and speeds it gives the club
  pose and spatial velocity of the weld model, which carries the two hand
  bushing frames (both on the left hand body, as in the OpenSim model);
* a one-body model of the club as a free joint (spec mass, centre of mass and
  inertia; spec gravity) integrated by MuJoCo's own RK4 at a small fixed step.

The bushing wrench is the OpenSim ``BushingForce`` law
(:mod:`src.shared.python.grip_contact.bushing_law`) applied through the
``mjcb_passive`` callback with ``mj_applyFT`` into ``qfrc_passive``.  MuJoCo
calls that callback inside every RK4 stage with ``d.time`` set to the stage
time, so the hand frames are re-evaluated at every stage.

Why not a soft weld equality.  A MuJoCo weld constraint (``solref``,
``solimp``, ``torquescale``) has one time constant and damping ratio (or one
direct stiffness and damping) shared by all six rows, acts on the quaternion
orientation error, and its force is scaled by the constraint-space inverse
inertia and the impedance ``d(r)``.  It cannot reproduce three translational
and three rotational stiffnesses and dampings per axis in the X-Y-Z angle
coordinates of the OpenSim bushing, so it is not a same-law comparison.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
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

ENGINE = "mujoco"
DEFAULT_TIMESTEP_S = 1.0e-4
SIDES = ("L", "R")
HandSource = Callable[[float], tuple[np.ndarray, np.ndarray]]


def _mujoco() -> Any:
    return import_module("mujoco")


def _inv(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def _quat_matrix(mj: Any, quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mj.mju_quat2Mat(mat, np.asarray(quat, float))
    return mat.reshape(3, 3)


def body_state(mj: Any, model: Any, data: Any, body: int) -> RigidBodyState:
    """World pose and spatial velocity of a body frame origin.

    ``mj_objectVelocity`` reports a body's linear velocity at its centre of
    mass (``xipos``); it is moved to the body-frame origin (``xpos``).
    Requires ``mj_comVel`` to have run.
    """
    vel = np.zeros(6)
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body, vel, 0)
    omega = vel[:3].copy()
    origin = data.xpos[body].copy()
    v_origin = vel[3:] + cross3(omega, origin - data.xipos[body])
    return RigidBodyState(data.xmat[body].reshape(3, 3).copy(), origin, v_origin, omega)


class WeldClubKinematics:
    """MuJoCo forward kinematics of the weld model's club (spec club frame)."""

    def __init__(self, spec_bytes: bytes, names: list[str]) -> None:
        mj = _mujoco()
        spec = json.loads(spec_bytes)
        xml, _ = export_full_body_mjcf(spec_bytes)
        self._mj = mj
        self.model = mj.MjModel.from_xml_string(xml)
        self.data = mj.MjData(self.model)
        joints = [self.model.joint(n) for n in names]
        self._qpos = np.array([int(j.qposadr[0]) for j in joints])
        self._dof = np.array([int(j.dofadr[0]) for j in joints])
        if self.model.nq != len(names):
            raise ValueError("every MJCF coordinate must be prescribed")
        self._body = int(self.model.body(spec["closure"]["body_b"]).id)
        site = self.model.site("native_closure_b")
        if int(site.bodyid[0]) != self._body:
            raise ValueError("native_closure_b is not on the club body")
        local = np.eye(4)
        local[:3, :3] = _quat_matrix(mj, site.quat)
        local[:3, 3] = site.pos
        # site pose in the MJCF body = offset @ placement_b
        placement_b = np.asarray(spec["closure"]["placement_b"], float)
        self._offset = local @ _inv(placement_b)

    def state(self, q: np.ndarray, qdot: np.ndarray) -> RigidBodyState:
        """Pose and spatial velocity of the spec club frame for ``(q, qdot)``."""
        mj, m, d = self._mj, self.model, self.data
        d.qpos[self._qpos] = q
        d.qvel[self._dof] = qdot
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        mj.mj_comVel(m, d)
        frame = body_state(mj, m, d, self._body).frame(self._offset)
        return RigidBodyState(
            frame.rotation, frame.position_m, frame.velocity_m_s, frame.omega_rad_s
        )


def club_mjcf(club: ClubDynamics, gravity: np.ndarray, timestep_s: float) -> str:
    """One free club body (spec frame) with spec inertia, RK4 at ``timestep_s``."""
    i = club.inertia_com_kg_m2
    full = (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])

    def nums(values: Any) -> str:
        return " ".join(format(float(v), ".17g") for v in values)

    return (
        '<mujoco model="grip_bushing_club">'
        f'<option timestep="{timestep_s:.17g}" integrator="RK4" '
        f'gravity="{nums(gravity)}"><flag contact="disable"/></option>'
        '<worldbody><body name="club"><freejoint name="club_free"/>'
        f'<inertial pos="{nums(club.com_m)}" mass="{club.mass_kg:.17g}" '
        f'fullinertia="{nums(full)}"/></body></worldbody></mujoco>'
    )


@dataclass
class _Sample:
    hands: dict[str, Any]
    club_frames: dict[str, Any]
    wrench: dict[str, BushingWrench]


class ClubOnBushings:
    """The free club driven by the hand frames through ``mjcb_passive``."""

    def __init__(
        self,
        spec_bytes: bytes,
        names: list[str],
        interface: GripInterface,
        timestep_s: float,
    ) -> None:
        mj = _mujoco()
        spec = json.loads(spec_bytes)
        self._mj = mj
        self.interface = interface
        self.kinematics = WeldClubKinematics(spec_bytes, names)
        gravity = np.asarray(spec["gravity_m_s2"], float)
        xml = club_mjcf(ClubDynamics.from_spec(spec), gravity, timestep_s)
        self.model = mj.MjModel.from_xml_string(xml)
        self.data = mj.MjData(self.model)
        self._body = int(self.model.body("club").id)
        self._offsets = {s: interface.frame(s).matrix() for s in SIDES}
        self.hand_source: HandSource | None = None
        self.last: _Sample | None = None

    def place(self, rotation: np.ndarray, position: np.ndarray) -> None:
        """Set the club pose (world) with zero velocity."""
        quat = np.zeros(4)
        self._mj.mju_mat2Quat(quat, np.asarray(rotation, float).ravel())
        self.data.qpos[:3] = position
        self.data.qpos[3:7] = quat
        self.data.qvel[:] = 0.0

    def _club_state(self) -> RigidBodyState:
        return body_state(self._mj, self.model, self.data, self._body)

    def set_passive_callback(self, callback: Any) -> None:
        """Install (or clear with ``None``) the MuJoCo passive callback."""
        self._mj.set_mjcb_passive(callback)

    def passive(self, m: Any, d: Any) -> None:
        """``mjcb_passive``: add both bushing wrenches to ``qfrc_passive``."""
        if m is not self.model or self.hand_source is None:
            return
        q, qdot = self.hand_source(float(d.time))
        hands = hand_frame_states(self.kinematics.state(q, qdot), self.interface)
        club = self._club_state()
        frames = {s: club.frame(self._offsets[s]) for s in SIDES}
        wrenches = {}
        for s in SIDES:
            w = bushing_wrench(self.interface.bushing, hands[s], frames[s])
            self._mj.mj_applyFT(
                m,
                d,
                w.force_n,
                w.moment_nm,
                frames[s].position_m,
                self._body,
                d.qfrc_passive,
            )
            wrenches[s] = w
        self.last = _Sample(hands, frames, wrenches)

    def evaluate(self) -> _Sample:
        """Run ``mj_forward`` at the current state and return the bushing sample."""
        self._mj.mj_forward(self.model, self.data)
        if self.last is None:
            raise RuntimeError("the passive callback did not run")
        return self.last


class _PassiveCallback:
    """Install ``mjcb_passive`` for the lifetime of a ``with`` block."""

    def __init__(self, target: ClubOnBushings) -> None:
        self._target = target

    def __enter__(self) -> ClubOnBushings:
        self._target.set_passive_callback(self._target.passive)
        return self._target

    def __exit__(self, *exc: object) -> None:
        self._target.set_passive_callback(None)


def _record(samples: list[_Sample], club_rot: list[np.ndarray]) -> dict[str, Any]:
    def side_pair(attr: str, sub: tuple[str, str]) -> dict[str, tuple]:
        return {
            s: tuple(
                np.array([getattr(getattr(x, attr)[s], k) for x in samples])
                for k in sub
            )
            for s in SIDES
        }

    return {
        "wrench": side_pair("wrench", ("force_n", "moment_nm")),
        "hand_pose": side_pair("hands", ("rotation", "position_m")),
        "club_frame_pose": side_pair("club_frames", ("rotation", "position_m")),
        "club_rotation": np.array(club_rot),
    }


def simulate_grip_bushing(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    interface: GripInterface | None = None,
    timestep_s: float = DEFAULT_TIMESTEP_S,
    t_end_s: float | None = None,
) -> GripKineticsSeries:
    """Integrate the free club on two bushings over the prescribed swing.

    The club starts at the weld pose of the first sample with zero velocity,
    as in the OpenSim reference.  Samples are taken at ``swing.time_s``.

    Raises:
        ValueError: if a sample interval is not a whole number of steps.
    """
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    sim = ClubOnBushings(spec_bytes, list(swing.names), interface, timestep_s)
    spline = CoordinateSpline(swing.time_s, swing.q)
    sim.hand_source = spline.evaluate
    times = swing.time_s
    if t_end_s is not None:
        times = times[times <= t_end_s + 1e-12]
    steps = np.diff(times) / timestep_s
    if not np.allclose(steps, np.round(steps), atol=1e-6):
        raise ValueError("sample interval must be a whole number of timesteps")
    weld0 = sim.kinematics.state(*spline.evaluate(float(times[0])))
    sim.place(weld0.rotation, weld0.position_m)
    sim.data.time = float(times[0])
    samples, club_rot = [], []
    with _PassiveCallback(sim):
        for k in range(times.size):
            if k:
                for _ in range(int(round(steps[k - 1]))):
                    sim._mj.mj_step(sim.model, sim.data)  # noqa: SLF001
            samples.append(sim.evaluate())
            club_rot.append(sim.data.xmat[sim._body].reshape(3, 3).copy())  # noqa: SLF001
    rec = _record(samples, club_rot)
    return GripKineticsSeries.from_frames(
        ENGINE,
        times,
        rec["wrench"],
        rec["hand_pose"],
        rec["club_frame_pose"],
        rec["club_rotation"],
        metadata={
            "integrator": "MuJoCo RK4 (fixed step)",
            "timestep_s": timestep_s,
            "force_law": "shared OpenSim BushingForce law via mjcb_passive",
            "kinematics": "MuJoCo full-body weld MJCF forward kinematics",
        },
    )


def probe_bushing_forces(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    translation_hand_m: np.ndarray,
    interface: GripInterface | None = None,
) -> BushingProbe:
    """Bushing forces for the club displaced by ``translation_hand_m``.

    The displacement is given in the (shared) hand-frame axes at the first
    sample's weld pose; hands and club are at rest.  The engine total is
    ``qfrc_passive`` of the free joint's translational dofs (world force).
    """
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    sim = ClubOnBushings(spec_bytes, list(swing.names), interface, DEFAULT_TIMESTEP_S)
    q0 = np.asarray(swing.q[0], float)
    still = np.zeros_like(q0)
    sim.hand_source = lambda _t: (q0, still)
    weld0 = sim.kinematics.state(q0, still)
    hand_rot = weld0.rotation @ np.asarray(interface.left.rotation, float)
    shift = hand_rot @ np.asarray(translation_hand_m, float)
    sim.place(weld0.rotation, weld0.position_m + shift)
    with _PassiveCallback(sim):
        sample = sim.evaluate()
    forces: Mapping[str, np.ndarray] = {s: sample.wrench[s].force_n for s in SIDES}
    return BushingProbe(
        force_n=dict(forces),
        hand_rotation=hand_rot,
        engine_total_force_n=np.array(sim.data.qfrc_passive[:3]),
    )
