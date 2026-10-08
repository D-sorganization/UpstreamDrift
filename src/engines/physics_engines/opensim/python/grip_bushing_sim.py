"""Driven-swing simulation of the OpenSim bushing grip (issue #11739, OSV-7).

The 44 body coordinates of the full-body spec are prescribed from a matched
trajectory; the club is a free body held only by the two grip bushings, so the
integrated club motion and the bushing records are the dynamics of the
compliant hand-club interface.  Sign convention (binding): every wrench is the
loading exerted BY THE HAND ON THE CLUB, expressed in the world frame; the
torque is the free torque at the hand grip point (the bushing moment
moved from the club body origin to the grip point).

OpenSim is imported lazily; callers must skip when it is unavailable.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.full_body_grip_topology import (
    CLUB_FREE_COORDINATES,
)
from src.engines.physics_engines.opensim.python.full_body_osim import (
    export_full_body_osim,
)
from src.shared.python.biomechanics.grip_wrench import (
    GripAnalysis,
    HandWrench,
    analyze_grip,
)
from src.shared.python.grip_contact import (
    ConditioningReport,
    GripInterface,
    condition_trajectory,
)

CLUB_BODY = "Clubhead"
TRACKED_BODIES = ("LF", "RF", "LGrip", "Grip")
_SIDES = (("L", "left"), ("R", "right"))


@dataclass(frozen=True)
class BushingRun:
    """Time series of one bushing-grip simulation (world frame, SI)."""

    time_s: np.ndarray
    force_on_club_n: dict[str, np.ndarray]  # side -> (n, 3)
    torque_on_club_nm: dict[str, np.ndarray]  # about the hand grip point
    grip_point_m: dict[str, np.ndarray]  # hand grip point in world
    deflection_m: dict[str, np.ndarray]  # (n, 3) in the hand-side frame axes
    rotation_deflection_rad: dict[str, np.ndarray]  # (n,) angle
    hand_rotation: dict[str, np.ndarray]  # (n, 3, 3) world <- hand-side frame
    club_position_m: np.ndarray
    club_rotation: np.ndarray  # (n, 3, 3) world <- club body
    club_com_m: np.ndarray  # (n, 3) world
    club_velocity_m_s: np.ndarray  # (n, 3) club body origin, world
    club_omega_rad_s: np.ndarray  # (n, 3) world
    body_positions_m: dict[str, np.ndarray]  # tracked body origins (n, 3), world


def _inv(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def _rotation(osim: Any, euler_xyz: np.ndarray) -> Any:
    """Simbody rotation from body-fixed X-Y-Z Euler angles (radians)."""
    rot = osim.Rotation()
    rot.setRotationToBodyFixedXYZ(
        osim.Vec3(float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2]))
    )
    return rot


def lowpass_trajectory(
    time_s: np.ndarray, q: np.ndarray, cutoff_hz: float = 25.0
) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass of every column of ``q``.

    The matched IK trajectories are sampled at ~360 Hz and carry frame-to-frame
    noise that a spline turns into spurious accelerations; the club inertial
    load is acceleration-driven, so the prescribed motion is filtered.

    Raises:
        ValueError: for a non-uniform time base or a cutoff above Nyquist.
    """
    from scipy.signal import butter, sosfiltfilt

    dt = np.diff(time_s)
    if dt.size < 20 or not np.allclose(dt, dt[0], rtol=1e-3):
        raise ValueError("time_s must be uniformly sampled with >= 21 samples")
    nyquist = 0.5 / float(dt[0])
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(f"cutoff_hz must lie in (0, {nyquist:.1f}) Hz")
    sos = butter(4, cutoff_hz / nyquist, output="sos")
    return np.asarray(sosfiltfilt(sos, q, axis=0))


def prepare_motion(
    time_s: np.ndarray,
    q: np.ndarray,
    names: Sequence[str],
    cutoff_hz: float = 25.0,
) -> tuple[np.ndarray, ConditioningReport]:
    """Condition a matched-IK trajectory for kinetics.

    Order: remove 2 pi branch flips, interpolate over IK-failure frames
    (:func:`condition_trajectory`), then zero-phase low-pass.  Returns the
    prepared coordinates and the conditioning report.
    """
    cleaned, report = condition_trajectory(time_s, q, names)
    return lowpass_trajectory(np.asarray(time_s, float), cleaned, cutoff_hz), report


def _osim() -> Any:
    return import_module("opensim")


def _transform_to_rt(transform: Any) -> tuple[np.ndarray, np.ndarray]:
    rot = np.array([[transform.R().get(i, j) for j in range(3)] for i in range(3)])
    pos = np.array([transform.p().get(i) for i in range(3)])
    return rot, pos


def weld_club_pose(
    spec_bytes: bytes, names: Sequence[str], q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World pose of the club body in the rigid-weld model for coordinates ``q``."""
    osim = _osim()
    xml, _ = export_full_body_osim(spec_bytes)
    model = _load(xml)
    state = model.initSystem()
    for name, value in zip(names, q, strict=True):
        model.getCoordinateSet().get(name).setValue(state, float(value), False)
    model.realizePosition(state)
    del osim
    return _transform_to_rt(
        model.getBodySet().get(CLUB_BODY).getTransformInGround(state)
    )


def _speed(points: list[np.ndarray], dt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.gradient(np.array(points), axis=0) / dt[:, None], axis=1)


def input_kinematics_report(
    spec_bytes: bytes,
    names: Sequence[str],
    time_s: np.ndarray,
    q: np.ndarray,
    interface: GripInterface | None = None,
) -> dict[str, Any]:
    """Consistency of a coordinate trajectory with the two-hand grip closure.

    In the rigid-weld model the club follows the left hand; the right hand is
    its own arm chain whose closure frame should coincide with the right grip
    frame on the club.  The residual (distance and angle between the two
    closure frames) is how far the input motion is from a physical two-hand
    grip.  Hand speeds are the speeds of the left grip point on the club and
    of the right closure frame on the right-hand body.
    """
    osim = _osim()
    spec = json.loads(spec_bytes)
    gi = interface or GripInterface.from_spec(spec)
    model = _load(export_full_body_osim(spec_bytes)[0])
    state = model.initSystem()
    weld = osim.WeldConstraint.safeDownCast(model.getConstraintSet().get(0))
    frame_a, frame_b = weld.getConnectee("frame1"), weld.getConnectee("frame2")
    club = model.getBodySet().get(CLUB_BODY)
    p_left = np.asarray(gi.left.position_m)
    dist, ang, left, right = [], [], [], []
    for row in np.asarray(q, float):
        for name, value in zip(names, row, strict=True):
            model.getCoordinateSet().get(name).setValue(state, float(value), False)
        model.realizePosition(state)
        ra, pa = _transform_to_rt(frame_a.getTransformInGround(state))
        rb, pb = _transform_to_rt(frame_b.getTransformInGround(state))
        dist.append(np.linalg.norm(pa - pb))
        ang.append(
            np.degrees(np.linalg.norm(Rotation.from_matrix(ra.T @ rb).as_rotvec()))
        )
        rc, pc = _transform_to_rt(club.getTransformInGround(state))
        left.append(rc @ p_left + pc)
        right.append(pa)
    dt = np.gradient(np.asarray(time_s, float))
    v_l, v_r = _speed(left, dt), _speed(right, dt)
    return {
        "closure_distance_m": np.array(dist),
        "closure_angle_deg": np.array(ang),
        "left_hand_speed_m_s": v_l,
        "right_hand_speed_m_s": v_r,
    }


def _load(xml: str) -> Any:
    osim = _osim()
    osim.Logger.setLevelString("Warn")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.osim"
        path.write_text(xml, encoding="utf-8")
        return osim.Model(str(path))


def _strip_non_bushing_forces(model: Any) -> None:
    forces = model.updForceSet()
    for i in range(forces.getSize() - 1, -1, -1):
        if not forces.get(i).getName().startswith("grip_bushing"):
            forces.remove(i)


class BushingGripSimulator:
    """Prescribed-arm, free-club simulation of the bushing grip model.

    Args:
        spec_bytes: ``full-body-v1`` spec JSON bytes.
        names: the 44 body coordinate names matching columns of ``q``.
        time_s: strictly increasing sample times, shape ``(n,)``.
        q: body coordinates, shape ``(n, len(names))``.
        interface: grip interface (default: built from the spec).
    """

    def __init__(
        self,
        spec_bytes: bytes,
        names: Sequence[str],
        time_s: np.ndarray,
        q: np.ndarray,
        interface: GripInterface | None = None,
    ) -> None:
        time_s, q = np.asarray(time_s, float), np.asarray(q, float)
        if q.ndim != 2 or q.shape != (time_s.size, len(names)):
            raise ValueError("q must have shape (len(time_s), len(names))")
        if time_s.size < 4 or not np.all(np.diff(time_s) > 0.0):
            raise ValueError("time_s must be strictly increasing with >= 4 samples")
        self.names, self.time_s, self.q = list(names), time_s, q
        self._osim = _osim()
        self.interface = interface
        xml, _ = export_full_body_osim(
            spec_bytes, grip_model="bushing", grip_interface=interface
        )
        self._weld_pose0 = weld_club_pose(spec_bytes, self.names, q[0])
        self._model = _load(xml)
        _strip_non_bushing_forces(self._model)
        self._tie_right_hand_to_left(json.loads(spec_bytes))
        self._prescribe()
        self._state = self._model.initSystem()
        self._init_club_pose()

    def _tie_right_hand_to_left(self, spec: dict[str, Any]) -> None:
        """Attach the right bushing's hand frame rigidly to the left hand body.

        The matched IK trajectories do not close the two-hand loop (the right
        hand is ~134 mm and ~50 deg off the right grip frame at address), so
        the right arm chain cannot be prescribed.  The right hand frame is
        instead fixed on the left hand body at ``T_L^-1 G_R`` (``T_L`` the
        club-side wrist frame, ``G_R`` the right grip frame), which is exactly
        where the rigid-weld model places it.  Both hands then follow the left
        arm chain of the matched motion.
        """
        osim = self._osim
        interface = self.interface or GripInterface.from_spec(spec)
        club = spec["closure"]["body_b"]
        wrist = next(j for j in spec["joints"] if j["child"] == club)
        offset = _inv(np.asarray(wrist["child_to_follower"], float)) @ (
            interface.right.matrix()
        )
        rot = Rotation.from_matrix(offset[:3, :3]).as_euler("XYZ")
        hand = self._model.getBodySet().get("LGrip")
        frame = osim.PhysicalOffsetFrame(
            "grip_right_tied_frame",
            hand,
            osim.Transform(
                _rotation(osim, rot),
                osim.Vec3(*(float(x) for x in offset[:3, 3])),
            ),
        )
        self._model.finalizeFromProperties()
        hand.addComponent(frame)
        force = osim.BushingForce.safeDownCast(
            self._model.updForceSet().get("grip_bushing_right")
        )
        force.connectSocket_frame1(frame)
        self._model.finalizeConnections()

    def _prescribe(self) -> None:
        osim = self._osim
        coords = self._model.updCoordinateSet()
        for k, name in enumerate(self.names):
            values = self.q[:, k]
            if np.ptp(values) < 1e-12:
                fn: Any = osim.Constant(float(values[0]))
            else:
                fn = osim.SimmSpline()
                fn.setName(name)
                for tk, vk in zip(self.time_s, values, strict=True):
                    fn.addPoint(float(tk), float(vk))
            coords.get(name).setPrescribedFunction(fn)
            coords.get(name).set_prescribed(True)

    def _init_club_pose(self) -> None:
        rot, pos = self._weld_pose0
        euler = Rotation.from_matrix(rot).as_euler("XYZ")
        coords = self._model.updCoordinateSet()
        for name, value in zip(CLUB_FREE_COORDINATES, [*euler, *pos], strict=True):
            coords.get(name).setValue(self._state, float(value), False)
        self._model.realizePosition(self._state)

    def probe_deflection(
        self, translation_m: Sequence[float], rotvec_rad: Sequence[float]
    ) -> BushingRun:
        """Evaluate the bushing records at the initial pose with the club displaced.

        The club is moved from its unloaded (weld) pose by ``translation_m``
        (world) and rotated by ``rotvec_rad`` (world, about the club body
        origin); no integration is performed, so velocity terms are zero.
        Returns a one-sample :class:`BushingRun` for ``F = K * delta`` checks.
        """
        rot0, pos0 = self._weld_pose0
        rot = Rotation.from_rotvec(np.asarray(rotvec_rad, float)).as_matrix() @ rot0
        pos = pos0 + np.asarray(translation_m, float)
        coords = self._model.updCoordinateSet()
        euler = Rotation.from_matrix(rot).as_euler("XYZ")
        state = self._state
        for name, value in zip(CLUB_FREE_COORDINATES, [*euler, *pos], strict=True):
            coords.get(name).setValue(state, float(value), False)
        self._model.realizePosition(state)
        return self._assemble(self.time_s[:1], [self._sample(state)])

    def run(self, t_end: float | None = None, accuracy: float = 1e-5) -> BushingRun:
        """Integrate to ``t_end`` and sample at the trajectory times."""
        osim = self._osim
        model, state = self._model, self._state
        stop = float(self.time_s[-1] if t_end is None else t_end)
        if not self.time_s[0] < stop <= self.time_s[-1]:
            raise ValueError("t_end must lie inside the trajectory time range")
        state.setTime(float(self.time_s[0]))
        manager = osim.Manager(model)
        manager.setIntegratorAccuracy(accuracy)
        manager.initialize(state)
        samples = self.time_s[self.time_s <= stop + 1e-12]
        rows = [self._sample(state)]
        for t in samples[1:]:
            state = manager.integrate(float(t))
            rows.append(self._sample(state))
        return self._assemble(samples, rows)

    def _frame(self, side_name: str, kind: str) -> Any:
        """Hand (frame1) or club (frame2) frame actually connected to the bushing."""
        force = self._osim.BushingForce.safeDownCast(
            self._model.getForceSet().get(f"grip_bushing_{side_name}")
        )
        socket = "frame1" if kind == "hand" else "frame2"
        return self._osim.PhysicalOffsetFrame.safeDownCast(force.getConnectee(socket))

    def _sample(self, state: Any) -> dict[str, Any]:
        model = self._model
        model.realizeDynamics(state)
        out: dict[str, Any] = {"t": state.getTime()}
        rot_c, pos_c = _transform_to_rt(
            model.getBodySet().get(CLUB_BODY).getTransformInGround(state)
        )
        out["club"] = (rot_c, pos_c)
        body = model.getBodySet().get(CLUB_BODY)
        com = body.findStationLocationInGround(state, body.getMassCenter())
        out["bodies"] = {
            name: np.array(
                [
                    model.getBodySet().get(name).getTransformInGround(state).p().get(i)
                    for i in range(3)
                ]
            )
            for name in TRACKED_BODIES
        }
        out["com"] = np.array([com.get(i) for i in range(3)])
        model.realizeVelocity(state)
        out["v"] = np.array(
            [body.getLinearVelocityInGround(state).get(i) for i in range(3)]
        )
        out["w"] = np.array(
            [body.getAngularVelocityInGround(state).get(i) for i in range(3)]
        )
        for side, name in _SIDES:
            force = self._osim.BushingForce.safeDownCast(
                model.getForceSet().get(f"grip_bushing_{name}")
            )
            rec = force.getRecordValues(state)
            vals = np.array([rec.get(i) for i in range(rec.size())])
            r1, p1 = _transform_to_rt(
                self._frame(name, "hand").getTransformInGround(state)
            )
            r2, p2 = _transform_to_rt(
                self._frame(name, "club").getTransformInGround(state)
            )
            out[side] = {
                "record": vals,
                "rot1": r1,
                "point": p2,
                "delta": r1.T @ (p2 - p1),
                "angle": float(
                    np.linalg.norm(Rotation.from_matrix(r1.T @ r2).as_rotvec())
                ),
            }
        return out

    def _assemble(self, samples: np.ndarray, rows: list[dict[str, Any]]) -> BushingRun:
        force, torque, point, delta, ang, rot1 = {}, {}, {}, {}, {}, {}
        for side, _ in _SIDES:
            rec = np.array([r[side]["record"] for r in rows])
            point[side] = np.array([r[side]["point"] for r in rows])
            force[side] = rec[:, 6:9]
            # OpenSim reports the wrench on body2 as (F, M about the body
            # origin); move it to the grip point so ``torque`` is the free
            # torque there, as :class:`HandWrench` requires.
            origin = np.array([r["club"][1] for r in rows])
            torque[side] = rec[:, 9:12] - np.cross(point[side] - origin, force[side])
            delta[side] = np.array([r[side]["delta"] for r in rows])
            ang[side] = np.array([r[side]["angle"] for r in rows])
            rot1[side] = np.array([r[side]["rot1"] for r in rows])
        return BushingRun(
            time_s=samples[: len(rows)],
            force_on_club_n=force,
            torque_on_club_nm=torque,
            grip_point_m=point,
            deflection_m=delta,
            rotation_deflection_rad=ang,
            hand_rotation=rot1,
            club_position_m=np.array([r["club"][1] for r in rows]),
            club_rotation=np.array([r["club"][0] for r in rows]),
            club_com_m=np.array([r["com"] for r in rows]),
            club_velocity_m_s=np.array([r["v"] for r in rows]),
            club_omega_rad_s=np.array([r["w"] for r in rows]),
            body_positions_m={
                name: np.array([r["bodies"][name] for r in rows])
                for name in TRACKED_BODIES
            },
        )


def analyze_run(run: BushingRun) -> list[GripAnalysis]:
    """Per-sample :func:`analyze_grip` of the bushing wrenches (force ON the club).

    The hand point is the club-side bushing frame origin; the torque is the
    bushing moment on the club.  ``split_method`` is ``"bushing"``.
    """
    out = []
    for i in range(run.time_s.size):
        hands = {
            side: HandWrench(
                side,
                tuple(run.grip_point_m[side][i]),
                tuple(run.force_on_club_n[side][i]),
                tuple(run.torque_on_club_nm[side][i]),
            )
            for side in ("L", "R")
        }
        out.append(
            analyze_grip(
                hands["L"],
                hands["R"],
                split_method="bushing",
                club_rotation=run.club_rotation[i],
                metadata={"solver": "opensim BushingForce"},
            )
        )
    return out


def hand_power_w(run: BushingRun, side: str) -> np.ndarray:
    """Power ``F.v + tau.omega`` delivered by one hand to the club, shape (n,).

    ``v`` is the velocity of the hand grip point as part of the club body.
    """
    if side not in ("L", "R"):
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")
    arm = run.grip_point_m[side] - run.club_position_m
    v = run.club_velocity_m_s + np.cross(run.club_omega_rad_s, arm)
    return np.einsum("ij,ij->i", run.force_on_club_n[side], v) + np.einsum(
        "ij,ij->i", run.torque_on_club_nm[side], run.club_omega_rad_s
    )
