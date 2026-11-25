"""Grip-driven flexible-shaft simulation (club fitting groundwork, #15684).

The club body follows a PRESCRIBED motion of the grip point (cubic spline of
the grip, C2 rotation spline of the orientation) and only the shaft chain
coordinates are integrated, ``M_cc qdd_c = f_c - M_cr a_root``, with
``f = qfrc_passive - qfrc_bias`` (springs, damping, gravity, Coriolis and
centrifugal terms) from MuJoCo. The shaft therefore sees the full inertial
loading of the hand motion, which a mocap-parented chain would not.

This is a SAME-HAND-MOTION study: the golfer does not adapt to the shaft, so
the head-speed change it reports is the kinematic effect of the bend for one
fixed hand path, not a speed-potential model of a player who re-times the
swing for a different shaft.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.club_fitting.grip_motion import (
    ClubPoseTrack,
    FittingSettings,
    GripMotion,
    grip_point_in_club,
    load_settings,
    unit_vector,
)
from src.shared.python.motion_matching import club_models as cm
from src.shared.python.motion_matching import club_shaft_chain as cs

logger = logging.getLogger(__name__)

GRAVITY_M_S2 = (0.0, 0.0, -9.80665)
_UP = np.array([0.0, 0.0, 1.0])
_QUASI_STATIC_ITERATIONS = 200
_QUASI_STATIC_TOL = 1e-12  # rad, fixed-point convergence of the initial bend
_HORIZONTAL_TOL = 1e-6


@dataclass(frozen=True)
class HeadGeometry:
    """Face centre and unit face normal in the club-body frame."""

    centre_club_m: np.ndarray
    normal_club: np.ndarray

    def __post_init__(self) -> None:
        c = np.asarray(self.centre_club_m, dtype=float)
        if c.shape != (3,) or not np.all(np.isfinite(c)):
            raise ValueError("centre_club_m must be a finite 3-vector")
        object.__setattr__(self, "centre_club_m", c)
        object.__setattr__(
            self, "normal_club", unit_vector("normal_club", self.normal_club)
        )


def chain_model(chain: cs.ShaftChain, dt_s: float) -> Any:
    """Clamped chain MJCF with a free joint on the club body (RK4 option)."""
    import mujoco

    from src.engines.physics_engines.mujoco.python.shaft_chain_mjcf import (
        export_clamped_shaft_mjcf,
    )

    doc = cs.clamped_shaft_document(chain, gravity=GRAVITY_M_S2)
    xml = export_clamped_shaft_mjcf(doc)
    tag = f'<body name="{doc["bodies"][1]["name"]}" pos="0 0 0" quat="1 0 0 0">'
    if tag not in xml or "<option gravity" not in xml:
        raise RuntimeError("unexpected clamped-shaft MJCF layout")
    xml = xml.replace(tag, tag + '<freejoint name="grip_root" />', 1)
    xml = xml.replace(
        "<option gravity", f'<option timestep="{dt_s}" integrator="RK4" gravity', 1
    )
    return mujoco.MjModel.from_xml_string(xml)


@dataclass(frozen=True)
class ShaftRun:
    """Shaft state and head kinematics along one grip-driven run.

    Deflections and ``*_delta_deg`` are bent minus straight (rigid club in the
    same hand motion); face delta + is more open, loft delta + is the face
    normal tilted up, lie delta + is a shaft axis closer to the ground.
    ``tip_deflection_m`` is the tip-site displacement caused by the bend.
    ``impact_chain_club_m`` holds the chain joint centres then the tip site
    at impact, ``(n_links + 1, 3)`` in the rigid club-body frame (``None``
    when not recorded; see :mod:`shaft_shape`).
    """

    times_s: np.ndarray
    chain_q: np.ndarray
    tip_deflection_m: np.ndarray
    head_centre_m: np.ndarray
    head_velocity_mps: np.ndarray
    rigid_centre_m: np.ndarray
    rigid_velocity_mps: np.ndarray
    tip_rotation: np.ndarray
    rigid_rotation: np.ndarray
    face_delta_deg: np.ndarray
    loft_delta_deg: np.ndarray
    lie_delta_deg: np.ndarray
    repaired_times_s: np.ndarray
    impact_chain_club_m: np.ndarray | None = None

    @property
    def head_speed_mps(self) -> np.ndarray:
        return np.linalg.norm(self.head_velocity_mps, axis=1)

    @property
    def rigid_head_speed_mps(self) -> np.ndarray:
        return np.linalg.norm(self.rigid_velocity_mps, axis=1)

    @property
    def peak_tip_deflection_m(self) -> float:
        return float(np.max(np.linalg.norm(self.tip_deflection_m, axis=1)))

    def peak_tip_deflection_until_m(self, t_end_s: float) -> float:
        """Peak tip deflection over ``times_s <= t_end_s``.

        Use the impact time: after it the grip motion is extrapolated and no
        ball load acts on the head, so the bend there is not physical.

        Raises:
            ValueError: ``t_end_s`` is before the first recorded time.
        """
        keep = self.times_s <= t_end_s
        if not np.any(keep):
            raise ValueError("t_end_s is before the run starts")
        return float(np.max(np.linalg.norm(self.tip_deflection_m[keep], axis=1)))

    def at(self, name: str, t: float) -> float:
        """Linear interpolation of the scalar series ``name`` at time ``t``."""
        return float(np.interp(t, self.times_s, getattr(self, name)))

    def at_vector_norm(self, name: str, t: float) -> float:
        """Norm of the (n, 3) series ``name`` interpolated at time ``t``."""
        series = getattr(self, name)
        return float(
            np.linalg.norm([np.interp(t, self.times_s, series[:, j]) for j in range(3)])
        )


def _angles(n: np.ndarray, x_t: np.ndarray, y_t: np.ndarray) -> tuple[float, float]:
    """``(face heading, loft)`` in degrees of a face normal (heading + = open)."""
    u = n / np.linalg.norm(n)
    return (
        math.degrees(math.atan2(-(u @ y_t), u @ x_t)),
        math.degrees(math.asin(float(np.clip(u @ _UP, -1.0, 1.0)))),
    )


def _lie_deg(axis: np.ndarray, y_t: np.ndarray) -> float:
    """Shaft angle above the ground seen along the target line."""
    return math.degrees(math.atan2(abs(axis @ _UP), abs(axis @ y_t)))


class _ChainSim:
    """Chain coordinates under the prescribed free-joint club-body motion."""

    def __init__(self, model: Any, motion: GripMotion) -> None:
        import mujoco

        self.mujoco, self.m, self.motion = mujoco, model, motion
        self.d = mujoco.MjData(model)
        self.nc = model.nv - 6
        self.M = np.zeros((model.nv, model.nv))
        self.stiff = np.asarray(model.jnt_stiffness)[1:].copy()

    def set_state(self, t: float, qc: np.ndarray, vc: np.ndarray) -> np.ndarray:
        R, po, vo, w, ao, al = self.motion.state(t)
        x, y, z, wq = Rotation.from_matrix(R).as_quat()
        self.d.qpos[:7], self.d.qpos[7:] = np.concatenate([po, [wq, x, y, z]]), qc
        self.d.qvel[:6], self.d.qvel[6:] = np.concatenate([vo, R.T @ w]), vc
        return np.concatenate([ao, R.T @ al])

    def accel(self, t: float, qc: np.ndarray, vc: np.ndarray) -> np.ndarray:
        a_root = self.set_state(t, qc, vc)
        self.mujoco.mj_forward(self.m, self.d)
        self.mujoco.mj_fullM(self.m, self.M, self.d.qM)
        f = self.d.qfrc_passive - self.d.qfrc_bias
        return np.linalg.solve(self.M[6:, 6:], f[6:] - self.M[6:, :6] @ a_root)

    def quasi_static(self, t: float) -> np.ndarray:
        """Chain deflection at rest relative to the grip (zero chain accel)."""
        qc, vc = np.zeros(self.nc), np.zeros(self.nc)
        for _ in range(_QUASI_STATIC_ITERATIONS):
            step = (self.M[6:, 6:] @ self.accel(t, qc, vc)) / self.stiff
            qc = qc + step
            if np.max(np.abs(step)) < _QUASI_STATIC_TOL:
                break
        return qc

    def run(self, t0: float, t1: float, dt: float, every: int):
        """RK4 integration; returns recorded ``times, qc, vc``."""
        qc, vc = self.quasi_static(t0), np.zeros(self.nc)
        rec = [(t0, qc, vc)]
        for k in range(int(round((t1 - t0) / dt))):
            t = t0 + k * dt
            k1v = self.accel(t, qc, vc)
            k2q = vc + dt / 2 * k1v
            k2v = self.accel(t + dt / 2, qc + dt / 2 * vc, k2q)
            k3q = vc + dt / 2 * k2v
            k3v = self.accel(t + dt / 2, qc + dt / 2 * k2q, k3q)
            k4q = vc + dt * k3v
            k4v = self.accel(t + dt, qc + dt * k3q, k4q)
            qc = qc + dt / 6 * (vc + 2 * k2q + 2 * k3q + k4q)
            vc = vc + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            if not np.all(np.isfinite(qc)):
                raise FloatingPointError(f"chain diverged at t={t:.4f} s")
            if (k + 1) % every == 0:
                rec.append((t0 + (k + 1) * dt, qc, vc))
        return tuple(np.array(c) for c in zip(*rec, strict=True))


def _head_series(sim: _ChainSim, times, qcs, vcs, head, target) -> dict:
    """Head kinematics and bend deltas along a run (bent minus straight)."""
    mj, m, d = sim.mujoco, sim.m, sim.d
    from src.engines.physics_engines.mujoco.python.shaft_chain_mjcf import TIP_SITE

    site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, TIP_SITE)
    tip_body = int(m.site_bodyid[site])
    off = head.centre_club_m - np.array([0.0, 0.0, cm.HEAD_HEIGHT_M])
    y_t = np.cross(_UP, target)
    jacp, n = np.zeros((3, m.nv)), len(times)
    out: dict[str, np.ndarray] = {
        k: np.zeros((n, 3)) for k in ("defl", "centre", "vel", "rc", "rv")
    }
    out |= {k: np.zeros((n, 3, 3)) for k in ("R_tip", "R_rigid")}
    out |= {k: np.zeros(n) for k in ("face", "loft", "lie")}
    for i, t in enumerate(times):
        sim.set_state(t, qcs[i], vcs[i])
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        R, p = d.site_xmat[site].reshape(3, 3).copy(), d.site_xpos[site].copy()
        centre = R @ off + p
        mj.mj_jac(m, d, jacp, None, centre, tip_body)
        d.qpos[7:] = 0.0  # straight chain: the rigid club in the same motion
        mj.mj_kinematics(m, d)
        Rs = d.site_xmat[site].reshape(3, 3).copy()
        Rm, _, vo, w, _, _ = sim.motion.state(t)
        out["defl"][i], out["R_tip"][i], out["R_rigid"][i] = (
            p - d.site_xpos[site],
            R,
            Rm,
        )
        out["centre"][i], out["vel"][i] = centre, jacp @ d.qvel
        out["rc"][i] = Rs @ off + d.site_xpos[site]
        out["rv"][i] = vo + np.cross(w, Rm @ head.centre_club_m)
        (fb, lb), (fs, ls) = (
            _angles(R @ head.normal_club, target, y_t),
            _angles(Rs @ head.normal_club, target, y_t),
        )
        out["face"][i], out["loft"][i] = fb - fs, lb - ls
        out["lie"][i] = _lie_deg(R[:, 1], y_t) - _lie_deg(Rs[:, 1], y_t)
    return out


def _impact_chain_points(sim: _ChainSim, times, qcs, impact_s: float) -> np.ndarray:
    """Chain joint centres then the tip site at impact, in the club-body frame.

    The chain coordinates are interpolated linearly at ``impact_s``.
    """
    mj, m, d = sim.mujoco, sim.m, sim.d
    from src.engines.physics_engines.mujoco.python.shaft_chain_mjcf import TIP_SITE

    qc = np.array([np.interp(impact_s, times, q) for q in np.asarray(qcs).T])
    sim.set_state(impact_s, qc, np.zeros_like(qc))
    mj.mj_kinematics(m, d)
    club = int(m.jnt_bodyid[0])
    joints = [int(m.body_jntadr[b]) for b in range(club + 1, m.nbody)]
    site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, TIP_SITE)
    world = np.vstack([d.xanchor[joints], d.site_xpos[site][None, :]])
    rotation = d.xmat[club].reshape(3, 3)
    return (world - d.xpos[club]) @ rotation


def simulate_grip_driven_shaft(
    track: ClubPoseTrack,
    chain: cs.ShaftChain,
    head: HeadGeometry,
    impact_time_s: float,
    settings: FittingSettings | None = None,
    target_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> ShaftRun:
    """Run ``chain`` under the grip motion of ``track`` up to just past impact.

    Simulation starts ``settings.lead_in_s`` before ``impact_time_s`` from the
    quasi-static bend and ends ``settings.post_impact_s`` after it. Only pose
    samples before impact drive the motion; the tail is extrapolated. Same
    hand motion for every shaft: the golfer does not adapt to the flex.

    Args:
        target_direction: horizontal target-line direction, only used to
            express the face, loft and lie deltas (world z is up).

    Postcondition: all series share ``times_s`` and are finite.

    Raises:
        ValueError: the track does not cover the lead-in, a non-finite
            impact time, or a target direction that is not horizontal.
    """
    cfg = settings or load_settings()
    if not math.isfinite(impact_time_s):
        raise ValueError("impact_time_s must be finite")
    target = unit_vector("target_direction", target_direction)
    if abs(target @ _UP) > _HORIZONTAL_TOL:
        raise ValueError("target_direction must be horizontal")
    t0 = impact_time_s - cfg.lead_in_s
    before = track.times_s[track.times_s < impact_time_s]
    if len(before) == 0 or track.times_s[0] > t0 - cfg.history_before_start_s:
        raise ValueError("pose track must cover the lead-in and its history")
    motion = GripMotion(
        track,
        grip_point_in_club(chain.club),
        t0 - cfg.history_before_start_s,
        float(before[-1]),
        cfg,
    )
    sim = _ChainSim(chain_model(chain, cfg.integration_dt_s), motion)
    t_end = impact_time_s + cfg.post_impact_s
    if t_end > motion.t_range[1]:
        raise ValueError("extend_s is too short to cover post_impact_s")
    times, qcs, vcs = sim.run(t0, t_end, cfg.integration_dt_s, cfg.record_every)
    o = _head_series(sim, times, qcs, vcs, head, target)
    logger.info(
        "grip-driven run: %d records, %d repaired",
        len(times),
        len(motion.repaired_times_s),
    )
    return ShaftRun(
        times,
        qcs,
        o["defl"],
        o["centre"],
        o["vel"],
        o["rc"],
        o["rv"],
        o["R_tip"],
        o["R_rigid"],
        o["face"],
        o["loft"],
        o["lie"],
        motion.repaired_times_s,
        _impact_chain_points(sim, times, qcs, impact_time_s),
    )
