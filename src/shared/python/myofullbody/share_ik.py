"""ROM-aware whole-body orientation IK with error sharing (issue #11689).

:meth:`mapping.MyoMapper.map_pose` solves one segment group at a time, proximal
first.  With ``rom_policy="extend"`` it reaches every orientation by leaving the
MyoFullBody joint ranges (axial rotation 23 degrees over, left hip rotation 46
degrees over), which makes the passive muscle force at those poses dominate the
reserve.  With ``"clamp"`` it stays in range but pushes the whole shortfall onto
the one segment whose joint saturates, and the segments below it inherit a
wrong parent.

This module solves all 16 segment orientations at once under hard joint limits.
The unknowns are the independent joints of :data:`mapping.SEGMENT_JOINTS` and a
small rotation ``delta`` of the pelvis target.  The cost is the sum of squared
orientation errors, each divided by a tolerance, so that when a limit binds the
shortfall is shared in proportion to ``1 / tolerance**2``: the proximal
chains (pelvis, thorax, femur) may give up to :data:`PROXIMAL_TOLERANCE_DEG` while
the arm segments hold tight.  When no limit binds the fit is exact, as before.

``velocity_map`` is the matching rate map: the same weighted least squares on the
segment angular velocities, with joints on a limit held.  It is the Gauss-Newton
(small residual) derivative of the pose solution, not the exact sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import require
from src.shared.python.myofullbody import anatomy, mapping, mapping_report

Array = np.ndarray
PROXIMAL_TOLERANCE_DEG = 15.0
PROXIMAL_SEGMENTS = ("pelvis", "thorax", "femur_l", "femur_r")
BOUND_TOL = 1e-7
SMALL_ANGLE = 1e-6


def _skew(v: Array) -> Array:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotation_vector(rot: Array) -> Array:
    """Rotation vector (log map) of a rotation matrix, accurate up to ~180 degrees."""
    cos_t = min(1.0, max(-1.0, 0.5 * (float(np.trace(rot)) - 1.0)))
    skew_part = 0.5 * np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]
    )
    sin_t = float(np.linalg.norm(skew_part))
    if cos_t > 0.0 or sin_t > 1e-3:
        theta = float(np.arctan2(sin_t, cos_t))
        return skew_part * (theta / sin_t) if sin_t > SMALL_ANGLE else skew_part
    return np.asarray(Rotation.from_matrix(rot).as_rotvec())  # near pi: robust path


def left_jacobian(rotvec: Array) -> Array:
    """Left Jacobian ``J_l`` of SO(3): ``Exp(r + dr) = Exp(J_l(r) dr) Exp(r)``."""
    theta = float(np.linalg.norm(rotvec))
    k = _skew(rotvec)
    if theta < SMALL_ANGLE:
        return np.eye(3) + 0.5 * k
    return (
        np.eye(3)
        + (1.0 - np.cos(theta)) / theta**2 * k
        + (theta - np.sin(theta)) / theta**3 * (k @ k)
    )


def left_jacobian_inv(rotvec: Array) -> Array:
    """Inverse of :func:`left_jacobian`: ``log(Exp(a) Exp(r)) ~ r + J_l^-1(r) a``."""
    theta = float(np.linalg.norm(rotvec))
    k = _skew(rotvec)
    if theta < SMALL_ANGLE:
        return np.eye(3) - 0.5 * k
    coeff = 1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * k + coeff * (k @ k)


def default_tolerances_deg() -> dict[str, float]:
    """Per-segment tolerance (deg): tight for distal 3-DOF chains, loose proximally."""
    tol = {s: mapping_report.tolerance_for(s) for s in anatomy.SEGMENTS}
    tol.update(dict.fromkeys(PROXIMAL_SEGMENTS, PROXIMAL_TOLERANCE_DEG))
    return tol


@dataclass(frozen=True)
class ShareConfig:
    """Tolerances (degrees) that weight the orientation errors of each segment.

    ``tolerance_deg`` overrides entries of :func:`default_tolerances_deg`.

    Raises:
        ValueError: on an unknown segment or a non-positive tolerance.
    """

    tolerance_deg: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.tolerance_deg.items():
            require(name in anatomy.SEGMENTS, f"unknown segment {name!r}")
            require(value > 0.0, f"tolerance of {name} must be positive")

    def weights(self) -> dict[str, float]:
        """``1 / tolerance`` in 1/rad for every segment."""
        merged = {**default_tolerances_deg(), **self.tolerance_deg}
        return {s: 1.0 / np.radians(merged[s]) for s in anatomy.SEGMENTS}


_JOINTS = tuple(j for names in mapping.SEGMENT_JOINTS.values() for j in names)


class _Problem:
    """One frame: targets, bounds and the residual with its analytic Jacobian."""

    def __init__(
        self, mapper: Any, q_spec: Array, config: ShareConfig, start: Array
    ) -> None:
        import mujoco

        self.mapper, self.mujoco = mapper, mujoco
        model = mapper.model
        mapper.spec.set(q_spec)
        self.target = {s: mapper.target(s) for s in anatomy.SEGMENTS}
        self.segments = [s for s in anatomy.SEGMENTS if s != "pelvis"]
        self.weight = config.weights()
        self.adr = np.array([mapper._joint_adr[j] for j in _JOINTS])
        self.dof = np.array([int(model.joint(j).dofadr[0]) for j in _JOINTS])
        self.lo, self.hi = mapper.bounds(_JOINTS, "clamp")
        rot_rel, pos_rel = mapper._root_rel
        self.root_rot0 = self.target["pelvis"] @ rot_rel.T
        spec_pos = np.array(
            [
                q_spec[mapper.spec_coordinate_index(n)]
                for n in mapping.SPEC_ROOT_TRANSLATION
            ]
        )
        self.pelvis_at = mapper.world @ spec_pos
        self.pos_rel = pos_rel
        self.start = start
        free_dofs = [
            int(model.jnt_dofadr[j])
            for j in range(model.njnt)
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        free_qadr = [
            int(model.jnt_qposadr[j])
            for j in range(model.njnt)
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.h_dof, self.h_qadr = np.array(free_dofs), np.array(free_qadr)
        self.nj = len(self.adr)
        self._cache: tuple[Array, Array, dict[str, Array]] | None = None

    def qpos(self, x: Array) -> Array:
        q = self.start.copy()
        q[self.adr] = x[: self.nj]
        rot = Rotation.from_rotvec(x[self.nj :]).as_matrix() @ self.root_rot0
        q[:3] = self.pelvis_at - rot @ self.pos_rel
        qx, qy, qz, qw = Rotation.from_matrix(rot).as_quat()
        q[3:7] = (qw, qx, qy, qz)
        return q

    def _errors(self, x: Array) -> tuple[Array, dict[str, Array]]:
        if self._cache is not None and np.array_equal(self._cache[0], x):
            return self._cache[1], self._cache[2]
        full = self.mapper.expand_coupled(self.qpos(x))
        data, model = self.mapper.data, self.mapper.model
        data.qpos[:] = full
        self.mujoco.mj_kinematics(model, data)
        self.mujoco.mj_comPos(model, data)
        rot = {
            s: rotation_vector(
                self.target[s].T
                @ np.asarray(data.xmat[self.mapper.myo_body[s]]).reshape(3, 3)
            )
            for s in self.segments
        }
        self._cache = (x.copy(), full, rot)
        return full, rot

    def residual(self, x: Array) -> Array:
        _, rot = self._errors(x)
        delta = x[self.nj :]
        return np.concatenate(
            [delta * self.weight["pelvis"]]
            + [rot[s] * self.weight[s] for s in self.segments]
        )

    def jacobian(self, x: Array) -> Array:
        full, rot = self._errors(x)
        data, model = self.mapper.data, self.mapper.model
        jac_q = self.mapper.coupled_jacobian(full)
        d_qfull = jac_q[np.ix_(self.h_qadr, self.adr)]  # (n_hinge, nj)
        root = left_jacobian(x[self.nj :])
        rows = [np.hstack([np.zeros((3, self.nj)), np.eye(3)]) * self.weight["pelvis"]]
        for s in self.segments:
            jacr = np.zeros((3, model.nv))
            self.mujoco.mj_jacBody(model, data, None, jacr, self.mapper.myo_body[s])
            omega = np.hstack([jacr[:, self.h_dof] @ d_qfull, root])
            m_t = self.target[s].T
            rows.append(self.weight[s] * left_jacobian_inv(rot[s]) @ m_t @ omega)
        return np.vstack(rows)


def map_pose(
    mapper: Any,
    q_spec: Array,
    guess: Array | None = None,
    config: ShareConfig | None = None,
) -> mapping.MappedPose:
    """Map one spec pose with every joint inside its limit and the error shared.

    Args:
        mapper: a :class:`mapping.MyoMapper` built with ``rom_policy="clamp"``.
        guess: previous-frame ``qpos`` (warm start, used directly); the first
            frame of a sequence should omit it so the multistart of the mapper
            supplies an in-range start.

    Returns:
        A pose whose ``error_deg`` holds the orientation error of every segment
        (``pelvis`` is the rotation given up from the target) and whose
        ``at_bound`` lists the joints on a limit.

    Raises:
        ValueError: if the mapper does not use the ``clamp`` ROM policy.
    """
    require(mapper.rom_policy == "clamp", "share IK needs rom_policy='clamp'")
    cfg = config or ShareConfig()
    start = (
        np.asarray(guess, dtype=float)
        if guess is not None
        else mapper.map_pose(q_spec, bounded=True).qpos
    )
    prob = _Problem(mapper, q_spec, cfg, start)
    x0 = np.concatenate([start[prob.adr], np.zeros(3)])
    lower = np.concatenate([prob.lo, -np.pi * np.ones(3)])
    upper = np.concatenate([prob.hi, np.pi * np.ones(3)])
    x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)
    result = least_squares(
        prob.residual, x0, jac=prob.jacobian, bounds=(lower, upper),
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200,
    )  # fmt: skip
    full, rot = prob._errors(result.x)
    errors = {"pelvis": float(np.degrees(np.linalg.norm(result.x[prob.nj :])))}
    errors.update({s: float(np.degrees(np.linalg.norm(v))) for s, v in rot.items()})
    joints = result.x[: prob.nj]
    hit = (np.abs(joints - prob.lo) < BOUND_TOL) | (
        np.abs(joints - prob.hi) < BOUND_TOL
    )
    at_bound = tuple(j for j, h in zip(_JOINTS, hit, strict=True) if h)
    return mapping.MappedPose(full, errors, at_bound, {})


def map_sequence(
    mapper: Any,
    q: Array,
    indices: Array,
    config: ShareConfig | None = None,
) -> list[mapping.MappedPose]:
    """Map the frames ``indices`` in order, each warm-started from the previous."""
    poses: list[mapping.MappedPose] = []
    guess: Array | None = None
    for k in indices:
        pose = map_pose(mapper, q[int(k)], guess, config)
        poses.append(pose)
        guess = pose.qpos
    return poses


def velocity_map(
    mapper: Any,
    q_spec: Array,
    pose: mapping.MappedPose,
    config: ShareConfig | None = None,
) -> Array:
    """``Phi`` (nv_myo, n_spec): MyoFullBody DOF rates per unit spec coordinate rate.

    Weighted least squares of all segment angular velocities at once (the weights
    of :class:`ShareConfig`) over the joints that are not on a limit and the root
    rotation; the dependent DOF rates follow the coupling slopes.
    """
    import mujoco

    cfg = config or ShareConfig()
    weight = cfg.weights()
    model = mapper.model
    mapper.spec.set(q_spec)
    mapper.data.qpos[:] = pose.qpos
    mujoco.mj_kinematics(model, mapper.data)
    mujoco.mj_comPos(model, mapper.data)
    coupling = mapper.dof_coupling(pose.qpos)
    root = int(model.joint("root").dofadr[0])
    held = set(pose.at_bound)
    unknown = list(range(root + 3, root + 6)) + [
        int(model.joint(j).dofadr[0]) for j in _JOINTS if j not in held
    ]
    eff, want = [], []
    for s in anatomy.SEGMENTS:
        rows, target = mapper._rate_rows(s, coupling)
        eff.append(weight[s] * rows[:, unknown])
        want.append(weight[s] * target)
    rates = np.zeros((model.nv, want[0].shape[1]))
    rates[unknown] = np.linalg.lstsq(np.vstack(eff), np.vstack(want), rcond=None)[0]
    return np.asarray(coupling @ rates)
