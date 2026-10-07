"""Spec skeleton to MyoFullBody mapping by segment-orientation IK (issue #11644).

Per frame, the spec coordinates ``q_spec`` (44) are turned into a MyoFullBody
pose by matching the world orientation of 16 anatomical segment frames
(:mod:`anatomy`), proximal segments first, with the 1 to 3 MyoFullBody
independent joints of each segment as unknowns (``ik.solve_orientation``).  The
pelvis sets the free joint, the thorax the three lumbar coordinates, each
humerus the three shoulder coordinates (the scapulothoracic chain follows by
the exact joint-equality couplings), and so on to the toes.

The same hierarchy gives the first-order map ``Phi = d q_myo / d q_spec`` used
to carry spec joint torques to muscle space (:mod:`redundancy`): the target
angular velocity of each segment is ``G * J_spec * qdot_spec`` and the unknown
rates are solved segment by segment against the MyoFullBody body Jacobians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import require
from src.shared.python.myofullbody import anatomy
from src.shared.python.myofullbody.couplings import JointCoupling
from src.shared.python.myofullbody.ik import OrientationFit, solve_orientations

Array = np.ndarray

SEGMENT_JOINTS: dict[str, tuple[str, ...]] = {
    "thorax": ("flex_extension", "lat_bending", "axial_rotation"),
    **{
        f"humerus_{s}": (f"elv_angle_{s}", f"shoulder_elv_{s}", f"shoulder_rot_{s}")
        for s in anatomy.SIDES
    },
    **{f"forearm_{s}": (f"elbow_flexion_{s}", f"pro_sup_{s}") for s in anatomy.SIDES},
    **{f"hand_{s}": (f"deviation_{s}", f"flexion_{s}") for s in anatomy.SIDES},
    **{
        f"femur_{s}": (f"hip_flexion_{s}", f"hip_adduction_{s}", f"hip_rotation_{s}")
        for s in anatomy.SIDES
    },
    **{f"tibia_{s}": (f"knee_angle_{s}",) for s in anatomy.SIDES},
    **{f"foot_{s}": (f"ankle_angle_{s}", f"subtalar_angle_{s}") for s in anatomy.SIDES},
    **{f"toes_{s}": (f"mtp_angle_{s}",) for s in anatomy.SIDES},
}
# Solve groups, proximal first.  Forearm and hand share one least-squares problem
# because MyoFullBody's elbow carries a ~12 degree carrying angle that the spec's
# orthogonal elbow lacks: the residual is split rather than pushed onto the wrist.
SEGMENT_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("thorax",), SEGMENT_JOINTS["thorax"]),
    *(
        item
        for s in anatomy.SIDES
        for item in (
            ((f"humerus_{s}",), SEGMENT_JOINTS[f"humerus_{s}"]),
            (
                (f"forearm_{s}", f"hand_{s}"),
                SEGMENT_JOINTS[f"forearm_{s}"] + SEGMENT_JOINTS[f"hand_{s}"],
            ),
            ((f"femur_{s}",), SEGMENT_JOINTS[f"femur_{s}"]),
            ((f"tibia_{s}",), SEGMENT_JOINTS[f"tibia_{s}"]),
            (
                (f"foot_{s}", f"toes_{s}"),
                SEGMENT_JOINTS[f"foot_{s}"] + SEGMENT_JOINTS[f"toes_{s}"],
            ),
        )
    ),
)
ELEVATION_PREFIX = "shoulder_elv_"
ROM_POLICIES = ("clamp", "extend")
EXTEND_MARGIN_RAD = np.pi
SPEC_ROOT_TRANSLATION = ("TranslationInputX", "TranslationInputY", "TranslationInputZ")


@dataclass
class MappedPose:
    """One mapped frame.

    Attributes:
        qpos: full MyoFullBody ``qpos`` (dependent joints substituted).
        error_deg: orientation error per segment (degrees).
        at_bound: names of joints sitting on a limit.
        clamp_deg: per joint, how far the unbounded optimum lies beyond the limit.
    """

    qpos: Array
    error_deg: dict[str, float]
    at_bound: tuple[str, ...]
    fits: dict[str, OrientationFit] = field(repr=False, default_factory=dict)


class SpecKinematics:
    """Forward kinematics and body Jacobians of the spec skeleton (MuJoCo export)."""

    def __init__(self, spec_bytes: bytes) -> None:
        import mujoco

        from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
            export_full_body_mjcf,
        )

        xml, _ = export_full_body_mjcf(spec_bytes)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.coordinate_order: tuple[str, ...] = tuple(
            json.loads(spec_bytes)["coordinate_order"]
        )
        self.qpos_adr = np.array(
            [self.model.joint(n).qposadr[0] for n in self.coordinate_order], dtype=int
        )
        self.dof_adr = np.array(
            [self.model.joint(n).dofadr[0] for n in self.coordinate_order], dtype=int
        )

    def set(self, q: Array) -> None:
        import mujoco

        require(q.shape == (len(self.coordinate_order),), "q must be in spec order")
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def rotation(self, body: int) -> Array:
        return np.asarray(self.data.xmat[body]).reshape(3, 3).copy()

    def position(self, body: int) -> Array:
        return np.asarray(self.data.xpos[body]).copy()

    def angular_jacobian(self, body: int) -> Array:
        """``(3, n_coord)`` world angular-velocity Jacobian in spec order."""
        import mujoco

        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, body)
        return jacr[:, self.dof_adr]


class MyoMapper:
    """Maps spec poses and rates onto MyoFullBody.

    Args:
        spec_bytes: the ``full-body-v1`` document of the bundle.
        myo_model: compiled MyoFullBody ``MjModel`` (from ``assets.load_myofullbody``).
        reference_q: spec pose (e.g. the address frame) at which the leg hinge axes
            are signed; the legs' own zero pose is a calibrated, twisted posture.
        rom_policy: ``"clamp"`` keeps every joint inside MyoFullBody's own limits
            (the honest default); ``"extend"`` widens each limit by
            :data:`EXTEND_MARGIN_RAD` so that the target orientation is reachable.
    """

    def __init__(
        self,
        spec_bytes: bytes,
        myo_model: Any,
        reference_q: Array,
        rom_policy: str = "clamp",
    ) -> None:
        import mujoco

        require(rom_policy in ROM_POLICIES, f"rom_policy must be one of {ROM_POLICIES}")
        self.rom_policy = rom_policy
        self.spec = SpecKinematics(spec_bytes)
        self.model = myo_model
        self.data = mujoco.MjData(myo_model)
        self.coupling = JointCoupling.from_model(myo_model)
        spec_frames = anatomy.segment_frames(
            self.spec.model,
            anatomy.spec_frame_defs(),
            np.zeros(self.spec.model.nq),
            self._spec_qpos(reference_q),
        )
        myo_frames = anatomy.segment_frames(
            myo_model, anatomy.myo_frame_defs(), np.asarray(myo_model.qpos0)
        )
        self.spec_body = spec_frames.body
        self.myo_body = myo_frames.body
        self.rotation_offset = {
            name: spec_frames.constant[name] @ myo_frames.constant[name].T
            for name in anatomy.SEGMENTS
        }
        self.world = anatomy.world_alignment(spec_frames, myo_frames)
        self._joint_adr = {
            name: int(myo_model.joint(name).qposadr[0])
            for names in SEGMENT_JOINTS.values()
            for name in names
        }
        self._root_rel = self._pelvis_in_root()
        self._limits = self._build_limits()

    # -- setup ---------------------------------------------------------------
    def _spec_qpos(self, q: Array) -> Array:
        qpos = np.zeros(self.spec.model.nq)
        qpos[self.spec.qpos_adr] = q
        return qpos

    def _pelvis_in_root(self) -> tuple[Array, Array]:
        """Pelvis rotation and origin in the free-joint frame at the identity root."""
        import mujoco

        q = np.asarray(self.model.qpos0).copy()
        q[:7] = [0, 0, 0, 1, 0, 0, 0]
        self.data.qpos[:] = self.coupling.expand(q)
        mujoco.mj_kinematics(self.model, self.data)
        body = self.myo_body["pelvis"]
        return (
            np.asarray(self.data.xmat[body]).reshape(3, 3).copy(),
            np.asarray(self.data.xpos[body]).copy(),
        )

    def _build_limits(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for name in self._joint_adr:
            jid = self.model.joint(name).id
            lo, hi = (
                self.model.jnt_range[jid]
                if self.model.jnt_limited[jid]
                else (-np.inf, np.inf)
            )
            out[name] = (float(lo), float(hi))
        return out

    def bounds(
        self, joints: tuple[str, ...], policy: str | None = None
    ) -> tuple[Array, Array]:
        """Lower and upper bounds of ``joints`` under the ROM policy."""
        policy = policy or self.rom_policy
        margin = EXTEND_MARGIN_RAD if policy == "extend" else 0.0
        lo = np.array([self._limits[j][0] - margin for j in joints])
        hi = np.array([self._limits[j][1] + margin for j in joints])
        for i, name in enumerate(joints):
            if name.startswith(ELEVATION_PREFIX):
                # The scapulothoracic polynomials are fitted for elevation >= 0, and
                # a negative elevation duplicates a positive one with the plane of
                # elevation turned by pi: never extrapolate across that branch.
                lo[i] = self._limits[name][0]
        return lo, hi

    # -- pose ------------------------------------------------------------------
    def target(self, segment: str) -> Array:
        """World orientation the MyoFullBody body of ``segment`` must take."""
        return (
            self.world
            @ self.spec.rotation(self.spec_body[segment])
            @ self.rotation_offset[segment]
        )

    def map_pose(
        self,
        q_spec: Array,
        guess: Array | None = None,
        *,
        bounded: bool = True,
        ground_offset: float = 0.0,
        multistart: bool | None = None,
    ) -> MappedPose:
        """Map one spec pose to MyoFullBody (``bounded=False`` ignores all limits).

        Args:
            guess: previous-frame ``qpos`` used as the warm start (continuity).
            multistart: also try neutral and mid-range starts (default: only
                when there is no ``guess``); slow, needed once per sequence.
            ground_offset: vertical shift (m) added to the root height.

        Postcondition: dependent joints satisfy their equalities exactly.
        """
        self.spec.set(q_spec)
        multistart = guess is None if multistart is None else multistart
        q = np.asarray(self.model.qpos0 if guess is None else guess, dtype=float).copy()
        self._set_root(q, q_spec, ground_offset)
        fits: dict[str, OrientationFit] = {}
        errors = {"pelvis": 0.0}
        for segments, joints in SEGMENT_GROUPS:
            adr = tuple(self._joint_adr[j] for j in joints)
            lo, hi = self.bounds(joints) if bounded else (None, None)
            box = None if lo is None else (lo, hi)
            targets = [(self.myo_body[n], self.target(n)) for n in segments]
            seeds = self._seeds(joints, adr) if multistart else ()
            prefer = self._mid_range(joints) if multistart else None
            if multistart and bounded:
                # Start inside the natural range and prefer the Euler branch closest
                # to mid-range, so that an extended fit does not land on an
                # equivalent but distant branch of the same orientation.
                clamped = solve_orientations(
                    self.model, self.data, self.coupling, q, targets, adr,
                    self.bounds(joints, "clamp"), seeds=seeds, prefer=prefer,
                )  # fmt: skip
                q[list(adr)] = clamped.values
                seeds, prefer = (), None
            fit = solve_orientations(
                self.model, self.data, self.coupling, q, targets, adr, box,
                seeds=seeds, prefer=prefer,
            )  # fmt: skip
            q[list(adr)] = fit.values
            fits[segments[0]] = fit
            for n, err in zip(segments, fit.errors_rad, strict=True):
                errors[n] = float(np.degrees(err))
            if len(segments) > 1:
                fits[segments[-1]] = fit
        full = self.coupling.expand(q)
        bound = tuple(
            j
            for segments, joints in SEGMENT_GROUPS
            for j, hit in zip(joints, fits[segments[0]].at_bound, strict=True)
            if hit
        )
        return MappedPose(full, errors, bound, fits)

    def _mid_range(self, joints: tuple[str, ...]) -> Array:
        lo, hi = self.bounds(joints, "clamp")
        return np.where(np.isfinite(lo + hi), 0.5 * (lo + hi), 0.0)

    def _seeds(
        self, joints: tuple[str, ...], adr: tuple[int, ...]
    ) -> tuple[Array, ...]:
        """Alternative starting points: the neutral posture and the mid-range."""
        lo, hi = self.bounds(joints, "clamp")
        zero = np.zeros(len(adr))
        return (zero, self._mid_range(joints), np.clip(zero, lo, hi))

    def _set_root(self, q: Array, q_spec: Array, ground_offset: float) -> None:
        """Free joint from the pelvis frame target and the spec root translation."""
        rot_rel, pos_rel = self._root_rel
        root_rot = self.target("pelvis") @ rot_rel.T
        spec_pos = np.array(
            [q_spec[self.spec.coordinate_order.index(n)] for n in SPEC_ROOT_TRANSLATION]
        )
        pelvis_at = self.world @ spec_pos
        q[:3] = pelvis_at - root_rot @ pos_rel + np.array([0.0, 0.0, ground_offset])
        x, y, z, w = Rotation.from_matrix(root_rot).as_quat()
        q[3:7] = (w, x, y, z)

    # -- rates -----------------------------------------------------------------
    def dof_coupling(self, qpos: Array) -> Array:
        """``(nv, nv)`` map from independent-DOF rates to all-DOF rates."""
        import mujoco

        m = self.model
        jac_q = self.coupling.jacobian(qpos)
        dof = np.zeros((m.nv, m.nv))
        qadr_to_dof: dict[int, int] = {}
        for j in range(m.njnt):
            qa, da = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                dof[da : da + 6, da : da + 6] = np.eye(6)
            else:
                qadr_to_dof[qa] = da
        for qa, da in qadr_to_dof.items():
            for qb, db in qadr_to_dof.items():
                dof[da, db] = jac_q[qa, qb]
        return dof

    def _rate_rows(self, segment: str, coupling: Array) -> tuple[Array, Array]:
        """MyoFullBody body Jacobian (over free DOFs) and target spec rate rows."""
        import mujoco

        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, self.myo_body[segment])
        want = self.world @ self.spec.angular_jacobian(self.spec_body[segment])
        return jacr @ coupling, want

    def velocity_map(self, q_spec: Array, pose: MappedPose) -> Array:
        """``Phi`` (nv_myo, 44): MyoFullBody DOF rates per unit spec coordinate rate.

        The hierarchy of :meth:`map_pose` is differentiated to first order; joints
        on a limit are held (zero rate).  Postcondition: dependent DOF rates follow
        the coupling slopes.
        """
        import mujoco

        self.spec.set(q_spec)
        m = self.model
        self.data.qpos[:] = pose.qpos
        mujoco.mj_kinematics(m, self.data)
        mujoco.mj_comPos(m, self.data)
        coupling = self.dof_coupling(pose.qpos)
        rates = np.zeros((m.nv, len(self.spec.coordinate_order)))
        root = int(m.joint("root").dofadr[0])
        pelvis = self._rate_rows("pelvis", coupling)
        rates[root : root + 6] = (
            np.linalg.pinv(pelvis[0][:, root : root + 6]) @ pelvis[1]
        )
        solved = list(range(root, root + 6))
        for segments, joints in SEGMENT_GROUPS:
            fit = pose.fits[segments[0]]
            own = [
                int(m.joint(j).dofadr[0])
                for j, hit in zip(joints, fit.at_bound, strict=True)
                if not hit
            ]
            if own:
                eff = np.vstack([self._rate_rows(n, coupling)[0] for n in segments])
                want = np.vstack([self._rate_rows(n, coupling)[1] for n in segments])
                rhs = want - eff[:, solved] @ rates[solved]
                rates[own] = np.linalg.pinv(eff[:, own]) @ rhs
            solved.extend(own)
        return coupling @ rates
