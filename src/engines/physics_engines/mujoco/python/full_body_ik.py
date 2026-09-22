"""MuJoCo full-body inverse kinematics and marker tracking adapter (FB-4).

Supplies forward kinematics (``pose_fn``), dual-grip weld loop-closure residuals,
least-squares inverse kinematics, and marker kinematics with ground support.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import (
    BaseFullBodyIK,
    _rotation_error,
    _validate_axis_spec,
)
from src.shared.python.motion_matching.marker_calibration import Pose

Array: TypeAlias = NDArray[np.float64]
Attachment = tuple[str, tuple[float, float, float]]


def _unit_ground_normal(ground: GroundPlane) -> Array:
    """Return ``ground.normal`` normalized with a scalar 3-vector norm.

    ``math.hypot`` is ~6x faster than ``np.linalg.norm`` for small 3D vectors.
    """
    n = np.asarray(ground.normal, dtype=float)
    return n / math.hypot(n[0], n[1], n[2])


class MujocoFullBodyIK(BaseFullBodyIK):
    """Full-body MuJoCo inverse kinematics adapter consuming a full-body spec."""

    def __init__(self, specification: Mapping[str, Any] | bytes | str) -> None:
        super().__init__(specification)
        spec_bytes = json.dumps(self.specification).encode("utf-8")
        self.model = NativeMujocoFullBodyModel(spec_bytes)
        self._mj_model = self.model.model
        self._mj_data = self.model.data
        self._metadata = self.model.metadata
        self.coordinate_order: tuple[str, ...] = tuple(self.model.coordinate_order)
        if len(self.coordinate_order) != 41:
            raise ValueError("Expected exactly 41 full-body coordinates")

        self._site_ids: dict[str, int] = {}
        self._body_ids: dict[str, int] = {}
        frame_sites = self._metadata.get("frame_sites", {})

        for b in self.marker_bodies:
            if b in frame_sites:
                self._site_ids[b] = self._mj_model.site(frame_sites[b]).id
            else:
                self._body_ids[b] = self._mj_model.body(b).id

        self._closure_a_id = self._mj_model.site("native_closure_a").id
        self._closure_b_id = self._mj_model.site("native_closure_b").id

        self._qpos_indices = np.array(
            [self.model._indices[name] for name in self.coordinate_order],
            dtype=np.int32,
        )

    def pose_fn(self, q: Array) -> dict[str, Pose]:
        """Compute world poses (R, t) for all bodies referenced by markers."""
        import mujoco

        q_arr = np.asarray(q, dtype=float)
        if q_arr.size != len(self.coordinate_order):
            raise ValueError(f"Expected {len(self.coordinate_order)} coordinates")

        self._mj_data.qpos[self._qpos_indices] = q_arr
        mujoco.mj_kinematics(self._mj_model, self._mj_data)

        poses: dict[str, Pose] = {}
        for b, sid in self._site_ids.items():
            r = self._mj_data.site_xmat[sid].reshape((3, 3)).copy()
            t = self._mj_data.site_xpos[sid].copy()
            poses[b] = (r, t)
        for b, bid in self._body_ids.items():
            r = self._mj_data.xmat[bid].reshape((3, 3)).copy()
            t = self._mj_data.xpos[bid].copy()
            poses[b] = (r, t)
        return poses

    def closure_residuals(self, q: Array) -> Array:
        """Evaluate position residual between dual-grip weld sites in world."""
        q_arr = np.asarray(q, dtype=float)
        if not np.array_equal(self._mj_data.qpos[self._qpos_indices], q_arr):
            import mujoco

            self._mj_data.qpos[self._qpos_indices] = q_arr
            mujoco.mj_kinematics(self._mj_model, self._mj_data)

        pa = self._mj_data.site_xpos[self._closure_a_id]
        pb = self._mj_data.site_xpos[self._closure_b_id]
        return np.asarray(pa - pb, dtype=float)


class FullBodyMarkerKinematics(BaseFullBodyIK):
    """MuJoCo marker kinematics and pose inverse kinematics adapter."""

    def __init__(
        self,
        adapter: NativeMujocoFullBodyModel,
        attachments: Mapping[str, Attachment],
    ) -> None:
        if not attachments:
            raise ValueError("At least one marker attachment is required")
        super().__init__(
            coordinate_order=tuple(adapter.coordinate_order),
            labels=tuple(attachments),
        )
        mj: Any = adapter._mj
        self._mj = mj
        self.adapter = adapter
        self.model = adapter.model
        self.data = adapter.data
        self._qpos = np.array(
            [self.model.joint(name).qposadr[0] for name in self.coordinate_order]
        )
        self._dof = np.array(
            [self.model.joint(name).dofadr[0] for name in self.coordinate_order]
        )
        sites = adapter.metadata["frame_sites"]
        self._body_ids: list[int] = []
        self._local: list[Array] = []
        for label, (body, offset) in attachments.items():
            local = np.asarray(offset, dtype=float)
            if local.shape != (3,) or not np.isfinite(local).all():
                raise ValueError(f"Marker {label} needs a finite 3-vector offset")
            if body in sites:
                site = self.model.site(sites[body]).id
                rot = np.zeros(9)
                mj.mju_quat2Mat(rot, self.model.site_quat[site])
                local = self.model.site_pos[site] + rot.reshape(3, 3) @ local
                body_id = int(self.model.site_bodyid[site])
            else:
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body)
                if body_id < 0:
                    raise ValueError(f"Marker {label} references unknown body {body}")
                frame = adapter.body_frames.get(body)
                if frame is None:
                    raise ValueError(f"Marker {label} body {body} has no joint frame")
                local = frame[:3, :3] @ local + frame[:3, 3]
            self._body_ids.append(body_id)
            self._local.append(local)
        self._closure = list(adapter._closure)
        self._spheres = {
            name: (int(info["site_id"]), float(info["radius"]))
            for name, info in adapter._spheres.items()
        }

    @property
    def nq(self) -> int:
        """Number of generalized coordinates in the model."""
        return len(self.coordinate_order)

    @property
    def sphere_names(self) -> tuple[str, ...]:
        """Names of the contact spheres carried by the model."""
        return tuple(self._spheres.keys())

    @property
    def closure_sites(self) -> tuple[str, str]:
        """Pair of site names (site_a, site_b) forming the loop closure."""
        return (self._closure[0], self._closure[1])

    @property
    def marker_bodies_and_offsets(self) -> dict[str, tuple[str, Array]]:
        """Map of marker label to (body_name, local_offset_in_mjcf_body)."""
        mj = self._mj
        model = self.model
        out: dict[str, tuple[str, Array]] = {}
        for label, body_id, local in zip(
            self.labels, self._body_ids, self._local, strict=True
        ):
            body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
            out[label] = (body_name, local.copy())
        return out

    def _set(self, q: Array) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self._qpos),) or not np.isfinite(q).all():
            raise ValueError("Coordinates must be a finite vector of model size")
        self.data.qpos[self._qpos] = q
        self._mj.mj_fwdPosition(self.model, self.data)

    def marker_positions(self, q: Array) -> Array:
        """World positions (markers, 3) of every attached marker at ``q``."""
        self._set(q)
        return self._positions()

    def _positions(self) -> Array:
        out = np.empty((len(self.labels), 3))
        for k, (body, local) in enumerate(
            zip(self._body_ids, self._local, strict=True)
        ):
            out[k] = self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ local
        return out

    def _marker_jacobian(self, positions: Array) -> Array:
        nv = self.model.nv
        jac = np.empty((len(self.labels), 3, nv))
        buffer = np.zeros((3, nv))
        for k, body in enumerate(self._body_ids):
            self._mj.mj_jac(self.model, self.data, buffer, None, positions[k], body)
            jac[k] = buffer[:, self._dof]
        return jac

    def body_poses(
        self, q: Array, bodies: Sequence[str]
    ) -> dict[str, tuple[Array, Array]]:
        """World pose ``(R, t)`` of each named spec frame or body at ``q``."""
        self._set(q)
        sites = self.adapter.metadata["frame_sites"]
        poses: dict[str, tuple[Array, Array]] = {}
        for body in bodies:
            if body in sites:
                site = self.model.site(sites[body]).id
                rotation = self.data.site_xmat[site].reshape(3, 3).copy()
                translation = self.data.site_xpos[site].copy()
            else:
                mj = self._mj
                mjt_obj = mj.mjtObj
                body_id = mj.mj_name2id(self.model, mjt_obj.mjOBJ_BODY, body)
                if body_id < 0:
                    raise ValueError(f"Unknown body {body}")
                frames = self.adapter.body_frames
                frame = frames.get(body)
                if frame is None:
                    raise ValueError(f"Body {body} has no joint frame")
                r_mj = self.data.xmat[body_id].reshape(3, 3)
                rotation = r_mj @ frame[:3, :3]
                translation = self.data.xpos[body_id] + r_mj @ frame[:3, 3]
            poses[body] = (rotation, translation)
        return poses

    def pose_fn(self, q: Array) -> dict[str, Pose]:
        """Compute world poses (R, t) for all bodies referenced by markers."""
        return self.body_poses(q, self.marker_bodies)

    def sphere_heights(self, q: Array, ground: GroundPlane) -> dict[str, float]:
        """Signed height of each contact sphere's lowest point above the plane."""
        self._set(q)
        return self._sphere_heights(ground)

    def _sphere_heights(self, ground: GroundPlane) -> dict[str, float]:
        n = _unit_ground_normal(ground)
        return {
            name: float(self.data.site_xpos[site] @ n - ground.height_m - radius)
            for name, (site, radius) in self._spheres.items()
        }

    def closure_error(self, q: Array) -> tuple[float, float]:
        """Position and orientation mismatch of the dual-grip weld at ``q``."""
        self._set(q)
        a, b = self._closure
        diff = self.data.site_xpos[a] - self.data.site_xpos[b]
        pos = math.hypot(
            diff[0], diff[1], diff[2]
        )  # ⚡ Bolt: math.hypot is ~6x faster than np.linalg.norm for small 3D vectors
        rot = _rotation_error(
            self.data.site_xmat[a].reshape(3, 3), self.data.site_xmat[b].reshape(3, 3)
        )
        return float(pos), float(
            math.hypot(rot[0], rot[1], rot[2])
        )  # ⚡ Bolt: math.hypot is ~6x faster than np.linalg.norm for small 3D vectors

    def _axis_rows(
        self,
        axis_targets: (
            Mapping[str, tuple[Sequence[float], Sequence[float], float]] | None
        ),
    ) -> list[tuple[int, Array, Array, float]]:
        sites = self.adapter.metadata["frame_sites"]
        out = []
        for frame, (body_axis, world_dir, weight) in (axis_targets or {}).items():
            if frame not in sites:
                raise ValueError(f"Unknown frame {frame}")
            a, d = _validate_axis_spec(body_axis, world_dir, weight)
            site = self.model.site(sites[frame]).id
            out.append((site, a, d, weight))
        return out

    def _append_axes(
        self,
        rows: list[Array],
        jacs: list[Array],
        axes: list[tuple[int, Array, Array, float]],
    ) -> None:
        nv = self.model.nv
        for site, axis, target, weight in axes:
            if weight <= 0:
                continue
            jr = np.zeros((3, nv))
            self._mj.mj_jacSite(self.model, self.data, None, jr, site)
            world_axis = self.data.site_xmat[site].reshape(3, 3) @ axis
            skew = np.array(
                [
                    [0.0, -world_axis[2], world_axis[1]],
                    [world_axis[2], 0.0, -world_axis[0]],
                    [-world_axis[1], world_axis[0], 0.0],
                ]
            )
            w = np.sqrt(weight)
            rows.append(w * (world_axis - target))
            jacs.append(w * (-skew @ jr)[:, self._dof])

    def _append_closure(
        self,
        rows: list[Array],
        jacs: list[Array],
        weight: float,
        rotation_weight: float | None = None,
    ) -> None:
        rotation_weight = weight if rotation_weight is None else rotation_weight
        if rotation_weight < 0:
            raise ValueError("Closure rotation weight must be nonnegative")
        if weight <= 0 and rotation_weight <= 0:
            return
        a, b = self._closure
        nv = self.model.nv
        jp_a, jr_a = np.zeros((3, nv)), np.zeros((3, nv))
        jp_b, jr_b = np.zeros((3, nv)), np.zeros((3, nv))
        self._mj.mj_jacSite(self.model, self.data, jp_a, jr_a, a)
        self._mj.mj_jacSite(self.model, self.data, jp_b, jr_b, b)
        if weight > 0:
            w = np.sqrt(weight)
            rows.append(w * (self.data.site_xpos[a] - self.data.site_xpos[b]))
            jacs.append(w * (jp_a - jp_b)[:, self._dof])
        if rotation_weight > 0:
            wr = np.sqrt(rotation_weight)
            rot = _rotation_error(
                self.data.site_xmat[a].reshape(3, 3),
                self.data.site_xmat[b].reshape(3, 3),
            )
            rows.append(wr * rot)
            jacs.append(wr * (jr_a - jr_b)[:, self._dof])

    def _append_ground(
        self,
        rows: list[Array],
        jacs: list[Array],
        ground: GroundPlane,
        weight: float,
        pinned: frozenset[str],
    ) -> None:
        if weight <= 0:
            return
        n = _unit_ground_normal(ground)
        nv = self.model.nv
        w = np.sqrt(weight)
        for name, (site, radius) in self._spheres.items():
            depth = float(self.data.site_xpos[site] @ n - ground.height_m - radius)
            if depth >= 0.0 and name not in pinned:
                continue
            jp = np.zeros((3, nv))
            self._mj.mj_jacSite(self.model, self.data, jp, None, site)
            rows.append(np.array([w * depth]))
            jacs.append(w * (n @ jp)[None, self._dof])

    def _append_anchors(
        self,
        rows: list[Array],
        jacs: list[Array],
        targets: Mapping[str, Array],
        weight: float,
    ) -> None:
        if weight <= 0 or not targets:
            return
        nv = self.model.nv
        w = np.sqrt(weight)
        for name, target in targets.items():
            site = self._spheres[name][0]
            jp = np.zeros((3, nv))
            self._mj.mj_jacSite(self.model, self.data, jp, None, site)
            rows.append(w * (self.data.site_xpos[site] - target))
            jacs.append(w * jp[:, self._dof])

    def _append_balance(
        self, rows: list[Array], jacs: list[Array], ground: GroundPlane, weight: float
    ) -> None:
        if weight <= 0:
            return
        n = _unit_ground_normal(ground)
        basis = np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, :2].T
        nv = self.model.nv
        self._mj.mj_comPos(self.model, self.data)
        jac_com = np.zeros((3, nv))
        self._mj.mj_jacSubtreeCom(self.model, self.data, jac_com, 1)
        com = self.data.subtree_com[1].copy()
        centres = np.zeros(3)
        jac_centres = np.zeros((3, nv))
        for site, _ in self._spheres.values():
            jp = np.zeros((3, nv))
            self._mj.mj_jacSite(self.model, self.data, jp, None, site)
            centres += self.data.site_xpos[site]
            jac_centres += jp
        centres /= len(self._spheres)
        jac_centres /= len(self._spheres)
        w = np.sqrt(weight)
        rows.append(w * basis @ (com - centres))
        jacs.append(w * (basis @ (jac_com - jac_centres))[:, self._dof])

    def _append_com_target(
        self,
        rows: list[Array],
        jacs: list[Array],
        ground: GroundPlane,
        goal: tuple[Array, float] | None,
    ) -> None:
        if goal is None or goal[1] <= 0:
            return
        basis = self._plane_basis(ground)
        nv = self.model.nv
        self._mj.mj_comPos(self.model, self.data)
        jac_com = np.zeros((3, nv))
        self._mj.mj_jacSubtreeCom(self.model, self.data, jac_com, 1)
        com = self.data.subtree_com[1].copy()
        w = np.sqrt(goal[1])
        rows.append(w * (basis @ com - goal[0]))
        jacs.append(w * (basis @ jac_com)[:, self._dof])

    def sphere_ground_points(self, q: Array, ground: GroundPlane) -> dict[str, Array]:
        """Projection of each contact sphere centre onto the ground plane at ``q``."""
        self._set(q)
        n = np.asarray(ground.normal, dtype=float)
        n = n / math.hypot(
            n[0], n[1], n[2]
        )  # ⚡ Bolt: math.hypot is ~6x faster than np.linalg.norm for small 3D vectors
        out = {}
        for name, (site, _) in self._spheres.items():
            c = self.data.site_xpos[site].copy()
            out[name] = c - (c @ n - ground.height_m) * n
        return out

    def com_plane_position(self, q: Array, ground: GroundPlane) -> Array:
        """Centre of mass expressed on the plane basis of ``_plane_basis``."""
        self._set(q)
        self._mj.mj_comPos(self.model, self.data)
        return np.asarray(self._plane_basis(ground) @ self.data.subtree_com[1])

    def support_offset(self, q: Array, ground: GroundPlane) -> float:
        """Distance in ground plane from CoM to sphere centroid at ``q``."""
        self._set(q)
        self._mj.mj_comPos(self.model, self.data)
        n = np.asarray(ground.normal, dtype=float)
        n = n / math.hypot(
            n[0], n[1], n[2]
        )  # ⚡ Bolt: math.hypot is ~6x faster than np.linalg.norm for small 3D vectors
        centres = np.mean(
            [self.data.site_xpos[site] for site, _ in self._spheres.values()], axis=0
        )
        offset = self.data.subtree_com[1] - centres
        offset = offset - (offset @ n) * n
        return float(
            math.hypot(offset[0], offset[1], offset[2])
        )  # ⚡ Bolt: math.hypot is ~6x faster than np.linalg.norm for small 3D vectors
