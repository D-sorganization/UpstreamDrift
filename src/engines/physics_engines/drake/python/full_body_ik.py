"""Drake full-body inverse kinematics and marker tracking adapter (FB-4).

Supplies forward kinematics (``pose_fn``), dual-grip weld loop-closure residuals,
least-squares inverse kinematics, and marker kinematics with ground support in Drake.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.drake.python.full_body_model import (
    FullBodyDrakeModel,
)
from src.shared.python.motion_matching import posture_metrics as post
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import (
    BaseFullBodyIK,
    _resolve_closure_weights,
    _rotation_error,
    _skew3,
    _validate_axis_spec,
)
from src.shared.python.motion_matching.marker_calibration import Pose
from src.shared.python.motion_matching.pipeline.constants import (
    FORWARD_AXIS,
    RIGHT_AXIS,
    UP_AXIS,
)

Array: TypeAlias = NDArray[np.float64]
Attachment = tuple[str, Sequence[float]]


class DrakeFullBodyIK(BaseFullBodyIK):
    """Full-body Drake inverse kinematics adapter consuming a full-body spec."""

    def __init__(
        self,
        specification: Mapping[str, Any] | bytes | str,
        attachments: Mapping[str, Attachment] | None = None,
    ) -> None:
        super().__init__(specification)
        self.model = FullBodyDrakeModel(self.specification)
        self.adapter = self.model
        self._plant = self.model.plant
        self._context = self.model.context
        self._metadata = self.model.metadata
        self._closure = self.model._closure
        self._world_frame = self._plant.world_frame()
        self.coordinate_order: tuple[str, ...] = tuple(self.model.names)
        if len(self.coordinate_order) not in (41, 44):
            raise ValueError(
                f"Expected 41 or 44 full-body coordinates, got {len(self.coordinate_order)}"
            )
        self.upper_body_coordinates: int = int(
            self.specification.get("upper_body_counts", {}).get("coordinates", 0)
        )
        self.model.upper_body_coordinates = self.upper_body_coordinates

        model = self.model
        spheres_dict = model._spheres
        self._spheres: dict[str, tuple[Any, float]] = {
            s_name: (info["frame"], float(info["radius_m"]))
            for s_name, info in spheres_dict.items()
        }

        self._init_attachments(attachments)

    def _init_attachments(self, attachments: Mapping[str, Attachment] | None) -> None:
        if attachments is not None:
            self._marker_info: dict[str, tuple[str, Array]] = {
                k: (b, np.asarray(off, dtype=float))
                for k, (b, off) in attachments.items()
            }
        else:
            raw = self.specification.get("marker_attachments", {})
            self._marker_info = {
                k: (v["body"], np.asarray(v["offset"], dtype=float))
                for k, v in raw.items()
            }
        self.labels = tuple(self._marker_info.keys())
        self.marker_bodies = sorted({b for b, _ in self._marker_info.values()})

        self._marker_frames: dict[str, tuple[Any, Array]] = {}
        for label, (body, offset) in self._marker_info.items():
            if body in self.model._frames:
                frame = self.model._frames[body]
            elif body in self._metadata["body_links"]:
                link_name = self._metadata["body_links"][body]
                body_obj = self._plant.GetBodyByName(link_name, self.model._instance)
                frame = body_obj.body_frame()
            else:
                raise ValueError(f"Body/frame {body} not found in Drake model")
            self._marker_frames[label] = (frame, offset)

    @property
    def nq(self) -> int:
        return len(self.coordinate_order)

    @property
    def sphere_names(self) -> tuple[str, ...]:
        return tuple(self._spheres.keys())

    @property
    def closure_sites(self) -> tuple[str, str]:
        return ("native_closure_a", "native_closure_b")

    @property
    def marker_bodies_and_offsets(self) -> dict[str, tuple[str, Array]]:
        return {
            label: (body, offset.copy())
            for label, (body, offset) in self._marker_info.items()
        }

    def _set(self, q: Array) -> None:
        q_arr = np.asarray(q, dtype=float)
        if q_arr.shape != (len(self.coordinate_order),) or not np.isfinite(q_arr).all():
            raise ValueError("Coordinates must be a finite vector of model size")
        self._plant.SetPositions(self._context, q_arr)

    def marker_positions(self, q: Array) -> Array:
        self._set(q)
        return self._positions()

    def _positions(self) -> Array:
        out = np.empty((len(self.labels), 3), dtype=float)
        for k, label in enumerate(self.labels):
            frame, offset = self._marker_frames[label]
            x_wf = self._plant.CalcRelativeTransform(
                self._context, self._world_frame, frame
            )
            out[k] = x_wf.rotation().matrix() @ offset + x_wf.translation()
        return out

    def _marker_jacobian(self, positions: Array) -> Array:
        nv = len(self.coordinate_order)
        jac = np.empty((len(self.labels), 3, nv), dtype=float)
        for k, label in enumerate(self.labels):
            frame, offset = self._marker_frames[label]
            jac[k] = self._plant.CalcJacobianTranslationalVelocity(
                self._context,
                self.model._wrt_v,
                frame,
                offset,
                self._world_frame,
                self._world_frame,
            )
        return jac

    def body_poses(
        self, q: Array, bodies: Sequence[str]
    ) -> dict[str, tuple[Array, Array]]:
        self._set(q)
        poses: dict[str, tuple[Array, Array]] = {}
        for b in bodies:
            if b in self.model._frames:
                x_wf = self._plant.CalcRelativeTransform(
                    self._context, self._world_frame, self.model._frames[b]
                )
                poses[b] = (x_wf.rotation().matrix().copy(), x_wf.translation().copy())
            elif b in self._metadata["body_links"]:
                link_name = self._metadata["body_links"][b]
                body_obj = self._plant.GetBodyByName(link_name, self.model._instance)
                x_wb = self._plant.EvalBodyPoseInWorld(self._context, body_obj)
                poses[b] = (x_wb.rotation().matrix().copy(), x_wb.translation().copy())
            else:
                raise ValueError(f"Body/frame {b} not found in Drake model")
        return poses

    def pose_fn(self, q: Array) -> dict[str, Pose]:
        return self.body_poses(q, self.marker_bodies)

    def _closure_transforms(self) -> tuple[Any, Any]:
        return (
            self._plant.CalcRelativeTransform(
                self._context, self._world_frame, self._closure[0]
            ),
            self._plant.CalcRelativeTransform(
                self._context, self._world_frame, self._closure[1]
            ),
        )

    def closure_residuals(self, q: Array) -> Array:
        self._set(q)
        x_wa, x_wb = self._closure_transforms()
        return np.asarray(x_wa.translation() - x_wb.translation(), dtype=float)

    def closure_error(self, q: Array) -> tuple[float, float]:
        self._set(q)
        x_wa, x_wb = self._closure_transforms()
        pos = np.linalg.norm(x_wa.translation() - x_wb.translation())
        rot = _rotation_error(x_wa.rotation().matrix(), x_wb.rotation().matrix())
        return float(pos), float(np.linalg.norm(rot))

    def sphere_heights(self, q: Array, ground: GroundPlane) -> dict[str, float]:
        self._set(q)
        return self._sphere_heights(ground)

    def _sphere_heights(self, ground: GroundPlane) -> dict[str, float]:
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        return {
            name: float(
                self._plant.CalcRelativeTransform(
                    self._context, self._world_frame, frame
                ).translation()
                @ n
                - ground.height_m
                - radius
            )
            for name, (frame, radius) in self._spheres.items()
        }

    def sphere_ground_points(self, q: Array, ground: GroundPlane) -> dict[str, Array]:
        self._set(q)
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        out = {}
        for name, (frame, _) in self._spheres.items():
            c = np.asarray(
                self._plant.CalcRelativeTransform(
                    self._context, self._world_frame, frame
                ).translation(),
                dtype=float,
            )
            out[name] = c - (c @ n - ground.height_m) * n
        return out

    def _axis_rows(
        self,
        axis_targets: (
            Mapping[str, tuple[Sequence[float], Sequence[float], float]] | None
        ),
    ) -> list[tuple[Any, Array, Array, float]]:
        frames = self.model._frames
        out = []
        for frame, (body_axis, world_dir, weight) in (axis_targets or {}).items():
            if frame not in frames:
                raise ValueError(f"Unknown frame {frame}")
            a, d = _validate_axis_spec(body_axis, world_dir, weight)
            frame_obj = frames[frame]
            out.append((frame_obj, a, d, weight))
        return out

    def _append_axes(
        self,
        rows: list[Array],
        jacs: list[Array],
        axes: list[tuple[Any, Array, Array, float]],
    ) -> None:
        for frame_obj, axis, target, weight in axes:
            if weight <= 0:
                continue
            x_wf = self._plant.CalcRelativeTransform(
                self._context, self._world_frame, frame_obj
            )
            world_axis = x_wf.rotation().matrix() @ axis
            jr = self._plant.CalcJacobianAngularVelocity(
                self._context,
                self.model._wrt_v,
                frame_obj,
                self._world_frame,
                self._world_frame,
            )
            skew = _skew3(world_axis)
            w = np.sqrt(weight)
            rows.append(w * (world_axis - target))
            jacs.append(w * (-skew @ jr))

    def _append_closure(
        self,
        rows: list[Array],
        jacs: list[Array],
        weight: float,
        rotation_weight: float | None = None,
    ) -> None:
        pos_weight, rot_weight, skip = _resolve_closure_weights(weight, rotation_weight)
        if skip:
            return
        a, b = self._closure
        x_wa = self._plant.CalcRelativeTransform(self._context, self._world_frame, a)
        x_wb = self._plant.CalcRelativeTransform(self._context, self._world_frame, b)
        jp_a = self._plant.CalcJacobianTranslationalVelocity(
            self._context,
            self.model._wrt_v,
            a,
            [0, 0, 0],
            self._world_frame,
            self._world_frame,
        )
        jp_b = self._plant.CalcJacobianTranslationalVelocity(
            self._context,
            self.model._wrt_v,
            b,
            [0, 0, 0],
            self._world_frame,
            self._world_frame,
        )
        if pos_weight > 0:
            w = np.sqrt(pos_weight)
            rows.append(w * (x_wa.translation() - x_wb.translation()))
            jacs.append(w * (jp_a - jp_b))
        if rot_weight > 0:
            wr = np.sqrt(rot_weight)
            rot = _rotation_error(x_wa.rotation().matrix(), x_wb.rotation().matrix())
            jr_a = self._plant.CalcJacobianAngularVelocity(
                self._context,
                self.model._wrt_v,
                a,
                self._world_frame,
                self._world_frame,
            )
            jr_b = self._plant.CalcJacobianAngularVelocity(
                self._context,
                self.model._wrt_v,
                b,
                self._world_frame,
                self._world_frame,
            )
            rows.append(wr * rot)
            jacs.append(wr * (jr_a - jr_b))

    def _frame_pos_and_jac(self, frame: Any) -> tuple[Array, Array]:
        pos = self._plant.CalcRelativeTransform(
            self._context, self._world_frame, frame
        ).translation()
        jp = self._plant.CalcJacobianTranslationalVelocity(
            self._context,
            self.model._wrt_v,
            frame,
            [0, 0, 0],
            self._world_frame,
            self._world_frame,
        )
        return pos, jp

    def _com_and_jac(self) -> tuple[Array, Array]:
        com = np.asarray(
            self._plant.CalcCenterOfMassPositionInWorld(self._context), dtype=float
        )
        jac_com = self._plant.CalcJacobianCenterOfMassTranslationalVelocity(
            self._context,
            self.model._wrt_v,
            self._world_frame,
            self._world_frame,
        )
        return com, jac_com

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
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        w = np.sqrt(weight)
        for name, (frame, radius) in self._spheres.items():
            pos, jp = self._frame_pos_and_jac(frame)
            depth = float(pos @ n - ground.height_m - radius)
            if depth >= 0.0 and name not in pinned:
                continue
            rows.append(np.array([w * depth]))
            jacs.append(w * (n @ jp)[None, :])

    def _append_anchors(
        self,
        rows: list[Array],
        jacs: list[Array],
        targets: Mapping[str, Array],
        weight: float,
    ) -> None:
        if weight <= 0 or not targets:
            return
        w = np.sqrt(weight)
        for name, target in targets.items():
            frame = self._spheres[name][0]
            pos, jp = self._frame_pos_and_jac(frame)
            rows.append(w * (pos - target))
            jacs.append(w * jp)

    def _append_balance(
        self, rows: list[Array], jacs: list[Array], ground: GroundPlane, weight: float
    ) -> None:
        if weight <= 0:
            return
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        basis = np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, :2].T
        nv = len(self.coordinate_order)
        com, jac_com = self._com_and_jac()
        centres = np.zeros(3)
        jac_centres = np.zeros((3, nv))
        for frame, _ in self._spheres.values():
            pos, jp = self._frame_pos_and_jac(frame)
            centres += pos
            jac_centres += jp
        centres /= len(self._spheres)
        jac_centres /= len(self._spheres)
        w = np.sqrt(weight)
        rows.append(w * basis @ (com - centres))
        jacs.append(w * (basis @ (jac_com - jac_centres)))

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
        com, jac_com = self._com_and_jac()
        w = np.sqrt(goal[1])
        rows.append(w * (basis @ com - goal[0]))
        jacs.append(w * (basis @ jac_com))

    def com_plane_position(self, q: Array, ground: GroundPlane) -> Array:
        self._set(q)
        com, _ = self._com_and_jac()
        return np.asarray(self._plane_basis(ground) @ com)

    def support_offset(self, q: Array, ground: GroundPlane) -> float:
        self._set(q)
        com, _ = self._com_and_jac()
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        centres = np.mean(
            [
                self._plant.CalcRelativeTransform(
                    self._context, self._world_frame, frame
                ).translation()
                for frame, _ in self._spheres.values()
            ],
            axis=0,
        )
        offset = com - centres
        offset = offset - (offset @ n) * n
        return float(np.linalg.norm(offset))

    def posture_summary(self, q: Array) -> dict[str, Any]:
        self._set(q)

        def site_pos(frame_name: str) -> Array:
            frame = self.model._frames[frame_name]
            return np.asarray(
                self._plant.CalcRelativeTransform(
                    self._context, self._world_frame, frame
                ).translation(),
                dtype=float,
            )

        p_r = self._plant.CalcRelativeTransform(
            self._context,
            self._world_frame,
            self.model._joints["hip_flexion_r"].frame_on_parent(),
        ).translation()
        p_l = self._plant.CalcRelativeTransform(
            self._context,
            self._world_frame,
            self.model._joints["hip_flexion_l"].frame_on_parent(),
        ).translation()
        hips = np.asarray((p_r + p_l) / 2.0, dtype=float)

        spine = site_pos("Spine")
        hub = site_pos("Hub")
        bend = post.spine_bend(
            spine - hips, hub - spine, UP_AXIS, FORWARD_AXIS, RIGHT_AXIS
        )
        links = {}
        for side in ("L", "R"):
            v = site_pos(f"{side}S") - hub
            links[side] = float(np.degrees(np.arcsin(-v[2] / np.linalg.norm(v))))
        return {
            "spine_bend_deg": bend.__dict__,
            "clavicle_link_below_horizontal_deg": links,
            "hips_to_shoulder_centre_m": float(
                np.linalg.norm((site_pos("LS") + site_pos("RS")) / 2.0 - hips)
            ),
        }
