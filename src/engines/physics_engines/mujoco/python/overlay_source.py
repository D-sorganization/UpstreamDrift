"""MuJoCo-backed contact and kinematics source for bundle overlays (NV-3, #11676).

The overlay wrenches (joint torques, ground reaction, weight) depend only on
the specification and a pose, so one MuJoCo evaluation of the specification
export serves every engine's native viewer; it is the same model the shared
contact law is validated on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.mujoco_axial_loads import (
    BodyWrench,
    ExternalWrench,
    GripClosure,
    MujocoAxialLoadSource,
)
from src.shared.python.force_overlay.bundle_provider import JointFrame

Vec3 = tuple[float, float, float]


class MujocoOverlaySource(NativeMujocoFullBodyModel):
    """Full-body MuJoCo adapter implementing ``ContactSource`` and ``KinematicsSource``."""

    def __init__(self, model_bytes: bytes) -> None:
        super().__init__(model_bytes)
        self.total_mass_kg: float = float(self.model.body_subtreemass[0])
        self._axial_source: MujocoAxialLoadSource | None = None

    def axial_load_frame(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        accelerations: Mapping[str, float],
        time_s: float,
        external_wrenches: Sequence[ExternalWrench] | None = None,
        closure: GripClosure | None = None,
    ) -> AxialLoadFrame | None:
        """Segment axial loads of a prescribed motion (inverse dynamics).

        Uses the ``kinematic`` axis policy (every body with a defined axis).
        Postcondition: tension positive, keyed by MuJoCo body name; ``None`` when
        the model has tendons, plugins or callbacks. The ground reaction comes from
        the shared contact law (``external_wrenches``), not MuJoCo contacts, which a
        prescribed pose lacks; see ``MujocoAxialLoadSource.sample_prescribed``.
        ``closure`` (from :meth:`closure_wrench`) is the physical grip-weld
        wrench; without it the segments on the grip loop report ``None``.
        """
        if self._axial_source is None:
            self._axial_source = MujocoAxialLoadSource(
                self.model, axis_policy="kinematic"
            )
        if external_wrenches is not None:
            external_wrenches = self.on_forward_line_of_action(
                external_wrenches, coordinates, rates
            )
        return self._axial_source.sample_prescribed(
            self._vector(coordinates),
            self._vector(rates),
            self._vector(accelerations),
            time_s,
            external_wrenches,
            closure,
        )

    def on_forward_line_of_action(
        self,
        wrenches: Sequence[ExternalWrench],
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
    ) -> list[ExternalWrench]:
        """Ground wrenches re-expressed where the forward model applies them.

        The overlay draws each contact force at its contact point, but the
        forward dynamics applies it at the contact sphere centre (its force
        line passes through the centre; no couple is added). A wrench whose
        point is a sphere's contact point is moved to that centre, force and
        free moment unchanged, so the inverse reproduces the forward solve.
        Other wrenches pass through. Only the inverse uses this; the drawn
        wrench keeps its contact point.
        """
        samples = self.evaluate_contact_samples(coordinates, rates)
        self._forward(coordinates)
        centres = [
            (
                np.asarray(sample.contact_point_m, float),
                self.data.site_xpos[self._spheres[name]["site_id"]].copy(),
            )
            for name, sample in samples.items()
        ]
        out: list[ExternalWrench] = []
        for wrench in wrenches:
            point = np.asarray(wrench.point_m, float)
            hit = next(
                (c for p, c in centres if np.allclose(p, point, atol=1e-9)), None
            )
            if hit is None or wrench.force_n is None:
                out.append(wrench)
                continue
            out.append(
                BodyWrench(
                    wrench.body,
                    (float(hit[0]), float(hit[1]), float(hit[2])),
                    tuple(float(x) for x in wrench.force_n),  # type: ignore[arg-type]
                    None
                    if wrench.torque_nm is None
                    else tuple(float(x) for x in wrench.torque_nm),  # type: ignore[arg-type]
                )
            )
        return out

    def closure_wrench(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float] | None,
    ) -> GripClosure:
        """Physical grip-weld closure wrench at the state, world frame (#13179).

        The 6-D multiplier ``lambda`` of the explicit KKT solve given the applied
        efforts: ``+lambda`` acts on the hand-side closure body at its site and
        ``-lambda`` on the club at its site (``GripClosure.wrenches``;
        ``multiplier_world`` is ``[force; torque]``). This is the public accessor
        for consumers that transport the grip load (e.g. onto a palm patch).

        Unavailable (``GripClosure.available`` False, with a ``reason``; never a
        zero wrench) when ``primitive_efforts`` is ``None`` or incomplete or the
        solve fails. Postcondition: the model state is left at the given state.
        """
        if primitive_efforts is None:
            return GripClosure.unavailable("no applied efforts for the closure solve")
        try:
            _, multiplier = self.solve_with_multipliers(
                coordinates, rates, primitive_efforts
            )
            site_a, site_b = self._closure
            mj, model = self._mj, self.model
            body_a, body_b = (
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.site_bodyid[s]))
                for s in (site_a, site_b)
            )
            return GripClosure.from_multiplier(
                body_a=body_a,
                body_b=body_b,
                point_a_m=self.data.site_xpos[site_a],
                point_b_m=self.data.site_xpos[site_b],
                multiplier=multiplier,
            )
        except (KeyError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            return GripClosure.unavailable(f"closure solve failed: {exc}")

    def club_frame_world(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World rotation (3x3) and origin of the *specification* club frame.

        This is the frame the club geometry, the grip frames and the palm-thumb
        patch are expressed in (``closure.placement_b`` is a pose in it). It is
        not the MJCF body frame, which the export re-centres: the weld site's
        world pose is ``club_frame @ placement_b``, so the club frame is
        ``site_pose @ inv(placement_b)``. Read-only (#13189).
        """
        self._forward(coordinates, rates)
        site = self._closure[1]
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(self.data.site_xmat[site], dtype=float).reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[site]
        placement = np.asarray(
            self._spec_for_partition["closure"]["placement_b"], dtype=float
        )
        frame = pose @ np.linalg.inv(placement)
        return frame[:3, :3].copy(), frame[:3, 3].copy()

    def _forward(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float] | None = None,
    ) -> None:
        self.data.qpos[:] = self._vector(coordinates)
        if rates is not None:
            self.data.qvel[:] = self._vector(rates)
            self._mj.mj_forward(self.model, self.data)
        else:
            self._mj.mj_fwdPosition(self.model, self.data)

    def bias_force(
        self,
        coordinates: Mapping[str, float] | np.ndarray,
        rates: Mapping[str, float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Generalised bias force vector h(q, v) = C(q, v)v + g(q) [N or N*m]."""
        if isinstance(coordinates, (np.ndarray, Sequence)) and not isinstance(
            coordinates, (str, bytes, Mapping)
        ):
            coords_map: Mapping[str, float] = dict(
                zip(
                    self.coordinate_order,
                    np.asarray(coordinates, dtype=float).tolist(),
                    strict=True,
                )
            )
        else:
            coords_map = coordinates
        if (
            rates is not None
            and isinstance(rates, (np.ndarray, Sequence))
            and not isinstance(rates, (str, bytes, Mapping))
        ):
            rates_map: Mapping[str, float] | None = dict(
                zip(
                    self.coordinate_order,
                    np.asarray(rates, dtype=float).tolist(),
                    strict=True,
                )
            )
        else:
            rates_map = rates
        self._forward(coords_map, rates_map)
        return np.asarray(self.data.qfrc_bias, dtype=float).copy()

    def point_jdot_v(
        self,
        point: Sequence[float] | np.ndarray,
        body: str | int,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """3-vector J̇(q, v) v [m/s^2] at ``point`` on ``body``."""
        self._forward(coordinates, rates)
        body_id = body if isinstance(body, int) else int(self.model.body(body).id)
        pt = np.asarray(point, dtype=float)
        jdot_pos = np.zeros((3, self.model.nv))
        jdot_rot = np.zeros((3, self.model.nv))
        self._mj.mj_jacDot(self.model, self.data, jdot_pos, jdot_rot, pt, body_id)
        return jdot_pos @ self.data.qvel

    def joint_frames(
        self, coordinates: Mapping[str, float]
    ) -> Mapping[str, JointFrame]:
        """World anchor and axis of every hinge coordinate at ``coordinates``."""
        self._forward(coordinates)
        mj: Any = self._mj
        out: dict[str, JointFrame] = {}
        for name in self.coordinate_order:
            j = self.model.joint(name).id
            if int(self.model.jnt_type[j]) != int(mj.mjtJoint.mjJNT_HINGE):
                continue
            body = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_BODY, int(self.model.jnt_bodyid[j])
            )
            anchor = self.data.xanchor[j]
            axis = self.data.xaxis[j]
            out[name] = JointFrame(
                body or "body",
                (float(anchor[0]), float(anchor[1]), float(anchor[2])),
                (float(axis[0]), float(axis[1]), float(axis[2])),
            )
        return out

    def center_of_mass_m(self, coordinates: Mapping[str, float]) -> Vec3:
        """System centre of mass in the world frame."""
        self._forward(coordinates)
        com = np.asarray(self.data.subtree_com[0], dtype=float)
        return (float(com[0]), float(com[1]), float(com[2]))
