"""MuJoCo full-body model adapter with shared rigid-ground contact and rigid closure.

Applies the shared Hunt-Crossley and regularized Coulomb contact law (FB-2) via
explicit spatial wrenches on the calcaneus bodies and enforces the six-dimensional
dual-grip weld closure via an explicit rigid solve.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)
from src.engines.physics_engines.mujoco.python.native_model import (
    _compute_closure_errors,
    _evaluate_weld_closure,
    _extract_frame_poses,
    _pack_vector,
    _prepare_forward_dynamics,
    _solve_kkt_with_multipliers,
)
from src.shared.python.biomechanics.grip_extraction import (
    closing_side_from_closure,
    closure_and_club_analysis,
)
from src.shared.python.biomechanics.grip_wrench import (
    GripAnalysis,
    to_overlay_wrenches,
)
from src.shared.python.engine_core.mujoco_compat import full_mass_matrix
from src.shared.python.force_overlay.contracts import OverlayWrench
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
    torsional_friction_moment,
)


class NativeMujocoFullBodyModel:
    """Full-body MuJoCo adapter combining closed-loop kinematics and shared contact."""

    def __init__(self, model_bytes: bytes) -> None:
        mj: Any = import_module("mujoco")
        self._mj = mj
        spec = json.loads(model_bytes)
        self.xml, self.metadata = export_full_body_mjcf(model_bytes)
        self.model_sha256 = self.metadata["model_sha256"]
        # Tikhonov term added to the weld KKT solve.  The default keeps the
        # canonical pipeline unchanged; same-input parity runs set it to 0 so
        # the solve is exact like Drake's and Pinocchio's (#11606).
        self.kkt_regularization: float = 1e-6
        self.model, self.data = self._build_model(self.xml)

        self.coordinate_order: list[str] = list(self.metadata["coordinate_order"])
        self.upper_body_coordinates = int(spec["upper_body_counts"]["coordinates"])
        # MJCF bodies sit at their joint follower frame; this maps a point given
        # in the specification's body frame into the MuJoCo body frame.
        self.body_frames: dict[str, np.ndarray] = {
            joint["child"]: np.linalg.inv(np.asarray(joint["child_to_follower"], float))
            for joint in spec["joints"]
        }
        self._indices: dict[str, int] = {}
        coordinate_units: list[str] = []
        native_units = {
            int(mj.mjtJoint.mjJNT_SLIDE): "m",
            int(mj.mjtJoint.mjJNT_HINGE): "rad",
        }
        for name in self.coordinate_order:
            j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError(f"Coordinate {name} missing from MuJoCo model")
            unit = native_units.get(int(self.model.jnt_type[j]))
            if unit is None:
                raise ValueError(
                    "Coordinate units require compiled scalar slide or hinge joints"
                )
            coordinate_units.append(unit)
            self._indices[name] = int(self.model.jnt_dofadr[j])
        self._coordinate_units = tuple(coordinate_units)

        if self.model.nq != len(self._indices) or self.model.nv != len(self._indices):
            raise ValueError("All coordinates must be scalar 1-DOF joints")

        self._sites = {
            name: self.model.site(site).id
            for name, site in self.metadata["frame_sites"].items()
        }
        self._closure = [self.model.site(f"native_closure_{s}").id for s in ("a", "b")]
        self.closing_hand_side: str = closing_side_from_closure(spec["closure"])

        # Contact setup
        contact_cfg = spec["contact"]
        self.contact_parameters = ContactParameters(**contact_cfg["parameters"])
        ground_cfg = contact_cfg["ground"]
        g = np.asarray(spec["gravity_m_s2"], dtype=float)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            raise ValueError("Nonzero gravity required for opposite_gravity policy")
        unit_g = -g / g_norm
        ground_normal = (float(unit_g[0]), float(unit_g[1]), float(unit_g[2]))
        ground_height = float(
            ground_cfg["height_m"] if ground_cfg["height_m"] is not None else 0.0
        )
        self.ground_plane = GroundPlane(normal=ground_normal, height_m=ground_height)

        # Optional spin-friction patch (#11671); absent means point contacts.
        torsion = contact_cfg.get("torsion") or {}
        self._torsion_patch_m = float(torsion.get("patch_radius_m", 0.0))
        self._torsion_rate_rad_s = float(torsion.get("transition_rad_s", 0.5))

        self._spheres: dict[str, dict[str, Any]] = {}
        for sphere in contact_cfg["spheres"]:
            s_name = sphere["name"]
            site_name = self.metadata["contact_sites"][s_name]
            site_id = self.model.site(site_name).id
            body_id = int(self.model.site_bodyid[site_id])
            radius = float(sphere["radius_m"])
            self._spheres[s_name] = {
                "site_id": site_id,
                "body_id": body_id,
                "radius": radius,
            }

        self._errors: tuple[np.ndarray, np.ndarray] | None = None

    def _build_model(self, xml: str) -> tuple[Any, Any]:
        """Compile ``xml`` into the ``(model, data)`` pair the adapter drives.

        Subclasses override this to supply a model held by another runtime
        (for example MyoSuite); the MJCF is the same specification export.
        """
        model_cls = self._mj.MjModel
        model = model_cls.from_xml_string(xml)
        return model, self._mj.MjData(model)

    @property
    def coordinate_units(self) -> tuple[str, ...]:
        """Return immutable units verified from this compiled scalar model."""
        return self._coordinate_units

    def _vector(self, values: Mapping[str, float]) -> np.ndarray:
        return _pack_vector(
            self._indices,
            values,
            self.model.nv,
            error_message="Coordinates and rates must be finite",
        )

    def frame_poses(self, coordinates: Mapping[str, float]) -> dict[str, np.ndarray]:
        """Compute 4x4 forward kinematics poses for all defined frame sites."""
        self.data.qpos[:] = self._vector(coordinates)
        self._mj.mj_kinematics(self.model, self.data)
        return _extract_frame_poses(self.data, self._sites)

    def kinematic_closure_residuals(
        self, coordinates: Mapping[str, float]
    ) -> np.ndarray:
        """Return detached world-space grip separation in metres at this pose."""
        self.frame_poses(coordinates)
        a, b = self._closure
        return np.asarray(self.data.site_xpos[a] - self.data.site_xpos[b], dtype=float)

    def evaluate_contact_samples(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
    ) -> dict[str, ContactSample]:
        """Evaluate the shared contact law for all contact spheres at current state."""
        self.data.qpos[:] = self._vector(coordinates)
        self.data.qvel[:] = self._vector(rates)
        self._mj.mj_fwdPosition(self.model, self.data)

        samples: dict[str, ContactSample] = {}
        for s_name, s_info in self._spheres.items():
            site_id = s_info["site_id"]
            radius = s_info["radius"]
            pos = self.data.site_xpos[site_id].copy()

            jac_pos = np.zeros((3, self.model.nv))
            self._mj.mj_jacSite(self.model, self.data, jac_pos, None, site_id)
            vel = jac_pos @ self.data.qvel

            sample = sphere_ground_contact(
                pos, vel, radius, self.ground_plane, self.contact_parameters
            )
            samples[s_name] = sample

        return samples

    def generalized_forces(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, ContactSample]]:
        """Return (bias, contact, samples) so that ``M a = effort + contact - bias``.

        ``bias`` is MuJoCo's Coriolis, centrifugal and gravity term; ``contact``
        is the shared law's spatial forces mapped through the sphere Jacobians.
        Postcondition: the model state is left at ``(coordinates, rates)`` with
        position and velocity stages computed.
        """
        mj, model, data = self._mj, self.model, self.data
        _prepare_forward_dynamics(
            mj, model, data, self._vector(coordinates), self._vector(rates)
        )
        samples = self.evaluate_contact_samples(coordinates, rates)
        mj.mj_fwdVelocity(model, data)
        data.xfrc_applied[:] = 0.0
        tau_contact = np.zeros(model.nv)
        for s_name, sample in samples.items():
            f_contact = sample.normal_force_n + sample.friction_force_n
            if np.linalg.norm(f_contact) <= 0.0:
                continue
            s_info = self._spheres[s_name]
            site_id = s_info["site_id"]
            body_id = s_info["body_id"]
            p_contact = data.site_xpos[site_id]
            r = p_contact - data.xipos[body_id]
            torque = np.cross(r, f_contact)
            data.xfrc_applied[body_id] += np.concatenate([f_contact, torque])

            jac_pos = np.zeros((3, model.nv))
            jac_rot = np.zeros((3, model.nv)) if self._torsion_patch_m > 0 else None
            mj.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
            tau_contact += jac_pos.T @ f_contact
            if jac_rot is not None:
                moment = torsional_friction_moment(
                    sample.normal_force_n,
                    jac_rot @ data.qvel,
                    self.ground_plane,
                    patch_radius_m=self._torsion_patch_m,
                    friction=self.contact_parameters.dynamic_friction,
                    transition_rad_s=self._torsion_rate_rad_s,
                )
                data.xfrc_applied[body_id, 3:] += moment
                tau_contact += jac_rot.T @ moment
        return data.qfrc_bias.copy(), tau_contact, samples

    def _constrained_solve(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shared KKT solve: dense ``(qacc, lambda)``; leaves ``data`` at the state."""
        mj, model, data = self._mj, self.model, self.data
        effort = self._vector(primitive_efforts)
        bias, tau_contact, _ = self.generalized_forces(coordinates, rates)

        mass = full_mass_matrix(mj, model, data)

        jac, drift = _evaluate_weld_closure(mj, model, data, self._closure)
        total_effort = effort + tau_contact - bias
        acceleration, multiplier = _solve_kkt_with_multipliers(
            mass, total_effort, jac, drift, regularization=self.kkt_regularization
        )

        if not np.isfinite(acceleration).all():
            raise FloatingPointError("Nonfinite full-body MuJoCo acceleration")

        self._errors = _compute_closure_errors(data, self._closure, jac)
        return acceleration, multiplier

    def _as_named(self, acceleration: np.ndarray) -> dict[str, float]:
        return {
            name: float(acceleration[index]) for name, index in self._indices.items()
        }

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Solve constrained dynamics with applied contact forces and dual-grip weld."""
        acceleration, _ = self._constrained_solve(coordinates, rates, primitive_efforts)
        return self._as_named(acceleration)

    def solve_with_multipliers(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> tuple[dict[str, float], np.ndarray]:
        """Constrained accelerations plus the 6-D weld multiplier (GCV-8, #11714).

        The accelerations are exactly those of :meth:`accelerations` (same
        solve, no extra arithmetic on them).  The multiplier ``lambda`` is
        ordered ``[force; torque]`` in the world frame: the closure applies
        ``+lambda`` at closure site ``a`` and ``-lambda`` at closure site ``b``
        (the club-side site), so the wrench the closing hand exerts ON THE CLUB
        is ``-lambda`` at site ``b``.  The holding hand (carried by the tree,
        no multiplier) follows from club Newton-Euler; see :meth:`grip_analysis`.
        """
        acceleration, multiplier = self._constrained_solve(
            coordinates, rates, primitive_efforts
        )
        return self._as_named(acceleration), multiplier.copy()

    def _club_newton_euler(self, acceleration: np.ndarray) -> dict[str, Any]:
        """Club kinematics for the Newton-Euler keywords at the last solve state."""
        mj, model, data = self._mj, self.model, self.data
        body = int(model.site_bodyid[self._closure[1]])
        com = np.array(data.xipos[body])
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        dotp, dotr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mj.mj_jacBodyCom(model, data, jacp, jacr, body)
        mj.mj_jacDot(model, data, dotp, dotr, com, body)
        rot = np.array(data.ximat[body]).reshape(3, 3)
        return {
            "mass_kg": float(model.body_mass[body]),
            "gravity_m_s2": np.array(model.opt.gravity),
            "com_m": com,
            "com_acceleration_m_s2": jacp @ acceleration + dotp @ data.qvel,
            "inertia_world_kg_m2": rot @ np.diag(model.body_inertia[body]) @ rot.T,
            "angular_velocity_rad_s": jacr @ data.qvel,
            "angular_acceleration_rad_s2": jacr @ acceleration + dotr @ data.qvel,
        }

    def grip_analysis(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> GripAnalysis:
        """Per-hand grip wrenches ON THE CLUB from the KKT multiplier (GCV-8).

        The closing hand (``closing_hand_side``) exerts ``-lambda`` at the
        club-side closure site.  The holding hand is carried by the kinematic
        tree and has no multiplier; its wrench follows from club Newton-Euler,
        ``F_hold = m (a_c - g) - F_close`` and ``tau_hold`` likewise about the
        club centre of mass (see
        :func:`~src.shared.python.biomechanics.grip_extraction.holding_hand_wrench`).
        The holding grip point is the club body origin (the wrist-joint
        follower frame); the split between hands is solver-defined and is
        recorded as ``split_method="constraint_multiplier"``.
        """
        acceleration, multiplier = self._constrained_solve(
            coordinates, rates, primitive_efforts
        )
        site_b = self._closure[1]
        body = int(self.model.site_bodyid[site_b])
        return closure_and_club_analysis(
            closing_side=self.closing_hand_side,
            closing_point_m=self.data.site_xpos[site_b],
            closing_force_n=-multiplier[:3],
            closing_torque_nm=-multiplier[3:],
            holding_point_m=self.data.xpos[body],
            club=self._club_newton_euler(acceleration),
            split_method="constraint_multiplier",
            metadata={
                "engine": "mujoco",
                "kkt_regularization": self.kkt_regularization,
            },
        )

    def grip_overlay_wrenches(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> list[OverlayWrench]:
        """``WrenchKind.GRIP`` overlay wrenches for the current state (ADR-0052)."""
        analysis = self.grip_analysis(coordinates, rates, primitive_efforts)
        return to_overlay_wrenches(analysis, source="mujoco:kkt_multiplier")

    def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return detached pose/rate residuals from the last acceleration call."""
        if self._errors is None:
            raise ValueError("Evaluate acceleration before closure errors")
        return self._errors[0].copy(), self._errors[1].copy()

    def evaluate_weld_closure(self) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate weld closure Jacobian and drift for dual-grip constraint."""
        return _evaluate_weld_closure(self._mj, self.model, self.data, self._closure)
