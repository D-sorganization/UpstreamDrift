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
    _solve_kkt_dynamics,
)
from src.shared.python.motion_matching.contact_law import (
    ContactSample,
    contact_setup_from_spec,
    sphere_ground_contact,
)


class NativeMujocoFullBodyModel:
    """Full-body MuJoCo adapter combining closed-loop kinematics and shared contact."""

    def __init__(self, model_bytes: bytes) -> None:
        mj: Any = import_module("mujoco")
        self._mj = mj
        spec = json.loads(model_bytes)
        self.xml, self.metadata = export_full_body_mjcf(model_bytes)
        self.model_sha256 = self.metadata["model_sha256"]
        self.model = mj.MjModel.from_xml_string(self.xml)
        self.data = mj.MjData(self.model)

        self.coordinate_order: list[str] = list(self.metadata["coordinate_order"])
        self._indices: dict[str, int] = {}
        for name in self.coordinate_order:
            j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError(f"Coordinate {name} missing from MuJoCo model")
            self._indices[name] = int(self.model.jnt_dofadr[j])

        if self.model.nq != len(self._indices) or self.model.nv != len(self._indices):
            raise ValueError("All coordinates must be scalar 1-DOF joints")

        self._sites = {
            name: self.model.site(site).id
            for name, site in self.metadata["frame_sites"].items()
        }
        self._closure = [self.model.site(f"native_closure_{s}").id for s in ("a", "b")]

        # Contact setup
        contact_cfg = spec["contact"]
        self.contact_parameters, self.ground_plane = contact_setup_from_spec(spec)

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

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Solve constrained dynamics with applied contact forces and dual-grip weld."""
        mj, model, data = self._mj, self.model, self.data
        effort = self._vector(primitive_efforts)
        _prepare_forward_dynamics(
            mj, model, data, self._vector(coordinates), self._vector(rates)
        )

        # Evaluate and apply contact forces
        samples = self.evaluate_contact_samples(coordinates, rates)
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
            mj.mj_jacSite(model, data, jac_pos, None, site_id)
            tau_contact += jac_pos.T @ f_contact

        mass = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, mass, data.qM)

        jac, drift = _evaluate_weld_closure(mj, model, data, self._closure)
        total_effort = effort + tau_contact - data.qfrc_bias
        acceleration = _solve_kkt_dynamics(mass, total_effort, jac, drift)

        if not np.isfinite(acceleration).all():
            raise FloatingPointError("Nonfinite full-body MuJoCo acceleration")

        self._errors = _compute_closure_errors(data, self._closure, jac)

        return {
            name: float(acceleration[index]) for name, index in self._indices.items()
        }

    def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return detached pose/rate residuals from the last acceleration call."""
        if self._errors is None:
            raise ValueError("Evaluate acceleration before closure errors")
        return self._errors[0].copy(), self._errors[1].copy()
