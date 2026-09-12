"""MuJoCo native tree with an explicit rigid six-dimensional grip solve.

Stock mj_step uses compliant equality constraints and is not this adapter's
execution path. Mass, bias, kinematics and Jacobian derivatives come from MuJoCo.
"""

from collections.abc import Mapping
from importlib import import_module
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.mujoco.python.native_mjcf import export_native_mjcf


class NativeMujocoModel:
    """Match the public native acceleration/frame protocol without feedback."""

    def __init__(self, model_bytes: bytes) -> None:
        mj: Any = import_module("mujoco")
        self._mj = mj
        self.xml, self.metadata = export_native_mjcf(model_bytes)
        self.model_sha256 = self.metadata["model_sha256"]
        self.model = mj.MjModel.from_xml_string(self.xml)
        self.data = mj.MjData(self.model)
        self._indices = {}
        for name in self.metadata["coordinate_order"]:
            j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError("Native coordinate missing from MuJoCo")
            self._indices[name] = int(self.model.jnt_dofadr[j])
        if self.model.nq != len(self._indices) or self.model.nv != len(self._indices):
            raise ValueError("Native coordinates must remain scalar")
        self._sites = {
            name: self.model.site(site).id
            for name, site in self.metadata["frame_sites"].items()
        }
        self._closure = [self.model.site(f"native_closure_{s}").id for s in ("a", "b")]
        self._errors: tuple[np.ndarray, np.ndarray] | None = None

    def _vector(self, values: Mapping[str, float]) -> np.ndarray:
        if set(values) != set(self._indices):
            raise ValueError("Provide exactly the native coordinate inventory")
        result = np.zeros(self.model.nv)
        for name, index in self._indices.items():
            result[index] = values[name]
        if not np.isfinite(result).all():
            raise ValueError("Native state and effort must be finite")
        return result

    def frame_poses(self, coordinates: Mapping[str, float]) -> dict[str, np.ndarray]:
        self.data.qpos[:] = self._vector(coordinates)
        self._mj.mj_kinematics(self.model, self.data)
        result = {}
        for name, index in self._sites.items():
            pose = np.eye(4)
            pose[:3, :3] = self.data.site_xmat[index].reshape(3, 3)
            pose[:3, 3] = self.data.site_xpos[index]
            result[name] = pose
        return result

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Solve M a - J.T lambda = effort-bias, J a = -Jdot v."""
        mj, model, data = self._mj, self.model, self.data
        data.qpos[:] = self._vector(coordinates)
        data.qvel[:] = self._vector(rates)
        effort = self._vector(primitive_efforts)
        mj.mj_fwdPosition(model, data)
        mj.mj_fwdVelocity(model, data)
        mass = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, mass, data.qM)
        jacobians, derivatives = [], []
        for site in self._closure:
            jac, derivative = np.zeros((6, model.nv)), np.zeros((6, model.nv))
            mj.mj_jacSite(model, data, jac[:3], jac[3:], site)
            mj.mj_jacDot(
                model,
                data,
                derivative[:3],
                derivative[3:],
                data.site_xpos[site],
                int(model.site_bodyid[site]),
            )
            jacobians.append(jac)
            derivatives.append(derivative)
        jac = jacobians[0] - jacobians[1]
        drift = (derivatives[0] - derivatives[1]) @ data.qvel
        unconstrained = np.linalg.solve(
            mass, np.column_stack((effort - data.qfrc_bias, jac.T))
        )
        free, response = unconstrained[:, 0], unconstrained[:, 1:]
        multiplier = np.linalg.solve(jac @ response, -drift - jac @ free)
        acceleration = free + response @ multiplier
        if not np.isfinite(acceleration).all():
            raise FloatingPointError("Nonfinite native MuJoCo acceleration")
        a, b = self._closure
        ra, rb = (data.site_xmat[i].reshape(3, 3) for i in (a, b))
        self._errors = (
            np.r_[
                data.site_xpos[a] - data.site_xpos[b],
                Rotation.from_matrix(ra @ rb.T).as_rotvec(),
            ],
            jac @ data.qvel,
        )
        return {
            name: float(acceleration[index]) for name, index in self._indices.items()
        }

    def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return detached pose/rate residuals from the last acceleration call."""
        if self._errors is None:
            raise ValueError("Evaluate acceleration before closure errors")
        return self._errors[0].copy(), self._errors[1].copy()
