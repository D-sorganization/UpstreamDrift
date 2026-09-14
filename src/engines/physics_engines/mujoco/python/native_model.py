"""MuJoCo native tree with an explicit rigid six-dimensional grip solve.

Stock mj_step uses compliant equality constraints and is not this adapter's
execution path. Mass, bias, kinematics and Jacobian derivatives come from MuJoCo.
"""

import hashlib
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.mujoco.python.native_mjcf import export_native_mjcf


def _pack_vector(
    indices: Mapping[str, int],
    values: Mapping[str, float],
    size: int,
    error_message: str = "State and effort must be finite",
) -> np.ndarray:
    """Pack coordinate mapping into dense array preserving DOF ordering."""
    if set(values) != set(indices):
        raise ValueError("Provide exactly the model coordinate inventory")
    result = np.zeros(size)
    for name, index in indices.items():
        result[index] = values[name]
    if not np.isfinite(result).all():
        raise ValueError(error_message)
    return result


def _extract_frame_poses(
    data: Any,
    sites: Mapping[str, int],
) -> dict[str, np.ndarray]:
    """Compute 4x4 forward kinematics poses for named frame sites."""
    result = {}
    for name, index in sites.items():
        pose = np.eye(4)
        pose[:3, :3] = data.site_xmat[index].reshape(3, 3)
        pose[:3, 3] = data.site_xpos[index]
        result[name] = pose
    return result


def _evaluate_weld_closure(
    mj: Any,
    model: Any,
    data: Any,
    closure_sites: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative 6D closure Jacobian and velocity drift."""
    jacobians, derivatives = [], []
    for site in closure_sites:
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
    return jac, drift


def _solve_kkt_dynamics(
    mass: np.ndarray,
    rhs_force: np.ndarray,
    jac: np.ndarray,
    drift: np.ndarray,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Solve M a - J.T lambda = rhs_force, J a = -drift."""
    rhs = np.column_stack((rhs_force, jac.T))
    try:
        unconstrained = np.linalg.solve(mass, rhs)
    except np.linalg.LinAlgError:
        mass_reg = mass + regularization * np.eye(mass.shape[0])
        try:
            unconstrained = np.linalg.solve(mass_reg, rhs)
        except np.linalg.LinAlgError:
            unconstrained, _, _, _ = np.linalg.lstsq(mass, rhs, rcond=None)

    free, response = unconstrained[:, 0], unconstrained[:, 1:]
    weld_matrix = jac @ response
    if regularization > 0.0:
        weld_matrix = weld_matrix + regularization * np.eye(weld_matrix.shape[0])
    try:
        multiplier = np.linalg.solve(weld_matrix, -drift - jac @ free)
    except np.linalg.LinAlgError:
        multiplier, _, _, _ = np.linalg.lstsq(
            weld_matrix, -drift - jac @ free, rcond=None
        )
    return free + response @ multiplier


def _compute_closure_errors(
    data: Any,
    closure_sites: Sequence[int],
    jac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 6D displacement/rotvec and velocity closure residuals."""
    a, b = closure_sites
    ra, rb = (data.site_xmat[i].reshape(3, 3) for i in (a, b))
    return (
        np.r_[
            data.site_xpos[a] - data.site_xpos[b],
            Rotation.from_matrix(ra @ rb.T).as_rotvec(),
        ],
        jac @ data.qvel,
    )


def _prepare_forward_dynamics(
    mj: Any,
    model: Any,
    data: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> None:
    """Set generalized coordinates/velocities and run position/velocity forward passes."""
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mj.mj_fwdPosition(model, data)
    mj.mj_fwdVelocity(model, data)


class NativeMujocoModel:
    """Match the public native acceleration/frame protocol without feedback."""

    @classmethod
    def from_native_bundle(
        cls, urdf_bytes: bytes, sidecar_bytes: bytes, model_bytes: bytes
    ) -> "NativeMujocoModel":
        """Validate the portable bundle, then convert canonical geometry to MJCF.

        This is not direct MuJoCo URDF parsing. The shared contract checks bundle
        identities and native semantics before any engine construction. Dynamics
        still uses the explicit rigid closure adapter, not stock mj_step.
        """
        from src.shared.python.motion_matching.native_urdf_contract import (
            validate_native_urdf_bundle,
        )

        metadata = validate_native_urdf_bundle(urdf_bytes, sidecar_bytes, model_bytes)
        engine = cls(model_bytes)
        engine.metadata.update(
            bundle_conversion="validated canonical geometry to MJCF; not direct URDF parsing",
            urdf_sha256=metadata["urdf_sha256"],
            sidecar_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
        )
        return engine

    def __init__(self, model_bytes: bytes) -> None:
        mj: Any = import_module("mujoco")
        self._mj = mj
        self.xml, self.metadata = export_native_mjcf(model_bytes)
        self.model_sha256 = self.metadata["model_sha256"]
        self.model = mj.MjModel.from_xml_string(self.xml)
        self.data = mj.MjData(self.model)
        self.coordinate_order = list(self.metadata["coordinate_order"])
        self._indices = {}
        for name in self.coordinate_order:
            j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError(f"Coordinate {name} missing from native model")
            self._indices[name] = int(self.model.jnt_dofadr[j])
        if self.model.nq != len(self._indices) or self.model.nv != len(self._indices):
            raise ValueError("All native coordinates must be scalar 1-DOF joints")
        self._sites = {
            name: self.model.site(site).id
            for name, site in self.metadata["frame_sites"].items()
        }
        self._closure = [self.model.site(f"native_closure_{s}").id for s in ("a", "b")]
        self._errors: tuple[np.ndarray, np.ndarray] | None = None

    def _vector(self, values: Mapping[str, float]) -> np.ndarray:
        return _pack_vector(
            self._indices,
            values,
            self.model.nv,
            error_message="Native state and effort must be finite",
        )

    def frame_poses(self, coordinates: Mapping[str, float]) -> dict[str, np.ndarray]:
        self.data.qpos[:] = self._vector(coordinates)
        self._mj.mj_kinematics(self.model, self.data)
        return _extract_frame_poses(self.data, self._sites)

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Solve M a - J.T lambda = effort-bias, J a = -Jdot v."""
        mj, model, data = self._mj, self.model, self.data
        effort = self._vector(primitive_efforts)
        _prepare_forward_dynamics(
            mj, model, data, self._vector(coordinates), self._vector(rates)
        )
        mass = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, mass, data.qM)
        jac, drift = _evaluate_weld_closure(mj, model, data, self._closure)
        acceleration = _solve_kkt_dynamics(mass, effort - data.qfrc_bias, jac, drift)
        if not np.isfinite(acceleration).all():
            raise FloatingPointError("Nonfinite native MuJoCo acceleration")
        self._errors = _compute_closure_errors(data, self._closure, jac)
        return {
            name: float(acceleration[index]) for name, index in self._indices.items()
        }

    def closure_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return detached pose/rate residuals from the last acceleration call."""
        if self._errors is None:
            raise ValueError("Evaluate acceleration before closure errors")
        return self._errors[0].copy(), self._errors[1].copy()
