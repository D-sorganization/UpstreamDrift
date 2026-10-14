"""MuJoCo and MJX inertial regressor provider (MOSAIC-03).

This module implements the MuJoCo-backed regressor provider by differentiating
inverse dynamics (`mj_inverse`) with respect to rigid-body inertial parameters.
Because inverse dynamics is strictly linear in the stacked link spatial parameters
pi = [m_i, h_ix, h_iy, h_iz, I_ixx, I_ixy, I_ixz, I_iyy, I_iyz, I_izz],
the Jacobian of `mj_inverse` with respect to pi yields the exact inertial regressor:

    tau = Y(q, v, a) @ pi

Closed-form bijections map between MuJoCo body mass/inertia (body_mass, body_ipos,
body_inertia, body_iquat) and the 10-vector spatial parameters / pseudo-inertia matrix.
For planar models, the active planar subspace columns (0, 1, 2, 9 per link) match
the planar chain oracle identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.inertial import (
    PLANAR_PARAMETERS_PER_BODY,
    SPATIAL_PARAMETERS_PER_BODY,
    parameters_from_pseudo_inertia,
    pseudo_inertia,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]

try:
    import mujoco

    _MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore[assignment]
    _MUJOCO_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    _MJX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    mjx = None
    _MJX_AVAILABLE = False


def is_mujoco_available() -> bool:
    """Return True if the mujoco package is installed and importable."""
    return _MUJOCO_AVAILABLE


def is_mjx_available() -> bool:
    """Return True if JAX and mujoco.mjx are available for vmap acceleration."""
    return _MJX_AVAILABLE


# ---------------------------------------------------------------------------
# Rotation representations and conversions
# ---------------------------------------------------------------------------


def quat_to_matrix(q: FloatArray) -> FloatArray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    require(q.shape == (4,), "quaternion must have shape (4,)", q.shape)
    norm = float(np.linalg.norm(q))
    require(norm > 1e-12, "quaternion must be non-zero", norm)
    w, x, y, z = (q / norm).tolist()

    R: FloatArray = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    return R


def matrix_to_quat(R: FloatArray) -> FloatArray:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z]."""
    require(R.shape == (3, 3), "R must have shape (3, 3)", R.shape)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1.0 + float(R[0, 0] - R[1, 1] - R[2, 2]), 1e-12)) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1.0 + float(R[1, 1] - R[0, 0] - R[2, 2]), 1e-12)) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + float(R[2, 2] - R[0, 0] - R[1, 1]), 1e-12)) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q: FloatArray = np.array([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm > 1e-12:
        q /= norm
    return q


# ---------------------------------------------------------------------------
# MuJoCo to / from spatial parameter mappings
# ---------------------------------------------------------------------------


def mujoco_to_spatial_parameters(
    mass: float, ipos: FloatArray, inertia: FloatArray, iquat: FloatArray
) -> FloatArray:
    """Map MuJoCo body inertia parameters to 10-vector spatial parameters about origin.

    MuJoCo body parameters:
    - mass: scalar mass m >= 0
    - ipos: 3-vector c = [cx, cy, cz], COM position in body frame
    - inertia: 3-vector [I_1, I_2, I_3], principal moments about COM
    - iquat: 4-vector [qw, qx, qy, qz], orientation of principal frame w.r.t body

    Spatial parameters:
    pi = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] about body frame origin.
    """
    require(mass >= 0.0, "mass must be non-negative", mass)
    require(ipos.shape == (3,), "ipos must have shape (3,)", ipos.shape)
    require(inertia.shape == (3,), "inertia must have shape (3,)", inertia.shape)
    require(bool(np.all(inertia >= 0.0)), "principal inertias must be non-negative")
    require(iquat.shape == (4,), "iquat must have shape (4,)", iquat.shape)

    m = float(mass)
    c = np.asarray(ipos, dtype=np.float64)
    R_iquat = quat_to_matrix(iquat)
    I_c = R_iquat @ np.diag(inertia) @ R_iquat.T

    # Parallel-axis theorem (Steiner's theorem) to origin
    c_dot_c = float(np.dot(c, c))
    I_o = I_c + m * (c_dot_c * np.eye(3, dtype=np.float64) - np.outer(c, c))
    h = m * c

    pi: FloatArray = np.array(
        [
            m,
            h[0],
            h[1],
            h[2],
            I_o[0, 0],
            I_o[0, 1],
            I_o[0, 2],
            I_o[1, 1],
            I_o[1, 2],
            I_o[2, 2],
        ],
        dtype=np.float64,
    )
    return pi


def spatial_to_mujoco_parameters(
    pi: FloatArray,
) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    """Map 10-vector spatial parameters about origin to MuJoCo body parameters.

    Returns:
    (mass, ipos, inertia, iquat)
    """
    require(pi.shape == (10,), "pi must have shape (10,)", pi.shape)
    require(bool(np.all(np.isfinite(pi))), "pi must be finite")
    m = float(pi[0])
    require(m >= 0.0, "mass must be non-negative", m)

    h = np.asarray(pi[1:4], dtype=np.float64)
    c: FloatArray = (h / m) if m > 1e-12 else np.zeros(3, dtype=np.float64)

    I_o: FloatArray = np.array(
        [
            [pi[4], pi[5], pi[6]],
            [pi[5], pi[7], pi[8]],
            [pi[6], pi[8], pi[9]],
        ],
        dtype=np.float64,
    )

    c_dot_c = float(np.dot(c, c))
    I_c = I_o - m * (c_dot_c * np.eye(3, dtype=np.float64) - np.outer(c, c))
    I_c = 0.5 * (I_c + I_c.T)

    eigvals, eigvecs = np.linalg.eigh(I_c)
    eigvals = np.maximum(eigvals, 0.0)

    # Ensure right-handed coordinate frame for principal axes (det = +1)
    if float(np.linalg.det(eigvecs)) < 0.0:
        eigvecs[:, 0] = -eigvecs[:, 0]

    iquat = matrix_to_quat(eigvecs)
    ipos: FloatArray = np.asarray(c, dtype=np.float64)
    inertia: FloatArray = np.asarray(eigvals, dtype=np.float64)

    return m, ipos, inertia, iquat


def model_to_spatial_parameters(model: Any) -> FloatArray:
    """Extract stacked spatial parameters for all moving bodies (1 to nbody-1)."""
    require(model is not None, "model must not be None")
    n_moving = model.nbody - 1
    require(n_moving >= 1, "model must contain at least one moving body")

    params_list: list[FloatArray] = []
    for b in range(1, model.nbody):
        pi_b = mujoco_to_spatial_parameters(
            model.body_mass[b],
            model.body_ipos[b],
            model.body_inertia[b],
            model.body_iquat[b],
        )
        params_list.append(pi_b)

    stacked: FloatArray = np.concatenate(params_list)
    return stacked


def spatial_parameters_to_pseudo_inertia(pi: FloatArray) -> FloatArray:
    """Convert spatial parameters of shape (10,) or (bodies, 10) to 4x4 pseudo-inertia."""
    pi_2d = pi[None, :] if pi.ndim == 1 else pi
    pseudo = pseudo_inertia(pi_2d)
    return np.asarray(pseudo[0] if pi.ndim == 1 else pseudo, dtype=np.float64)


def pseudo_inertia_to_spatial_parameters(pseudo: FloatArray) -> FloatArray:
    """Convert 4x4 pseudo-inertia matrix to 10-vector spatial parameters."""
    pseudo_3d = pseudo[None, ...] if pseudo.ndim == 2 else pseudo
    params = parameters_from_pseudo_inertia(pseudo_3d)
    return np.asarray(params[0] if pseudo.ndim == 2 else params, dtype=np.float64)


def _extract_marker_info(markers: Any) -> tuple[Any, Any]:
    require(markers is not None, "markers must not be None")
    if hasattr(markers, "link_index") and hasattr(markers, "offsets"):
        return markers.link_index, markers.offsets
    if hasattr(markers, "body_id") and hasattr(markers, "offsets"):
        return markers.body_id, markers.offsets
    raise ValueError(f"Unsupported marker container type: {type(markers)}")


def _resolve_body_and_offset(
    moving_bodies: list[int],
    link_idx: int,
    offset: Any,
) -> tuple[int, FloatArray]:
    body_id = int(link_idx)
    if body_id < moving_bodies[0]:
        body_id = moving_bodies[body_id]
    offset_3d: FloatArray = np.zeros(3, dtype=np.float64)
    raw_offset = np.asarray(offset, dtype=np.float64)
    offset_3d[: raw_offset.size] = raw_offset
    return body_id, offset_3d


# ---------------------------------------------------------------------------
# MujocoRegressorModel
# ---------------------------------------------------------------------------


@dataclass
class MujocoRegressorModel:
    """MuJoCo multibody regressor provider implementing RegressorModel."""

    model: Any
    data: Any
    planar: bool = False
    input_matrix_override: FloatArray | None = None

    def __post_init__(self) -> None:
        require(
            _MUJOCO_AVAILABLE, "mujoco must be installed to use MujocoRegressorModel"
        )
        require(self.model is not None, "model must not be None")
        require(self.data is not None, "data must not be None")
        require(self.model.nbody >= 2, "model must have at least one moving body")

        self._moving_bodies: list[int] = list(range(1, self.model.nbody))
        self._n_moving: int = len(self._moving_bodies)

        # Planar subspace column indices: for body i, columns [m, hx, hy, Izz]
        # In spatial parameters: m=0, hx=1, hy=2, Izz=9
        planar_cols: list[int] = []
        for i in range(self._n_moving):
            base = i * SPATIAL_PARAMETERS_PER_BODY
            planar_cols.extend([base + 0, base + 1, base + 2, base + 9])
        self._planar_columns: list[int] = planar_cols

        for b in self._moving_bodies:
            self.model.body_sameframe[b] = 0
        mujoco.mj_setConst(self.model, self.data)

    @classmethod
    def _from_mjmodel(
        cls,
        model: Any,
        planar: bool = False,
        input_matrix: FloatArray | None = None,
    ) -> MujocoRegressorModel:
        return cls(
            model=model,
            data=mujoco.MjData(model),
            planar=planar,
            input_matrix_override=input_matrix,
        )

    @classmethod
    def from_xml_string(
        cls,
        xml: str,
        planar: bool = False,
        input_matrix: FloatArray | None = None,
    ) -> MujocoRegressorModel:
        """Create a MujocoRegressorModel from an MJCF XML string."""
        require(_MUJOCO_AVAILABLE, "mujoco must be installed")
        return cls._from_mjmodel(
            mujoco.MjModel.from_xml_string(xml),
            planar=planar,
            input_matrix=input_matrix,
        )

    @classmethod
    def from_xml_path(
        cls,
        path: str,
        planar: bool = False,
        input_matrix: FloatArray | None = None,
    ) -> MujocoRegressorModel:
        """Create a MujocoRegressorModel from an MJCF XML file path."""
        require(_MUJOCO_AVAILABLE, "mujoco must be installed")
        return cls._from_mjmodel(
            mujoco.MjModel.from_xml_path(path),
            planar=planar,
            input_matrix=input_matrix,
        )

    # ------------------------------------------------------------------ properties
    @property
    def n_dof(self) -> int:
        """Number of generalized velocity coordinates (degrees of freedom)."""
        return int(self.model.nv)

    @property
    def n_bodies(self) -> int:
        """Number of moving bodies."""
        return self._n_moving

    @property
    def n_parameters(self) -> int:
        """Number of identified parameters (4 per body if planar, 10 otherwise)."""
        if self.planar:
            return PLANAR_PARAMETERS_PER_BODY * self._n_moving
        return SPATIAL_PARAMETERS_PER_BODY * self._n_moving

    @property
    def input_matrix(self) -> FloatArray:
        """Selection matrix B mapping inputs to joint torques."""
        if self.input_matrix_override is not None:
            return self.input_matrix_override
        return np.eye(self.n_dof, dtype=np.float64)

    @property
    def spatial_parameters(self) -> FloatArray:
        """Current spatial parameters of moving bodies (sliced if planar)."""
        full_params = model_to_spatial_parameters(self.model)
        if self.planar:
            return np.asarray(full_params[self._planar_columns], dtype=np.float64)
        return full_params

    # ------------------------------------------------------------------ mutators
    def set_body_parameters(
        self,
        body_id: int,
        mass: float,
        ipos: FloatArray,
        inertia: FloatArray,
        iquat: FloatArray,
    ) -> None:
        """Set MuJoCo body parameters directly."""
        require(
            1 <= body_id < self.model.nbody,
            "body_id must correspond to a moving body",
            body_id,
        )
        require(mass >= 0.0, "mass must be non-negative", mass)
        require(ipos.shape == (3,), "ipos must be (3,)", ipos.shape)
        require(inertia.shape == (3,), "inertia must be (3,)", inertia.shape)
        require(iquat.shape == (4,), "iquat must be (4,)", iquat.shape)

        self.model.body_mass[body_id] = float(mass)
        self.model.body_ipos[body_id] = np.asarray(ipos, dtype=np.float64)
        self.model.body_inertia[body_id] = np.asarray(inertia, dtype=np.float64)
        self.model.body_iquat[body_id] = np.asarray(iquat, dtype=np.float64)
        self.model.body_sameframe[body_id] = 0
        mujoco.mj_setConst(self.model, self.data)

    # ------------------------------------------------------------------ dynamics
    def inverse_dynamics(
        self, q: FloatArray, v: FloatArray, a: FloatArray
    ) -> FloatArray:
        """Compute generalized forces tau = M(q) a + C(q, v) v + g(q) via mj_inverse."""
        require(
            q.shape == (self.model.nq,) or q.shape == (self.n_dof,),
            "q shape mismatch",
            q.shape,
        )
        require(v.shape == (self.n_dof,), "v shape mismatch", v.shape)
        require(a.shape == (self.n_dof,), "a shape mismatch", a.shape)
        require(
            bool(
                np.all(np.isfinite(q))
                and np.all(np.isfinite(v))
                and np.all(np.isfinite(a))
            ),
            "state and acceleration must be finite",
        )

        self.data.qpos[: q.size] = q
        self.data.qvel[:] = v
        self.data.qacc[:] = a
        mujoco.mj_inverse(self.model, self.data)
        tau: FloatArray = np.asarray(self.data.qfrc_inverse.copy(), dtype=np.float64)
        return tau

    def _eval_inverse_for_body(
        self,
        b: int,
        mass: float,
        ipos: FloatArray,
        inertia: FloatArray,
        iquat: FloatArray,
        q: FloatArray,
        v: FloatArray,
        a: FloatArray,
    ) -> FloatArray:
        self.model.body_mass[b] = mass
        self.model.body_ipos[b] = ipos
        self.model.body_inertia[b] = inertia
        self.model.body_iquat[b] = iquat
        self.model.body_sameframe[b] = 0
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[: q.size] = q
        self.data.qvel[:] = v
        self.data.qacc[:] = a
        mujoco.mj_inverse(self.model, self.data)
        return np.asarray(self.data.qfrc_inverse.copy(), dtype=np.float64)

    def _restore_params(
        self,
        params: Sequence[tuple[float, FloatArray, FloatArray, FloatArray]],
        q: FloatArray,
        v: FloatArray,
        a: FloatArray,
    ) -> None:
        for b_idx, b in enumerate(self._moving_bodies):
            m_o, ipos_o, inertia_o, iquat_o = params[b_idx]
            self.model.body_mass[b] = m_o
            self.model.body_ipos[b] = ipos_o
            self.model.body_inertia[b] = inertia_o
            self.model.body_iquat[b] = iquat_o
            self.model.body_sameframe[b] = 0
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[: q.size] = q
        self.data.qvel[:] = v
        self.data.qacc[:] = a
        mujoco.mj_inverse(self.model, self.data)

    def compute_regressor(
        self, q: FloatArray, v: FloatArray, a: FloatArray, h_step: float = 1e-6
    ) -> FloatArray:
        """Compute the regressor matrix Y(q, v, a) such that tau = Y @ pi."""
        require(h_step > 0.0, "h_step must be positive", h_step)
        # Precondition checks on states
        self.inverse_dynamics(q, v, a)

        # Cache original parameters for all moving bodies
        original_params: list[tuple[float, FloatArray, FloatArray, FloatArray]] = []
        for b in self._moving_bodies:
            original_params.append(
                (
                    float(self.model.body_mass[b]),
                    np.asarray(self.model.body_ipos[b].copy(), dtype=np.float64),
                    np.asarray(self.model.body_inertia[b].copy(), dtype=np.float64),
                    np.asarray(self.model.body_iquat[b].copy(), dtype=np.float64),
                )
            )

        nv = self.n_dof
        n_bodies = self._n_moving
        Y_full: FloatArray = np.zeros(
            (nv, SPATIAL_PARAMETERS_PER_BODY * n_bodies), dtype=np.float64
        )

        try:
            for b_idx, b in enumerate(self._moving_bodies):
                m_o, ipos_o, inertia_o, iquat_o = original_params[b_idx]
                pi_orig = mujoco_to_spatial_parameters(m_o, ipos_o, inertia_o, iquat_o)

                for k in range(SPATIAL_PARAMETERS_PER_BODY):
                    # Forward perturbation
                    pi_plus = pi_orig.copy()
                    pi_plus[k] += h_step
                    m_p, c_p, I_p, q_p = spatial_to_mujoco_parameters(pi_plus)
                    tau_plus = self._eval_inverse_for_body(
                        b, m_p, c_p, I_p, q_p, q, v, a
                    )

                    # Backward perturbation
                    pi_minus = pi_orig.copy()
                    pi_minus[k] -= h_step
                    m_m, c_m, I_m, q_m = spatial_to_mujoco_parameters(pi_minus)
                    tau_minus = self._eval_inverse_for_body(
                        b, m_m, c_m, I_m, q_m, q, v, a
                    )

                    col = b_idx * SPATIAL_PARAMETERS_PER_BODY + k
                    Y_full[:, col] = (tau_plus - tau_minus) / (2.0 * h_step)

                # Restore this body before moving to the next
                self._eval_inverse_for_body(b, m_o, ipos_o, inertia_o, iquat_o, q, v, a)
        finally:
            self._restore_params(original_params, q, v, a)

        if self.planar:
            Y_planar: FloatArray = np.asarray(
                Y_full[:, self._planar_columns], dtype=np.float64
            )
            return Y_planar
        return Y_full

    def batch_compute_regressor(
        self, q_batch: FloatArray, v_batch: FloatArray, a_batch: FloatArray
    ) -> FloatArray:
        """Compute regressor for a batch of N time nodes, returning (N, n_dof, n_parameters)."""
        require(q_batch.ndim == 2, "q_batch must be 2-D", q_batch.shape)
        require(v_batch.ndim == 2, "v_batch must be 2-D", v_batch.shape)
        require(a_batch.ndim == 2, "a_batch must be 2-D", a_batch.shape)
        N = q_batch.shape[0]
        require(
            v_batch.shape[0] == N and a_batch.shape[0] == N,
            "batch sizes must match",
            (N, v_batch.shape[0], a_batch.shape[0]),
        )

        n_p = self.n_parameters
        nv = self.n_dof
        Y_batch: FloatArray = np.zeros((N, nv, n_p), dtype=np.float64)

        for i in range(N):
            Y_batch[i] = self.compute_regressor(q_batch[i], v_batch[i], a_batch[i])

        return Y_batch

    def regressor(self, q: FloatArray, v: FloatArray, a: FloatArray) -> FloatArray:
        """Compute regressor for single node or batch (implements RegressorModel)."""
        if q.ndim == 1:
            return self.compute_regressor(q, v, a)
        return self.batch_compute_regressor(q, v, a)

    # ------------------------------------------------------------------ markers
    def marker_positions(self, q: FloatArray, markers: Any) -> FloatArray:
        """Compute world positions of attached markers."""
        link_indices, offsets = _extract_marker_info(markers)

        n_markers = len(link_indices)
        is_batched = q.ndim == 2
        batch_size = q.shape[0] if is_batched else 1
        q_2d = q if is_batched else q[None, :]

        positions: FloatArray = np.zeros((batch_size, n_markers, 3), dtype=np.float64)

        for b_i in range(batch_size):
            self.data.qpos[: q_2d.shape[1]] = q_2d[b_i]
            mujoco.mj_fwdPosition(self.model, self.data)

            for m in range(n_markers):
                body_id, offset_3d = _resolve_body_and_offset(
                    self._moving_bodies, link_indices[m], offsets[m]
                )
                R_body = self.data.xmat[body_id].reshape(3, 3)
                p_body = self.data.xpos[body_id]
                positions[b_i, m] = p_body + R_body @ offset_3d

        if not is_batched:
            return np.asarray(positions[0], dtype=np.float64)
        return positions

    def marker_jacobians(
        self, q: FloatArray, markers: Any
    ) -> tuple[FloatArray, FloatArray]:
        """Compute marker Jacobians w.r.t coordinates q and geometry s."""
        link_indices, offsets = _extract_marker_info(markers)

        n_markers = len(link_indices)
        is_batched = q.ndim == 2
        batch_size = q.shape[0] if is_batched else 1
        q_2d = q if is_batched else q[None, :]

        nv = self.n_dof
        J_q: FloatArray = np.zeros((batch_size, n_markers, 3, nv), dtype=np.float64)
        J_s: FloatArray = np.zeros((batch_size, n_markers, 3, 3), dtype=np.float64)

        jacp: FloatArray = np.zeros((3, nv), dtype=np.float64)

        for b_i in range(batch_size):
            self.data.qpos[: q_2d.shape[1]] = q_2d[b_i]
            mujoco.mj_fwdPosition(self.model, self.data)

            for m in range(n_markers):
                body_id, offset_3d = _resolve_body_and_offset(
                    self._moving_bodies, link_indices[m], offsets[m]
                )
                R_body = self.data.xmat[body_id].reshape(3, 3)
                p_body = self.data.xpos[body_id]
                pt = p_body + R_body @ offset_3d

                jacp.fill(0.0)
                mujoco.mj_jac(self.model, self.data, jacp, None, pt, body_id)
                J_q[b_i, m] = jacp
                J_s[b_i, m] = R_body

        if not is_batched:
            return (
                np.asarray(J_q[0], dtype=np.float64),
                np.asarray(J_s[0], dtype=np.float64),
            )
        return J_q, J_s
