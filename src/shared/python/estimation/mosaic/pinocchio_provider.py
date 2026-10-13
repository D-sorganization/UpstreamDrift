"""Pinocchio RegressorModel provider for MOSAIC (issue #11534, MOSAIC-02).

Exposes inverse dynamics in linear-regressor form Y(q, v, a) pi = tau using
Pinocchio's computeJointTorqueRegressor, vectorized/batched over N nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import PreconditionError, require
from src.shared.python.estimation.mosaic.planar_chain import (
    PlanarChain,
    PlanarMarkerSet,
)

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

SPATIAL_PARAMETERS_PER_BODY = 10
PLANAR_COLUMNS_IN_SPATIAL = (0, 1, 2, 9)  # m, h_x, h_y, I_zz


def is_pinocchio_available() -> bool:
    """Return True if real Pinocchio is installed and importable (not a MagicMock)."""
    try:
        import sys

        pin = sys.modules.get("pinocchio")
        if pin is not None and type(pin).__module__ == "unittest.mock":
            return False

        import pinocchio as pin_mod

        if type(pin_mod).__module__ == "unittest.mock":
            return False
        if getattr(pin_mod, "__file__", None) is None:
            return False
        return (
            hasattr(pin_mod, "Model")
            and type(pin_mod.Model).__module__ != "unittest.mock"
        )
    except (ImportError, AttributeError):
        return False


def require_pinocchio() -> Any:
    """Import and return the real pinocchio module, or raise PreconditionError."""
    require(is_pinocchio_available(), "Real Pinocchio library is required")
    import pinocchio as pin

    return pin


def planar_regressor_from_spatial(Y_spatial: FloatArray, n_links: int) -> FloatArray:
    """Extract planar regressor columns (m, h_x, h_y, I_zz) from spatial regressor.

    Args:
        Y_spatial: (N, n_dof, 10 * n_links) or (n_dof, 10 * n_links) spatial
          regressor.
        n_links: Number of serial links.

    Returns:
        (N, n_dof, 4 * n_links) planar regressor matching PlanarChain.regressor.
    """
    require(Y_spatial.ndim in (2, 3), "Y_spatial must be 2-D or 3-D")
    require(n_links > 0, "n_links must be positive")
    spatial_cols = 10 * n_links
    require(
        Y_spatial.shape[-1] >= spatial_cols,
        "Y_spatial column count must be >= 10 * n_links",
    )

    cols: list[int] = []
    for i in range(n_links):
        base = 10 * i
        for offset in PLANAR_COLUMNS_IN_SPATIAL:
            cols.append(base + offset)

    return Y_spatial[..., cols]


def build_planar_pinocchio_model(chain: PlanarChain) -> Any:
    """Construct a Pinocchio Model matching the geometry and gravity of a PlanarChain."""
    pin = require_pinocchio()

    model = pin.Model()
    g_2d = chain.gravity
    model.gravity.linear = np.array([g_2d[0], g_2d[1], 0.0], dtype=np.float64)
    model.gravity.angular = np.zeros(3, dtype=np.float64)

    parent_joint = 0  # universe
    n_links = chain.n_links

    for i in range(n_links):
        joint_name = f"joint_{i + 1}"
        if i == 0:
            joint_placement = pin.SE3.Identity()
        else:
            prev_len = float(chain.link_lengths[i - 1])
            joint_placement = pin.SE3(
                np.eye(3, dtype=np.float64),
                np.array([prev_len, 0.0, 0.0], dtype=np.float64),
            )

        joint_id = model.addJoint(
            parent_joint, pin.JointModelRZ(), joint_placement, joint_name
        )

        inertia = pin.Inertia(
            1.0, np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
        )
        model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())
        parent_joint = joint_id

    return model


@dataclass(frozen=True)
class PinocchioMarkerSet:
    """Markers rigidly attached to Pinocchio joints at local 3-D offsets (metres)."""

    parent_joint_id: IntArray
    offsets: FloatArray

    def __post_init__(self) -> None:
        require(self.parent_joint_id.ndim == 1, "parent_joint_id must be 1-D")
        require(
            self.offsets.shape == (self.parent_joint_id.size, 3),
            "offsets must be (markers, 3)",
        )
        require(
            bool(np.all(self.parent_joint_id >= 0)),
            "parent_joint_id must be non-negative",
        )

    @property
    def n_markers(self) -> int:
        return int(self.parent_joint_id.size)


class PinocchioRegressorModel:
    """Pinocchio-backed implementation of the MOSAIC RegressorModel protocol."""

    def __init__(
        self,
        model: Any,
        input_matrix: FloatArray | None = None,
    ) -> None:
        require(model is not None, "model must not be None")
        self._model = model
        self._data = model.createData()

        if input_matrix is not None:
            require(input_matrix.ndim == 2, "input_matrix must be 2-D")
            require(
                input_matrix.shape[0] == self._model.nv,
                "input_matrix rows must equal model.nv",
            )
            self._input_matrix = np.asarray(input_matrix, dtype=np.float64)
        else:
            self._input_matrix = np.eye(self._model.nv, dtype=np.float64)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def data(self) -> Any:
        return self._data

    @property
    def n_dof(self) -> int:
        """Number of velocity degrees of freedom (model.nv)."""
        return int(self._model.nv)

    @property
    def n_parameters(self) -> int:
        """Number of spatial inertial parameters across moving joints."""
        return int(SPATIAL_PARAMETERS_PER_BODY * (self._model.njoints - 1))

    @property
    def input_matrix(self) -> FloatArray:
        return self._input_matrix

    def regressor(
        self,
        q: FloatArray,
        v: FloatArray,
        a: FloatArray,
    ) -> FloatArray:
        """Batched joint torque regressor Y(q, v, a) with shape (N, nv, n_parameters)."""
        pin = require_pinocchio()
        q_arr = np.asarray(q, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        a_arr = np.asarray(a, dtype=np.float64)

        require(q_arr.ndim == 2, "q must be (N, nq)", q_arr.shape)
        require(v_arr.ndim == 2, "v must be (N, nv)", v_arr.shape)
        require(a_arr.ndim == 2, "a must be (N, nv)", a_arr.shape)
        batch = q_arr.shape[0]
        require(
            v_arr.shape[0] == batch and a_arr.shape[0] == batch,
            "batch sizes of q, v, a must match",
        )
        require(
            q_arr.shape[1] == self._model.nq,
            f"q dimension must be {self._model.nq}",
            q_arr.shape,
        )
        require(
            v_arr.shape[1] == self._model.nv,
            f"v dimension must be {self._model.nv}",
            v_arr.shape,
        )
        require(
            a_arr.shape[1] == self._model.nv,
            f"a dimension must be {self._model.nv}",
            a_arr.shape,
        )
        require(bool(np.all(np.isfinite(q_arr))), "q contains non-finite values")
        require(bool(np.all(np.isfinite(v_arr))), "v contains non-finite values")
        require(bool(np.all(np.isfinite(a_arr))), "a contains non-finite values")

        nv = self._model.nv
        n_params = self.n_parameters
        Y = np.zeros((batch, nv, n_params), dtype=np.float64)

        for i in range(batch):
            Y[i] = pin.computeJointTorqueRegressor(
                self._model, self._data, q_arr[i], v_arr[i], a_arr[i]
            )

        return Y

    def _normalize_markers(self, markers: Any) -> tuple[IntArray, FloatArray, bool]:
        """Convert either PinocchioMarkerSet or PlanarMarkerSet to joint IDs and 3-D offsets."""
        if isinstance(markers, PinocchioMarkerSet):
            return markers.parent_joint_id, markers.offsets, False
        if isinstance(markers, PlanarMarkerSet):
            joint_ids = np.asarray(markers.link_index + 1, dtype=np.int64)
            offsets_2d = np.asarray(markers.offsets, dtype=np.float64)
            offsets_3d = np.pad(offsets_2d, ((0, 0), (0, 1)), mode="constant")
            return joint_ids, offsets_3d, True
        raise PreconditionError(f"Unsupported marker set type: {type(markers)}")

    def _validate_q(self, q: FloatArray) -> FloatArray:
        """Validate joint position trajectory array."""
        q_arr = np.asarray(q, dtype=np.float64)
        require(q_arr.ndim == 2, "q must be (N, nq)", q_arr.shape)
        require(
            q_arr.shape[1] == self._model.nq,
            f"q dimension must be {self._model.nq}",
            q_arr.shape,
        )
        require(bool(np.all(np.isfinite(q_arr))), "q contains non-finite values")
        return q_arr

    def marker_positions(self, q: FloatArray, markers: Any) -> FloatArray:
        """World positions of attached markers."""
        pin = require_pinocchio()
        q_arr = self._validate_q(q)

        joint_ids, offsets, is_planar = self._normalize_markers(markers)
        n_markers = joint_ids.size
        batch = q_arr.shape[0]
        out_dim = 2 if is_planar else 3
        positions = np.zeros((batch, n_markers, out_dim), dtype=np.float64)

        for t in range(batch):
            pin.forwardKinematics(self._model, self._data, q_arr[t])
            for m in range(n_markers):
                j_id = int(joint_ids[m])
                p_world = self._data.oMi[j_id].act(offsets[m])
                positions[t, m] = p_world[:out_dim]

        return positions

    def marker_jacobians(
        self, q: FloatArray, markers: Any
    ) -> tuple[FloatArray, FloatArray]:
        """Analytic Jacobians of marker positions w.r.t joint coordinates and marker offsets."""
        pin = require_pinocchio()
        q_arr = self._validate_q(q)

        joint_ids, offsets, is_planar = self._normalize_markers(markers)
        n_markers = joint_ids.size
        batch = q_arr.shape[0]
        nv = self._model.nv
        out_dim = 2 if is_planar else 3
        geom_dim = 2 if is_planar else 3

        jac_q = np.zeros((batch, n_markers, out_dim, nv), dtype=np.float64)
        jac_geom = np.zeros((batch, n_markers, out_dim, geom_dim), dtype=np.float64)

        for t in range(batch):
            pin.forwardKinematics(self._model, self._data, q_arr[t])
            pin.computeJointJacobians(self._model, self._data, q_arr[t])
            for m in range(n_markers):
                j_id = int(joint_ids[m])
                p_offset = offsets[m]
                oMi = self._data.oMi[j_id]
                J6 = pin.getJointJacobian(
                    self._model,
                    self._data,
                    j_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                p_world_offset = oMi.rotation @ p_offset
                J_point = J6[:3, :] - pin.skew(p_world_offset) @ J6[3:, :]
                jac_q[t, m] = J_point[:out_dim, :]
                jac_geom[t, m] = oMi.rotation[:out_dim, :geom_dim]

        return jac_q, jac_geom
