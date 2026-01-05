"""Manipulability Analysis Module for Pinocchio Engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)


@dataclass
class EllipsoidParams:
    """Parameters defining a 3D ellipsoid for visualization."""

    radii: np.ndarray  # (3,)
    axes: np.ndarray  # (3, 3) rotation matrix, columns are axes
    center: np.ndarray  # (3,)


@dataclass
class ManipulabilityResult:
    """Container for manipulability metrics."""

    body_name: str
    J_trans: np.ndarray  # Translational Jacobian (3xN)
    mobility_matrix: np.ndarray  # J J^T (Velocity)
    force_matrix: np.ndarray  # (J J^T)^-1 (Force)
    velocity_ellipsoid: EllipsoidParams
    force_ellipsoid: EllipsoidParams
    condition_number: float
    volume: float  # Proportional to det(J J^T)
    manipulability_index: float  # sqrt(det(J J^T))


class PinocchioManipulabilityAnalyzer:
    """Analyzes mechanism manipulability using Pinocchio."""

    def __init__(self, model: pin.Model, data: pin.Data) -> None:
        """Initialize the analyzer."""
        self.model = model
        self.data = data

    def compute_metrics(
        self, body_name: str, q: np.ndarray | None = None
    ) -> ManipulabilityResult | None:
        """Compute manipulability metrics for a specific frame/body."""
        # Resolve frame ID (Pinocchio tends to use Frames for end effectors)
        if not self.model.existFrame(body_name):
            # Try body name if frame not found?
            if self.model.existBodyName(body_name):
                # Get the body's frame, usually named same or similar
                # For now just fail gracefully
                return None
            return None

        fid = self.model.getFrameId(body_name)
        
        # Ensure data is up to date. If q provided, update kinematics.
        # If q is None, assume data is current.
        if q is not None:
             pin.forwardKinematics(self.model, self.data, q)
             pin.updateFramePlacements(self.model, self.data)
             pin.computeJointJacobians(self.model, self.data)

        # Get Full Jacobian (6xN), locally world aligned
        try:
            J = pin.getFrameJacobian(
                self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
        except ValueError:
            return None

        # Translational part (top 3 rows)
        J_trans = J[:3, :]

        # 2. Compute Core Matrices
        # Velocity Manipulability Matrix M_v = J * J^T
        M_v = J_trans @ J_trans.T

        # Force Manipulability Matrix M_f = (J * J^T)^-1
        try:
            M_f = np.linalg.inv(M_v)
        except np.linalg.LinAlgError:
            M_f = np.linalg.pinv(M_v)

        # 3. Eigen Decomposition
        try:
            eig_val_v, eig_vec_v = np.linalg.eigh(M_v)
        except np.linalg.LinAlgError:
            return None

        idx_v = eig_val_v.argsort()[::-1]
        eig_val_v = eig_val_v[idx_v]
        eig_vec_v = eig_vec_v[:, idx_v]

        radii_v = np.sqrt(np.maximum(eig_val_v, 1e-9))
        radii_f = 1.0 / np.maximum(radii_v, 1e-9)

        cond_num = radii_v[0] / radii_v[-1] if radii_v[-1] > 1e-9 else float("inf")
        manip_index = np.prod(radii_v)

        # Position of the frame
        pos = self.data.oMf[fid].translation

        return ManipulabilityResult(
            body_name=body_name,
            J_trans=J_trans,
            mobility_matrix=M_v,
            force_matrix=M_f,
            velocity_ellipsoid=EllipsoidParams(radii_v, eig_vec_v, pos),
            force_ellipsoid=EllipsoidParams(radii_f, eig_vec_v, pos),
            condition_number=cond_num,
            volume=manip_index,
            manipulability_index=manip_index,
        )

    def find_potential_bodies(self) -> list[str]:
        """Find frame names that look like points of interest."""
        candidates = ["hand", "wrist", "arm", "club", "head", "grip"]
        found = []
        for frame in self.model.frames:
            name = frame.name
            name_lower = name.lower()
            if any(c in name_lower for c in candidates):
                found.append(name)
        return sorted(list(set(found)))
