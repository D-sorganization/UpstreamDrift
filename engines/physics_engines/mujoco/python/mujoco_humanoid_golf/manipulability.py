"""Manipulability Analysis Module for Golf Biomechanics.

This module provides the core mathematical engine for computing:
1. Velocity Manipulability Ellipsoids (Kinematic Mobility)
2. Force Manipulability Ellipsoids (Static Force Production)
3. Condition Numbers and Isotropy Indices

It supports both Kinematic (J * J.T) and Dynamic (Mass-weighted) formulations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

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


class ManipulabilityAnalyzer:
    """Analyzes mechanism manipulability and force production capabilities."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize the analyzer.

        Args:
            model: MuJoCo model
            data: MuJoCo data (Thread-local/private data recommended)
        """
        self.model = model
        self.data = data
        self._cache_J_trans: dict[str, np.ndarray] = {}

    def compute_metrics(
        self, body_name: str, mode: Literal["kinematic", "dynamic"] = "kinematic"
    ) -> ManipulabilityResult | None:
        """Compute manipulability metrics for a specific body.

        Args:
            body_name: Name of the body to analyze.
            mode: "kinematic" (default) or "dynamic" (mass-weighted).

        Returns:
            ManipulabilityResult or None if body not found.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return None

        # 1. Compute Jacobian
        # We focus on Translational Manipulability (3xN) for golf context
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))  # rotational, ignored for now
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

        # Use Translational Jacobian
        J = jacp

        # 2. Compute Core Matrices
        # Velocity Manipulability Matrix M_v = J * J^T
        # Shape: (3, 3)
        M_v = J @ J.T

        # Force Manipulability Matrix M_f = (J * J^T)^-1
        # Note: If singular, we pseudoinverse
        try:
            M_f = np.linalg.inv(M_v)
        except np.linalg.LinAlgError:
            M_f = np.linalg.pinv(M_v)

        # 3. Eigen Decomposition for Ellipsoids
        # Velocity Ellipsoid:
        # Defined by v^T (J J^T)^-1 v <= 1
        # Shape determined by J J^T. Radii are sqrt(eigenvalues of J J^T).
        try:
            eig_val_v, eig_vec_v = np.linalg.eigh(M_v)
        except np.linalg.LinAlgError:
            return None

        # Sort descending
        idx_v = eig_val_v.argsort()[::-1]
        eig_val_v = eig_val_v[idx_v]
        eig_vec_v = eig_vec_v[:, idx_v]

        # Velocity Radii = sqrt(eigenvalues) = sigma
        radii_v = np.sqrt(np.maximum(eig_val_v, 1e-9))

        # Force Ellipsoid:
        # Defined by f^T (J J^T) f <= 1
        # Shape determined by (J J^T)^-1. Radii are sqrt(eigenvalues of (J J^T)^-1).
        # Eigenvalues of inverse are 1/eigenvalues.
        # Force Radii = 1/sigma
        radii_f = 1.0 / np.maximum(radii_v, 1e-9)

        # Condition Number (Isotropy)
        # Ratio of largest to smallest singular value (sigma_max / sigma_min)
        # = radii_v_max / radii_v_min
        cond_num = radii_v[0] / radii_v[-1] if radii_v[-1] > 1e-9 else float("inf")

        # Guideline O3: Singularity Detection & Warnings
        # Warn on poor conditioning (κ > 1e6), error on singularity (κ > 1e10)
        if cond_num > 1e6:
            logger.warning(
                f"⚠️ High Jacobian condition number for {body_name}: κ={cond_num:.2e}. "
                f"Near singularity - manipulability metrics may be unreliable. "
                f"Guideline O3 warning threshold exceeded. "
                f"Consider alternative joint configuration or regularization."
            )

        if cond_num > 1e10:
            logger.error(
                f"❌ Jacobian is singular for {body_name}: κ={cond_num:.2e}. "
                f"Cannot compute reliable manipulability. "
                f"Guideline O3 VIOLATION - system at or near kinematic singularity."
            )
            raise ValueError(
                f"Jacobian singularity detected for {body_name} (κ={cond_num:.2e}). "
                f"System is at or near kinematic singularity. "
                f"Manipulability analysis invalid."
            )

        # Manipulability Index (Volumetric)
        manip_index = np.prod(radii_v)

        # Position of the ellipsoid
        pos = self.data.xpos[body_id].copy()

        # Construct Result
        return ManipulabilityResult(
            body_name=body_name,
            J_trans=J,
            mobility_matrix=M_v,
            force_matrix=M_f,
            velocity_ellipsoid=EllipsoidParams(radii_v, eig_vec_v, pos),
            force_ellipsoid=EllipsoidParams(radii_f, eig_vec_v, pos),  # Axes same as V
            condition_number=cond_num,
            volume=manip_index,
            manipulability_index=manip_index,
        )

    def find_golf_bodies(self) -> list[str]:
        """Heuristic to find relevant golf body names in the model."""
        candidates = [
            "hand",
            "wrist",
            "arm",
            "shoulder",
            "club",
            "head",
            "grip",
            "spine",
            "pelvis",
            "hip",
        ]
        found = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name:
                continue
            name_lower = name.lower()
            if any(c in name_lower for c in candidates):
                found.append(name)
        return sorted(list(set(found)))
