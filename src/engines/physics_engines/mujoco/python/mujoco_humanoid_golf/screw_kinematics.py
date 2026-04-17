"""Screw-theoretic kinematics (Guideline C3 - Required).

This module implements screw theory analysis per project design guidelines
Section C3: "Instantaneous screw axis (ISA) / twist extraction at key task
points. Visualization of screw axis and pitch where meaningful."

Screw theory provides a unified geometric framework for describing rigid body
motion, combining rotation and translation into a single entity (the twist).

Reference: docs/assessments/project_design_guidelines.qmd Section C3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.screw_theory import (
    ScrewAxis,
    Twist,
    compute_screw_axis,
    compute_screw_endpoints,
)

if TYPE_CHECKING:
    import mujoco

logger = get_logger(__name__)


class ScrewKinematicsAnalyzer:
    """Analyze screw-theoretic kinematics (Guideline C3).

    This is a REQUIRED feature per project design guidelines Section C3.
    Implements:
    - Twist extraction from Jacobians
    - Instantaneous Screw Axis (ISA) computation
    - Pitch calculation
    - Screw visualization support

    Example:
        >>> model = mujoco.MjModel.from_xml_path("humanoid.xml")
        >>> analyzer = ScrewKinematicsAnalyzer(model)
        >>>
        >>> # Extract twist for clubhead
        >>> body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "club_head")
        >>> twist = analyzer.compute_twist(qpos, qvel, body_id)
        >>> print(f"Angular velocity: {twist.angular}")
        >>> print(f"Linear velocity: {twist.linear}")
        >>>
        >>> # Compute ISA
        >>> screw = analyzer.compute_screw_axis(twist)
        >>> print(f"Screw axis direction: {screw.axis_direction}")
        >>> print(f"Pitch: {screw.pitch} m/rad")
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize screw kinematics analyzer.

        Args:
            model: MuJoCo model
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model

        # Thread-safe data structure
        import mujoco

        self._data = mujoco.MjData(model)

    def compute_twist(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        body_id: int,
        reference_point: np.ndarray | None = None,
    ) -> Twist:
        """Compute spatial twist for a body.

        The twist is a 6D vector [ω; v] where:
        - ω is angular velocity (3D)
        - v is linear velocity at reference point (3D)

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            body_id: Body ID to analyze
            reference_point: Point for linear velocity [3] (default: body COM)

        Returns:
            Twist with angular and linear velocities
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        import mujoco

        # Set state
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel

        # Forward kinematics
        mujoco.mj_forward(self.model, self._data)

        # Get Jacobians at body COM
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        mujoco.mj_jacBodyCom(self.model, self._data, jacp, jacr, body_id)

        # Compute twist: [ω; v] = J * qvel
        angular = jacr @ qvel
        linear = jacp @ qvel

        # Reference point (default: COM)
        if reference_point is None:
            reference_point = self._data.xpos[body_id].copy()

        body = self.model.body(body_id)

        return Twist(
            angular=angular,
            linear=linear,
            body_name=body.name,
            reference_point=reference_point,
        )

    def compute_screw_axis(
        self,
        twist: Twist,
        singularity_threshold: float = 1e-6,
    ) -> ScrewAxis:
        """Compute Instantaneous Screw Axis from twist.
        Delegates to shared abstractions.
        """
        return compute_screw_axis(twist, singularity_threshold)

    def analyze_key_points(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        body_names: list[str],
    ) -> dict[str, tuple[Twist, ScrewAxis]]:
        """Analyze screw kinematics for key task points.

        Per Guideline C3, analyzes multiple key points:
        - Clubhead, grip
        - Left hand, right hand
        - Forearms, upper arms, torso

        Args:
            qpos: Joint positions [nv]
            qvel: Joint velocities [nv]
            body_names: List of body names to analyze

        Returns:
            Dict mapping body name to (twist, screw_axis) tuple
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        import mujoco

        results = {}

        for name in body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

            if body_id == -1:
                logger.warning(f"Body '{name}' not found in model")
                continue

            twist = self.compute_twist(qpos, qvel, body_id)
            screw = self.compute_screw_axis(twist)

            results[name] = (twist, screw)

        return results

    def visualize_screw_axis(
        self,
        screw: ScrewAxis,
        length: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate line segment for screw axis visualization.
        Delegates to shared generic logic.
        """
        return compute_screw_endpoints(screw, length)

    def compute_manipulability_screw(
        self,
        qpos: np.ndarray,
        body_id: int,
    ) -> float:
        """Compute manipulability measure in screw coordinates.

        This is the volume of the manipulability ellipsoid, which measures
        how "easy" it is to move the end-effector in all directions.

        Args:
            qpos: Joint positions [nv]
            body_id: Body ID to analyze

        Returns:
            Manipulability measure (dimensionless)
        """
        if qpos is None:
            raise ValueError("qpos must be provided")
        import mujoco

        # Set state
        self._data.qpos[:] = qpos
        self._data.qvel[:] = np.zeros(self.model.nv)

        mujoco.mj_forward(self.model, self._data)

        # Get 6D Jacobian (stacked angular + linear)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        mujoco.mj_jacBodyCom(self.model, self._data, jacp, jacr, body_id)

        # Stack into 6×nv Jacobian: J = [jacr; jacp]
        J = np.vstack([jacr, jacp])

        # Manipulability: μ = √det(J J^T)
        # For redundant systems (nv > 6), use pseudoinverse
        if self.model.nv >= 6:
            JJT = J @ J.T
            # Compute determinant (if full rank)
            try:
                manip = float(np.sqrt(np.linalg.det(JJT)))
            except np.linalg.LinAlgError:
                manip = 0.0
        else:
            # Underdetermined (nv < 6)
            manip = 0.0

        return manip
