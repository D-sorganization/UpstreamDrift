"""Screw-theoretic kinematics (Guideline C3 - Required).

This module implements screw theory analysis per project design guidelines
Section C3: "Instantaneous screw axis (ISA) / twist extraction at key task
points. Visualization of screw axis and pitch where meaningful."

Screw theory provides a unified geometric framework for describing rigid body
motion, combining rotation and translation into a single entity (the twist).

Reference: docs/assessments/project_design_guidelines.qmd Section C3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

logger = logging.getLogger(__name__)


@dataclass
class Twist:
    """Spatial twist (6D velocity) in screw theory.

    A twist represents the instantaneous motion of a rigid body as a screw
    motion: rotation about an axis combined with translation along that axis.

    Attributes:
        angular: Angular velocity vector [3] (rad/s)
        linear: Linear velocity vector [3] (m/s)
        body_name: Name of the body this twist describes
        reference_point: Point where linear velocity is measured [3] (m)
    """

    angular: np.ndarray
    linear: np.ndarray
    body_name: str
    reference_point: np.ndarray


@dataclass
class ScrewAxis:
    """Instantaneous Screw Axis (ISA) representation.

    Per Guideline C3, this represents the instantaneous axis of rotation
    and translation (the screw axis) for a rigid body motion.

    Attributes:
        axis_direction: Unit vector along screw axis [3] (dimensionless)
        axis_point: A point on the screw axis [3] (m)
        pitch: Screw pitch (h = v_parallel / ω) [m/rad]
                Special cases:
                - h = 0: Pure rotation about axis
                - h = ∞: Pure translation along axis
                - h finite: Helical motion (螺旋运动)
        angular_magnitude: Magnitude of angular velocity (rad/s)
        linear_magnitude: Magnitude of linear velocity (m/s)
        is_singular: True if motion is pure translation (ω ≈ 0)
    """

    axis_direction: np.ndarray
    axis_point: np.ndarray
    pitch: float
    angular_magnitude: float
    linear_magnitude: float
    is_singular: bool


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

        Per Guideline C3, extracts the screw axis representation:
        - Axis direction (unit vector)
        - Axis location (point on axis)
        - Pitch (ratio of translation to rotation)

        Args:
            twist: Spatial twist to analyze
            singularity_threshold: Threshold for detecting pure translation

        Returns:
            ScrewAxis with complete representation
        """
        ω = twist.angular
        v = twist.linear
        r = twist.reference_point

        # Check for singular case (pure translation, ω ≈ 0)
        ω_mag = float(np.linalg.norm(ω))
        v_mag = float(np.linalg.norm(v))

        if ω_mag < singularity_threshold:
            # Pure translation: screw axis is at infinity
            # Direction is along velocity
            if v_mag > singularity_threshold:
                axis_dir = v / v_mag
            else:
                # No motion at all
                axis_dir = np.array([0.0, 0.0, 1.0])  # Arbitrary

            axis_point = r.copy()
            pitch = float("inf")
            is_singular = True

        else:
            # General case: screw motion
            is_singular = False

            # 1. Axis direction: ŝ = ω / |ω|
            axis_dir = ω / ω_mag

            # 2. Pitch: h = (ω · v) / |ω|²
            # This is the component of v parallel to ω, divided by ω magnitude
            pitch = float(np.dot(ω, v) / (ω_mag**2))

            # 3. Axis location: Find point q on axis closest to reference point
            # The axis passes through q such that:
            # q = r + (ω × v) / |ω|²
            # This is the "moment" of the twist

            axis_point = r + np.cross(ω, v) / (ω_mag**2)

        return ScrewAxis(
            axis_direction=axis_dir,
            axis_point=axis_point,
            pitch=pitch,
            angular_magnitude=ω_mag,
            linear_magnitude=v_mag,
            is_singular=is_singular,
        )

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

        Per Guideline C3, provides data for visualizing the screw axis.

        Args:
            screw: Screw axis to visualize
            length: Length of axis segment to draw [m]

        Returns:
            Tuple of (start_point, end_point) for line segment [3], [3]
        """
        if screw.is_singular:
            # Pure translation: draw along velocity direction
            start = screw.axis_point
            end = screw.axis_point + screw.axis_direction * length
        else:
            # Draw axis segment centered at axis_point
            start = screw.axis_point - screw.axis_direction * (length / 2)
            end = screw.axis_point + screw.axis_direction * (length / 2)

        return start, end

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


def plot_screw_axis_3d(
    ax: any,  # type: ignore[valid-type]
    screw: ScrewAxis,
    length: float = 0.5,
    color: str = "blue",
    label: str | None = None,
) -> None:
    """Plot screw axis in 3D matplotlib axes.

    Helper function for visualizing screw axes.

    Args:
        ax: Matplotlib 3D axes
        screw: Screw axis to plot
        length: Length of axis to draw
        color: Color for the axis
        label: Label for legend
    """
    # Compute visualization points directly without __new__ code smell
    if screw.is_singular:
        # Pure translation: draw along velocity direction
        start = screw.axis_point
        end = screw.axis_point + screw.axis_direction * length
    else:
        # Draw axis segment centered at axis_point
        start = screw.axis_point - screw.axis_direction * (length / 2)
        end = screw.axis_point + screw.axis_direction * (length / 2)

    # Draw axis as line
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        linewidth=3,
        label=label,
    )

    # Draw arrow at end
    arrow_length = length * 0.1
    arrow = end - start
    arrow_norm = arrow / np.linalg.norm(arrow)

    ax.quiver(
        end[0],
        end[1],
        end[2],
        arrow_norm[0],
        arrow_norm[1],
        arrow_norm[2],
        length=arrow_length,
        color=color,
        arrow_length_ratio=0.3,
    )

    # Annotate pitch if not singular
    if not screw.is_singular and abs(screw.pitch) < 10:
        mid = (start + end) / 2
        ax.text(
            mid[0],
            mid[1],
            mid[2],
            f"h={screw.pitch:.3f}",
            fontsize=8,
        )
