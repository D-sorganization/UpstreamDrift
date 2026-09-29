"""Screw-theoretic kinematics adapter for the Pinocchio engine.

Implements Guideline C3 (ISA/twist extraction) using Pinocchio's Jacobian
API.  The adapter is importable even when ``pinocchio`` is not installed;
calling any computational method will raise ``ImportError`` at that point.

Reference: docs/assessments/project_design_guidelines.qmd Section C3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.engine_core.engine_availability import PINOCCHIO_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.screw_theory import (
    ScrewAxis,
    Twist,
    compute_screw_axis,
    compute_screw_endpoints,
)

if TYPE_CHECKING:
    import pinocchio as pin

if PINOCCHIO_AVAILABLE:
    import pinocchio as pin  # noqa: F811

logger = get_logger(__name__)


class PinocchioScrewKinematics:
    """Screw-theoretic kinematics for a Pinocchio rigid-body model.

    Wraps Pinocchio's ``computeJointJacobians`` / ``getFrameJacobian`` to
    extract spatial twists and instantaneous screw axes (ISA) per
    Guideline C3.

    Example::

        engine = PinocchioPhysicsEngine()
        engine.load_from_path("robot.urdf")
        sk = PinocchioScrewKinematics(engine.model, engine.data)
        twist = sk.compute_twist(q, v, "end_effector")
        screw = sk.compute_screw_axis(twist)
        logger.info("Pitch: %.4f m/rad", screw.pitch)
    """

    def __init__(self, model: pin.Model, data: pin.Data) -> None:
        """Initialise with a loaded Pinocchio model and data.

        Args:
            model: Pinocchio model (joint topology, inertias, etc.).
            data: Pinocchio data (mutable computation buffer).
        """
        if not PINOCCHIO_AVAILABLE:
            raise ImportError(
                "pinocchio is not installed. "
                "Install it to use PinocchioScrewKinematics."
            )
        if not (model is not None):
            raise ValueError("model must be provided")
        if not (data is not None):
            raise ValueError("data must be provided")
        self.model = model
        self.data = data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_twist(
        self,
        q: np.ndarray,
        v: np.ndarray,
        frame_name: str,
        reference_frame: pin.ReferenceFrame | None = None,
    ) -> Twist:
        """Compute spatial twist for a named frame.

        Runs forward kinematics and Jacobian computation then multiplies
        J(q) · v to obtain the 6D twist [ω; v] at the given frame.

        Args:
            q: Joint positions [nq].
            v: Joint velocities [nv].
            frame_name: Name of frame or body to query.
            reference_frame: Pinocchio reference frame for the Jacobian.
                Defaults to ``pin.ReferenceFrame.LOCAL_WORLD_ALIGNED``.

        Returns:
            ``Twist`` with angular and linear components.

        Raises:
            ValueError: If ``frame_name`` is not found in the model.
            ImportError: If pinocchio is not installed.
        """
        if not (q is not None):
            raise ValueError("q must be provided")
        if not (v is not None):
            raise ValueError("v must be provided")

        if reference_frame is None:
            reference_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Look up frame ID
        frame_id = self.model.getFrameId(frame_name)
        if frame_id >= self.model.nframes:
            raise ValueError(f"Frame '{frame_name}' not found in Pinocchio model")

        # Forward kinematics + Jacobians
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)

        # 6×nv Jacobian: rows 0-2 angular, rows 3-5 linear
        J = pin.getFrameJacobian(self.model, self.data, frame_id, reference_frame)

        spatial_vel = J @ v
        angular = spatial_vel[:3]
        linear = spatial_vel[3:]

        ref_point = self.data.oMf[frame_id].translation.copy()

        return Twist(
            angular=angular,
            linear=linear,
            body_name=frame_name,
            reference_point=ref_point,
        )

    def compute_screw_axis(
        self,
        twist: Twist,
        singularity_threshold: float = 1e-6,
    ) -> ScrewAxis:
        """Compute ISA from a twist. Delegates to shared module."""
        return compute_screw_axis(twist, singularity_threshold)

    def analyze_key_points(
        self,
        q: np.ndarray,
        v: np.ndarray,
        frame_names: list[str],
    ) -> dict[str, tuple[Twist, ScrewAxis]]:
        """Compute twist and screw axis for multiple named frames.

        Args:
            q: Joint positions [nq].
            v: Joint velocities [nv].
            frame_names: Pinocchio frame names to analyze.

        Returns:
            Dict mapping frame name → (Twist, ScrewAxis).
        """
        if not (q is not None):
            raise ValueError("q must be provided")
        results: dict[str, tuple[Twist, ScrewAxis]] = {}
        for name in frame_names:
            try:
                twist = self.compute_twist(q, v, name)
                screw = self.compute_screw_axis(twist)
                results[name] = (twist, screw)
            except ValueError as exc:
                logger.warning("Skipping frame '%s': %s", name, exc)
        return results

    def visualize_screw_axis(
        self,
        screw: ScrewAxis,
        length: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate line segment for screw axis visualization.

        Delegates to the shared ``compute_screw_endpoints`` utility.

        Args:
            screw: Screw axis to visualize.
            length: Length of axis segment [m].

        Returns:
            Tuple of (start_point, end_point) [3], [3].
        """
        return compute_screw_endpoints(screw, length)
