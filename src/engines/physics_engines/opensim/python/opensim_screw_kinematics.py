"""Screw-theoretic kinematics adapter for the OpenSim engine.

Implements Guideline C3 (ISA/twist extraction) using OpenSim's Jacobian /
velocity API.  The adapter is importable even when ``opensim`` is not
installed; calling any computational method will raise ``ImportError``.

Reference: docs/assessments/project_design_guidelines.qmd Section C3
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.engine_core.engine_availability import OPENSIM_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.screw_theory import (
    ScrewAxis,
    Twist,
    compute_screw_axis,
    compute_screw_endpoints,
)

logger = get_logger(__name__)


class OpenSimScrewKinematics:
    """Screw-theoretic kinematics for an OpenSim musculoskeletal model.

    Uses OpenSim's ``Model.getMatterSubsystem().calcBodyJacobian`` to extract
    spatial twists and instantaneous screw axes (ISA) per Guideline C3.

    The adapter is intentionally lightweight — it holds a reference to the
    shared ``Model`` and ``State`` objects managed by ``OpenSimPhysicsEngine``
    and delegates all screw-axis math to the shared ``screw_theory`` module.

    Example::

        engine = OpenSimPhysicsEngine()
        engine.load_from_path("model.osim")
        sk = OpenSimScrewKinematics(engine._model, engine._state)
        twist = sk.compute_twist("radius")
        screw = sk.compute_screw_axis(twist)
        logger.info("ISA pitch: %.4f m/rad", screw.pitch)
    """

    def __init__(self, model: Any, state: Any) -> None:
        """Initialise with a loaded OpenSim model and state.

        Args:
            model: ``opensim.Model`` instance with initialised system.
            state: ``opensim.State`` instance for current simulation state.

        Raises:
            ImportError: If ``opensim`` is not installed.
        """
        if not OPENSIM_AVAILABLE:
            raise ImportError(
                "opensim is not installed. "
                "Install the OpenSim Python bindings to use OpenSimScrewKinematics."
            )
        assert model is not None, "model must be provided"
        assert state is not None, "state must be provided"
        self.model = model
        self.state = state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_twist(self, body_name: str) -> Twist:
        """Compute spatial twist for a named body.

        Queries the body's angular and linear velocity at its origin from the
        current state using OpenSim's built-in velocity API.

        Args:
            body_name: Name of the body (frame) to query.

        Returns:
            ``Twist`` with angular and linear components.

        Raises:
            ValueError: If ``body_name`` is not found in the model.
        """

        body_set = self.model.getBodySet()
        idx = body_set.getIndex(body_name)
        if idx < 0:
            raise ValueError(f"Body '{body_name}' not found in OpenSim model")

        body = body_set.get(idx)

        # Realise to velocity stage
        self.model.realizeVelocity(self.state)

        # Angular velocity of the body expressed in ground frame
        ang_vec3 = body.getAngularVelocityInGround(self.state)
        angular = np.array([ang_vec3[0], ang_vec3[1], ang_vec3[2]])

        # Linear velocity of the body origin in ground frame
        lin_vec3 = body.getLinearVelocityInGround(self.state)
        linear = np.array([lin_vec3[0], lin_vec3[1], lin_vec3[2]])

        # Body origin position in ground frame
        pos_vec3 = body.getPositionInGround(self.state)
        ref_point = np.array([pos_vec3[0], pos_vec3[1], pos_vec3[2]])

        return Twist(
            angular=angular,
            linear=linear,
            body_name=body_name,
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
        body_names: list[str],
    ) -> dict[str, tuple[Twist, ScrewAxis]]:
        """Compute twist and screw axis for multiple bodies.

        Args:
            body_names: OpenSim body names to analyze.

        Returns:
            Dict mapping body name → (Twist, ScrewAxis).
        """
        results: dict[str, tuple[Twist, ScrewAxis]] = {}
        for name in body_names:
            try:
                twist = self.compute_twist(name)
                screw = self.compute_screw_axis(twist)
                results[name] = (twist, screw)
            except ValueError as exc:
                logger.warning("Skipping body '%s': %s", name, exc)
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
