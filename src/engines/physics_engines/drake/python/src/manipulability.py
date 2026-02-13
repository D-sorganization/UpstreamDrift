"""Manipulability Analysis for Drake physics engine.

Computes Force and Mobility ellipsoids/matrices using Drake's MultibodyPlant.
"""

from dataclasses import dataclass

import numpy as np

# Import from centralized availability module (DRY compliance)
from src.shared.python.engine_core.engine_availability import DRAKE_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger

# Try to import Drake types for actual usage
if DRAKE_AVAILABLE:
    from pydrake.all import (
        BodyIndex,
        Context,
        JacobianWrtVariable,
        MultibodyPlant,
    )
else:
    # Dummies for type checking when Drake is not installed
    MultibodyPlant = object  # type: ignore[misc, assignment]  # dummy fallback
    Context = object  # type: ignore[misc, assignment]  # dummy fallback
    BodyIndex = object  # type: ignore[misc, assignment]  # dummy fallback
    JacobianWrtVariable = object  # type: ignore[misc, assignment]  # dummy fallback


logger = get_logger(__name__)


@dataclass
class EllipsoidParams:
    """Parameters defining an ellipsoid."""

    radii: np.ndarray
    axes: np.ndarray  # Column vectors (eigenvectors)
    center: np.ndarray


@dataclass
class ManipulabilityResult:
    """Results of manipulability analysis for a single body."""

    body_name: str
    cartesian_pos: np.ndarray
    mobility_ellipsoid: EllipsoidParams | None
    force_ellipsoid: EllipsoidParams | None
    condition_number: float
    isotropy: float
    manipulability_index: float


class DrakeManipulabilityAnalyzer:
    """Computes manipulability metrics using Drake."""

    def __init__(self, plant: MultibodyPlant) -> None:
        """Initialize with a Drake MultibodyPlant."""
        self.plant = plant
        if DRAKE_AVAILABLE:
            self.world_frame = plant.world_frame()

    def find_potential_bodies(self) -> list[str]:
        """Return a list of body names suitable for analysis."""
        if not DRAKE_AVAILABLE:
            return []

        bodies = []
        for i in range(self.plant.num_bodies()):
            body = self.plant.get_body(BodyIndex(i))
            name = body.name()
            # Filter dummy/world bodies if needed
            if name != "world" and "dummy" not in name:
                bodies.append(name)

        # Heuristic sort: Put club/hands first
        priority = ["club", "club_head", "right_hand", "left_hand"]

        def sort_key(b: str) -> tuple[int, str]:
            for i, p in enumerate(priority):
                if p in b.lower():
                    return (i, b)
            return (len(priority), b)

        return sorted(bodies, key=sort_key)

    def _compute_translational_jacobian(self, context: Context, body):
        return self.plant.CalcJacobianTranslationalVelocity(
            context,
            JacobianWrtVariable.kV,
            body.body_frame(),
            np.zeros(3),  # type: ignore[arg-type]  # pydrake accepts ndarray at runtime
            self.world_frame,
            self.world_frame,
        )

    def _decompose_mobility_matrix(self, mobility_matrix):
        eigvals_v, eigvecs_v = np.linalg.eigh(mobility_matrix)
        idx = np.argsort(eigvals_v)[::-1]
        eigvals_v = eigvals_v[idx]
        eigvecs_v = eigvecs_v[:, idx]
        radii_v = np.sqrt(np.maximum(eigvals_v, 1e-9))
        return radii_v, eigvecs_v

    def _check_condition_number(self, name, cond):
        if cond > 1e6:
            logger.warning(
                f"High Jacobian condition number for {name}: k={cond:.2e}. "
                f"Near singularity - manipulability metrics may be unreliable. "
                f"Guideline O3 warning threshold exceeded."
            )

        if cond > 1e10:
            logger.error(
                f"Jacobian is singular for {name}: k={cond:.2e}. "
                f"Guideline O3 VIOLATION - system at kinematic singularity."
            )
            return False
        return True

    def _build_result_for_body(
        self, context: Context, name, body, radii_v, eigvecs_v, cond
    ):
        isotropy = 1.0 / cond if cond > 0 else 0.0
        manip_index = np.prod(radii_v)
        radii_f = 1.0 / np.maximum(radii_v, 1e-9)

        pose = self.plant.EvalBodyPoseInWorld(context, body)
        cartesian_pos = pose.translation()

        return ManipulabilityResult(
            body_name=name,
            cartesian_pos=cartesian_pos,
            mobility_ellipsoid=EllipsoidParams(radii_v, eigvecs_v, cartesian_pos),
            force_ellipsoid=EllipsoidParams(radii_f, eigvecs_v, cartesian_pos),
            condition_number=cond,
            isotropy=isotropy,
            manipulability_index=manip_index,
        )

    def compute_metrics(
        self, context: Context, body_names: list[str]
    ) -> list[ManipulabilityResult]:
        """Compute manipulability metrics for the specified bodies.

        Args:
            context: The plant Context (must have positions/velocities set).
            body_names: List of body names to analyze.

        Returns:
            List of ManipulabilityResult.
        """
        if not DRAKE_AVAILABLE:
            return []

        results = []

        for name in body_names:
            if not self.plant.HasBodyNamed(name):
                continue

            body = self.plant.GetBodyByName(name)
            J_full = self._compute_translational_jacobian(context, body)
            mobility_matrix = J_full @ J_full.T

            try:
                radii_v, eigvecs_v = self._decompose_mobility_matrix(mobility_matrix)
                cond = radii_v[0] / radii_v[-1] if radii_v[-1] > 1e-9 else float("inf")

                if not self._check_condition_number(name, cond):
                    continue

            except np.linalg.LinAlgError:
                continue

            res = self._build_result_for_body(
                context, name, body, radii_v, eigvecs_v, cond
            )
            results.append(res)

        return results
