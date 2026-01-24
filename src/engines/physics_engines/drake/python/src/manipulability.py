"""Manipulability Analysis for Drake physics engine.

Computes Force and Mobility ellipsoids/matrices using Drake's MultibodyPlant.
"""

from dataclasses import dataclass

import numpy as np

from src.shared.python.logging_config import get_logger

# Try to import Drake. If failing, we define dummies or rely on user env.
try:
    from pydrake.all import (
        BodyIndex,
        Context,
        JacobianWrtVariable,
        MultibodyPlant,
    )

    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    # Dummies for type checking
    MultibodyPlant = object
    Context = object
    BodyIndex = object
    JacobianWrtVariable = object


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

    def __init__(self, plant: MultibodyPlant):
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

            # Compute Translational Jacobian J (3 x nv)
            # Frame B: body frame
            # p_BoBp_B: Point of interest in B. We use origin (0,0,0).
            # Frame A: World
            # Frame E: World (Expressed in)
            J_full = self.plant.CalcJacobianTranslationalVelocity(
                context,
                JacobianWrtVariable.kV,
                body.body_frame(),
                np.zeros(3),
                self.world_frame,
                self.world_frame,
            )

            # J is 3 x nv.
            # For redundant manipulators, J J^T (3x3) gives mobility.

            # Mobility: Mv = J J^T
            mobility_matrix = J_full @ J_full.T

            # Eigen Decomposition
            try:
                eigvals_v, eigvecs_v = np.linalg.eigh(mobility_matrix)
                # Sort descending
                idx = np.argsort(eigvals_v)[::-1]
                eigvals_v = eigvals_v[idx]
                eigvecs_v = eigvecs_v[:, idx]

                # Radii (Velocity) = sqrt(lambda)
                radii_v = np.sqrt(np.maximum(eigvals_v, 1e-9))

                # Axes are eigenvectors

                # Condition Number
                cond = radii_v[0] / radii_v[-1] if radii_v[-1] > 1e-9 else float("inf")

                # Guideline O3: Singularity Detection & Warnings
                # Warn on poor conditioning (κ > 1e6), error on singularity (κ > 1e10)
                if cond > 1e6:
                    logger.warning(
                        f"⚠️ High Jacobian condition number for {name}: κ={cond:.2e}. "
                        f"Near singularity - manipulability metrics may be unreliable. "
                        f"Guideline O3 warning threshold exceeded."
                    )

                if cond > 1e10:
                    logger.error(
                        f"❌ Jacobian is singular for {name}: κ={cond:.2e}. "
                        f"Guideline O3 VIOLATION - system at kinematic singularity."
                    )
                    # Drake: Continue instead of raising to allow
                    # processing other bodies
                    continue

                # Isotropy (1/cond)
                isotropy = 1.0 / cond if cond > 0 else 0.0

                # Manipulability Index (Measure of volume ~ prod(radii))
                manip_index = np.prod(radii_v)

            except np.linalg.LinAlgError:
                continue

            # Force Ellipsoid
            # Mf = (J J^T)^-1
            # Radii = 1 / sqrt(lambda_v)
            radii_f = 1.0 / np.maximum(radii_v, 1e-9)
            # Axes are same as velocity

            # Position
            pose = self.plant.EvalBodyPoseInWorld(context, body)
            cartesian_pos = pose.translation()

            res = ManipulabilityResult(
                body_name=name,
                cartesian_pos=cartesian_pos,
                mobility_ellipsoid=EllipsoidParams(radii_v, eigvecs_v, cartesian_pos),
                force_ellipsoid=EllipsoidParams(radii_f, eigvecs_v, cartesian_pos),
                condition_number=cond,
                isotropy=isotropy,
                manipulability_index=manip_index,
            )
            results.append(res)

        return results
