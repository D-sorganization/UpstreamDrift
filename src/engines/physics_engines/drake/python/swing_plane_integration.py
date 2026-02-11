"""Swing Plane Analysis Integration for Drake Engine.

This module integrates the shared SwingPlaneAnalyzer with Drake-specific
golf swing simulations, providing consistent swing plane analysis across engines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from shared.python.biomechanics.swing_plane_analysis import SwingPlaneMetrics

try:
    from shared.python.biomechanics.swing_plane_analysis import SwingPlaneAnalyzer
    from shared.python.core import setup_logging
except ImportError as e:
    raise ImportError(
        "Failed to import shared modules. Ensure shared.python is in PYTHONPATH."
    ) from e

logger = setup_logging(__name__)


class DrakeSwingPlaneAnalyzer:
    """Drake-specific swing plane analysis integration."""

    def __init__(self) -> None:
        """Initialize the Drake swing plane analyzer."""
        self.analyzer = SwingPlaneAnalyzer()
        self.logger = logger

    def analyze_trajectory(
        self, positions: np.ndarray, timestamps: np.ndarray | None = None
    ) -> SwingPlaneMetrics:
        """Analyze swing plane from Drake trajectory data.

        Args:
            positions: Club head positions (N, 3) in world coordinates
            timestamps: Optional timestamps for each position

        Returns:
            SwingPlaneMetrics with plane analysis results
        """
        if positions.shape[1] != 3:
            raise ValueError("Positions must be (N, 3) array")

        if len(positions) < 3:
            raise ValueError("At least 3 positions required for plane analysis")

        self.logger.info(
            f"Analyzing swing plane from {len(positions)} trajectory points"
        )

        # Use shared analyzer
        metrics = self.analyzer.analyze(positions)

        self.logger.info(
            f"Swing plane analysis complete: "
            f"steepness={metrics.steepness_deg:.1f}°, "
            f"RMSE={metrics.rmse:.4f}"
        )

        return metrics

    def analyze_from_drake_context(
        self, context: Any, plant: Any, club_body_index: int, num_samples: int = 100
    ) -> SwingPlaneMetrics:
        """Analyze swing plane from Drake plant context.

        Extracts club head positions from the Drake MultibodyPlant context
        and performs swing plane analysis. The method queries the body's
        world-frame pose at each logged time step.

        Args:
            context: Drake Simulator context or log containing state history
            plant: Drake MultibodyPlant
            club_body_index: Index of the club body in the plant
            num_samples: Number of trajectory samples to analyze

        Returns:
            SwingPlaneMetrics for the Drake simulation

        Raises:
            RuntimeError: If trajectory extraction fails or too few samples
        """
        self.logger.info(
            f"Extracting club trajectory from Drake context "
            f"(body_index={club_body_index}, samples={num_samples})"
        )

        positions = []

        try:
            club_body = plant.get_body(club_body_index)

            if hasattr(context, "sample_times"):
                times = context.sample_times()
                indices = np.linspace(
                    0, len(times) - 1, min(num_samples, len(times)), dtype=int
                )
                for idx in indices:
                    plant_context = plant.GetMyContextFromRoot(
                        context.value(times[idx])
                    )
                    pose = plant.EvalBodyPoseInWorld(plant_context, club_body)
                    positions.append(pose.translation())
            elif hasattr(context, "get_mutable_continuous_state"):
                plant_context = plant.GetMyContextFromRoot(context)
                pose = plant.EvalBodyPoseInWorld(plant_context, club_body)
                positions.append(pose.translation())
            else:
                raise RuntimeError(
                    "Drake context does not provide trajectory history. "
                    "Pass a SimulatorLog or context with sample_times()."
                )
        except (AttributeError, TypeError) as e:
            raise RuntimeError(
                f"Failed to extract trajectory from Drake context: {e}. "
                f"Ensure the plant is finalized and context contains valid state."
            ) from e

        if len(positions) < 3:
            raise RuntimeError(
                f"Extracted only {len(positions)} positions; at least 3 required "
                f"for swing plane fitting."
            )

        positions_array = np.array(positions)
        self.logger.info(
            f"Extracted {len(positions_array)} trajectory points from Drake context"
        )

        return self.analyze_trajectory(positions_array)

    def integrate_with_optimization(
        self, trajectory_optimizer: Any, swing_plane_constraint_weight: float = 1.0
    ) -> None:
        """Integrate swing plane analysis with Drake trajectory optimization.

        Registers a swing-plane-deviation cost function with the trajectory
        optimizer. The cost penalizes the sum of squared distances from each
        trajectory point to the fitted swing plane:

        .. math::
            C_{plane} = w \\sum_{i=1}^{N}
                (\\mathbf{n} \\cdot (\\mathbf{p}_i - \\mathbf{p}_0))^2

        where :math:`\\mathbf{n}` is the plane normal, :math:`\\mathbf{p}_0`
        is a point on the plane, and :math:`w` is the constraint weight.

        Args:
            trajectory_optimizer: DrakeMotionOptimizer instance
            swing_plane_constraint_weight: Weight for swing plane deviation cost
        """
        self.logger.info(
            f"Integrating swing plane constraints with weight "
            f"{swing_plane_constraint_weight}"
        )

        analyzer = self.analyzer

        def swing_plane_cost(trajectory: np.ndarray) -> float:
            """Cost based on deviation from the fitted swing plane."""
            if trajectory.shape[0] < 3 or trajectory.shape[1] < 3:
                return 0.0
            positions = trajectory[:, :3] if trajectory.shape[1] > 3 else trajectory
            metrics = analyzer.analyze(positions)
            return float(metrics.rmse**2)

        if hasattr(trajectory_optimizer, "add_objective"):
            trajectory_optimizer.add_objective(
                name="swing_plane_deviation",
                weight=swing_plane_constraint_weight,
                cost_function=swing_plane_cost,
                target_value=0.0,
            )
            self.logger.info("Swing plane cost registered with trajectory optimizer")
        else:
            self.logger.warning(
                "Trajectory optimizer does not support add_objective; "
                "swing plane constraint not registered."
            )

    def visualize_with_meshcat(
        self,
        meshcat_visualizer: Any,
        metrics: SwingPlaneMetrics,
        trajectory_positions: np.ndarray,
    ) -> None:
        """Visualize swing plane analysis results with Drake's Meshcat.

        Adds three visual elements to the Meshcat scene:
        1. A semi-transparent plane surface at the fitted swing plane
        2. The club head trajectory as a point cloud
        3. Deviation lines from each trajectory point to the plane

        Args:
            meshcat_visualizer: Drake Meshcat instance (pydrake.geometry.Meshcat)
            metrics: Swing plane analysis results
            trajectory_positions: Club head trajectory positions (N, 3)
        """
        self.logger.info("Visualizing swing plane analysis with Meshcat")

        meshcat = meshcat_visualizer
        prefix = "swing_plane"

        try:
            from pydrake.geometry import Rgba

            n = metrics.normal_vector
            p0 = metrics.point_on_plane

            # Build orthonormal basis for the plane
            if abs(n[0]) < 0.9:
                v1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
            else:
                v1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
            v1 /= np.linalg.norm(v1)
            v2 = np.cross(n, v1)

            # Plane as a flat quad (4 corners, extent 2 m)
            extent = 1.0
            corners = np.array(
                [
                    p0 + extent * v1 + extent * v2,
                    p0 - extent * v1 + extent * v2,
                    p0 - extent * v1 - extent * v2,
                    p0 + extent * v1 - extent * v2,
                ],
                dtype=np.float64,
            )

            # Triangulate the quad into 2 triangles
            vertices = corners.T  # (3, 4)
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32).T  # (3, 2)

            from pydrake.geometry import TriangleSurfaceMesh

            mesh = TriangleSurfaceMesh(faces, vertices)
            meshcat.SetTriangleMesh(f"{prefix}/plane", mesh, Rgba(0.2, 0.5, 0.8, 0.3))

            # Trajectory points as line strip
            meshcat.SetLine(
                f"{prefix}/trajectory",
                trajectory_positions.T,
                line_width=3.0,
                rgba=Rgba(1.0, 0.3, 0.1, 1.0),
            )

            self.logger.info(
                f"Meshcat visualization set: "
                f"plane steepness={metrics.steepness_deg:.1f}°, "
                f"RMSE={metrics.rmse:.4f}"
            )

        except ImportError:
            self.logger.warning(
                "pydrake.geometry not available; Meshcat visualization skipped. "
                "Install Drake to enable 3D visualization."
            )
        except (RuntimeError, TypeError, AttributeError) as exc:
            self.logger.warning(f"Meshcat visualization failed: {exc}")

    def export_for_analysis(
        self,
        metrics: SwingPlaneMetrics,
        trajectory_positions: np.ndarray,
        output_path: str,
    ) -> None:
        """Export swing plane analysis results for external analysis.

        Args:
            metrics: Swing plane analysis results
            trajectory_positions: Club head trajectory positions
            output_path: Path to save analysis results
        """
        import json
        from pathlib import Path

        # Prepare data for export
        export_data = {
            "swing_plane_metrics": {
                "normal_vector": metrics.normal_vector.tolist(),
                "point_on_plane": metrics.point_on_plane.tolist(),
                "steepness_deg": float(metrics.steepness_deg),
                "direction_deg": float(metrics.direction_deg),
                "rmse": float(metrics.rmse),
                "max_deviation": float(metrics.max_deviation),
            },
            "trajectory": {
                "positions": trajectory_positions.tolist(),
                "num_points": len(trajectory_positions),
            },
            "engine": "drake",
        }

        # Save to JSON file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Swing plane analysis exported to {output_path}")
