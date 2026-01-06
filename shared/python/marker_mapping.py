"""Marker-to-model mapping and fitting (Guideline A2 - Mandatory).

This module implements marker-to-model registration per project design
guidelines Section A2: "Landmark mapping to model frames. Segment pose
reconstruction (rigid body fit). Residuals, outlier detection, and
diagnostics plots. Logged uncertainty / fit quality metrics."

Reference: docs/assessments/project_design_guidelines.qmd Section A2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import logging

if TYPE_CHECKING:
    import mujoco

logger = logging.getLogger(__name__)


@dataclass
class MarkerMapping:
    """Mapping from physical marker to model body/point.

    Attributes:
        marker_name: Name of physical marker
        body_name: Name of model body
        body_offset: Offset from body COM in body frame [3] (m)
    """

    marker_name: str
    body_name: str
    body_offset: np.ndarray


@dataclass
class RegistrationResult:
    """Result of marker-to-model registration.

    Per Guideline A2, includes residuals, quality metrics, and outlier detection.

    Attributes:
        success: Whether registration succeeded
        transformation: SE(3) transformation (4×4 matrix)
        residuals: Per-marker residuals [N] (m)
        rms_error: RMS of residuals (m)
        max_error: Maximum residual (m)
        outlier_indices: Indices of outlier markers
        fit_quality: Quality score [0,1] (1 = perfect)
        num_markers_used: Number of markers after outlier removal
        condition_number: Condition number of solving system
    """

    success: bool
    transformation: np.ndarray
    residuals: np.ndarray
    rms_error: float
    max_error: float
    outlier_indices: list[int]
    fit_quality: float
    num_markers_used: int
    condition_number: float


class MarkerToModelMapper:
    """Map physical markers to model and fit poses (Guideline A2).

    This is a MANDATORY feature per project design guidelines Section A2.
    Implements:
    - Landmark mapping to model frames
    - Rigid body pose reconstruction (6-DOF fitting)
    - Residual computation and outlier detection
    - Fit quality metrics and diagnostics

    Example:
        >>> model = mujoco.MjModel.from_xml_path("humanoid.xml")
        >>> mapper = MarkerToModelMapper(model)
        >>>
        >>> # Define marker mappings
        >>> mappings = [
        ...     MarkerMapping("LSHO", "left_shoulder", np.array([0.05, 0, 0])),
        ...     MarkerMapping("RSHO", "right_shoulder", np.array([0.05, 0, 0])),
        ... ]
        >>> mapper.add_mappings(mappings)
        >>>
        >>> # Fit to observed markers
        >>> marker_positions = np.array([[0.2, 0.3, 1.4], [0.2, -0.3, 1.4]])
        >>> result = mapper.fit_segment_pose("torso", marker_positions)
        >>> print(f"RMS error: {result.rms_error} m")
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Initialize mapper.

        Args:
            model: MuJoCo model
        """
        self.model = model
        self._mappings: dict[str, list[MarkerMapping]] = {}

        # Thread-safe data
        import mujoco

        self._data = mujoco.MjData(model)

    def add_mapping(self, mapping: MarkerMapping) -> None:
        """Add a marker mapping.

        Args:
            mapping: Marker-to-body mapping
        """
        if mapping.body_name not in self._mappings:
            self._mappings[mapping.body_name] = []

        self._mappings[mapping.body_name].append(mapping)
        logger.info(
            f"Added mapping: marker '{mapping.marker_name}' → "
            f"body '{mapping.body_name}' at offset {mapping.body_offset}"
        )

    def add_mappings(self, mappings: list[MarkerMapping]) -> None:
        """Add multiple marker mappings.

        Args:
            mappings: List of marker mappings
        """
        for mapping in mappings:
            self.add_mapping(mapping)

    def fit_segment_pose(
        self,
        body_name: str,
        marker_positions: np.ndarray,
        outlier_threshold: float = 3.0,
        max_iterations: int = 3,
    ) -> RegistrationResult:
        """Fit segment pose to observed markers (6-DOF rigid body fit).

        Per Guideline A2, performs robust registration with outlier detection.
        Uses iterative weighted least squares with outlier rejection.

        Args:
            body_name: Name of body segment to fit
            marker_positions: Observed marker positions [N × 3] (m)
            outlier_threshold: Outlier threshold in standard deviations
            max_iterations: Maximum outlier rejection iterations

        Returns:
            RegistrationResult with transformation and diagnostics
        """
        if body_name not in self._mappings:
            return RegistrationResult(
                success=False,
                transformation=np.eye(4),
                residuals=np.array([]),
                rms_error=float("inf"),
                max_error=float("inf"),
                outlier_indices=[],
                fit_quality=0.0,
                num_markers_used=0,
                condition_number=float("inf"),
            )

        mappings = self._mappings[body_name]

        if len(mappings) != len(marker_positions):
            logger.error(
                f"Marker count mismatch: {len(mappings)} mappings, "
                f"{len(marker_positions)} positions"
            )
            return RegistrationResult(
                success=False,
                transformation=np.eye(4),
                residuals=np.array([]),
                rms_error=float("inf"),
                max_error=float("inf"),
                outlier_indices=[],
                fit_quality=0.0,
                num_markers_used=0,
                condition_number=float("inf"),
            )

        # Iterative outlier rejection
        inlier_mask = np.ones(len(marker_positions), dtype=bool)
        transformation = np.eye(4)

        for iteration in range(max_iterations):
            # Get inlier markers
            inlier_observed = marker_positions[inlier_mask]
            inlier_offsets = np.array(
                [
                    mappings[i].body_offset
                    for i in range(len(mappings))
                    if inlier_mask[i]
                ]
            )

            if len(inlier_observed) < 3:
                logger.warning("Too few markers for robust fit")
                break

            # Fit rigid transformation (Kabsch algorithm)
            transformation = self._fit_rigid_transform(inlier_offsets, inlier_observed)

            # Compute residuals
            all_offsets = np.array([m.body_offset for m in mappings])
            predicted = self._apply_transform(transformation, all_offsets)
            residuals = np.linalg.norm(marker_positions - predicted, axis=1)

            # Detect outliers
            inlier_residuals = residuals[inlier_mask]
            if len(inlier_residuals) > 0:
                std = np.std(inlier_residuals)
                new_outliers = residuals > outlier_threshold * std

                if np.array_equal(new_outliers, ~inlier_mask):
                    break  # Converged

                inlier_mask = ~new_outliers

        # Final metrics
        final_residuals = residuals[inlier_mask]
        rms = (
            float(np.sqrt(np.mean(final_residuals**2)))
            if len(final_residuals) > 0
            else float("inf")
        )
        max_err = (
            float(np.max(final_residuals)) if len(final_residuals) > 0 else float("inf")
        )

        outlier_idx = [i for i in range(len(marker_positions)) if not inlier_mask[i]]

        # Fit quality: exp(-rms/threshold)
        fit_quality = float(np.exp(-rms / 0.01))  # 1cm reference

        # Condition number (from SVD of point cloud)
        if len(inlier_observed) >= 3:
            centered = inlier_observed - np.mean(inlier_observed, axis=0)
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            cond = float(s[0] / s[-1]) if s[-1] > 1e-10 else float("inf")
        else:
            cond = float("inf")

        return RegistrationResult(
            success=True,
            transformation=transformation,
            residuals=final_residuals,
            rms_error=rms,
            max_error=max_err,
            outlier_indices=outlier_idx,
            fit_quality=fit_quality,
            num_markers_used=int(np.sum(inlier_mask)),
            condition_number=cond,
        )

    def _fit_rigid_transform(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Fit rigid transformation using Kabsch algorithm.

        Args:
            source_points: Source points [N × 3]
            target_points: Target points [N × 3]

        Returns:
            4×4 SE(3) transformation matrix
        """
        # Center point clouds
        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points, axis=0)

        source_centered = source_points - source_center
        target_centered = target_points - target_center

        # Compute rotation via SVD
        H = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_center - R @ source_center

        # Construct 4×4 matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def _apply_transform(
        self,
        transformation: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Apply rigid transformation to points.

        Args:
            transformation: 4×4 SE(3) matrix
            points: Points [N × 3]

        Returns:
            Transformed points [N × 3]
        """
        # Convert to homogeneous coordinates
        points_h = np.hstack([points, np.ones((len(points), 1))])

        # Apply transformation
        transformed_h = (transformation @ points_h.T).T

        return transformed_h[:, :3]

    def plot_registration_diagnostics(
        self,
        result: RegistrationResult,
        marker_names: list[str] | None = None,
    ) -> None:
        """Plot registration diagnostics (Guideline A2).

        Args:
            result: Registration result
            marker_names: Optional marker names for labeling
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Residuals bar plot
        ax = axes[0, 0]
        marker_idx = np.arange(len(result.residuals))
        ax.bar(marker_idx, result.residuals * 1000)  # Convert to mm
        ax.axhline(y=result.rms_error * 1000, color="r", linestyle="--", label="RMS")
        ax.set_xlabel("Marker Index")
        ax.set_ylabel("Residual [mm]")
        ax.set_title("Per-Marker Residuals")
        ax.legend()
        ax.grid(True)

        # Residual distribution
        ax = axes[0, 1]
        ax.hist(result.residuals * 1000, bins=20, edgecolor="black")
        ax.axvline(x=result.rms_error * 1000, color="r", linestyle="--", label="RMS")
        ax.set_xlabel("Residual [mm]")
        ax.set_ylabel("Count")
        ax.set_title("Residual Distribution")
        ax.legend()
        ax.grid(True)

        # Fit quality metrics
        ax = axes[1, 0]
        metrics = {
            "RMS Error\n[mm]": result.rms_error * 1000,
            "Max Error\n[mm]": result.max_error * 1000,
            "Fit Quality\n[0-1]": result.fit_quality,
            "Markers Used": result.num_markers_used,
        }
        ax.bar(metrics.keys(), metrics.values())
        ax.set_title("Fit Quality Metrics")
        ax.grid(True)

        # Summary text
        ax = axes[1, 1]
        ax.axis("off")
        summary = f"""
Registration Summary:
────────────────────
Success: {result.success}
RMS Error: {result.rms_error*1000:.2f} mm
Max Error: {result.max_error*1000:.2f} mm
Fit Quality: {result.fit_quality:.3f}
Markers Used: {result.num_markers_used}
Outliers: {len(result.outlier_indices)}
Condition #: {result.condition_number:.1e}
        """
        ax.text(
            0.1,
            0.5,
            summary,
            fontsize=12,
            family="monospace",
            verticalalignment="center",
        )

        plt.tight_layout()
        plt.show()
