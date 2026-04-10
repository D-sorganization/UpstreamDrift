"""Inverse-kinematics helpers for the A3 data-fitting pipeline."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from src.shared.python.logging_pkg.logging_config import get_logger

from .data_fitting_models import FitResult

logger = get_logger(__name__)


class InverseKinematicsSolver:
    """Solve inverse kinematics from marker positions to joint angles."""

    def __init__(
        self,
        segment_lengths: dict[str, float],
        joint_names: list[str],
        tolerance: float = 1e-6,
        max_iterations: int = 100,
    ) -> None:
        if not (segment_lengths is not None):
            raise ValueError("segment_lengths must be provided")
        self.segment_lengths = segment_lengths
        self.joint_names = joint_names
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        logger.info(
            f"IK solver initialized with {len(joint_names)} joints, "
            f"{len(segment_lengths)} segments"
        )

    def solve_analytical_2d(
        self,
        target_position: np.ndarray,
        segment1_length: float,
        segment2_length: float,
    ) -> tuple[float, float]:
        """Solve 2-link planar IK analytically."""
        x, y = target_position[:2]
        L1, L2 = segment1_length, segment2_length

        d = np.sqrt(x**2 + y**2)
        if d > L1 + L2:
            raise ValueError(
                f"Target at distance {d:.3f}m is unreachable (max: {L1 + L2:.3f}m)"
            )
        if d < abs(L1 - L2):
            raise ValueError(f"Target at distance {d:.3f}m is too close")

        cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2 = np.arccos(cos_theta2)

        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return float(theta1), float(theta2)

    def solve_numerical(
        self,
        target_positions: np.ndarray,
        initial_angles: np.ndarray | None = None,
    ) -> FitResult:
        """Solve IK numerically using optimization."""
        if not (target_positions is not None):
            raise ValueError("target_positions must be provided")
        n_joints = len(self.joint_names)

        if initial_angles is None:
            initial_angles = np.zeros(n_joints)

        def residual_func(angles: np.ndarray) -> np.ndarray:
            predicted = self._forward_kinematics(angles)
            return (predicted - target_positions).flatten()

        result = optimize.least_squares(
            residual_func,
            initial_angles,
            method="lm",
            ftol=self.tolerance,
            max_nfev=self.max_iterations,
        )

        residuals = residual_func(result.x)
        rms = float(np.sqrt(np.mean(residuals**2)))
        total_variance = np.var(target_positions.flatten())
        r_squared = (
            1.0 - (np.var(residuals) / total_variance) if total_variance > 0 else 0.0
        )

        try:
            jac = result.jac
            if jac is not None and jac.size > 0:
                s = np.linalg.svd(jac, compute_uv=False)
                cond = float(s[0] / s[-1]) if s[-1] > 1e-10 else float("inf")
            else:
                cond = float("inf")
        except (ValueError, TypeError, RuntimeError):
            cond = float("inf")

        return FitResult(
            success=result.success,
            parameters={
                name: float(angle)
                for name, angle in zip(self.joint_names, result.x, strict=False)
            },
            residuals=residuals,
            rms_error=rms,
            r_squared=float(r_squared),
            condition_number=cond,
            iterations=result.nfev,
            message=result.message,
        )

    def _forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """Compute forward kinematics for given joint angles."""
        if not (angles is not None):
            raise ValueError("angles must be provided")
        positions = []
        x, y, z = 0.0, 0.0, 0.0
        cumulative_angle = 0.0

        for joint_name, angle in zip(self.joint_names, angles, strict=False):
            cumulative_angle += angle
            segment_name = joint_name.replace("_joint", "")
            length = self.segment_lengths.get(segment_name, 0.3)

            x += length * np.cos(cumulative_angle)
            y += length * np.sin(cumulative_angle)
            positions.append([x, y, z])

        return np.array(positions)
