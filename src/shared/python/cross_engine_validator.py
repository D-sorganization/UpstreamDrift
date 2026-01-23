"""Cross-engine validation framework for numerical consistency.

Per Guideline M2 and P3 from docs/assessments/project_design_guidelines.qmd:
- M2: Cross-engine comparison tests required
- P3: Tolerance-based deviation reporting mandatory

This module provides automated cross-engine validation to ensure MuJoCo, Drake,
and Pinocchio produce consistent results within specified tolerances.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of cross-engine validation.

    Attributes:
        passed: Whether validation passed (deviation within tolerance)
        metric_name: Name of the metric being compared (e.g., "position", "torque")
        max_deviation: Maximum deviation found between engines
        tolerance: Tolerance threshold that was applied
        engine1: Name of first engine
        engine2: Name of second engine
        message: Detailed message (empty if passed, error description if failed)
        severity: Classification of deviation severity (PASSED/WARNING/ERROR/BLOCKER)
    """

    passed: bool
    metric_name: str
    max_deviation: float
    tolerance: float
    engine1: str
    engine2: str
    message: str
    severity: str = "PASSED"  # PASSED, WARNING, ERROR, BLOCKER


class CrossEngineValidator:
    """Validates numerical consistency across physics engines.

    Implements tolerance-based validation per Guideline P3:
    - Positions: ±1e-6 m
    - Velocities: ±1e-5 m/s
    - Accelerations: ±1e-4 m/s²
    - Torques: ±1e-3 N⋅m (or <10% RMS for large magnitudes)
    - Jacobians: ±1e-8 (element-wise)

    Example:
        >>> validator = CrossEngineValidator()
        >>> mujoco_pos = np.array([1.0, 2.0, 3.0])
        >>> drake_pos = np.array([1.0000001, 2.0000001, 3.0000001])
        >>> result = validator.compare_states(
        ...     "MuJoCo", mujoco_pos,
        ...     "Drake", drake_pos,
        ...     metric="position"
        ... )
        >>> assert result.passed
        >>> print(f"Deviation: {result.max_deviation:.2e}")
        Deviation: 1.00e-07
    """

    # Tolerance specifications from Guideline P3
    TOLERANCES = {
        "position": 1e-6,  # meters
        "velocity": 1e-5,  # m/s
        "acceleration": 1e-4,  # m/s²
        "torque": 1e-3,  # N⋅m
        "jacobian": 1e-8,  # dimensionless
    }

    # Severity thresholds (Assessment C Finding C-003)
    # Classify deviation severity by multiples of tolerance
    WARNING_THRESHOLD = 2.0  # 2× tolerance → warning (acceptable with caution)
    ERROR_THRESHOLD = 10.0  # 10× tolerance → error (investigation required)
    BLOCKER_THRESHOLD = 100.0  # 100× tolerance → blocker (fundamental model error)

    def compare_states(
        self,
        engine1_name: str,
        engine1_state: np.ndarray,
        engine2_name: str,
        engine2_state: np.ndarray,
        metric: Literal[
            "position", "velocity", "acceleration", "torque", "jacobian"
        ] = "position",
    ) -> ValidationResult:
        """Compare states from two engines against tolerance targets.

        Args:
            engine1_name: Name of first engine (e.g., "MuJoCo")
            engine1_state: State array from first engine
            engine2_name: Name of second engine (e.g., "Drake")
            engine2_state: State array from second engine
            metric: Type of metric being compared (determines tolerance)

        Returns:
            ValidationResult with pass/fail status and deviation details

        Raises:
            ValueError: If metric is not recognized
        """
        if metric not in self.TOLERANCES:
            raise ValueError(
                f"Unknown metric '{metric}'. Valid metrics: {list(self.TOLERANCES.keys())}"
            )

        # Shape consistency check
        if engine1_state.shape != engine2_state.shape:
            return ValidationResult(
                passed=False,
                metric_name=metric,
                max_deviation=np.inf,
                tolerance=self.TOLERANCES[metric],
                engine1=engine1_name,
                engine2=engine2_name,
                message=f"Shape mismatch: {engine1_state.shape} vs {engine2_state.shape}",
            )

        # Compute deviation
        deviation = np.abs(engine1_state - engine2_state)
        max_dev = float(np.max(deviation))
        tol = self.TOLERANCES[metric]

        # Classify severity (C-003 remediation)
        passed, severity = self._classify_severity(max_dev, tol)

        # Log with appropriate severity level
        self._log_result(
            severity=severity,
            engine1_name=engine1_name,
            engine2_name=engine2_name,
            metric=metric,
            max_dev=max_dev,
            tol=tol,
            deviation=deviation,
            engine1_state=engine1_state,
            engine2_state=engine2_state,
        )

        return ValidationResult(
            passed=passed,
            metric_name=metric,
            max_deviation=max_dev,
            tolerance=tol,
            engine1=engine1_name,
            engine2=engine2_name,
            message=self._build_message(severity, max_dev, tol),
            severity=severity,
        )

    def _classify_severity(self, max_dev: float, tolerance: float) -> tuple[bool, str]:
        """Classify deviation severity based on threshold multipliers (C-003).

        Args:
            max_dev: Maximum deviation observed.
            tolerance: Base tolerance threshold.

        Returns:
            Tuple of (passed, severity_level).
        """
        ratio = max_dev / tolerance if tolerance > 0 else float("inf")

        if ratio <= 1.0:
            return True, "PASSED"
        elif ratio <= self.WARNING_THRESHOLD:
            return True, "WARNING"  # Acceptable with caution
        elif ratio <= self.ERROR_THRESHOLD:
            return False, "ERROR"  # Investigation required
        else:
            return False, "BLOCKER"  # Fundamental model error

    def _build_message(self, severity: str, max_dev: float, tol: float) -> str:
        """Build appropriate message based on severity."""
        if severity == "PASSED":
            return ""
        elif severity == "WARNING":
            return f"Deviation {max_dev:.2e} acceptable but exceeds base tolerance {tol:.2e}"
        elif severity == "ERROR":
            return f"Deviation {max_dev:.2e} exceeds tolerance {tol:.2e} - investigation required"
        else:  # BLOCKER
            return f"CRITICAL: Deviation {max_dev:.2e} is >{self.BLOCKER_THRESHOLD}× tolerance - fundamental error"

    def _log_result(
        self,
        severity: str,
        engine1_name: str,
        engine2_name: str,
        metric: str,
        max_dev: float,
        tol: float,
        deviation: np.ndarray,
        engine1_state: np.ndarray,
        engine2_state: np.ndarray,
    ) -> None:
        """Log validation result with appropriate severity level."""
        ratio = max_dev / tol if tol > 0 else float("inf")
        worst_idx = int(np.argmax(deviation))

        base_msg = (
            f"Cross-engine validation ({severity}):\n"
            f"  Engines: {engine1_name} vs {engine2_name}\n"
            f"  Metric: {metric}\n"
            f"  Max deviation: {max_dev:.2e} ({ratio:.1f}× tolerance)\n"
            f"  Tolerance threshold: {tol:.2e}"
        )

        if severity == "PASSED":
            logger.info(f"✅ {base_msg}")
        elif severity == "WARNING":
            logger.warning(
                f"⚠️ {base_msg}\n"
                f"  Status: Acceptable with caution (2-{self.ERROR_THRESHOLD:.0f}× tolerance)"
            )
        elif severity == "ERROR":
            logger.error(
                f"❌ {base_msg}\n"
                f"  Deviation location: index {worst_idx}\n"
                f"  {engine1_name} value: {engine1_state.flat[worst_idx]:.6e}\n"
                f"  {engine2_name} value: {engine2_state.flat[worst_idx]:.6e}\n"
                f"  Possible causes:\n"
                f"    - Integration method differences\n"
                f"    - Timestep size mismatch\n"
                f"    - Constraint handling differences\n"
                f"  ACTION: Investigate before using results"
            )
        else:  # BLOCKER
            logger.critical(
                f"🚫 BLOCKER - {base_msg}\n"
                f"  Deviation location: index {worst_idx}\n"
                f"  {engine1_name} value: {engine1_state.flat[worst_idx]:.6e}\n"
                f"  {engine2_name} value: {engine2_state.flat[worst_idx]:.6e}\n"
                f"  FUNDAMENTAL MODEL ERROR - DO NOT USE FOR PUBLICATION"
            )

    def compare_torques_with_rms(
        self,
        engine1_name: str,
        engine1_torques: np.ndarray,
        engine2_name: str,
        engine2_torques: np.ndarray,
        rms_threshold_pct: float = 10.0,
    ) -> ValidationResult:
        """Compare torques with RMS percentage threshold.

        For large torque magnitudes, a percentage-based RMS comparison is more
        appropriate than absolute tolerance. Per Guideline P3: <10% RMS difference.

        Args:
            engine1_name: Name of first engine
            engine1_torques: Torque array from first engine [N⋅m]
            engine2_name: Name of second engine
            engine2_torques: Torque array from second engine [N⋅m]
            rms_threshold_pct: Maximum allowed RMS difference as percentage (default: 10%)

        Returns:
            ValidationResult with RMS comparison details
        """
        if engine1_torques.shape != engine2_torques.shape:
            return ValidationResult(
                passed=False,
                metric_name="torque_rms",
                max_deviation=np.inf,
                tolerance=rms_threshold_pct,
                engine1=engine1_name,
                engine2=engine2_name,
                message=f"Shape mismatch: {engine1_torques.shape} vs {engine2_torques.shape}",
            )

        # RMS difference
        rms_diff = np.sqrt(np.mean((engine1_torques - engine2_torques) ** 2))
        rms_mag = np.sqrt(np.mean(engine1_torques**2))

        if rms_mag < 1e-10:  # Avoid division by zero
            rms_pct = 0.0 if rms_diff < 1e-10 else 100.0
        else:
            rms_pct = 100.0 * rms_diff / rms_mag

        passed = rms_pct < rms_threshold_pct

        if not passed:
            logger.error(
                f"❌ Torque RMS difference EXCEEDS threshold (Guideline P3 VIOLATION):\n"
                f"  Engines: {engine1_name} vs {engine2_name}\n"
                f"  RMS difference: {rms_pct:.2f}%\n"
                f"  Threshold: {rms_threshold_pct:.2f}%\n"
                f"  Absolute RMS diff: {rms_diff:.4f} N⋅m\n"
                f"  Absolute RMS magnitude: {rms_mag:.4f} N⋅m"
            )
        else:
            logger.info(
                f"✅ Torque RMS validation PASSED:\n"
                f"  Engines: {engine1_name} vs {engine2_name}\n"
                f"  RMS difference: {rms_pct:.2f}% < threshold: {rms_threshold_pct:.2f}%"
            )

        return ValidationResult(
            passed=passed,
            metric_name="torque_rms",
            max_deviation=rms_pct,
            tolerance=rms_threshold_pct,
            engine1=engine1_name,
            engine2=engine2_name,
            message=(
                ""
                if passed
                else f"RMS difference {rms_pct:.2f}% exceeds {rms_threshold_pct:.2f}%"
            ),
        )
