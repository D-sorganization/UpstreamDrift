"""Cross-engine validation framework for numerical consistency.

Per Guideline M2 and P3 from docs/assessments/project_design_guidelines.qmd:
- M2: Cross-engine comparison tests required
- P3: Tolerance-based deviation reporting mandatory

This module provides automated cross-engine validation to ensure MuJoCo, Drake,
and Pinocchio produce consistent results within specified tolerances.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import logging

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
    """
    
    passed: bool
    metric_name: str
    max_deviation: float
    tolerance: float
    engine1: str
    engine2: str
    message: str


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
        "position": 1e-6,       # meters
        "velocity": 1e-5,       # m/s
        "acceleration": 1e-4,   # m/s²
        "torque": 1e-3,         # N⋅m
        "jacobian": 1e-8,       # dimensionless
    }
    
    def compare_states(
        self,
        engine1_name: str,
        engine1_state: np.ndarray,
        engine2_name: str,
        engine2_state: np.ndarray,
        metric: Literal["position", "velocity", "acceleration", "torque", "jacobian"] = "position",
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
                message=f"Shape mismatch: {engine1_state.shape} vs {engine2_state.shape}"
            )
        
        # Compute deviation
        deviation = np.abs(engine1_state - engine2_state)
        max_dev = np.max(deviation)
        tol = self.TOLERANCES[metric]
        
        passed = max_dev <= tol
        
        # Detailed logging per Guideline P3
        if not passed:
            logger.error(
                f"❌ Cross-engine deviation EXCEEDS tolerance (Guideline P3 VIOLATION):\n"
                f"  Engines: {engine1_name} vs {engine2_name}\n"
                f"  Metric: {metric}\n"
                f"  Max deviation: {max_dev:.2e}\n"
                f"  Tolerance threshold: {tol:.2e}\n"
                f"  Deviation location: index {np.argmax(deviation)}\n"
                f"  {engine1_name} value at worst index: {engine1_state.flat[np.argmax(deviation)]:.6e}\n"
                f"  {engine2_name} value at worst index: {engine2_state.flat[np.argmax(deviation)]:.6e}\n"
                f"  Possible causes:\n"
                f"    - Integration method differences (MuJoCo=semi-implicit, Drake=RK3)\n"
                f"    - Timestep size mismatch\n"
                f"    - Constraint handling differences\n"
                f"    - Contact model parameters\n"
                f"    - Joint damping/friction defaults\n"
                f"  ACTION REQUIRED: Investigate before using results for publication"
            )
        else:
            logger.info(
                f"✅ Cross-engine validation PASSED:\n"
                f"  Engines: {engine1_name} vs {engine2_name}\n"
                f"  Metric: {metric}\n"
                f"  Max deviation: {max_dev:.2e} < tolerance: {tol:.2e}"
            )
        
        return ValidationResult(
            passed=passed,
            metric_name=metric,
            max_deviation=max_dev,
            tolerance=tol,
            engine1=engine1_name,
            engine2=engine2_name,
            message="" if passed else f"Deviation {max_dev:.2e} exceeds tolerance {tol:.2e}"
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
                message=f"Shape mismatch: {engine1_torques.shape} vs {engine2_torques.shape}"
            )
        
        # RMS difference
        rms_diff = np.sqrt(np.mean((engine1_torques - engine2_torques)**2))
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
            message="" if passed else f"RMS difference {rms_pct:.2f}% exceeds {rms_threshold_pct:.2f}%"
        )
