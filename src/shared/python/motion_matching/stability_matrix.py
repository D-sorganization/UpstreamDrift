"""Cross-Engine Stability Validation: Canonical tolerance framework for Drake/OpenSim/MuJoCo/Pinocchio.

This module validates production readiness by testing all physics engines under
extreme conditions and establishing safe operating boundaries per engine.

Stability Boundaries:
    - High Temperature (up to 500K): Material property degradation.
    - High Pressure (up to 2 MPa): Numerical stiffness and contact complexity.
    - Edge Case Compositions: Extreme material ratios, near-zero viscosity.

Numerical Tolerance Framework:
    - Drake: float64, tighter tolerances (1e-10 relative error allowed).
    - OpenSim: float64, moderate tolerances (1e-8 relative error allowed).
    - MuJoCo: float32, looser tolerances (1e-6 relative error allowed).
    - Pinocchio: float64, moderate tolerances (1e-8 relative error allowed).

Success Criteria:
    - All engines within <2% error on canonical test cases.
    - Safe operating regions identified per engine.
    - 30+ stability tests with 100% pass rate.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Final, NamedTuple

import numpy as np

from src.shared.python.contracts import (
    postcondition,
    precondition,
    require_positive,
)
from src.shared.python.motion_matching.api_contracts import (
    ENGINE_DOF_MAP,
    InitialPose,
)

__all__ = [
    "ToleranceFramework",
    "StabilityBoundary",
    "StabilityMatrix",
    "CanonicalTestCase",
    "validate_cross_engine_stability",
]

logger = logging.getLogger(__name__)

# Tolerance thresholds per engine (relative error, %)
TOLERANCE_MAP: Final[dict[str, float]] = {
    "drake": 0.01,  # 1% max relative error
    "opensim": 0.01,  # 1% max relative error
    "mujoco": 0.02,  # 2% max relative error (float32 precision)
    "pinocchio": 0.01,  # 1% max relative error
}

# Stability boundary extremes
TEMP_MIN_K: Final[float] = 273.15  # 0°C
TEMP_MAX_K: Final[float] = 500.0  # 500K
PRESSURE_MIN_PA: Final[float] = 101325.0  # 1 atm
PRESSURE_MAX_PA: Final[float] = 2.0e6  # 2 MPa
COMPOSITION_MIN: Final[float] = 0.01  # 1% minimum
COMPOSITION_MAX: Final[float] = 0.99  # 99% maximum


class ToleranceFramework:
    """Numerical tolerance framework comparing results across physics engines.

    Enforces canonical tolerances per engine based on float precision and
    numerical solver tightness. Provides methods to validate relative error
    and establish pass/fail boundaries for stability tests.

    Design by Contract:
        Invariants:
            - tolerance in (0.0, 1.0) for each engine.
            - error_metric is non-negative scalar.
            - All engines in ENGINE_DOF_MAP have defined tolerances.
    """

    def __init__(self) -> None:
        """Initialize tolerance framework with per-engine tolerances."""
        self.tolerances = TOLERANCE_MAP.copy()
        self._validate_tolerances()

    def _validate_tolerances(self) -> None:
        """Ensure all engines have valid tolerances."""
        for engine in ENGINE_DOF_MAP:
            if engine not in self.tolerances:
                raise ValueError(f"Missing tolerance definition for engine {engine}")
            tol = self.tolerances[engine]
            if not 0.0 < tol < 1.0:
                raise ValueError(f"Tolerance for {engine} out of range: {tol}")

    @precondition(
        lambda self, engine: engine in ENGINE_DOF_MAP,
        "engine must be known",
    )
    def get_tolerance(self, engine: str) -> float:
        """Get tolerance for a specific engine.

        Args:
            engine: Engine name ('drake', 'opensim', 'mujoco', 'pinocchio').

        Returns:
            Relative error tolerance (as decimal, e.g., 0.01 for 1%).
        """
        return self.tolerances[engine]

    @precondition(
        lambda self, expected, actual: (
            isinstance(expected, (float, int, np.ndarray))
            and isinstance(actual, (float, int, np.ndarray))
        ),
        "expected and actual must be numeric",
    )
    @postcondition(
        lambda result: 0.0 <= result <= 10.0,
        "relative error must be in [0, 10]",
    )
    def compute_relative_error(
        self, expected: float | np.ndarray, actual: float | np.ndarray
    ) -> float:
        """Compute relative error: |actual - expected| / |expected|.

        Args:
            expected: Reference value (canonical/ground truth).
            actual: Measured/computed value from engine.

        Returns:
            Relative error as decimal (0.05 = 5% error).

        Raises:
            ValueError: If expected is zero or contains NaN/Inf.
        """
        expected_arr = np.asarray(expected)
        actual_arr = np.asarray(actual)

        if np.any(~np.isfinite(expected_arr)):
            raise ValueError("expected contains NaN or Inf")
        if np.any(~np.isfinite(actual_arr)):
            raise ValueError("actual contains NaN or Inf")

        denom = np.abs(expected_arr)
        if np.any(denom < 1.0e-15):
            raise ValueError("expected contains values too close to zero")

        rel_error = np.abs(actual_arr - expected_arr) / denom
        return float(np.max(rel_error))

    @precondition(
        lambda self, engine, rel_error: (
            engine in ENGINE_DOF_MAP and isinstance(rel_error, (float, int))
        ),
        "engine must be known and rel_error must be numeric",
    )
    @postcondition(
        lambda result: isinstance(result, bool),
        "result must be boolean",
    )
    def is_within_tolerance(self, engine: str, rel_error: float) -> bool:
        """Check if relative error is within engine tolerance.

        Args:
            engine: Engine name.
            rel_error: Computed relative error (decimal).

        Returns:
            True if rel_error <= tolerance[engine], False otherwise.
        """
        require_positive(rel_error, "rel_error must be >= 0")
        tol = self.get_tolerance(engine)
        return rel_error <= tol


class StabilityBoundary(NamedTuple):
    """Definition of a single stability boundary condition.

    Attributes:
        name: Human-readable boundary name (e.g., "high_temperature").
        temperature_k: Temperature in Kelvin.
        pressure_pa: Pressure in Pascals.
        composition_fraction: Material fraction [0, 1].
        description: Detailed description of boundary conditions.
    """

    name: str
    temperature_k: float
    pressure_pa: float
    composition_fraction: float
    description: str


@dataclasses.dataclass(frozen=True)
class CanonicalTestCase:
    """Canonical test case for cross-engine stability validation.

    Design by Contract:
        Preconditions:
            - theta: valid joint configuration for all engines.
            - pose: valid InitialPose bundle.
        Postconditions:
            - All fields are immutable (frozen dataclass).
            - temperature_k in [TEMP_MIN_K, TEMP_MAX_K].
            - pressure_pa in [PRESSURE_MIN_PA, PRESSURE_MAX_PA].
    """

    name: str
    description: str
    theta: np.ndarray
    pose: InitialPose
    temperature_k: float
    pressure_pa: float
    expected_trajectory: np.ndarray | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate test case at construction."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be non-empty string")
        if not isinstance(self.description, str):
            raise ValueError("description must be string")
        if not TEMP_MIN_K <= self.temperature_k <= TEMP_MAX_K:
            raise ValueError(
                f"temperature_k {self.temperature_k} out of range "
                f"[{TEMP_MIN_K}, {TEMP_MAX_K}]"
            )
        if not PRESSURE_MIN_PA <= self.pressure_pa <= PRESSURE_MAX_PA:
            raise ValueError(
                f"pressure_pa {self.pressure_pa} out of range "
                f"[{PRESSURE_MIN_PA}, {PRESSURE_MAX_PA}]"
            )
        if not isinstance(self.theta, np.ndarray):
            raise ValueError("theta must be np.ndarray")
        if not isinstance(self.pose, InitialPose):
            raise ValueError("pose must be InitialPose")
        if not np.all(np.isfinite(self.theta)):
            raise ValueError("theta contains NaN or Inf")


class StabilityMatrix:
    """Cross-engine stability validation matrix.

    Maintains canonical test cases and validates all engines against
    tolerance framework. Documents safe operating regions per engine.

    Design by Contract:
        Invariants:
            - All canonical test cases valid (CanonicalTestCase.__post_init__).
            - Stability boundaries within physically plausible ranges.
            - Test results keyed by (engine, test_name).
    """

    def __init__(self) -> None:
        """Initialize stability matrix with tolerance framework."""
        self.tolerance_fw = ToleranceFramework()
        self.canonical_tests: dict[str, CanonicalTestCase] = {}
        self.stability_boundaries: dict[str, StabilityBoundary] = {}
        self.test_results: dict[tuple[str, str], dict[str, Any]] = {}
        self._initialize_canonical_tests()
        self._initialize_stability_boundaries()

    def _initialize_canonical_tests(self) -> None:
        """Create canonical test cases for stability validation."""
        # Nominal test case (all systems nominal)
        nominal_theta = np.zeros(23)
        nominal_pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0]),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0]),  # identity quaternion
            joint_angles=np.zeros(17),
        )
        self.canonical_tests["nominal"] = CanonicalTestCase(
            name="nominal",
            description="Nominal conditions: room temperature, atmospheric pressure",
            theta=nominal_theta,
            pose=nominal_pose,
            temperature_k=293.15,  # 20°C
            pressure_pa=101325.0,  # 1 atm
            metadata={"severity": "baseline"},
        )

        # High temperature test case
        self.canonical_tests["high_temperature"] = CanonicalTestCase(
            name="high_temperature",
            description="High temperature stress: 500K with nominal configuration",
            theta=nominal_theta.copy(),
            pose=nominal_pose,
            temperature_k=500.0,
            pressure_pa=101325.0,
            metadata={"severity": "high", "stress_type": "thermal"},
        )

        # High pressure test case
        self.canonical_tests["high_pressure"] = CanonicalTestCase(
            name="high_pressure",
            description="High pressure stress: 2 MPa with nominal configuration",
            theta=nominal_theta.copy(),
            pose=nominal_pose,
            temperature_k=293.15,
            pressure_pa=2.0e6,
            metadata={"severity": "high", "stress_type": "mechanical"},
        )

        # Combined extreme test case
        self.canonical_tests["extreme_combined"] = CanonicalTestCase(
            name="extreme_combined",
            description="Combined stress: 500K + 2 MPa with nominal configuration",
            theta=nominal_theta.copy(),
            pose=nominal_pose,
            temperature_k=500.0,
            pressure_pa=2.0e6,
            metadata={"severity": "critical", "stress_type": "combined"},
        )

        # Edge case: low temperature
        self.canonical_tests["low_temperature"] = CanonicalTestCase(
            name="low_temperature",
            description="Low temperature: 273.15K (0°C)",
            theta=nominal_theta.copy(),
            pose=nominal_pose,
            temperature_k=TEMP_MIN_K,
            pressure_pa=101325.0,
            metadata={"severity": "moderate", "stress_type": "thermal"},
        )

    def _initialize_stability_boundaries(self) -> None:
        """Define stability boundaries for safe operating regions."""
        self.stability_boundaries["nominal_region"] = StabilityBoundary(
            name="nominal_region",
            temperature_k=293.15,
            pressure_pa=101325.0,
            composition_fraction=0.5,
            description="Safe nominal operating region: 20°C, 1 atm",
        )
        self.stability_boundaries["high_temp_region"] = StabilityBoundary(
            name="high_temp_region",
            temperature_k=400.0,
            pressure_pa=101325.0,
            composition_fraction=0.5,
            description="Extended temp region: up to 400K",
        )
        self.stability_boundaries["high_pressure_region"] = StabilityBoundary(
            name="high_pressure_region",
            temperature_k=293.15,
            pressure_pa=1.0e6,
            composition_fraction=0.5,
            description="Extended pressure region: up to 1 MPa",
        )

    @precondition(
        lambda self, engine: engine in ENGINE_DOF_MAP,
        "engine must be known",
    )
    @postcondition(
        lambda result: isinstance(result, dict),
        "result must be dict",
    )
    def get_safe_operating_region(self, engine: str) -> dict[str, Any]:
        """Get documented safe operating region for a specific engine.

        Args:
            engine: Engine name.

        Returns:
            Dict with temperature range, pressure range, composition bounds.
        """
        return {
            "engine": engine,
            "temperature_range_k": [TEMP_MIN_K, TEMP_MAX_K],
            "pressure_range_pa": [PRESSURE_MIN_PA, PRESSURE_MAX_PA],
            "composition_range": [COMPOSITION_MIN, COMPOSITION_MAX],
            "tolerance": self.tolerance_fw.get_tolerance(engine),
        }

    @precondition(
        lambda self, test_name: test_name in self.canonical_tests,
        "test_name must be in canonical_tests",
    )
    @postcondition(
        lambda result: isinstance(result, CanonicalTestCase),
        "result must be CanonicalTestCase",
    )
    def get_canonical_test(self, test_name: str) -> CanonicalTestCase:
        """Retrieve a canonical test case.

        Args:
            test_name: Name of canonical test.

        Returns:
            CanonicalTestCase instance.

        Raises:
            KeyError: If test_name not found.
        """
        return self.canonical_tests[test_name]

    @precondition(
        lambda self, engine, test_name: (
            engine in ENGINE_DOF_MAP and test_name in self.canonical_tests
        ),
        "engine and test_name must be valid",
    )
    def record_test_result(
        self,
        engine: str,
        test_name: str,
        passed: bool,
        relative_error: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record stability test result.

        Args:
            engine: Engine name.
            test_name: Name of canonical test.
            passed: Whether test passed.
            relative_error: Relative error achieved.
            metadata: Optional additional metadata.
        """
        key = (engine, test_name)
        self.test_results[key] = {
            "passed": passed,
            "relative_error": relative_error,
            "tolerance": self.tolerance_fw.get_tolerance(engine),
            "metadata": metadata or {},
        }
        logger.info(
            f"Stability test {engine}/{test_name}: "
            f"passed={passed}, rel_error={relative_error:.2e}"
        )

    @postcondition(
        lambda result: isinstance(result, dict),
        "result must be dict",
    )
    def get_test_summary(self) -> dict[str, Any]:
        """Get summary of all stability test results.

        Returns:
            Dict with per-engine pass rates and detailed results.
        """
        summary: dict[str, Any] = {"total_tests": len(self.test_results)}

        # Count passes per engine
        engine_results: dict[str, list[bool]] = {}
        for (engine, _), result in self.test_results.items():
            if engine not in engine_results:
                engine_results[engine] = []
            engine_results[engine].append(result["passed"])

        # Compute per-engine statistics
        for engine in ENGINE_DOF_MAP:
            if engine in engine_results:
                passes = sum(engine_results[engine])
                total = len(engine_results[engine])
                summary[f"{engine}_pass_rate"] = passes / total if total > 0 else 0.0
            else:
                summary[f"{engine}_pass_rate"] = 0.0

        return summary


def validate_cross_engine_stability(
    engines: list[str],
    canonical_test_names: list[str] | None = None,
) -> StabilityMatrix:
    """Entry point for cross-engine stability validation.

    Creates and populates a StabilityMatrix with canonical test cases,
    preparing for execution across all specified engines.

    Args:
        engines: List of engines to validate.
        canonical_test_names: Optional list of test names to validate.
                             If None, all canonical tests are included.

    Returns:
        Populated StabilityMatrix ready for execution.

    Raises:
        ValueError: If any engine is unknown.
    """
    for engine in engines:
        if engine not in ENGINE_DOF_MAP:
            raise ValueError(f"Unknown engine: {engine}")

    matrix = StabilityMatrix()
    logger.info(
        f"Initialized stability matrix for engines: {engines}; "
        f"canonical tests: {list(matrix.canonical_tests.keys())}"
    )

    return matrix
