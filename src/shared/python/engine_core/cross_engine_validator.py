"""Cross-engine validation framework for numerical consistency.

Per Guideline M2 and P3 from docs/assessments/project_design_guidelines.qmd:
- M2: Cross-engine comparison tests required
- P3: Tolerance-based deviation reporting mandatory

This module provides automated cross-engine validation to ensure MuJoCo, Drake,
and Pinocchio produce consistent results within specified tolerances.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import yaml

from src.shared.python.core.contracts import ContractChecker, invariant, require
from src.shared.python.engine_core.capabilities import CapabilityLevel
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

MetricName = Literal[
    "position",
    "velocity",
    "acceleration",
    "torque",
    "jacobian",
    "state_round_trip",
    "end_effector",
    "mass",
    "center_of_mass",
    "inertia",
]


class _StateRemapAdapter(Protocol):
    def from_canonical(self, state: Mapping[str, np.ndarray]) -> object: ...

    def to_canonical(self, state: object) -> Mapping[str, object]: ...


class _ForwardKinematicsAdapter(Protocol):
    def forward_kinematics(
        self,
        state: Mapping[str, np.ndarray],
    ) -> Mapping[str, Mapping[str, object]]: ...


class _DynamicsAdapter(Protocol):
    def inverse_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
    ) -> np.ndarray: ...

    def forward_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray: ...


class _MassPropertiesAdapter(Protocol):
    def exported_mass_properties(self) -> Mapping[str, object]: ...


class _DifferentialAdapter(Protocol):
    def differential_state(self, state: np.ndarray) -> np.ndarray: ...


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


@dataclass(frozen=True)
class DivergenceEntry:
    """Registered, explained cross-engine divergence."""

    id: str
    check_name: str
    metric_name: str
    engines: tuple[str, str]
    tolerance: float
    rationale: str


@dataclass(frozen=True)
class ConformanceCheckResult:
    """Result for one canonical-v2 conformance check."""

    check_name: str
    engine_name: str
    passed: bool
    skipped: bool = False
    message: str = ""
    validation: ValidationResult | None = None
    divergence: DivergenceEntry | None = None


@dataclass(frozen=True)
class ConformanceReference:
    """Reference data used by the canonical-v2 conformance harness."""

    canonical_state: Mapping[str, np.ndarray]
    rigid_dofs: Sequence[str]
    quaternion_dofs: Sequence[str]
    fk_poses: Mapping[str, Mapping[str, np.ndarray]]
    inverse_dynamics_q: np.ndarray
    inverse_dynamics_v: np.ndarray
    inverse_dynamics_a: np.ndarray
    mass_properties: Mapping[str, np.ndarray]
    differential_state: np.ndarray


class DivergenceRegistry:
    """Machine-readable registry for explained cross-engine divergences."""

    def __init__(self, entries: Sequence[DivergenceEntry] | None = None) -> None:
        self._entries = tuple(entries or ())

    @classmethod
    def from_yaml(cls, path: str | Path) -> DivergenceRegistry:
        """Load a divergence registry from YAML."""
        registry_path = Path(path)
        with registry_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        entries = raw.get("divergences", [])
        if not isinstance(entries, list):
            raise ValueError("divergence registry must contain a divergences list")
        return cls([_parse_divergence_entry(entry) for entry in entries])

    def find(
        self,
        *,
        check_name: str,
        metric_name: str,
        engine1: str,
        engine2: str,
    ) -> DivergenceEntry | None:
        """Return the matching divergence entry, if one is registered."""
        engine_pair = {engine1, engine2}
        for entry in self._entries:
            if entry.check_name != check_name or entry.metric_name != metric_name:
                continue
            if set(entry.engines) == engine_pair:
                return entry
        return None


@invariant(
    lambda self: all(v > 0 for v in self.TOLERANCES.values()),
    "All tolerance values must be positive",
)
class CrossEngineValidator(ContractChecker):
    """Validates numerical consistency across physics engines.

    Implements tolerance-based validation per Guideline P3:
    - Positions: ±1e-6 m
    - Velocities: ±1e-5 m/s
    - Accelerations: ±1e-4 m/s²
    - Torques: ±1e-3 N⋅m (or <10% RMS for large magnitudes)
    - Jacobians: ±1e-8 (element-wise)

    Design by Contract:
        Invariants:
            - All tolerance values are positive
            - Severity thresholds are ordered correctly

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

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for CrossEngineValidator."""
        return [
            (
                lambda: all(v > 0 for v in self.TOLERANCES.values()),
                "All tolerance values must be positive",
            ),
            (
                lambda: (
                    self.WARNING_THRESHOLD
                    < self.ERROR_THRESHOLD
                    < self.BLOCKER_THRESHOLD
                ),
                "Severity thresholds must be ordered: WARNING < ERROR < BLOCKER",
            ),
        ]

    # Tolerance specifications from Guideline P3
    TOLERANCES = {
        "position": 1e-6,  # meters
        "velocity": 1e-5,  # m/s
        "acceleration": 1e-4,  # m/s²
        "torque": 1e-3,  # N⋅m
        "jacobian": 1e-8,  # dimensionless
        "state_round_trip": 1e-9,  # canonical-v2 state remap identity
        "end_effector": 5e-3,  # meters; v1 parity spec target
        "mass": 1e-9,  # kilograms
        "center_of_mass": 1e-9,  # meters
        "inertia": 1e-9,  # kg m²
    }

    # Severity thresholds (Assessment C Finding C-003)
    # Classify deviation severity by multiples of tolerance
    WARNING_THRESHOLD = 2.0  # 2× tolerance → warning (acceptable with caution)
    ERROR_THRESHOLD = 10.0  # 10× tolerance → error (investigation required)
    BLOCKER_THRESHOLD = 100.0  # 100× tolerance → blocker (fundamental model error)

    def _require_capability(
        self,
        adapter: object,
        check_name: str,
        capability: str,
    ) -> ConformanceCheckResult | None:
        """Gate a check on an advertised capability.

        Returns ``None`` when the adapter advertises ``capability`` (the caller
        should proceed). Returns a *passing* skip when the adapter genuinely
        does not advertise it, and a *failing* result when the ``supports()``
        query raises — so a capability-silent or throwing adapter can no longer
        clear the gate with zero validation (issue #6891).
        """
        try:
            advertised = _supports_capability(adapter, capability)
        except (
            AttributeError,
            TypeError,
            NotImplementedError,
            ValueError,
            RuntimeError,
        ) as exc:
            logger.warning(
                "capability query for %r raised on %s: %s",
                capability,
                _adapter_name(adapter),
                exc,
            )
            return _capability_error(adapter, check_name, capability, exc)
        if not advertised:
            return _capability_skip(adapter, check_name, capability)
        return None

    def validate_round_trip_state_remap(
        self,
        adapter: object,
        reference: ConformanceReference,
    ) -> ConformanceCheckResult:
        """Validate ``from_canonical(to_canonical(q)) == q`` for canonical-v2."""
        check_name = "round_trip_state_remap"
        missing = _missing_methods(adapter, ("to_canonical", "from_canonical"))
        if missing:
            return _skipped_result(adapter, check_name, missing)

        remap_adapter = cast(_StateRemapAdapter, adapter)
        native_state = remap_adapter.from_canonical(reference.canonical_state)
        round_tripped = remap_adapter.to_canonical(native_state)
        dof_names = [*reference.rigid_dofs, *reference.quaternion_dofs]
        expected = _flatten_named_arrays(reference.canonical_state, dof_names)
        actual = _flatten_named_arrays(round_tripped, dof_names)
        validation = self.compare_states(
            _adapter_name(adapter),
            actual,
            "canonical-v2",
            expected,
            metric="state_round_trip",
        )
        return _conformance_result(adapter, check_name, validation)

    def validate_forward_kinematics(
        self,
        adapter: object,
        reference: ConformanceReference,
    ) -> ConformanceCheckResult:
        """Validate FK against known reference poses."""
        check_name = "forward_kinematics_reference_pose"
        gate = self._require_capability(adapter, check_name, "forward_sim")
        if gate is not None:
            return gate
        missing = _missing_methods(adapter, ("forward_kinematics",))
        if missing:
            return _skipped_result(adapter, check_name, missing)

        fk_adapter = cast(_ForwardKinematicsAdapter, adapter)
        actual_poses = fk_adapter.forward_kinematics(reference.canonical_state)
        expected = _flatten_pose_map(reference.fk_poses)
        actual = _flatten_pose_map(actual_poses)
        validation = self.compare_states(
            _adapter_name(adapter),
            actual,
            "canonical-v2-reference",
            expected,
            metric="end_effector",
        )
        return _conformance_result(adapter, check_name, validation)

    def validate_inverse_forward_dynamics_consistency(
        self,
        adapter: object,
        reference: ConformanceReference,
    ) -> ConformanceCheckResult:
        """Validate inverse-dynamics torques reproduce acceleration via FD."""
        check_name = "inverse_forward_dynamics_consistency"
        gate = self._require_capability(adapter, check_name, "inverse_dynamics")
        if gate is not None:
            return gate
        gate = self._require_capability(adapter, check_name, "forward_sim")
        if gate is not None:
            return gate
        missing = _missing_methods(adapter, ("inverse_dynamics", "forward_dynamics"))
        if missing:
            return _skipped_result(adapter, check_name, missing)

        dynamics_adapter = cast(_DynamicsAdapter, adapter)
        tau = dynamics_adapter.inverse_dynamics(
            reference.inverse_dynamics_q,
            reference.inverse_dynamics_v,
            reference.inverse_dynamics_a,
        )
        actual_acceleration = dynamics_adapter.forward_dynamics(
            reference.inverse_dynamics_q,
            reference.inverse_dynamics_v,
            tau,
        )
        validation = self.compare_states(
            _adapter_name(adapter),
            np.asarray(actual_acceleration, dtype=float),
            "canonical-v2-reference",
            np.asarray(reference.inverse_dynamics_a, dtype=float),
            metric="acceleration",
        )
        return _conformance_result(adapter, check_name, validation)

    def validate_post_export_mass_properties(
        self,
        adapter: object,
        reference: ConformanceReference,
    ) -> list[ConformanceCheckResult]:
        """Validate exported mass, CoM, and inertia against canonical CC-3."""
        check_name = "post_export_mass_properties"
        gate = self._require_capability(adapter, check_name, "mass_matrix")
        if gate is not None:
            return [gate]
        missing = _missing_methods(adapter, ("exported_mass_properties",))
        if missing:
            return [_skipped_result(adapter, check_name, missing)]

        mass_adapter = cast(_MassPropertiesAdapter, adapter)
        actual = mass_adapter.exported_mass_properties()
        results = []
        mass_metrics: tuple[MetricName, ...] = ("mass", "center_of_mass", "inertia")
        for metric_name in mass_metrics:
            validation = self.compare_states(
                _adapter_name(adapter),
                np.asarray(actual[metric_name], dtype=float),
                "canonical-v2-reference",
                np.asarray(reference.mass_properties[metric_name], dtype=float),
                metric=metric_name,
            )
            results.append(_conformance_result(adapter, check_name, validation))
        return results

    def validate_differential_against_reference(
        self,
        adapter: object,
        reference_adapter: object,
        reference: ConformanceReference,
        divergence_registry: DivergenceRegistry | None = None,
    ) -> ConformanceCheckResult:
        """Validate differential behavior against a reference engine."""
        check_name = "differential_cross_engine_reference"
        gate = self._require_capability(adapter, check_name, "forward_sim")
        if gate is not None:
            return gate
        missing = _missing_methods(adapter, ("differential_state",))
        if missing:
            return _skipped_result(adapter, check_name, missing)

        differential_adapter = cast(_DifferentialAdapter, adapter)
        reference_differential = cast(_DifferentialAdapter, reference_adapter)
        actual = differential_adapter.differential_state(reference.differential_state)
        expected = reference_differential.differential_state(
            reference.differential_state
        )
        validation = self.compare_states(
            _adapter_name(adapter),
            np.asarray(actual, dtype=float),
            _adapter_name(reference_adapter),
            np.asarray(expected, dtype=float),
            metric="position",
        )
        return self.enforce_divergence_registry(
            check_name,
            validation,
            divergence_registry,
        )

    def enforce_divergence_registry(
        self,
        check_name: str,
        validation: ValidationResult,
        divergence_registry: DivergenceRegistry | None,
    ) -> ConformanceCheckResult:
        """Apply the "unregistered divergence = failure" rule."""
        engine_name = validation.engine1
        if validation.passed:
            return ConformanceCheckResult(
                check_name=check_name,
                engine_name=engine_name,
                passed=True,
                validation=validation,
            )
        if divergence_registry is None:
            message = f"unregistered divergence: {validation.message}"
            return ConformanceCheckResult(
                check_name=check_name,
                engine_name=engine_name,
                passed=False,
                message=message,
                validation=validation,
            )

        entry = divergence_registry.find(
            check_name=check_name,
            metric_name=validation.metric_name,
            engine1=validation.engine1,
            engine2=validation.engine2,
        )
        if entry is None:
            message = f"unregistered divergence: {validation.message}"
            return ConformanceCheckResult(
                check_name=check_name,
                engine_name=engine_name,
                passed=False,
                message=message,
                validation=validation,
            )
        if validation.max_deviation > entry.tolerance:
            message = (
                f"registered divergence {entry.id} exceeded registry tolerance "
                f"{entry.tolerance:.2e}: {validation.message}"
            )
            return ConformanceCheckResult(
                check_name=check_name,
                engine_name=engine_name,
                passed=False,
                message=message,
                validation=validation,
                divergence=entry,
            )
        return ConformanceCheckResult(
            check_name=check_name,
            engine_name=engine_name,
            passed=True,
            message=f"accepted registered divergence {entry.id}: {entry.rationale}",
            validation=validation,
            divergence=entry,
        )

    def compare_states(
        self,
        engine1_name: str,
        engine1_state: np.ndarray,
        engine2_name: str,
        engine2_state: np.ndarray,
        metric: MetricName = "position",
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
        if max_dev is None:
            raise ValueError("max_dev must be provided")
        ratio = max_dev / tolerance if tolerance > 0 else float("inf")

        if ratio <= 1.0:
            return True, "PASSED"
        if ratio <= self.WARNING_THRESHOLD:
            return True, "WARNING"  # Acceptable with caution
        if ratio <= self.ERROR_THRESHOLD:
            return False, "ERROR"  # Investigation required
        return False, "BLOCKER"  # Fundamental model error

    def _build_message(self, severity: str, max_dev: float, tol: float) -> str:
        """Build appropriate message based on severity."""
        if severity is None:
            raise ValueError("severity must be provided")
        if severity == "PASSED":  # noqa: SIM116
            return ""
        if severity == "WARNING":
            return f"Deviation {max_dev:.2e} acceptable but exceeds base tolerance {tol:.2e}"
        if severity == "ERROR":
            return f"Deviation {max_dev:.2e} exceeds tolerance {tol:.2e} - investigation required"
        # BLOCKER
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
        if severity is None:
            raise ValueError("severity must be provided")
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
        if engine1_name is None:
            raise ValueError("engine1_name must be provided")
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
        diff = engine1_torques - engine2_torques
        # ⚡ Bolt: Using np.vdot avoids intermediate array allocation compared to np.mean(diff**2)
        rms_diff = np.sqrt(np.vdot(diff, diff) / diff.size)
        rms_mag = np.sqrt(
            np.vdot(engine1_torques, engine1_torques) / engine1_torques.size
        )

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


def _parse_divergence_entry(raw: object) -> DivergenceEntry:
    """Parse one divergence registry entry."""
    if not isinstance(raw, Mapping):
        raise ValueError("each divergence registry entry must be a mapping")
    engines = raw.get("engines", ())
    if not isinstance(engines, Sequence) or isinstance(engines, str):
        raise ValueError("divergence entry engines must list two engine names")
    require(len(engines) == 2, "divergence entry engines must list two names")
    tolerance = float(raw.get("tolerance", 0.0))
    require(tolerance > 0.0, "divergence entry tolerance must be positive")
    entry_id = str(raw.get("id", "")).strip()
    check_name = str(raw.get("check", "")).strip()
    metric_name = str(raw.get("metric", "")).strip()
    rationale = str(raw.get("rationale", "")).strip()
    require(bool(entry_id), "divergence entry id must be provided")
    require(bool(check_name), "divergence entry check must be provided")
    require(bool(metric_name), "divergence entry metric must be provided")
    require(bool(rationale), "divergence entry rationale must be provided")
    return DivergenceEntry(
        id=entry_id,
        check_name=check_name,
        metric_name=metric_name,
        engines=(str(engines[0]), str(engines[1])),
        tolerance=tolerance,
        rationale=rationale,
    )


def _adapter_name(adapter: object) -> str:
    """Return a stable display name for an adapter-like object."""
    return str(getattr(adapter, "engine_name", adapter.__class__.__name__))


def _missing_methods(adapter: object, method_names: Sequence[str]) -> tuple[str, ...]:
    """Return required method names not implemented by ``adapter``."""
    return tuple(
        name for name in method_names if not callable(getattr(adapter, name, None))
    )


def _skipped_result(
    adapter: object,
    check_name: str,
    missing_methods: Sequence[str],
) -> ConformanceCheckResult:
    """Build a failure result for an adapter missing a required method.

    Reaching this point means the adapter advertises (or unconditionally owns,
    for capability-free checks) the capability under test yet does not implement
    the required method(s). That is a real conformance failure, not a free skip:
    a half-implemented adapter must NOT clear the CC-8 gate. See issue #6891.
    """
    missing = ", ".join(missing_methods)
    return ConformanceCheckResult(
        check_name=check_name,
        engine_name=_adapter_name(adapter),
        passed=False,
        skipped=False,
        message=(f"missing adapter method(s) for advertised capability: {missing}"),
    )


def _capability_skip(
    adapter: object,
    check_name: str,
    capability: str,
) -> ConformanceCheckResult:
    """Build a capability-aware skip for an engine that lacks a capability.

    A genuine missing capability is a legitimate skip (``passed=True,
    skipped=True``): the engine truthfully reports it does not support the
    feature, so there is nothing to validate. This is distinct from an adapter
    that *advertises* a capability but cannot back it up (see ``_skipped_result``
    and ``_capability_error``).
    """
    return ConformanceCheckResult(
        check_name=check_name,
        engine_name=_adapter_name(adapter),
        passed=True,
        skipped=True,
        message=f"capability not supported: {capability}",
    )


def _capability_error(
    adapter: object,
    check_name: str,
    capability: str,
    exc: Exception,
) -> ConformanceCheckResult:
    """Build a failure result when a ``supports()`` query raises.

    A capability descriptor whose ``supports()`` raises cannot be trusted to
    truthfully report a missing capability, so it must not be routed to a
    passing skip (which previously let a capability-silent adapter clear the
    gate with zero validation). See issue #6891.
    """
    return ConformanceCheckResult(
        check_name=check_name,
        engine_name=_adapter_name(adapter),
        passed=False,
        skipped=False,
        message=(
            f"capability query for {capability!r} raised {type(exc).__name__}: {exc}"
        ),
    )


def _conformance_result(
    adapter: object,
    check_name: str,
    validation: ValidationResult,
) -> ConformanceCheckResult:
    """Convert a numeric validation result into a conformance result."""
    return ConformanceCheckResult(
        check_name=check_name,
        engine_name=_adapter_name(adapter),
        passed=validation.passed,
        message=validation.message,
        validation=validation,
    )


def _supports_capability(adapter: object, capability: str) -> bool:
    """Return whether an adapter advertises a capability.

    This supports both the current ``EngineCapabilities`` fields and the
    canonical ``supports()`` query contract from PR #6824.

    Raises:
        A ``supports()`` query that itself raises is propagated to the caller
        rather than swallowed into a ``False`` (which previously routed a
        capability-silent adapter to a passing skip — issue #6891). Callers use
        :meth:`CrossEngineValidator._require_capability` to convert the raised
        error into a failing conformance result.
    """
    descriptor = _capability_descriptor(adapter)
    if descriptor is None:
        return False

    supports = getattr(descriptor, "supports", None)
    if callable(supports):
        return bool(supports(capability))

    raw = getattr(descriptor, capability, None)
    if raw is None:
        raw = getattr(descriptor, f"has_{capability}", None)
    if isinstance(raw, CapabilityLevel):
        return raw is not CapabilityLevel.NONE
    if isinstance(raw, bool):
        return raw
    return False


def _capability_descriptor(adapter: object) -> object | None:
    """Return a capability descriptor from an adapter-like object."""
    get_capabilities = getattr(adapter, "get_capabilities", None)
    if callable(get_capabilities):
        return get_capabilities()
    return getattr(adapter, "capabilities", None)


def _flatten_named_arrays(
    named_arrays: Mapping[str, object],
    names: Sequence[str],
) -> np.ndarray:
    """Flatten named canonical state arrays in deterministic order."""
    values = []
    for name in names:
        require(name in named_arrays, f"canonical state missing {name}")
        values.append(np.asarray(named_arrays[name], dtype=float).reshape(-1))
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def _flatten_pose_map(poses: Mapping[str, Mapping[str, object]]) -> np.ndarray:
    """Flatten nested pose maps in deterministic order."""
    values = []
    for pose_name in sorted(poses):
        pose = poses[pose_name]
        for body_name in sorted(pose):
            values.append(np.asarray(pose[body_name], dtype=float).reshape(-1))
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)
