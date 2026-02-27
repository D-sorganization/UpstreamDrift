"""Protocol interfaces for upstream drift tools.

These protocols define the structural typing contracts for process
calculators, data processors, and UI state management. They enable
loose coupling between the shared calculation engines, GUI layers,
and the FastAPI backend.

Usage:
    def size_equipment(calc: ProcessCalculator) -> dict[str, float]:
        inputs = calc.get_default_inputs()
        return calc.calculate(inputs)

    def run_validated(calc: Calculator, inputs: dict) -> CalculationResult:
        vr = calc.validate_inputs(inputs)
        if not vr.valid:
            raise ValueError(vr.errors)
        return calc.calculate(inputs)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CalculationResult:
    """Immutable container for calculator outputs.

    Attributes:
        values:   Mapping of result-name to numeric value.
        units:    Mapping of result-name to its unit string.
        warnings: Non-fatal messages generated during the calculation.
        metadata: Arbitrary extra information (timing, solver stats, etc.).
    """

    values: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Outcome of an input-validation pass.

    Attributes:
        valid:    ``True`` when all checks passed.
        errors:   Fatal problems that prevent calculation.
        warnings: Non-fatal issues the user should be aware of.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Calculator(Protocol):
    """Unified protocol that every calculator must satisfy.

    Implementations provide ``calculate`` and ``validate_inputs`` methods
    together with ``name`` and ``version`` properties so that callers can
    treat all calculators uniformly.
    """

    @property
    def name(self) -> str:
        """Human-readable calculator name."""
        ...

    @property
    def version(self) -> str:
        """Semantic version string (e.g. ``'1.2.0'``)."""
        ...

    def calculate(self, inputs: dict[str, Any]) -> CalculationResult:
        """Run the calculation and return a :class:`CalculationResult`."""
        ...

    def validate_inputs(self, inputs: dict[str, Any]) -> ValidationResult:
        """Validate *inputs* before calculation."""
        ...


@runtime_checkable
class ProcessCalculator(Protocol):
    """Protocol for process engineering calculators.

    All calculators in ``process_calculators/`` share the pattern of
    accepting a dict (or dataclass) of inputs and returning a dict
    (or dataclass) of results.

    Implementations: FlareCalculator, BaghouseCalculator,
    AcidGasDewpointCalculator, FinancialCalculator, etc.
    """

    def calculate(self, inputs: Any) -> Any:
        """Run the calculation and return results."""
        ...


@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for tabular data transformation operations.

    Covers the common pattern in DataProcessor and its pipeline
    stages: accept data, transform it, return the result.
    """

    def transform(self, data: Any) -> Any:
        """Apply a transformation and return the result."""
        ...


@runtime_checkable
class StateSerializable(Protocol):
    """Protocol for objects that can save/restore UI state.

    Used by CalculatorStateMixin and any widget that persists
    its configuration across sessions.
    """

    def save_state(self) -> dict[str, Any]:
        """Serialize current state to a dictionary."""
        ...

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore state from a previously serialized dictionary."""
        ...


@runtime_checkable
class UnitConverter(Protocol):
    """Protocol for unit conversion operations.

    Implementations should convert a numeric value from one unit
    to another within the same physical dimension.
    """

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert *value* from *from_unit* to *to_unit*."""
        ...


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class InputValidator:
    """Reusable validation primitives for calculator inputs.

    All ``require_*`` / ``validate_*`` methods raise :class:`ValueError`
    on failure so they can be composed inside ``Calculator.validate_inputs``
    implementations.

    Example::

        v = InputValidator()
        v.require_positive("flow_rate", inputs["flow_rate"])
        v.validate_temperature(inputs["temperature"])
    """

    # -- scalar checks -----------------------------------------------------

    @staticmethod
    def require_positive(name: str, value: float) -> None:
        """Raise :class:`ValueError` if *value* is not strictly positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def require_in_range(name: str, value: float, low: float, high: float) -> None:
        """Raise :class:`ValueError` if *value* is outside [*low*, *high*]."""
        if value < low or value > high:
            raise ValueError(f"{name} must be in range [{low}, {high}], got {value}")

    # -- dict checks -------------------------------------------------------

    @staticmethod
    def require_keys(inputs: dict[str, Any], required_keys: set[str]) -> None:
        """Raise :class:`ValueError` if any *required_keys* are missing."""
        missing = required_keys - set(inputs)
        if missing:
            raise ValueError(f"Missing required keys: {sorted(missing)}")

    # -- domain-specific checks --------------------------------------------

    @staticmethod
    def validate_temperature(value: float) -> None:
        """Temperature must be > 0 K (absolute zero excluded)."""
        if value <= 0:
            raise ValueError(f"Temperature must be > 0 K, got {value}")

    @staticmethod
    def validate_pressure(value: float) -> None:
        """Pressure must be > 0 Pa."""
        if value <= 0:
            raise ValueError(f"Pressure must be > 0 Pa, got {value}")

    @staticmethod
    def validate_composition(
        composition: dict[str, float], *, tolerance: float = 1e-6
    ) -> None:
        """Composition fractions must sum to ~1.0 within *tolerance*.

        Also rejects negative fractions.
        """
        for species, fraction in composition.items():
            if fraction < 0:
                raise ValueError(
                    f"Composition fraction for '{species}' is negative: {fraction}"
                )
        total = math.fsum(composition.values())
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"Composition fractions must sum to 1.0 (got {total})")
