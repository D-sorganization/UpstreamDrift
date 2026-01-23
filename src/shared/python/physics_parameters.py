"""Physics parameter registry for Golf Modeling Suite.

This module provides a central registry for all physics parameters
with validation, units, and source citations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.shared.python import constants


class ParameterCategory(Enum):
    """Categories of physics parameters."""

    BALL = "ball"
    CLUB = "club"
    ENVIRONMENT = "environment"
    BIOMECHANICS = "biomechanics"
    SIMULATION = "simulation"


@dataclass
class PhysicsParameter:
    """A physics parameter with validation and metadata."""

    name: str
    value: Any
    unit: str
    category: ParameterCategory
    description: str
    source: str
    min_value: float | None = None
    max_value: float | None = None
    # True for values that shouldn't change (e.g., USGA rules)
    is_constant: bool = False

    def validate(self, new_value: Any) -> tuple[bool, str]:
        """Validate a new value against constraints.

        Args:
            new_value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.is_constant:
            return False, f"{self.name} is a constant and cannot be changed"

        # Type check
        if not isinstance(new_value, int | float):
            return False, f"{self.name} must be numeric"

        # Range check
        if self.min_value is not None and new_value < self.min_value:
            return (
                False,
                f"{self.name} must be >= {self.min_value} {self.unit}",
            )

        if self.max_value is not None and new_value > self.max_value:
            return (
                False,
                f"{self.name} must be <= {self.max_value} {self.unit}",
            )

        return True, ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "unit": self.unit,
            "category": self.category.value,
            "description": self.description,
            "source": self.source,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "is_constant": self.is_constant,
        }


class PhysicsParameterRegistry:
    """Central registry for all physics parameters."""

    def __init__(self) -> None:
        """Initialize the parameter registry."""
        self.parameters: dict[str, PhysicsParameter] = {}
        self._load_default_parameters()

    def _load_default_parameters(self) -> None:
        """Load default parameters from various sources."""
        # Ball parameters (USGA Rules)
        self.register(
            PhysicsParameter(
                name="BALL_MASS",
                value=constants.GOLF_BALL_MASS_KG,
                unit="kg",
                category=ParameterCategory.BALL,
                description="Golf ball mass (USGA maximum)",
                source="USGA Rules of Golf, Appendix III",
                min_value=constants.GOLF_BALL_MASS_KG,
                max_value=constants.GOLF_BALL_MASS_KG,
                is_constant=True,  # Exact per rules
            )
        )

        self.register(
            PhysicsParameter(
                name="BALL_DIAMETER",
                value=constants.GOLF_BALL_DIAMETER_M,
                unit="m",
                category=ParameterCategory.BALL,
                description="Golf ball diameter (USGA minimum)",
                source="USGA Rules of Golf, Appendix III",
                min_value=constants.GOLF_BALL_DIAMETER_M,
                max_value=0.10,  # Reasonable upper bound
                is_constant=False,
            )
        )

        self.register(
            PhysicsParameter(
                name="BALL_RADIUS",
                value=0.021335,  # m
                unit="m",
                category=ParameterCategory.BALL,
                description="Golf ball radius",
                source="Calculated from USGA diameter",
                min_value=0.021335,
                max_value=0.05,
                is_constant=False,
            )
        )

        # Club parameters
        self.register(
            PhysicsParameter(
                name="CLUB_MASS",
                value=0.310,  # kg
                unit="kg",
                category=ParameterCategory.CLUB,
                description="Driver club mass (typical)",
                source="Typical driver specifications",
                min_value=0.200,
                max_value=0.500,
                is_constant=False,
            )
        )

        self.register(
            PhysicsParameter(
                name="CLUB_LENGTH",
                value=1.143,  # m (45 inches)
                unit="m",
                category=ParameterCategory.CLUB,
                description="Driver club length (typical)",
                source="Typical driver specifications",
                min_value=0.900,
                max_value=1.219,  # 48 inches (USGA limit)
                is_constant=False,
            )
        )

        self.register(
            PhysicsParameter(
                name="CLUB_HEAD_MASS",
                value=0.200,  # kg
                unit="kg",
                category=ParameterCategory.CLUB,
                description="Driver club head mass",
                source="Typical driver specifications",
                min_value=0.150,
                max_value=0.250,
                is_constant=False,
            )
        )

        # Environment parameters
        self.register(
            PhysicsParameter(
                name="GRAVITY",
                value=constants.GRAVITY_M_S2,
                unit="m/s²",
                category=ParameterCategory.ENVIRONMENT,
                description="Standard gravity (sea level, 45° latitude)",
                source="NIST",
                min_value=constants.GRAVITY_M_S2,
                max_value=constants.GRAVITY_M_S2,
                is_constant=True,
            )
        )

        self.register(
            PhysicsParameter(
                name="AIR_DENSITY",
                value=constants.AIR_DENSITY_SEA_LEVEL_KG_M3,
                unit="kg/m³",
                category=ParameterCategory.ENVIRONMENT,
                description="Air density at sea level, 15°C",
                source="ISA Standard Atmosphere",
                min_value=0.9,  # High altitude
                max_value=1.3,  # Sea level, cold
                is_constant=False,
            )
        )

        self.register(
            PhysicsParameter(
                name="DRAG_COEFFICIENT",
                value=constants.GOLF_BALL_DRAG_COEFFICIENT,
                unit="dimensionless",
                category=ParameterCategory.BALL,
                description="Golf ball drag coefficient (typical)",
                source="Golf ball aerodynamics literature",
                min_value=0.20,
                max_value=0.30,
                is_constant=False,
            )
        )

        # Biomechanics parameters
        self.register(
            PhysicsParameter(
                name="HUMAN_HEIGHT",
                value=1.75,  # m
                unit="m",
                category=ParameterCategory.BIOMECHANICS,
                description="Average human height",
                source="WHO statistics",
                min_value=1.40,
                max_value=2.20,
                is_constant=False,
            )
        )

        self.register(
            PhysicsParameter(
                name="HUMAN_MASS",
                value=75.0,  # kg
                unit="kg",
                category=ParameterCategory.BIOMECHANICS,
                description="Average human mass",
                source="WHO statistics",
                min_value=40.0,
                max_value=150.0,
                is_constant=False,
            )
        )

        # Simulation parameters
        self.register(
            PhysicsParameter(
                name="TIMESTEP",
                value=0.001,  # s (1 ms)
                unit="s",
                category=ParameterCategory.SIMULATION,
                description="Simulation timestep",
                source="Typical physics simulation",
                min_value=0.0001,
                max_value=0.01,
                is_constant=False,
            )
        )

        self.register(
            PhysicsParameter(
                name="SWING_DURATION",
                value=3.0,  # s
                unit="s",
                category=ParameterCategory.SIMULATION,
                description="Typical golf swing duration",
                source="Golf biomechanics research",
                min_value=1.0,
                max_value=5.0,
                is_constant=False,
            )
        )

    def register(self, param: PhysicsParameter) -> None:
        """Register a parameter.

        Args:
            param: Parameter to register
        """
        self.parameters[param.name] = param

    def get(self, name: str) -> PhysicsParameter | None:
        """Get a parameter by name.

        Args:
            name: Parameter name

        Returns:
            Parameter or None if not found
        """
        return self.parameters.get(name)

    def set(self, name: str, value: Any) -> tuple[bool, str]:
        """Set a parameter value with validation.

        Args:
            name: Parameter name
            value: New value

        Returns:
            Tuple of (success, error_message)
        """
        param = self.parameters.get(name)
        if param is None:
            return False, f"Parameter {name} not found"

        is_valid, error_msg = param.validate(value)
        if is_valid:
            param.value = value
            return True, ""

        return False, error_msg

    def get_by_category(self, category: ParameterCategory) -> list[PhysicsParameter]:
        """Get all parameters in a category.

        Args:
            category: Parameter category

        Returns:
            List of parameters in category
        """
        return [
            param for param in self.parameters.values() if param.category == category
        ]

    def get_all_categories(self) -> list[ParameterCategory]:
        """Get all parameter categories.

        Returns:
            List of categories
        """
        return list(ParameterCategory)

    def export_to_dict(self) -> dict[str, Any]:
        """Export all parameters to dictionary.

        Returns:
            Dictionary of parameters
        """
        return {name: param.to_dict() for name, param in self.parameters.items()}

    def export_to_json(self, filepath: Path | str) -> None:
        """Export parameters to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self.export_to_dict(), f, indent=2)

    def import_from_json(self, filepath: Path | str) -> int:
        """Import parameters from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Number of parameters imported
        """
        with open(filepath) as f:
            data = json.load(f)

        count = 0
        for name, param_data in data.items():
            # Only update value if parameter exists
            if name in self.parameters:
                self.parameters[name]
                success, _ = self.set(name, param_data["value"])
                if success:
                    count += 1

        return count

    def get_summary(self) -> str:
        """Get human-readable summary of all parameters.

        Returns:
            Formatted summary
        """
        lines = [
            "",
            "=" * 80,
            "Physics Parameter Registry",
            "=" * 80,
            "",
        ]

        for category in ParameterCategory:
            params = self.get_by_category(category)
            if not params:
                continue

            lines.append(f"{category.value.upper()}")
            lines.append("-" * 80)

            for param in params:
                const_marker = " [CONSTANT]" if param.is_constant else ""
                lines.append(f"  {param.name}{const_marker}")
                lines.append(f"    Value: {param.value} {param.unit}")
                lines.append(f"    Description: {param.description}")
                lines.append(f"    Source: {param.source}")

                if not param.is_constant:
                    range_str = ""
                    if param.min_value is not None:
                        range_str += f"min={param.min_value}"
                    if param.max_value is not None:
                        if range_str:
                            range_str += ", "
                        range_str += f"max={param.max_value}"
                    if range_str:
                        lines.append(f"    Range: {range_str} {param.unit}")

                lines.append("")

        lines.append("=" * 80)
        lines.append("")

        return "\n".join(lines)


# Global registry instance
_registry: PhysicsParameterRegistry | None = None


def get_registry() -> PhysicsParameterRegistry:
    """Get the global physics parameter registry.

    Returns:
        Global registry instance
    """
    global _registry
    if _registry is None:
        _registry = PhysicsParameterRegistry()
    return _registry
