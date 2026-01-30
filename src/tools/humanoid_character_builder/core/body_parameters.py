"""
Body parameter definitions for humanoid character builder.

This module provides dataclasses for specifying humanoid body parameters
including overall build, individual segment scaling, and appearance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BuildType(Enum):
    """Predefined body build types."""

    ECTOMORPH = "ectomorph"  # Lean, long limbs
    MESOMORPH = "mesomorph"  # Athletic, muscular
    ENDOMORPH = "endomorph"  # Heavier build
    AVERAGE = "average"  # Average proportions
    CUSTOM = "custom"  # User-defined


class GenderModel(Enum):
    """Gender model for anthropometric calculations."""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"  # Average of male/female


@dataclass
class Vector3:
    """Simple 3D vector for scaling and offsets."""

    x: float = 1.0
    y: float = 1.0
    z: float = 1.0

    def as_tuple(self) -> tuple[float, float, float]:
        """Return as tuple."""
        return (self.x, self.y, self.z)

    def __iter__(self):
        """Allow unpacking."""
        return iter([self.x, self.y, self.z])

    @classmethod
    def uniform(cls, value: float) -> Vector3:
        """Create uniform scale vector."""
        return cls(value, value, value)

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float]) -> Vector3:
        """Create from tuple."""
        return cls(t[0], t[1], t[2])


@dataclass
class RGBA:
    """RGBA color specification."""

    r: float = 0.8
    g: float = 0.7
    b: float = 0.6
    a: float = 1.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return as tuple."""
        return (self.r, self.g, self.b, self.a)

    def as_hex(self) -> str:
        """Return as hex color string."""
        return f"#{int(self.r * 255):02x}{int(self.g * 255):02x}{int(self.b * 255):02x}"


@dataclass
class SegmentParameters:
    """
    Parameters for an individual body segment.

    These can override the automatically computed values.
    """

    # Scaling relative to default
    scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))

    # Mass override (None = compute from anthropometry)
    mass_kg: float | None = None

    # Inertia override (None = compute from mesh/primitive)
    inertia_override: dict[str, float] | None = None

    # Center of mass offset from geometric center (None = compute)
    com_offset: Vector3 | None = None

    # Visual appearance
    color: RGBA | None = None

    # Custom mesh path (None = generate)
    mesh_path: str | None = None

    # Collision mesh simplification factor (0.0-1.0)
    collision_simplification: float = 0.5

    def has_mass_override(self) -> bool:
        """Check if mass is manually specified."""
        return self.mass_kg is not None

    def has_inertia_override(self) -> bool:
        """Check if inertia is manually specified."""
        return self.inertia_override is not None


@dataclass
class AppearanceParameters:
    """
    Visual appearance parameters for the character.

    These primarily affect mesh generation and textures,
    not physics properties.
    """

    # Skin tone (affects texture/material)
    skin_tone: RGBA = field(default_factory=lambda: RGBA(0.87, 0.72, 0.53, 1.0))

    # Face parameters (for future MakeHuman integration)
    face_preset: str = "default"

    # Age affects skin texture and subtle geometry
    age_years: float = 30.0

    # Muscularity affects surface detail
    muscle_definition: float = 0.5  # 0.0 = smooth, 1.0 = highly defined

    # Body hair density (for future use)
    body_hair: float = 0.0  # 0.0 = none, 1.0 = full

    # Eye color
    eye_color: RGBA = field(default_factory=lambda: RGBA(0.4, 0.3, 0.2, 1.0))

    # Hair parameters (for future use)
    hair_style: str = "none"
    hair_color: RGBA = field(default_factory=lambda: RGBA(0.2, 0.15, 0.1, 1.0))


@dataclass
class BodyParameters:
    """
    Complete body parameters for humanoid character generation.

    This is the main configuration object that specifies all aspects
    of the humanoid model to be generated.

    Example:
        params = BodyParameters(
            height_m=1.80,
            mass_kg=80.0,
            build_type=BuildType.ATHLETIC,
        )
    """

    # === Primary Parameters ===

    # Total height in meters
    height_m: float = 1.75

    # Total body mass in kilograms
    mass_kg: float = 75.0

    # Build type preset
    build_type: BuildType = BuildType.AVERAGE

    # Gender model for anthropometric ratios
    gender_model: GenderModel = GenderModel.NEUTRAL

    # === Build Factors (0.0 - 1.0 normalized) ===

    # Muscularity: 0.0 = lean, 1.0 = very muscular
    muscularity: float = 0.5

    # Body fat percentage: 0.0 = very lean, 1.0 = high body fat
    body_fat_factor: float = 0.2

    # === Proportion Factors (1.0 = default, relative scaling) ===

    # Shoulder width relative to height
    shoulder_width_factor: float = 1.0

    # Hip width relative to height
    hip_width_factor: float = 1.0

    # Arm length relative to height
    arm_length_factor: float = 1.0

    # Leg length relative to height
    leg_length_factor: float = 1.0

    # Torso length relative to height
    torso_length_factor: float = 1.0

    # Head size relative to height
    head_scale_factor: float = 1.0

    # Neck length relative to height
    neck_length_factor: float = 1.0

    # Hand size relative to arm length
    hand_scale_factor: float = 1.0

    # Foot size relative to leg length
    foot_scale_factor: float = 1.0

    # === Individual Segment Overrides ===
    # Keys are segment names from HUMANOID_SEGMENTS
    segment_overrides: dict[str, SegmentParameters] = field(default_factory=dict)

    # === Appearance ===
    appearance: AppearanceParameters = field(default_factory=AppearanceParameters)

    # === Metadata ===
    name: str = "humanoid"
    description: str = ""

    def get_segment_params(self, segment_name: str) -> SegmentParameters:
        """
        Get parameters for a specific segment.

        Returns the override if specified, otherwise a default SegmentParameters.
        """
        return self.segment_overrides.get(segment_name, SegmentParameters())

    def set_segment_override(
        self, segment_name: str, params: SegmentParameters
    ) -> None:
        """Set override parameters for a specific segment."""
        self.segment_overrides[segment_name] = params

    def get_effective_gender_factor(self) -> float:
        """
        Get gender factor for interpolation (0.0 = female, 1.0 = male).

        Used for blending anthropometric data.
        """
        if self.gender_model == GenderModel.MALE:
            return 1.0
        elif self.gender_model == GenderModel.FEMALE:
            return 0.0
        else:
            return 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "height_m": self.height_m,
            "mass_kg": self.mass_kg,
            "build_type": self.build_type.value,
            "gender_model": self.gender_model.value,
            "muscularity": self.muscularity,
            "body_fat_factor": self.body_fat_factor,
            "shoulder_width_factor": self.shoulder_width_factor,
            "hip_width_factor": self.hip_width_factor,
            "arm_length_factor": self.arm_length_factor,
            "leg_length_factor": self.leg_length_factor,
            "torso_length_factor": self.torso_length_factor,
            "head_scale_factor": self.head_scale_factor,
            "neck_length_factor": self.neck_length_factor,
            "hand_scale_factor": self.hand_scale_factor,
            "foot_scale_factor": self.foot_scale_factor,
            "name": self.name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyParameters:
        """Create from dictionary."""
        # Handle enum conversions
        if "build_type" in data and isinstance(data["build_type"], str):
            data["build_type"] = BuildType(data["build_type"])
        if "gender_model" in data and isinstance(data["gender_model"], str):
            data["gender_model"] = GenderModel(data["gender_model"])

        # Filter to known fields
        known_fields = {
            "height_m",
            "mass_kg",
            "build_type",
            "gender_model",
            "muscularity",
            "body_fat_factor",
            "shoulder_width_factor",
            "hip_width_factor",
            "arm_length_factor",
            "leg_length_factor",
            "torso_length_factor",
            "head_scale_factor",
            "neck_length_factor",
            "hand_scale_factor",
            "foot_scale_factor",
            "name",
            "description",
        }
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def validate(self) -> list[str]:
        """
        Validate parameters and return list of errors.

        Returns empty list if valid.
        """
        errors = []

        if self.height_m <= 0:
            errors.append("height_m must be positive")
        if self.height_m < 0.5 or self.height_m > 3.0:
            errors.append("height_m should be between 0.5 and 3.0 meters")

        if self.mass_kg <= 0:
            errors.append("mass_kg must be positive")
        if self.mass_kg < 10 or self.mass_kg > 500:
            errors.append("mass_kg should be between 10 and 500 kg")

        # Check factors are reasonable
        factor_fields = [
            "muscularity",
            "body_fat_factor",
            "shoulder_width_factor",
            "hip_width_factor",
            "arm_length_factor",
            "leg_length_factor",
            "torso_length_factor",
            "head_scale_factor",
        ]
        for field_name in factor_fields:
            value = getattr(self, field_name)
            if value < 0:
                errors.append(f"{field_name} must be non-negative")
            if value > 3.0:
                errors.append(f"{field_name} is unusually large (> 3.0)")

        return errors


# Convenience factory functions
def create_athletic_body(
    height_m: float = 1.80, mass_kg: float = 80.0
) -> BodyParameters:
    """Create athletic body type parameters."""
    return BodyParameters(
        height_m=height_m,
        mass_kg=mass_kg,
        build_type=BuildType.MESOMORPH,
        muscularity=0.7,
        body_fat_factor=0.15,
        shoulder_width_factor=1.1,
        name="athletic_humanoid",
    )


def create_average_body(
    height_m: float = 1.75, mass_kg: float = 75.0
) -> BodyParameters:
    """Create average body type parameters."""
    return BodyParameters(
        height_m=height_m,
        mass_kg=mass_kg,
        build_type=BuildType.AVERAGE,
        muscularity=0.5,
        body_fat_factor=0.2,
        name="average_humanoid",
    )


def create_heavy_body(height_m: float = 1.78, mass_kg: float = 100.0) -> BodyParameters:
    """Create heavier body type parameters."""
    return BodyParameters(
        height_m=height_m,
        mass_kg=mass_kg,
        build_type=BuildType.ENDOMORPH,
        muscularity=0.4,
        body_fat_factor=0.35,
        hip_width_factor=1.1,
        name="heavy_humanoid",
    )
