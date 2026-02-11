"""Turf Properties for Putting Green Simulation.

This module defines the physical properties of putting green turf including
stimp rating, friction coefficients, grain direction, and environmental effects.

Design by Contract:
    - Stimp rating must be between 1 and 15 (realistic range)
    - Friction coefficients must be positive and less than 1
    - Grain direction is automatically normalized

References:
    - USGA Stimpmeter: Green speed measurement standard
    - Haake, S.J. (1989). Ball-turf interaction. Science and Golf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.core.physics_constants import GRAVITY_M_S2


class GrassType(Enum):
    """Types of grass commonly found on putting greens."""

    BENT_GRASS = "bent_grass"
    BERMUDA = "bermuda"
    POA_ANNUA = "poa_annua"
    FESCUE = "fescue"
    RYE_GRASS = "rye_grass"
    PASPALUM = "paspalum"
    ZOYSIA = "zoysia"

    @property
    def default_grain_strength(self) -> float:
        """Default grain strength for this grass type."""
        grain_strengths = {
            GrassType.BENT_GRASS: 0.1,
            GrassType.BERMUDA: 0.4,  # Bermuda has strong grain
            GrassType.POA_ANNUA: 0.15,
            GrassType.FESCUE: 0.2,
            GrassType.RYE_GRASS: 0.15,
            GrassType.PASPALUM: 0.35,
            GrassType.ZOYSIA: 0.25,
        }
        return grain_strengths.get(self, 0.15)

    @property
    def default_height_mm(self) -> float:
        """Default height of cut for this grass type."""
        heights = {
            GrassType.BENT_GRASS: 3.0,
            GrassType.BERMUDA: 3.5,
            GrassType.POA_ANNUA: 3.2,
            GrassType.FESCUE: 4.0,
            GrassType.RYE_GRASS: 3.5,
            GrassType.PASPALUM: 3.5,
            GrassType.ZOYSIA: 3.8,
        }
        return heights.get(self, 3.5)


class TurfCondition(Enum):
    """Environmental conditions affecting turf behavior."""

    DRY = "dry"
    NORMAL = "normal"
    WET = "wet"
    DEWY = "dewy"
    FROSTED = "frosted"
    RECENTLY_MOWED = "recently_mowed"
    AFTERNOON_GROWTH = "afternoon_growth"

    @property
    def friction_multiplier(self) -> float:
        """Multiplier applied to base friction coefficient."""
        multipliers = {
            TurfCondition.DRY: 0.9,
            TurfCondition.NORMAL: 1.0,
            TurfCondition.WET: 1.3,
            TurfCondition.DEWY: 1.15,
            TurfCondition.FROSTED: 1.4,
            TurfCondition.RECENTLY_MOWED: 0.95,
            TurfCondition.AFTERNOON_GROWTH: 1.05,
        }
        return multipliers.get(self, 1.0)


@dataclass(frozen=True)
class TurfProperties:
    """Physical properties of putting green turf.

    This dataclass encapsulates all turf characteristics that affect
    ball rolling behavior.

    Attributes:
        stimp_rating: USGA Stimpmeter reading (feet ball rolls from ramp)
        grass_type: Type of grass on the green
        condition: Current environmental condition
        grain_direction: Direction of grass grain (2D unit vector)
        grain_strength: Strength of grain effect (0-1)
        height_of_cut_mm: Grass height in millimeters
        compaction_factor: Soil compaction (0-1, 1 = very firm)
        rolling_friction_coefficient: Override friction (None = auto-compute)

    Design by Contract:
        Preconditions:
            - 1 <= stimp_rating <= 15
            - 0 <= grain_strength <= 1
            - 0 < compaction_factor <= 1
            - height_of_cut_mm > 0
    """

    stimp_rating: float = 10.0
    grass_type: GrassType = GrassType.BENT_GRASS
    condition: TurfCondition = TurfCondition.NORMAL
    grain_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    grain_strength: float | None = None
    height_of_cut_mm: float | None = None
    compaction_factor: float = 0.8
    _friction_override: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize properties."""
        # Validate stimp rating
        if not 1 <= self.stimp_rating <= 15:
            raise ValueError(
                f"stimp_rating must be between 1 and 15, got {self.stimp_rating}"
            )

        # Normalize grain direction
        grain_mag = np.linalg.norm(self.grain_direction)
        if grain_mag > 0:
            object.__setattr__(
                self, "grain_direction", self.grain_direction / grain_mag
            )
        else:
            object.__setattr__(self, "grain_direction", np.array([1.0, 0.0]))

        # Set defaults based on grass type if not provided
        if self.grain_strength is None:
            object.__setattr__(
                self, "grain_strength", self.grass_type.default_grain_strength
            )

        if self.height_of_cut_mm is None:
            object.__setattr__(
                self, "height_of_cut_mm", self.grass_type.default_height_mm
            )

    @property
    def rolling_friction_coefficient(self) -> float:
        """Compute rolling friction coefficient from stimp rating.

        The relationship between stimp and friction is derived from
        energy conservation on the stimpmeter ramp:

        μ ≈ 0.196 / stimp

        This comes from:
        - Ramp angle: 20°
        - Ball release height: 6 inches
        - Energy loss to friction during roll
        """
        if self._friction_override is not None:
            return self._friction_override

        # Physics-based calculation
        # From stimpmeter: ball rolls 'stimp' feet from 20° ramp
        # Using energy conservation: μ ≈ h * sin(θ) / (stimp * cos(θ))
        # Simplified: μ ≈ 0.196 / stimp (approximately)
        base_friction = 0.196 / self.stimp_rating

        # Adjust for height of cut (longer grass = more friction)
        # Note: height_of_cut_mm is guaranteed non-None after __post_init__
        assert self.height_of_cut_mm is not None
        height_factor = 1.0 + 0.05 * (self.height_of_cut_mm - 3.0) / 2.0

        return base_friction * height_factor

    @property
    def effective_friction(self) -> float:
        """Get effective friction including condition effects."""
        return self.rolling_friction_coefficient * self.condition.friction_multiplier

    def compute_grain_effect(self, velocity_direction: np.ndarray) -> float:
        """Compute grain effect multiplier for given velocity direction.

        With grain: ball rolls faster (negative friction adjustment)
        Against grain: ball rolls slower (positive friction adjustment)

        Args:
            velocity_direction: Unit vector of ball velocity direction

        Returns:
            Multiplier for friction adjustment (-1 to +1)
        """
        if np.linalg.norm(velocity_direction) < 1e-10:
            return 0.0

        v_dir = velocity_direction / np.linalg.norm(velocity_direction)
        # Dot product gives alignment: +1 with grain, -1 against grain
        alignment = np.dot(v_dir, self.grain_direction)

        # Return signed effect: negative = faster, positive = slower
        return -alignment * self.grain_strength

    def compute_deceleration(self, velocity: np.ndarray) -> np.ndarray:
        """Compute deceleration vector from friction and grain.

        Args:
            velocity: Ball velocity vector [m/s]

        Returns:
            Deceleration vector [m/s²] (opposing motion)
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-10:
            return np.zeros(2)

        v_dir = velocity / speed

        # Base friction deceleration
        base_decel = self.effective_friction * GRAVITY_M_S2

        # Grain effect modifies deceleration
        grain_effect = self.compute_grain_effect(v_dir)
        modified_decel = base_decel * (1.0 + grain_effect)

        # Deceleration opposes motion
        return -modified_decel * v_dir

    def apply_grain_to_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Apply grain effect to velocity (slight directional bias).

        Strong grain can slightly curve the ball path toward
        the grain direction.

        Args:
            velocity: Current velocity vector

        Returns:
            Modified velocity with grain effect
        """
        speed = np.linalg.norm(velocity)
        if speed < 0.05:  # Grain effect negligible at very low speeds
            return velocity

        v_dir = velocity / speed

        # Cross-grain component causes slight curve
        # Perpendicular to velocity in direction of grain
        cross_grain = self.grain_direction - np.dot(self.grain_direction, v_dir) * v_dir
        cross_mag = np.linalg.norm(cross_grain)

        if cross_mag < 1e-10:
            return velocity

        cross_grain = cross_grain / cross_mag

        # Apply small cross-grain velocity component
        # Effect is proportional to grain strength and inversely to speed
        # Note: grain_strength is guaranteed non-None after __post_init__
        assert self.grain_strength is not None
        curve_amount = self.grain_strength * 0.01 / (1.0 + speed)
        return velocity + curve_amount * cross_grain * speed

    def compute_speed_factor(self) -> float:
        """Compute overall speed factor for the turf.

        Returns:
            Multiplier for ball speed (higher = faster)
        """
        # Higher compaction = faster (less energy absorption)
        compaction_effect = 0.8 + 0.2 * self.compaction_factor

        # Condition effect (inverse of friction multiplier)
        condition_effect = 1.0 / self.condition.friction_multiplier

        return compaction_effect * condition_effect

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stimp_rating": self.stimp_rating,
            "grass_type": self.grass_type.value,
            "condition": self.condition.value,
            "grain_direction": self.grain_direction.tolist(),
            "grain_strength": self.grain_strength,
            "height_of_cut_mm": self.height_of_cut_mm,
            "compaction_factor": self.compaction_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurfProperties:
        """Deserialize from dictionary."""
        return cls(
            stimp_rating=data.get("stimp_rating", 10.0),
            grass_type=GrassType(data.get("grass_type", "bent_grass")),
            condition=TurfCondition(data.get("condition", "normal")),
            grain_direction=np.array(data.get("grain_direction", [1.0, 0.0])),
            grain_strength=data.get("grain_strength"),
            height_of_cut_mm=data.get("height_of_cut_mm"),
            compaction_factor=data.get("compaction_factor", 0.8),
        )

    @classmethod
    def create_preset(cls, name: str) -> TurfProperties:
        """Create turf properties from a named preset.

        Available presets:
            - tournament_fast: Fast tournament conditions (stimp 11-12)
            - tournament_standard: Standard tournament (stimp 10-11)
            - municipal_slow: Public course slow greens (stimp 7-8)
            - augusta_like: Augusta National style (stimp 13+)
            - practice_green: Typical practice putting green

        Args:
            name: Name of the preset

        Returns:
            TurfProperties configured for the preset

        Raises:
            ValueError: If preset name is unknown
        """
        presets = {
            "tournament_fast": cls(
                stimp_rating=12.0,
                grass_type=GrassType.BENT_GRASS,
                condition=TurfCondition.NORMAL,
                grain_strength=0.08,
                height_of_cut_mm=2.8,
                compaction_factor=0.9,
            ),
            "tournament_standard": cls(
                stimp_rating=10.5,
                grass_type=GrassType.BENT_GRASS,
                condition=TurfCondition.NORMAL,
                grain_strength=0.1,
                height_of_cut_mm=3.2,
                compaction_factor=0.85,
            ),
            "municipal_slow": cls(
                stimp_rating=8.0,
                grass_type=GrassType.POA_ANNUA,
                condition=TurfCondition.NORMAL,
                grain_strength=0.2,
                height_of_cut_mm=4.0,
                compaction_factor=0.7,
            ),
            "augusta_like": cls(
                stimp_rating=13.5,
                grass_type=GrassType.BENT_GRASS,
                condition=TurfCondition.DRY,
                grain_strength=0.05,
                height_of_cut_mm=2.5,
                compaction_factor=0.95,
            ),
            "practice_green": cls(
                stimp_rating=9.5,
                grass_type=GrassType.BENT_GRASS,
                condition=TurfCondition.NORMAL,
                grain_strength=0.12,
                height_of_cut_mm=3.5,
                compaction_factor=0.8,
            ),
            "bermuda_summer": cls(
                stimp_rating=9.0,
                grass_type=GrassType.BERMUDA,
                condition=TurfCondition.DRY,
                grain_strength=0.4,
                grain_direction=np.array([0.707, 0.707]),  # 45 degrees
                height_of_cut_mm=3.5,
                compaction_factor=0.85,
            ),
            "links_fescue": cls(
                stimp_rating=10.0,
                grass_type=GrassType.FESCUE,
                condition=TurfCondition.NORMAL,
                grain_strength=0.2,
                height_of_cut_mm=4.0,
                compaction_factor=0.75,
            ),
            "wet_morning": cls(
                stimp_rating=10.0,
                grass_type=GrassType.BENT_GRASS,
                condition=TurfCondition.DEWY,
                grain_strength=0.1,
                height_of_cut_mm=3.2,
                compaction_factor=0.8,
            ),
        }

        if name not in presets:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(presets.keys())}"
            )

        return presets[name]
