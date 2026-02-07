"""Putter Stroke Model for Putting Simulation.

This module implements the physics of putter-ball interaction including
impact dynamics, energy transfer, and spin generation.

Physics Model:
    1. Putter strikes ball with given speed and angle
    2. Collision transfers energy based on COR and mass ratio
    3. Loft and attack angle create initial backspin
    4. Off-center hits create sidespin and reduce efficiency
    5. Ball launches with initial velocity and spin

References:
    - Cochran, A. & Stobbs, J. (1968). The Search for the Perfect Swing.
    - Penner, A.R. (2002). The Physics of Putting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.engines.physics_engines.putting_green.python.ball_roll_physics import BallState
from src.shared.python.physics_constants import (
    DEG_TO_RAD,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)


class PutterType(Enum):
    """Types of putters with different characteristics."""

    BLADE = "blade"
    MALLET = "mallet"
    FACE_BALANCED = "face_balanced"
    TOE_HANG = "toe_hang"
    CENTER_SHAFTED = "center_shafted"

    @property
    def default_toe_hang(self) -> float:
        """Default toe hang angle for this putter type [degrees]."""
        toe_hangs = {
            PutterType.BLADE: 45.0,
            PutterType.MALLET: 20.0,
            PutterType.FACE_BALANCED: 0.0,
            PutterType.TOE_HANG: 60.0,
            PutterType.CENTER_SHAFTED: 0.0,
        }
        return toe_hangs.get(self, 30.0)

    @property
    def default_moi(self) -> float:
        """Default moment of inertia [g⋅cm²]."""
        mois = {
            PutterType.BLADE: 3000.0,
            PutterType.MALLET: 5000.0,
            PutterType.FACE_BALANCED: 4500.0,
            PutterType.TOE_HANG: 3500.0,
            PutterType.CENTER_SHAFTED: 4000.0,
        }
        return mois.get(self, 4000.0)


@dataclass
class StrokeParameters:
    """Parameters defining a putting stroke.

    Attributes:
        speed: Clubhead speed at impact [m/s]
        direction: Direction of stroke (2D unit vector)
        face_angle: Face angle relative to path [degrees, + = open]
        attack_angle: Attack angle [degrees, - = descending]
        impact_location: Impact point on face [m, m] (0,0 = center)
        tempo: Stroke tempo multiplier (affects dynamics)
    """

    speed: float
    direction: np.ndarray
    face_angle: float = 0.0
    attack_angle: float = 0.0
    impact_location: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    tempo: float = 1.0

    def __post_init__(self) -> None:
        """Normalize direction vector."""
        self.direction = np.array(self.direction, dtype=np.float64)
        mag = np.linalg.norm(self.direction)
        if mag > 0:
            self.direction = self.direction / mag
        else:
            self.direction = np.array([1.0, 0.0])

        self.impact_location = np.array(self.impact_location, dtype=np.float64)

    @property
    def effective_direction(self) -> np.ndarray:
        """Get effective ball direction accounting for face angle.

        The ball starts at roughly 80% of face angle relative to path.
        """
        # Rotate direction by ~80% of face angle
        effective_angle = self.face_angle * 0.8 * DEG_TO_RAD
        cos_a, sin_a = np.cos(effective_angle), np.sin(effective_angle)

        return np.array(
            [
                cos_a * self.direction[0] - sin_a * self.direction[1],
                sin_a * self.direction[0] + cos_a * self.direction[1],
            ]
        )

    @classmethod
    def from_backstroke_length(
        cls,
        backstroke_length: float,
        direction: np.ndarray,
        tempo: float = 1.0,
        face_angle: float = 0.0,
        attack_angle: float = 0.0,
    ) -> StrokeParameters:
        """Create stroke parameters from backstroke length.

        Longer backstroke = faster stroke speed.

        Args:
            backstroke_length: Length of backstroke [m]
            direction: Stroke direction
            tempo: Tempo multiplier
            face_angle: Face angle [degrees]
            attack_angle: Attack angle [degrees]

        Returns:
            StrokeParameters instance
        """
        # Empirical relationship: speed ≈ 4 * backstroke_length * tempo
        speed = 4.0 * backstroke_length * tempo
        return cls(
            speed=speed,
            direction=direction,
            face_angle=face_angle,
            attack_angle=attack_angle,
            tempo=tempo,
        )

    @classmethod
    def for_target_distance(
        cls,
        distance: float,
        stimp_rating: float,
        direction: np.ndarray,
        face_angle: float = 0.0,
        slope_percent: float = 0.0,
    ) -> StrokeParameters:
        """Estimate stroke parameters for target distance.

        Args:
            distance: Target roll distance [m]
            stimp_rating: Green stimp rating
            direction: Stroke direction
            face_angle: Face angle [degrees]
            slope_percent: Average slope along putt [%]

        Returns:
            StrokeParameters tuned for distance
        """
        # Physics-based estimation:
        # Distance ≈ v₀² / (2 * μ * g) where μ ≈ 0.196/stimp
        mu = 0.196 / stimp_rating

        # Adjust for slope
        slope_factor = 1.0 + slope_percent / 100.0 * 2.0

        # Required initial velocity
        # d = v² / (2μg) → v = √(2μgd)
        v_required = np.sqrt(2 * mu * GRAVITY_M_S2 * distance * slope_factor)

        # Account for COR (ball speed ≈ 1.4 * clubhead speed for typical putter)
        clubhead_speed = v_required / 1.4

        return cls(
            speed=clubhead_speed,
            direction=direction,
            face_angle=face_angle,
            attack_angle=-2.0,  # Slight descending blow typical
        )

    @classmethod
    def create_preset(cls, name: str, direction: np.ndarray) -> StrokeParameters:
        """Create preset stroke parameters.

        Presets:
            - lag_putt: Gentle stroke for distance control
            - aggressive: Firm stroke to take break out
            - practice: Moderate controlled stroke

        Args:
            name: Preset name
            direction: Stroke direction

        Returns:
            StrokeParameters for preset
        """
        presets = {
            "lag_putt": cls(speed=1.5, direction=direction, attack_angle=-1.0),
            "aggressive": cls(speed=3.0, direction=direction, attack_angle=-3.0),
            "practice": cls(speed=2.0, direction=direction, attack_angle=-2.0),
            "tap_in": cls(speed=0.8, direction=direction, attack_angle=0.0),
        }

        if name not in presets:
            raise ValueError(f"Unknown preset: {name}")

        return presets[name]


class PutterStroke:
    """Physics model for putter-ball interaction.

    Computes the resulting ball state from a putting stroke including
    velocity, spin, and energy transfer.
    """

    # Standard putter properties
    DEFAULT_LOFT_DEG = 3.5
    DEFAULT_MASS_KG = 0.350  # 350g typical putter head mass
    DEFAULT_COR = 0.78

    def __init__(
        self,
        putter_type: PutterType = PutterType.MALLET,
        loft_deg: float | None = None,
        mass: float | None = None,
        coefficient_of_restitution: float | None = None,
        insert_type: str = "metal",
    ) -> None:
        """Initialize putter model.

        Args:
            putter_type: Type of putter
            loft_deg: Loft angle [degrees]
            mass: Head mass [kg]
            coefficient_of_restitution: COR for impact
            insert_type: Face insert material ("metal", "polymer", "milled")
        """
        self.putter_type = putter_type
        self.loft = loft_deg or self.DEFAULT_LOFT_DEG
        self.mass = mass or self.DEFAULT_MASS_KG
        self.insert_type = insert_type

        # Set COR based on insert type if not specified
        if coefficient_of_restitution is not None:
            self.coefficient_of_restitution = coefficient_of_restitution
        else:
            insert_cors = {
                "metal": 0.82,
                "milled": 0.80,
                "polymer": 0.75,
                "insert": 0.73,
            }
            self.coefficient_of_restitution = insert_cors.get(insert_type, 0.78)

        # Sweet spot size based on putter type MOI
        self.sweet_spot_size = self._compute_sweet_spot_size()

    def _compute_sweet_spot_size(self) -> float:
        """Compute effective sweet spot size from MOI."""
        # Higher MOI = larger sweet spot
        moi = self.putter_type.default_moi
        # Sweet spot radius: approximately sqrt(MOI/4000) * 0.02 meters
        return np.sqrt(moi / 4000.0) * 0.02

    def execute_stroke(
        self,
        ball_position: np.ndarray,
        params: StrokeParameters,
    ) -> BallState:
        """Execute stroke and compute resulting ball state.

        Args:
            ball_position: Ball position on green [m, m]
            params: Stroke parameters

        Returns:
            Initial ball state after impact
        """
        # Compute launch velocity
        launch_velocity = self.compute_launch_velocity(params)

        # Compute initial spin
        initial_spin = self.compute_initial_spin(params)

        return BallState(
            position=ball_position,
            velocity=launch_velocity,
            spin=initial_spin,
        )

    def compute_launch_velocity(self, params: StrokeParameters) -> np.ndarray:
        """Compute ball launch velocity from impact.

        Uses collision physics with COR and mass ratio.

        Args:
            params: Stroke parameters

        Returns:
            Launch velocity [m/s] as 2D vector
        """
        # Mass ratio
        m1 = self.mass  # Putter head
        m2 = GOLF_BALL_MASS_KG  # Ball

        # Velocity transfer with COR
        # v_ball = (m1 * v_club * (1 + e)) / (m1 + m2)
        e = self.coefficient_of_restitution
        v_transfer = (m1 * params.speed * (1 + e)) / (m1 + m2)

        # Off-center impact reduces velocity
        impact_offset = float(np.linalg.norm(params.impact_location))
        efficiency = self._compute_impact_efficiency(impact_offset)

        ball_speed = v_transfer * efficiency

        # Direction accounts for face angle
        direction = params.effective_direction

        return ball_speed * direction

    def compute_initial_spin(self, params: StrokeParameters) -> np.ndarray:
        """Compute initial ball spin from impact.

        Spin is generated by:
        1. Putter loft (backspin)
        2. Attack angle (modifies backspin)
        3. Off-center impact (sidespin)

        Args:
            params: Stroke parameters

        Returns:
            Spin vector [rad/s] as 3D (around x, y, z axes)
        """
        # Effective loft at impact
        effective_loft = self.loft - params.attack_angle  # Descending adds loft

        # Backspin from loft
        # Simplified model: spin ≈ (v * sin(loft)) / r * friction_factor
        loft_rad = effective_loft * DEG_TO_RAD
        backspin = (params.speed * np.sin(loft_rad) * 50) / GOLF_BALL_RADIUS_M

        # Backspin axis is perpendicular to ball direction
        v_dir = params.effective_direction
        # Backspin rotates about axis perpendicular to direction
        # Using right-hand rule: for ball moving in +x, backspin is about -y
        backspin_axis = np.array([-v_dir[1], v_dir[0], 0])

        # Sidespin from off-center impact
        toe_offset = params.impact_location[0]  # Positive = toe hit
        # Gear effect: toe hit creates hook spin, heel hit creates slice spin
        sidespin = -toe_offset * params.speed * 200 / GOLF_BALL_RADIUS_M

        # Combine spins
        spin = backspin * backspin_axis
        spin[2] = sidespin  # z-component for sidespin

        return spin

    def _compute_impact_efficiency(self, offset: float) -> float:
        """Compute energy transfer efficiency based on impact location.

        Impacts away from sweet spot lose energy to putter rotation.

        Args:
            offset: Distance from sweet spot [m]

        Returns:
            Efficiency factor (0-1)
        """
        if offset <= 0:
            return 1.0

        # Gaussian falloff from sweet spot
        # 50% efficiency at 2 * sweet_spot_size
        sigma = self.sweet_spot_size
        efficiency = np.exp(-0.5 * (offset / sigma) ** 2)

        return max(0.5, efficiency)  # Minimum 50% efficiency

    def estimate_required_speed(
        self,
        distance: float,
        stimp_rating: float,
        slope_percent: float = 0.0,
    ) -> float:
        """Estimate clubhead speed required for given distance.

        Args:
            distance: Target distance [m]
            stimp_rating: Green stimp rating
            slope_percent: Slope along putt [%] (+ = uphill)

        Returns:
            Required clubhead speed [m/s]
        """
        # Friction coefficient from stimp
        mu = 0.196 / stimp_rating

        # Adjust for slope
        slope_adjustment = 1.0 + slope_percent / 100.0 * 2.5

        # Required ball speed: v = √(2μgd)
        ball_speed = np.sqrt(2 * mu * GRAVITY_M_S2 * distance * slope_adjustment)

        # Convert to clubhead speed
        m1, m2 = self.mass, GOLF_BALL_MASS_KG
        e = self.coefficient_of_restitution

        # v_ball = (m1 * v_club * (1 + e)) / (m1 + m2)
        # v_club = v_ball * (m1 + m2) / (m1 * (1 + e))
        clubhead_speed = ball_speed * (m1 + m2) / (m1 * (1 + e))

        return clubhead_speed

    def compute_aim_point(
        self,
        ball_position: np.ndarray,
        target: np.ndarray,
        break_amount: float,
    ) -> np.ndarray:
        """Compute aim point to account for break.

        Args:
            ball_position: Current ball position [m, m]
            target: Target (hole) position [m, m]
            break_amount: Expected break distance [m] (+ = right to left)

        Returns:
            Aim point to start ball toward
        """
        # Direction to target
        to_target = target - ball_position
        distance = np.linalg.norm(to_target)

        if distance < 1e-10:
            return target

        target_dir = to_target / distance

        # Perpendicular direction (90 degrees counterclockwise)
        perp_dir = np.array([-target_dir[1], target_dir[0]])

        # Aim point is offset from target
        aim_point = target + perp_dir * break_amount

        return aim_point
