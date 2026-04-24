from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from src.shared.python.core.physics_constants import (
    DRIVER_COR,
    DRIVER_MOI_KG_M2,
    TYPICAL_CONTACT_DURATION_S,
)


class ImpactModelType(Enum):
    """Types of impact physics models."""

    RIGID_BODY = auto()  # Instantaneous impulse with COR
    SPRING_DAMPER = auto()  # Kelvin-Voigt viscoelastic
    FINITE_TIME = auto()  # Impulse-momentum with duration


@dataclass
class PreImpactState:
    """State of ball and clubhead immediately before impact.

    Attributes:
        clubhead_velocity: Clubhead velocity [m/s] (3,)
        clubhead_angular_velocity: Clubhead angular velocity [rad/s] (3,)
        clubhead_orientation: Clubface normal vector [unitless] (3,)
        ball_position: Ball center position [m] (3,)
        ball_velocity: Ball velocity [m/s] (3,)
        ball_angular_velocity: Ball spin [rad/s] (3,)
        clubhead_mass: Effective clubhead mass [kg]
        clubhead_loft: Clubface loft angle [rad]
        clubhead_lie: Clubface lie angle [rad]
        clubhead_moi: Clubhead moment of inertia about CG [kg·m²]
        impact_offset: Impact location offset from CG on clubface [m] (2,) [horizontal, vertical]
    """

    clubhead_velocity: np.ndarray
    clubhead_angular_velocity: np.ndarray
    clubhead_orientation: np.ndarray
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    ball_angular_velocity: np.ndarray
    clubhead_mass: float = 0.200  # [kg] Typical driver head
    clubhead_loft: float = np.radians(10.5)  # [rad] Driver loft
    clubhead_lie: float = np.radians(60.0)  # [rad] Lie angle
    clubhead_moi: float = float(DRIVER_MOI_KG_M2)  # [kg·m²] MOI about CG
    impact_offset: np.ndarray | None = None  # [m] (2,) offset from CG


@dataclass
class PostImpactState:
    """State of ball and clubhead immediately after impact.

    Attributes:
        ball_velocity: Ball launch velocity [m/s] (3,)
        ball_angular_velocity: Ball spin [rad/s] (3,)
        clubhead_velocity: Clubhead velocity after impact [m/s] (3,)
        clubhead_angular_velocity: Clubhead angular velocity after [rad/s] (3,)
        contact_duration: Duration of contact [s]
        energy_transfer: Kinetic energy transferred to ball [J]
        impact_location: Location of impact on clubface [m] (2,) [x, y from center]
    """

    ball_velocity: np.ndarray
    ball_angular_velocity: np.ndarray
    clubhead_velocity: np.ndarray
    clubhead_angular_velocity: np.ndarray
    contact_duration: float
    energy_transfer: float
    impact_location: np.ndarray


@dataclass
class ImpactParameters:
    """Parameters for impact model.

    Attributes:
        cor: Coefficient of restitution (0-1)
        contact_duration: Contact time [s]
        contact_stiffness: Spring stiffness for compliant model [N/m]
        contact_damping: Damping for compliant model [N·s/m]
        friction_coefficient: Ball-face friction
        gear_effect_factor: Gear effect spin amplification (0-1)
    """

    cor: float = float(DRIVER_COR)
    contact_duration: float = float(TYPICAL_CONTACT_DURATION_S)
    contact_stiffness: float = 1e6  # [N/m]
    contact_damping: float = 1e3  # [N·s/m]
    friction_coefficient: float = 0.4
    gear_effect_factor: float = 0.5
    gear_effect_h_scale: float = 100.0
    gear_effect_v_scale: float = 50.0


@dataclass
class ImpactEvent:
    """Complete record of a single impact event.

    Issue #758: Surface pre-impact and post-impact states in recorder outputs.

    Attributes:
        timestamp: Simulation time when impact occurred [s]
        pre_state: State before impact
        post_state: State after impact
        energy_balance: Energy analysis results
        impact_id: Unique identifier for this impact
        model_type: Type of impact model used
    """

    timestamp: float
    pre_state: PreImpactState
    post_state: PostImpactState
    energy_balance: dict[str, float]
    impact_id: int
    model_type: ImpactModelType
