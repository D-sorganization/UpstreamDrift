"""MJCF models for golf swing systems.

Includes simple pendulum models and biomechanically realistic golf swing models
with anthropometric body segments and two-handed grip.

Refactored: Model definitions split across sub-modules for maintainability.
All public names are re-exported here for backwards compatibility.

Sub-modules:
    ``models_pendulum``          – Chaotic, double, triple pendulum, 2-link, gimbal
    ``models_swing``             – Upper-body and full-body swing
    ``models_advanced``          – Advanced biomechanical swing (scapula, flex shaft)
    ``models_club_generation``   – External paths, CMU humanoid, club XML generators
"""

from __future__ import annotations

# Re-export shared constants (used by consumers that import from models directly)
from src.shared.python.core import constants  # noqa: F401
from src.shared.python.physics.equipment import CLUB_CONFIGS  # noqa: F401
from src.shared.python.physics.physics_parameters import (
    get_parameter_registry,  # noqa: F401
)

from .models_advanced import (  # noqa: F401
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
)
from .models_club_generation import (  # noqa: F401
    HUMANOID_CM_JOINTS,
    MYOARM_SIMPLE_PATH,
    MYOBODY_PATH,
    MYOUPPERBODY_PATH,
    generate_flexible_club_xml,
    generate_rigid_club_xml,
    load_humanoid_cm_xml,
)
from .models_pendulum import (  # noqa: F401
    CHAOTIC_PENDULUM_XML,
    DOUBLE_PENDULUM_XML,
    GIMBAL_JOINT_DEMO_XML,
    TRIPLE_PENDULUM_XML,
    TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML,
)
from .models_swing import (  # noqa: F401
    BALL_MASS,
    BALL_RADIUS,
    FULL_BODY_GOLF_SWING_XML,
    UPPER_BODY_GOLF_SWING_XML,
)

GRAVITY_M_S2 = float(constants.GRAVITY_M_S2)
DEFAULT_TIME_STEP = float(constants.DEFAULT_TIME_STEP)
