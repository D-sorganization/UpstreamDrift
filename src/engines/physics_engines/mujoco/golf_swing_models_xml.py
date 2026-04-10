"""MJCF XML model definitions.

Split from head_models.py for maintainability.
XML strings extracted to _golf_swing_*_xml.py modules to keep each file
under 500 LOC. All public names remain importable from this module.
"""

from __future__ import annotations

from ._golf_swing_advanced_xml import ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
from ._golf_swing_full_body_xml import FULL_BODY_GOLF_SWING_XML
from ._golf_swing_upper_body_xml import UPPER_BODY_GOLF_SWING_XML

# MYO simulator model paths
# Upper body model (19 DOF, 20 actuators): Torso + head + both arms
MYOUPPERBODY_PATH = "myo_sim/body/myoupperbody.xml"

# Full body model (52 DOF, 290 actuators): Complete musculoskeletal system
# Includes torso, head, arms, and legs with muscle-tendon units
MYOBODY_PATH = "myo_sim/body/myobody.xml"

# Simplified arm model (bilateral, 14 DOF): Both arms with simplified torso
MYOARM_SIMPLE_PATH = "myo_sim/arm/myoarm_simple.xml"


# GOLF CLUB CONFIGURATIONS
# ==============================================================================

# Golf club parameters (realistic values)
CLUB_CONFIGS: dict[str, dict[str, float | list[float]]] = {
    "driver": {
        "grip_length": 0.28,
        "grip_radius": 0.0145,
        "grip_mass": 0.050,
        "shaft_length": 1.10,  # Total shaft length
        "shaft_radius": 0.0062,
        "shaft_mass": 0.065,
        "head_mass": 0.198,
        "head_size": [0.062, 0.048, 0.038],
        "total_length": 1.16,
        "club_loft": 0.17,  # 10 degrees
        "flex_stiffness": [180, 150, 120],  # Upper, middle, lower
    },
    "iron_7": {
        "grip_length": 0.26,
        "grip_radius": 0.0140,
        "grip_mass": 0.048,
        "shaft_length": 0.94,
        "shaft_radius": 0.0058,
        "shaft_mass": 0.072,
        "head_mass": 0.253,
        "head_size": [0.038, 0.025, 0.045],
        "total_length": 0.95,
        "club_loft": 0.56,  # 32 degrees
        "flex_stiffness": [220, 200, 180],
    },
    "wedge": {
        "grip_length": 0.25,
        "grip_radius": 0.0138,
        "grip_mass": 0.045,
        "shaft_length": 0.89,
        "shaft_radius": 0.0056,
        "shaft_mass": 0.078,
        "head_mass": 0.288,
        "head_size": [0.032, 0.022, 0.048],
        "total_length": 0.90,
        "club_loft": 0.96,  # 55 degrees
        "flex_stiffness": [240, 220, 200],
    },
}

__all__ = [
    "ADVANCED_BIOMECHANICAL_GOLF_SWING_XML",
    "CLUB_CONFIGS",
    "FULL_BODY_GOLF_SWING_XML",
    "MYOARM_SIMPLE_PATH",
    "MYOBODY_PATH",
    "MYOUPPERBODY_PATH",
    "UPPER_BODY_GOLF_SWING_XML",
]
