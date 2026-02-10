"""Golf equipment specifications and configurations.

This module provides standard specifications for golf clubs and balls,
serving as a Single Source of Truth for equipment parameters across
different physics engines.
"""

from __future__ import annotations

# Golf club parameters (realistic values)
# All units in SI (meters, kg, radians) unless otherwise specified
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
        "club_loft": 0.17,  # ~10 degrees
        "flex_stiffness": [
            180.0,
            150.0,
            120.0,
        ],  # Upper, middle, lower (Nm/rad or similar)
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
        "club_loft": 0.56,  # ~32 degrees
        "flex_stiffness": [220.0, 200.0, 180.0],
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
        "club_loft": 0.96,  # ~55 degrees
        "flex_stiffness": [240.0, 220.0, 200.0],
    },
}


def get_club_config(club_type: str) -> dict[str, float | list[float]]:
    """Retrieve configuration for a specific club type.

    Args:
        club_type: "driver", "iron_7", or "wedge"

    Returns:
        Dictionary of club parameters

    Raises:
        ValueError: If club_type is not found
    """
    if club_type not in CLUB_CONFIGS:
        valid_types = ", ".join(CLUB_CONFIGS.keys())
        raise ValueError(
            f"Invalid club_type '{club_type}'. Must be one of: {valid_types}"
        )
    return CLUB_CONFIGS[club_type]
