"""Named authoring defaults for HRG-GB1 (issue #15123).

Grip geometry angles live here as named defaults, not as unnamed literals in
projection code. Force and contact numbers are resolved only through provider
delegates outside this package.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[int] = 1

# Legacy GripPose() compatibility (PALM_DIAGONAL_DEG / placement defaults).
LEGACY_CLUB_CROSS_ANGLE_DEG: Final[float] = 40.0
LEGACY_LEAD_V_AZIMUTH_DEG: Final[float] = 20.0
LEGACY_TRAIL_V_AZIMUTH_DEG: Final[float] = 20.0
LEGACY_LEAD_THUMB_AZIMUTH_DEG: Final[float] = 10.0
LEGACY_TRAIL_THUMB_AZIMUTH_DEG: Final[float] = 0.0

# Dieter neutral grip thumb targets (degrees, target-side sign convention).
NEUTRAL_LEAD_THUMB_AZIMUTH_DEG: Final[float] = 22.5
NEUTRAL_TRAIL_THUMB_AZIMUTH_DEG: Final[float] = 45.0

DEFAULT_CURL_ORDER: Final[tuple[str, ...]] = (
    "lead_flat_pose",
    "lead_fingers_index_to_pinky",
    "lead_palm_wrap",
    "lead_thumb_press",
    "trail_flat_pose",
    "trail_fingers",
    "trail_palm_wrap",
    "trail_thumb_press",
)

DEFAULT_FORCE_PROVIDER_NAME: Final[str] = "unspecified"
DEFAULT_CONTACT_PROVIDER_NAME: Final[str] = "unspecified"
