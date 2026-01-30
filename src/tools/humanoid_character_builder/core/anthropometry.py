"""
Anthropometric data for humanoid character builder.

Based on de Leva (1996) "Adjustments to Zatsiorsky-Seluyanov's
segment inertia parameters" and other biomechanics literature.

This module provides anthropometric ratios for computing:
- Segment masses from total body mass
- Segment lengths from total height
- Segment dimensions (width, depth)
- Center of mass locations
- Gyration radii for inertia estimation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SegmentAnthropometry:
    """Anthropometric data for a single body segment."""

    # Mass as fraction of total body mass
    mass_ratio: float

    # Length as fraction of total height (standing height)
    length_ratio: float

    # Center of mass location as fraction of segment length
    # (from proximal end)
    com_proximal_ratio: float

    # Gyration radii as fractions of segment length
    # (for inertia calculation: I = m * k^2 * L^2)
    gyration_sagittal: float  # About mediolateral axis
    gyration_transverse: float  # About anteroposterior axis
    gyration_longitudinal: float  # About longitudinal axis

    # Width/depth as fractions of segment length (approximate)
    width_ratio: float = 0.2
    depth_ratio: float = 0.15


@dataclass
class AnthropometryData:
    """
    Complete anthropometric dataset for humanoid model.

    Contains data for male and female models that can be
    interpolated based on gender factor.
    """

    male: dict[str, SegmentAnthropometry]
    female: dict[str, SegmentAnthropometry]

    def get_segment_data(
        self, segment_name: str, gender_factor: float = 0.5
    ) -> SegmentAnthropometry:
        """
        Get interpolated segment data.

        Args:
            segment_name: Name of the body segment
            gender_factor: 0.0 = female, 1.0 = male, 0.5 = neutral

        Returns:
            Interpolated SegmentAnthropometry
        """
        male_data = self.male.get(segment_name)
        female_data = self.female.get(segment_name)

        if male_data is None or female_data is None:
            # Fall back to neutral data
            return male_data or female_data or _get_default_segment()

        # Linear interpolation
        return SegmentAnthropometry(
            mass_ratio=_lerp(female_data.mass_ratio, male_data.mass_ratio, gender_factor),
            length_ratio=_lerp(
                female_data.length_ratio, male_data.length_ratio, gender_factor
            ),
            com_proximal_ratio=_lerp(
                female_data.com_proximal_ratio,
                male_data.com_proximal_ratio,
                gender_factor,
            ),
            gyration_sagittal=_lerp(
                female_data.gyration_sagittal,
                male_data.gyration_sagittal,
                gender_factor,
            ),
            gyration_transverse=_lerp(
                female_data.gyration_transverse,
                male_data.gyration_transverse,
                gender_factor,
            ),
            gyration_longitudinal=_lerp(
                female_data.gyration_longitudinal,
                male_data.gyration_longitudinal,
                gender_factor,
            ),
            width_ratio=_lerp(
                female_data.width_ratio, male_data.width_ratio, gender_factor
            ),
            depth_ratio=_lerp(
                female_data.depth_ratio, male_data.depth_ratio, gender_factor
            ),
        )


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def _get_default_segment() -> SegmentAnthropometry:
    """Get default segment data for unknown segments."""
    return SegmentAnthropometry(
        mass_ratio=0.01,
        length_ratio=0.05,
        com_proximal_ratio=0.5,
        gyration_sagittal=0.3,
        gyration_transverse=0.3,
        gyration_longitudinal=0.15,
    )


# =============================================================================
# de Leva (1996) Anthropometric Data
# =============================================================================

# Male segment data
_MALE_SEGMENTS = {
    "head": SegmentAnthropometry(
        mass_ratio=0.0694,
        length_ratio=0.1395,
        com_proximal_ratio=0.5976,
        gyration_sagittal=0.362,
        gyration_transverse=0.376,
        gyration_longitudinal=0.312,
        width_ratio=0.5,
        depth_ratio=0.6,
    ),
    "neck": SegmentAnthropometry(
        mass_ratio=0.0240,
        length_ratio=0.052,
        com_proximal_ratio=0.50,
        gyration_sagittal=0.38,
        gyration_transverse=0.38,
        gyration_longitudinal=0.20,
        width_ratio=0.6,
        depth_ratio=0.5,
    ),
    "thorax": SegmentAnthropometry(
        mass_ratio=0.2160,
        length_ratio=0.170,
        com_proximal_ratio=0.4486,
        gyration_sagittal=0.372,
        gyration_transverse=0.347,
        gyration_longitudinal=0.191,
        width_ratio=0.8,
        depth_ratio=0.5,
    ),
    "lumbar": SegmentAnthropometry(
        mass_ratio=0.1390,
        length_ratio=0.108,
        com_proximal_ratio=0.45,
        gyration_sagittal=0.35,
        gyration_transverse=0.33,
        gyration_longitudinal=0.18,
        width_ratio=0.7,
        depth_ratio=0.5,
    ),
    "pelvis": SegmentAnthropometry(
        mass_ratio=0.1117,
        length_ratio=0.078,
        com_proximal_ratio=0.61,
        gyration_sagittal=0.615,
        gyration_transverse=0.551,
        gyration_longitudinal=0.587,
        width_ratio=1.5,
        depth_ratio=0.6,
    ),
    "upper_arm": SegmentAnthropometry(
        mass_ratio=0.0271,
        length_ratio=0.186,
        com_proximal_ratio=0.5772,
        gyration_sagittal=0.285,
        gyration_transverse=0.269,
        gyration_longitudinal=0.158,
        width_ratio=0.18,
        depth_ratio=0.18,
    ),
    "forearm": SegmentAnthropometry(
        mass_ratio=0.0162,
        length_ratio=0.146,
        com_proximal_ratio=0.4574,
        gyration_sagittal=0.276,
        gyration_transverse=0.265,
        gyration_longitudinal=0.121,
        width_ratio=0.16,
        depth_ratio=0.14,
    ),
    "hand": SegmentAnthropometry(
        mass_ratio=0.0061,
        length_ratio=0.108,
        com_proximal_ratio=0.7900,
        gyration_sagittal=0.628,
        gyration_transverse=0.513,
        gyration_longitudinal=0.401,
        width_ratio=0.5,
        depth_ratio=0.15,
    ),
    "thigh": SegmentAnthropometry(
        mass_ratio=0.1416,
        length_ratio=0.245,
        com_proximal_ratio=0.4095,
        gyration_sagittal=0.329,
        gyration_transverse=0.329,
        gyration_longitudinal=0.149,
        width_ratio=0.25,
        depth_ratio=0.22,
    ),
    "shin": SegmentAnthropometry(
        mass_ratio=0.0433,
        length_ratio=0.246,
        com_proximal_ratio=0.4459,
        gyration_sagittal=0.255,
        gyration_transverse=0.249,
        gyration_longitudinal=0.103,
        width_ratio=0.16,
        depth_ratio=0.14,
    ),
    "foot": SegmentAnthropometry(
        mass_ratio=0.0137,
        length_ratio=0.152,
        com_proximal_ratio=0.4415,
        gyration_sagittal=0.257,
        gyration_transverse=0.245,
        gyration_longitudinal=0.124,
        width_ratio=0.35,
        depth_ratio=0.25,
    ),
    # Virtual segments (minimal mass)
    "shoulder": SegmentAnthropometry(
        mass_ratio=0.0050,
        length_ratio=0.040,
        com_proximal_ratio=0.50,
        gyration_sagittal=0.30,
        gyration_transverse=0.30,
        gyration_longitudinal=0.15,
        width_ratio=0.5,
        depth_ratio=0.4,
    ),
    "hip": SegmentAnthropometry(
        mass_ratio=0.0010,
        length_ratio=0.000,
        com_proximal_ratio=0.50,
        gyration_sagittal=0.30,
        gyration_transverse=0.30,
        gyration_longitudinal=0.30,
        width_ratio=0.5,
        depth_ratio=0.5,
    ),
}

# Female segment data (adjusted from de Leva)
_FEMALE_SEGMENTS = {
    "head": SegmentAnthropometry(
        mass_ratio=0.0668,
        length_ratio=0.1395,
        com_proximal_ratio=0.5894,
        gyration_sagittal=0.330,
        gyration_transverse=0.359,
        gyration_longitudinal=0.318,
        width_ratio=0.48,
        depth_ratio=0.55,
    ),
    "neck": SegmentAnthropometry(
        mass_ratio=0.0220,
        length_ratio=0.052,
        com_proximal_ratio=0.50,
        gyration_sagittal=0.36,
        gyration_transverse=0.36,
        gyration_longitudinal=0.18,
        width_ratio=0.55,
        depth_ratio=0.45,
    ),
    "thorax": SegmentAnthropometry(
        mass_ratio=0.2040,
        length_ratio=0.160,
        com_proximal_ratio=0.4430,
        gyration_sagittal=0.357,
        gyration_transverse=0.339,
        gyration_longitudinal=0.186,
        width_ratio=0.75,
        depth_ratio=0.48,
    ),
    "lumbar": SegmentAnthropometry(
        mass_ratio=0.1380,
        length_ratio=0.100,
        com_proximal_ratio=0.46,
        gyration_sagittal=0.34,
        gyration_transverse=0.32,
        gyration_longitudinal=0.17,
        width_ratio=0.65,
        depth_ratio=0.48,
    ),
    "pelvis": SegmentAnthropometry(
        mass_ratio=0.1220,
        length_ratio=0.078,
        com_proximal_ratio=0.63,
        gyration_sagittal=0.640,
        gyration_transverse=0.580,
        gyration_longitudinal=0.620,
        width_ratio=1.6,
        depth_ratio=0.62,
    ),
    "upper_arm": SegmentAnthropometry(
        mass_ratio=0.0255,
        length_ratio=0.173,
        com_proximal_ratio=0.5754,
        gyration_sagittal=0.278,
        gyration_transverse=0.260,
        gyration_longitudinal=0.148,
        width_ratio=0.16,
        depth_ratio=0.16,
    ),
    "forearm": SegmentAnthropometry(
        mass_ratio=0.0138,
        length_ratio=0.138,
        com_proximal_ratio=0.4559,
        gyration_sagittal=0.261,
        gyration_transverse=0.257,
        gyration_longitudinal=0.094,
        width_ratio=0.14,
        depth_ratio=0.12,
    ),
    "hand": SegmentAnthropometry(
        mass_ratio=0.0056,
        length_ratio=0.100,
        com_proximal_ratio=0.7474,
        gyration_sagittal=0.531,
        gyration_transverse=0.454,
        gyration_longitudinal=0.335,
        width_ratio=0.45,
        depth_ratio=0.14,
    ),
    "thigh": SegmentAnthropometry(
        mass_ratio=0.1478,
        length_ratio=0.249,
        com_proximal_ratio=0.3612,
        gyration_sagittal=0.364,
        gyration_transverse=0.369,
        gyration_longitudinal=0.162,
        width_ratio=0.28,
        depth_ratio=0.24,
    ),
    "shin": SegmentAnthropometry(
        mass_ratio=0.0481,
        length_ratio=0.257,
        com_proximal_ratio=0.4352,
        gyration_sagittal=0.271,
        gyration_transverse=0.267,
        gyration_longitudinal=0.093,
        width_ratio=0.14,
        depth_ratio=0.12,
    ),
    "foot": SegmentAnthropometry(
        mass_ratio=0.0129,
        length_ratio=0.143,
        com_proximal_ratio=0.4014,
        gyration_sagittal=0.299,
        gyration_transverse=0.279,
        gyration_longitudinal=0.139,
        width_ratio=0.32,
        depth_ratio=0.22,
    ),
    # Virtual segments
    "shoulder": SegmentAnthropometry(
        mass_ratio=0.0045,
        length_ratio=0.035,
        com_proximal_ratio=0.50,
        gyration_sagittal=0.28,
        gyration_transverse=0.28,
        gyration_longitudinal=0.14,
        width_ratio=0.45,
        depth_ratio=0.38,
    ),
    "hip": SegmentAnthropometry(
        mass_ratio=0.0010,
        length_ratio=0.000,
        com_proximal_ratio=0.50,
        gyration_sagittal=0.30,
        gyration_transverse=0.30,
        gyration_longitudinal=0.30,
        width_ratio=0.5,
        depth_ratio=0.5,
    ),
}

# Create the global anthropometry data
DE_LEVA_DATA = AnthropometryData(male=_MALE_SEGMENTS, female=_FEMALE_SEGMENTS)

# Mapping from full segment names to anthropometry keys
_SEGMENT_NAME_MAP = {
    "head": "head",
    "neck": "neck",
    "thorax": "thorax",
    "lumbar": "lumbar",
    "pelvis": "pelvis",
    "left_shoulder": "shoulder",
    "right_shoulder": "shoulder",
    "left_upper_arm": "upper_arm",
    "right_upper_arm": "upper_arm",
    "left_forearm": "forearm",
    "right_forearm": "forearm",
    "left_hand": "hand",
    "right_hand": "hand",
    "left_hip": "hip",
    "right_hip": "hip",
    "left_thigh": "thigh",
    "right_thigh": "thigh",
    "left_shin": "shin",
    "right_shin": "shin",
    "left_foot": "foot",
    "right_foot": "foot",
}


def get_anthropometry_key(segment_name: str) -> str:
    """Map segment name to anthropometry data key."""
    return _SEGMENT_NAME_MAP.get(segment_name, segment_name)


def get_segment_mass_ratio(
    segment_name: str, gender_factor: float = 0.5
) -> float:
    """
    Get segment mass as fraction of total body mass.

    Args:
        segment_name: Name of the body segment
        gender_factor: 0.0 = female, 1.0 = male

    Returns:
        Mass ratio (0.0 - 1.0)
    """
    key = get_anthropometry_key(segment_name)
    data = DE_LEVA_DATA.get_segment_data(key, gender_factor)
    return data.mass_ratio


def get_segment_length_ratio(
    segment_name: str, gender_factor: float = 0.5
) -> float:
    """
    Get segment length as fraction of total height.

    Args:
        segment_name: Name of the body segment
        gender_factor: 0.0 = female, 1.0 = male

    Returns:
        Length ratio (0.0 - 1.0)
    """
    key = get_anthropometry_key(segment_name)
    data = DE_LEVA_DATA.get_segment_data(key, gender_factor)
    return data.length_ratio


def estimate_segment_masses(
    total_mass_kg: float, gender_factor: float = 0.5
) -> dict[str, float]:
    """
    Estimate mass for all segments based on total body mass.

    Args:
        total_mass_kg: Total body mass in kg
        gender_factor: 0.0 = female, 1.0 = male

    Returns:
        Dict mapping segment name to mass in kg
    """
    masses = {}
    for segment_name in _SEGMENT_NAME_MAP.keys():
        ratio = get_segment_mass_ratio(segment_name, gender_factor)
        masses[segment_name] = total_mass_kg * ratio
    return masses


def estimate_segment_dimensions(
    total_height_m: float, gender_factor: float = 0.5
) -> dict[str, dict[str, float]]:
    """
    Estimate dimensions for all segments based on total height.

    Args:
        total_height_m: Total standing height in meters
        gender_factor: 0.0 = female, 1.0 = male

    Returns:
        Dict mapping segment name to dimensions dict with
        'length', 'width', 'depth' keys
    """
    dimensions = {}
    for segment_name in _SEGMENT_NAME_MAP.keys():
        key = get_anthropometry_key(segment_name)
        data = DE_LEVA_DATA.get_segment_data(key, gender_factor)

        length = total_height_m * data.length_ratio
        dimensions[segment_name] = {
            "length": length,
            "width": length * data.width_ratio,
            "depth": length * data.depth_ratio,
        }
    return dimensions


def estimate_segment_inertia_from_gyration(
    segment_name: str,
    mass_kg: float,
    length_m: float,
    gender_factor: float = 0.5,
) -> dict[str, float]:
    """
    Estimate segment inertia using gyration radii.

    Uses I = m * k^2 * L^2 where k is the radius of gyration
    as a fraction of segment length.

    Args:
        segment_name: Name of the body segment
        mass_kg: Segment mass in kg
        length_m: Segment length in meters
        gender_factor: 0.0 = female, 1.0 = male

    Returns:
        Dict with ixx, iyy, izz (about principal axes at COM)
    """
    key = get_anthropometry_key(segment_name)
    data = DE_LEVA_DATA.get_segment_data(key, gender_factor)

    # Inertia about each principal axis
    # Sagittal (mediolateral): rotation in sagittal plane
    # Transverse (anteroposterior): rotation in transverse plane
    # Longitudinal: rotation about long axis

    ixx = mass_kg * (data.gyration_sagittal * length_m) ** 2
    iyy = mass_kg * (data.gyration_transverse * length_m) ** 2
    izz = mass_kg * (data.gyration_longitudinal * length_m) ** 2

    return {"ixx": ixx, "iyy": iyy, "izz": izz, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}


def get_com_location(
    segment_name: str, length_m: float, gender_factor: float = 0.5
) -> tuple[float, float, float]:
    """
    Get center of mass location relative to segment frame origin.

    Assumes segment is oriented along Z axis with proximal end at origin.

    Args:
        segment_name: Name of the body segment
        length_m: Segment length in meters
        gender_factor: 0.0 = female, 1.0 = male

    Returns:
        (x, y, z) COM position in meters
    """
    key = get_anthropometry_key(segment_name)
    data = DE_LEVA_DATA.get_segment_data(key, gender_factor)

    # COM is located along the longitudinal axis
    com_z = length_m * data.com_proximal_ratio
    return (0.0, 0.0, com_z)
