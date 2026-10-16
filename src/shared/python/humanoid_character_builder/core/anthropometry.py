# TRACKED_TASK: see #2310 — architecture debt extraction schedule

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

from __future__ import annotations  # noqa: E402, F404

from dataclasses import dataclass  # noqa: E402

from src.shared.python.body_segment_parameters.de_leva_1996 import (  # noqa: E402
    Sex,
    table4_row,
)
from src.shared.python.humanoid_character_builder.contracts import (
    precondition,  # noqa: E402
)


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
        if segment_name is None:
            raise ValueError("segment_name must be provided")
        male_data = self.male.get(segment_name)
        female_data = self.female.get(segment_name)

        if male_data is None or female_data is None:
            # Fall back to neutral data
            return male_data or female_data or _get_default_segment()

        # Linear interpolation
        return SegmentAnthropometry(
            mass_ratio=_lerp(
                female_data.mass_ratio, male_data.mass_ratio, gender_factor
            ),
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

#: Table 4 row (de Leva 1996) behind each model segment. Mass, CM position and
#: radii come from that one row; CM is measured from the row's origin landmark
#: (cranial for head and trunk parts, proximal for limbs). The head spans
#: vertex to cervicale, so it is the head-and-neck row; the thorax, lumbar and
#: pelvis are the upper, middle and lower trunk measured from the cervicale; the
#: hand reaches the third fingertip; the shin ends at the ankle joint centre.
DE_LEVA_SOURCE_ROWS: dict[str, str] = {
    "head": "head_vert_cerv",
    "thorax": "upt_cerv_xyph",
    "lumbar": "mpt_xyph_omph",
    "pelvis": "lpt_omph_midh",
    "upper_arm": "upper_arm_sjc_ejc",
    "forearm": "forearm_ejc_wjc",
    "hand": "hand_wjc_dac3",
    "thigh": "thigh_hjc_kjc",
    "shin": "shank_kjc_ajc",
    "foot": "foot_heel_ttip",
}

#: Model-only segments de Leva does not list: ``(parent, mass ratio each)``.
#: Their mass is carved out of the parent's row so the body still sums to one.
VIRTUAL_SEGMENTS: dict[str, tuple[str, float]] = {
    "neck": ("head", 0.005),
    "shoulder": ("thorax", 0.005),
    "hip": ("pelvis", 0.001),
}

#: Placeholder CM and radii of the virtual segments (not de Leva values).
_VIRTUAL_INERTIA: dict[str, tuple[float, float, float, float]] = {
    "neck": (0.50, 0.38, 0.38, 0.20),
    "shoulder": (0.50, 0.30, 0.30, 0.15),
    "hip": (0.50, 0.30, 0.30, 0.30),
}

#: Model geometry, not de Leva: (length/stature, width/length, depth/length).
#: Lengths are the builder's skeleton proportions; the de Leva CM and radii
#: fractions are applied to them.
_GEOMETRY: dict[str, dict[str, tuple[float, float, float]]] = {
    "male": {
        "head": (0.1395, 0.5, 0.6),
        "neck": (0.052, 0.6, 0.5),
        "thorax": (0.170, 0.8, 0.5),
        "lumbar": (0.108, 0.7, 0.5),
        "pelvis": (0.078, 1.5, 0.6),
        "upper_arm": (0.186, 0.18, 0.18),
        "forearm": (0.146, 0.16, 0.14),
        "hand": (0.108, 0.5, 0.15),
        "thigh": (0.245, 0.25, 0.22),
        "shin": (0.246, 0.16, 0.14),
        "foot": (0.152, 0.35, 0.25),
        "shoulder": (0.040, 0.5, 0.4),
        "hip": (0.02, 0.5, 0.5),
    },
    "female": {
        "head": (0.1395, 0.48, 0.55),
        "neck": (0.052, 0.55, 0.45),
        "thorax": (0.16, 0.75, 0.48),
        "lumbar": (0.10, 0.65, 0.48),
        "pelvis": (0.078, 1.6, 0.62),
        "upper_arm": (0.173, 0.16, 0.16),
        "forearm": (0.138, 0.14, 0.12),
        "hand": (0.10, 0.45, 0.14),
        "thigh": (0.249, 0.28, 0.24),
        "shin": (0.257, 0.14, 0.12),
        "foot": (0.143, 0.32, 0.22),
        "shoulder": (0.035, 0.45, 0.38),
        "hip": (0.02, 0.5, 0.5),
    },
}

#: Whole-body mass closure tolerance: Table 4 prints two decimals of a percent.
_MASS_SUM_TOL = 2e-4


def virtual_segment_count(key: str) -> int:
    """Number of model segments that use anthropometry ``key`` (2 for bilateral)."""
    return sum(1 for value in _SEGMENT_NAME_MAP.values() if value == key)


def _carved_mass(parent: str) -> float:
    return sum(
        share * virtual_segment_count(virtual)
        for virtual, (owner, share) in VIRTUAL_SEGMENTS.items()
        if owner == parent
    )


def _build_segments(sex: Sex) -> dict[str, SegmentAnthropometry]:
    """Model segments for ``sex``, each a whole Table 4 row on model geometry."""
    geometry = _GEOMETRY[sex]
    segments: dict[str, SegmentAnthropometry] = {}
    for key, row_id in DE_LEVA_SOURCE_ROWS.items():
        row = table4_row(row_id, sex)
        length, width, depth = geometry[key]
        segments[key] = SegmentAnthropometry(
            mass_ratio=row.mass_fraction - _carved_mass(key),
            length_ratio=length,
            com_proximal_ratio=row.com_fraction,
            gyration_sagittal=row.radii[0],
            gyration_transverse=row.radii[1],
            gyration_longitudinal=row.radii[2],
            width_ratio=width,
            depth_ratio=depth,
        )
    for key, (_parent, share) in VIRTUAL_SEGMENTS.items():
        length, width, depth = geometry[key]
        com, sagittal, transverse, longitudinal = _VIRTUAL_INERTIA[key]
        segments[key] = SegmentAnthropometry(
            mass_ratio=share,
            length_ratio=length,
            com_proximal_ratio=com,
            gyration_sagittal=sagittal,
            gyration_transverse=transverse,
            gyration_longitudinal=longitudinal,
            width_ratio=width,
            depth_ratio=depth,
        )
    check_mass_closure(segments, sex)
    return segments


def check_mass_closure(segments: dict[str, SegmentAnthropometry], label: str) -> None:
    """Raise ``ValueError`` unless the model's segment masses sum to the body.

    Every named model segment counts once, so bilateral keys count twice.
    """
    total = sum(segments[key].mass_ratio for key in _SEGMENT_NAME_MAP.values())
    if abs(total - 1.0) > _MASS_SUM_TOL:
        raise ValueError(
            f"{label} segment mass ratios sum to {total:.5f}, not 1 "
            f"(tolerance {_MASS_SUM_TOL})"
        )


_MALE_SEGMENTS = _build_segments("male")
_FEMALE_SEGMENTS = _build_segments("female")

# Create the global anthropometry data
DE_LEVA_DATA = AnthropometryData(male=_MALE_SEGMENTS, female=_FEMALE_SEGMENTS)


def get_anthropometry_key(segment_name: str) -> str:
    """Map segment name to anthropometry data key."""
    return _SEGMENT_NAME_MAP.get(segment_name, segment_name)


def get_segment_mass_ratio(segment_name: str, gender_factor: float = 0.5) -> float:
    """
    Get segment mass as fraction of total body mass.

    Args:
        segment_name: Name of the body segment
        gender_factor: 0.0 = female, 1.0 = male

    Returns:
        Mass ratio (0.0 - 1.0)
    """
    if segment_name is None:
        raise ValueError("segment_name must be provided")
    key = get_anthropometry_key(segment_name)
    data = DE_LEVA_DATA.get_segment_data(key, gender_factor)
    return data.mass_ratio


def get_segment_length_ratio(segment_name: str, gender_factor: float = 0.5) -> float:
    """
    Get segment length as fraction of total height.

    Args:
        segment_name: Name of the body segment
        gender_factor: 0.0 = female, 1.0 = male

    Returns:
        Length ratio (0.0 - 1.0)
    """
    if segment_name is None:
        raise ValueError("segment_name must be provided")
    key = get_anthropometry_key(segment_name)
    data = DE_LEVA_DATA.get_segment_data(key, gender_factor)
    return data.length_ratio


def get_urdf_mass_ratios(gender_factor: float = 1.0) -> dict[str, float]:
    """Get mass ratios for URDF builder, including composite segments."""
    return {
        "pelvis": get_segment_mass_ratio("pelvis", gender_factor),
        "lumbar": get_segment_mass_ratio("lumbar", gender_factor),
        "thorax": get_segment_mass_ratio("thorax", gender_factor),
        "torso": get_segment_mass_ratio("lumbar", gender_factor)
        + get_segment_mass_ratio("thorax", gender_factor),
        "neck": get_segment_mass_ratio("neck", gender_factor),
        "head": get_segment_mass_ratio("head", gender_factor),
        "upper_arm": get_segment_mass_ratio("upper_arm", gender_factor),
        "forearm": get_segment_mass_ratio("forearm", gender_factor),
        "hand": get_segment_mass_ratio("hand", gender_factor),
        "thigh": get_segment_mass_ratio("thigh", gender_factor),
        "shin": get_segment_mass_ratio("shin", gender_factor),
        "foot": get_segment_mass_ratio("foot", gender_factor),
    }


@precondition(lambda total_mass_kg: total_mass_kg > 0, "Total mass must be positive")
@precondition(
    # Accept the leading argument too: with ``f(75.0)`` the contracts layer calls
    # the condition with the raw positional args, so a bare ``gender_factor``
    # parameter would receive the mass (#12706).
    lambda total_mass_kg, gender_factor=0.5: 0.0 <= gender_factor <= 1.0,
    "Gender factor must be between 0 and 1",
)
def estimate_segment_masses(
    total_mass_kg: float, gender_factor: float = 0.5
) -> dict[str, float]:
    """
    Estimate mass for all segments based on total body mass.

    Args:
        total_mass_kg: Total body mass in kg
        gender_factor: 0.0 = female, 1.0 = male

    Returns:
        Dict mapping segment name to mass in kg. Postcondition: the masses are
        the table fractions of total_mass_kg and sum to it within the
        table's printed precision; a table that does not close raises
        ValueError rather than being rescaled (#13010).
    """
    if total_mass_kg is None:
        raise ValueError("total_mass_kg must be provided")
    for table, label in ((DE_LEVA_DATA.male, "male"), (DE_LEVA_DATA.female, "female")):
        check_mass_closure(table, label)
    return {
        segment_name: total_mass_kg
        * get_segment_mass_ratio(segment_name, gender_factor)
        for segment_name in _SEGMENT_NAME_MAP
    }


@precondition(
    lambda total_height_m: total_height_m > 0, "Total height must be positive"
)
@precondition(
    # Accept the leading argument too: with ``f(1.8)`` the contracts layer calls
    # the condition with the raw positional args, so a bare ``gender_factor``
    # parameter would receive the height (#12706).
    lambda total_height_m, gender_factor=0.5: 0.0 <= gender_factor <= 1.0,
    "Gender factor must be between 0 and 1",
)
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
    if total_height_m is None:
        raise ValueError("total_height_m must be provided")
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


@precondition(lambda mass_kg: mass_kg > 0, "Mass must be positive")
@precondition(lambda length_m: length_m > 0, "Length must be positive")
@precondition(
    lambda gender_factor: 0.0 <= gender_factor <= 1.0,
    "Gender factor must be between 0 and 1",
)
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
    if segment_name is None:
        raise ValueError("segment_name must be provided")
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


@precondition(lambda length_m: length_m > 0, "Length must be positive")
@precondition(
    lambda gender_factor: 0.0 <= gender_factor <= 1.0,
    "Gender factor must be between 0 and 1",
)
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
    if segment_name is None:
        raise ValueError("segment_name must be provided")
    key = get_anthropometry_key(segment_name)
    data = DE_LEVA_DATA.get_segment_data(key, gender_factor)

    # COM is located along the longitudinal axis
    com_z = length_m * data.com_proximal_ratio
    return (0.0, 0.0, com_z)


# =============================================================================
# URDF Builder GUI Specific Ratios
# =============================================================================

# Composite mass ratios (torso = lumbar + thorax) derived from male canonical data
URDF_MASS_RATIOS: dict[str, float] = {
    "pelvis": _MALE_SEGMENTS["pelvis"].mass_ratio,
    "lumbar": _MALE_SEGMENTS["lumbar"].mass_ratio,
    "thorax": _MALE_SEGMENTS["thorax"].mass_ratio,
    "torso": _MALE_SEGMENTS["lumbar"].mass_ratio + _MALE_SEGMENTS["thorax"].mass_ratio,
    "neck": _MALE_SEGMENTS["neck"].mass_ratio,
    "head": _MALE_SEGMENTS["head"].mass_ratio,
    "upper_arm": _MALE_SEGMENTS["upper_arm"].mass_ratio,
    "forearm": _MALE_SEGMENTS["forearm"].mass_ratio,
    "hand": _MALE_SEGMENTS["hand"].mass_ratio,
    "thigh": _MALE_SEGMENTS["thigh"].mass_ratio,
    "shin": _MALE_SEGMENTS["shin"].mass_ratio,
    "foot": _MALE_SEGMENTS["foot"].mass_ratio,
}

# Height ratios using URDF builder measurement conventions.
# Some segment conventions differ from the simple humanoid builder ratios.
URDF_HEIGHT_RATIOS: dict[str, float] = {
    "pelvis": 0.078,
    "torso": 0.278,
    "head": 0.139,
    "thigh": 0.245,
    "shin": 0.246,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.058,
    "foot": 0.039,
}


def get_simple_mass_ratios(gender_factor: float = 0.5) -> dict[str, float]:
    """Get simple mass ratios for standard segments."""
    keys = [
        "head",
        "neck",
        "thorax",
        "lumbar",
        "pelvis",
        "upper_arm",
        "forearm",
        "hand",
        "thigh",
        "shin",
        "foot",
    ]
    return {k: DE_LEVA_DATA.get_segment_data(k, gender_factor).mass_ratio for k in keys}


def get_simple_length_ratios(gender_factor: float = 0.5) -> dict[str, float]:
    """Get simple length ratios for standard segments."""
    keys = [
        "head",
        "neck",
        "thorax",
        "lumbar",
        "pelvis",
        "upper_arm",
        "forearm",
        "hand",
        "thigh",
        "shin",
        "foot",
    ]
    return {
        k: DE_LEVA_DATA.get_segment_data(k, gender_factor).length_ratio for k in keys
    }
