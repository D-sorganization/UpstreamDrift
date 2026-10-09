"""Engine-agnostic grip interface description (issue #11739, OSV-7)."""

from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.damping import (
    DEFAULT_DAMPING_RATIO,
    design_damping,
    modal_damping,
)
from src.shared.python.grip_contact.couple_check import (
    DEFAULT_COUPLE_NOISE_FLOOR_NM,
    ClubKinematics,
    CoupleConsistency,
    couple_consistency,
    peak_squeeze_n,
    required_hand_moment_nm,
)
from src.shared.python.grip_contact.force_decomposition import (
    ForceDecomposition,
    decompose_hand_forces,
)
from src.shared.python.grip_contact.interface import GripFrame, GripInterface
from src.shared.python.grip_contact.parameters import (
    BushingParameters,
    ContactMaterial,
    default_bushing,
)
from src.shared.python.grip_contact.swing_input import (
    FIXTURE_DT_S,
    CoordinateSwing,
    load_coordinate_swing,
    map_coordinates,
)
from src.shared.python.grip_contact.trajectory_conditioning import (
    ConditioningReport,
    condition_trajectory,
    detect_ik_outliers,
    first_discontinuity_time,
    unwrap_angular,
)

__all__ = [
    "DEFAULT_COUPLE_NOISE_FLOOR_NM",
    "ClubKinematics",
    "CoupleConsistency",
    "couple_consistency",
    "peak_squeeze_n",
    "required_hand_moment_nm",
    "DEFAULT_DAMPING_RATIO",
    "FIXTURE_DT_S",
    "BushingParameters",
    "ClubDynamics",
    "ConditioningReport",
    "ContactMaterial",
    "CoordinateSwing",
    "ForceDecomposition",
    "GripFrame",
    "GripInterface",
    "condition_trajectory",
    "decompose_hand_forces",
    "default_bushing",
    "design_damping",
    "detect_ik_outliers",
    "first_discontinuity_time",
    "load_coordinate_swing",
    "map_coordinates",
    "modal_damping",
    "unwrap_angular",
]
