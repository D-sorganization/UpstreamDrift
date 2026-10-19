"""Engine-agnostic grip interface description (issue #11739, OSV-7)."""

from src.shared.python.grip_contact.bushing_law import (
    BushingState,
    BushingWrench,
    bushing_wrench,
)
from src.shared.python.grip_contact.closure_series import (
    GripClosureSeries,
    closure_series_from_residuals,
)
from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.contact_topology import (
    GRIP,
    ContactPair,
    ContactSite,
    ContactTopology,
    GripStyle,
    HandAnatomy,
    contact_topology,
)
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
from src.shared.python.grip_contact.interface import (
    R_APPEARANCE_TO_SHARED,
    R_SHARED_TO_APPEARANCE,
    GripFrame,
    GripInterface,
    appearance_to_shared_grip_frame,
    shared_to_appearance_grip_frame,
)
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    ContactSphere,
    PairWrench,
    capsule_capsule,
    sphere_capsule,
)
from src.shared.python.grip_contact.parameters import (
    BushingParameters,
    ContactMaterial,
    default_bushing,
)
from src.shared.python.grip_contact.prescribed_motion import (
    CoordinateSpline,
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.swing_input import (
    FIXTURE_DT_S,
    CoordinateSwing,
    load_coordinate_swing,
    map_coordinates,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    TopologySideWrench,
    build_topology_model,
    map_site_to_hand_frame,
    topology_side_wrench,
)
from src.shared.python.grip_contact.trajectory_conditioning import (
    ConditioningReport,
    condition_trajectory,
    detect_ik_outliers,
    first_discontinuity_time,
    unwrap_angular,
)

__all__ = [
    "BushingState",
    "BushingWrench",
    "ContactCapsule",
    "ContactPair",
    "ContactSite",
    "ContactSphere",
    "ContactTopology",
    "CoordinateSpline",
    "GRIP",
    "GripStyle",
    "HandAnatomy",
    "PairWrench",
    "RigidBodyState",
    "bushing_wrench",
    "capsule_capsule",
    "contact_topology",
    "hand_frame_states",
    "sphere_capsule",
    "GripClosureSeries",
    "closure_series_from_residuals",
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
    "R_APPEARANCE_TO_SHARED",
    "R_SHARED_TO_APPEARANCE",
    "appearance_to_shared_grip_frame",
    "shared_to_appearance_grip_frame",
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
    "TopologyContactModel",
    "TopologySideWrench",
    "build_topology_model",
    "map_site_to_hand_frame",
    "topology_side_wrench",
]
