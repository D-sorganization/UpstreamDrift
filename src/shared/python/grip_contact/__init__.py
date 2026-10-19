"""Engine-agnostic grip interface description (issue #11739, OSV-7)."""

from src.shared.python.grip_contact.bushing_law import (
    BushingState,
    BushingWrench,
    bushing_wrench,
)
from src.shared.python.grip_contact.capability_inventory import (
    DEFAULT_INVENTORY_PATH,
    ExcludedModel,
    GripCapabilityInventory,
    GripFeatureCapability,
    ModelGripCapability,
    load_grip_capability_inventory,
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
from src.shared.python.grip_contact.interface import GripFrame, GripInterface
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
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    GripStepReport,
    HandClubBinding,
    ModelGripAdapter,
    SiteBinding,
    apply_report,
    bind_rigid_hand_sites,
    evaluate_contacts,
    site_transform_in_hand_frame,
    validate_binding,
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
    "BodyStates",
    "DEFAULT_INVENTORY_PATH",
    "ExcludedModel",
    "GripCapabilityInventory",
    "GripFeatureCapability",
    "ModelGripCapability",
    "load_grip_capability_inventory",
    "BushingState",
    "BushingWrench",
    "ContactCapsule",
    "ContactPair",
    "ContactSite",
    "ContactSphere",
    "ContactTopology",
    "CoordinateSpline",
    "GRIP",
    "GripCapabilities",
    "GripStepReport",
    "GripStyle",
    "HandAnatomy",
    "HandClubBinding",
    "ModelGripAdapter",
    "PairWrench",
    "RigidBodyState",
    "SiteBinding",
    "apply_report",
    "bind_rigid_hand_sites",
    "bushing_wrench",
    "capsule_capsule",
    "evaluate_contacts",
    "contact_topology",
    "hand_frame_states",
    "site_transform_in_hand_frame",
    "sphere_capsule",
    "validate_binding",
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
