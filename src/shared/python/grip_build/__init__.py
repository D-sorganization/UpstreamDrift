"""Engine-free procedural grip build (HRG-GB2, #15124)."""

from src.shared.python.grip_build.builder import (
    BUILDER_VERSION,
    FlatHandCurlBuilder,
    build_grip,
    build_lead_fingers,
    thumb_radial_azimuth_deg,
)
from src.shared.python.grip_build.full_builder import (
    BUILDER_VERSION as GB3_BUILDER_VERSION,
)
from src.shared.python.grip_build.receipt import (
    BUILD_RECEIPT_SCHEMA_VERSION,
    BuildReceipt,
    BuildStageRecord,
    BuildStatus,
    FullGripBuildResult,
    GripBuildResult,
)
from src.shared.python.grip_build.frozen_template import (
    FROZEN_TEMPLATE_SCHEMA_ID,
    FROZEN_TEMPLATE_SCHEMA_VERSION,
    TEMPLATE_BUILDER_VERSION,
    CapsulePose,
    FrozenGripTemplate,
    FrozenTemplateError,
    FrozenTemplateStampError,
    HandTemplate,
    build_frozen_template,
    load_frozen_template,
    verify_template_stamp,
    write_frozen_template,
)
from src.shared.python.grip_build.registry import (
    DEFAULT_BUILDER_NAME,
    get_contact_distance_model,
    get_grip_builder,
    register_contact_distance_model,
    register_grip_builder,
)

__all__ = [
    "BUILDER_VERSION",
    "BUILD_RECEIPT_SCHEMA_VERSION",
    "BuildReceipt",
    "BuildStageRecord",
    "BuildStatus",
    "FlatHandCurlBuilder",
    "FROZEN_TEMPLATE_SCHEMA_ID",
    "FROZEN_TEMPLATE_SCHEMA_VERSION",
    "TEMPLATE_BUILDER_VERSION",
    "CapsulePose",
    "FrozenGripTemplate",
    "FrozenTemplateError",
    "FrozenTemplateStampError",
    "HandTemplate",
    "build_frozen_template",
    "load_frozen_template",
    "verify_template_stamp",
    "write_frozen_template",
    "GripBuildResult",
    "GB3_BUILDER_VERSION",
    "FullGripBuildResult",
    "build_grip",
    "build_lead_fingers",
    "thumb_radial_azimuth_deg",
    "DEFAULT_BUILDER_NAME",
    "get_contact_distance_model",
    "get_grip_builder",
    "register_contact_distance_model",
    "register_grip_builder",
]
