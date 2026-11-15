"""HRG-GB1 versioned grip authoring spec (issue #15123).

Frames and thumb azimuth sign convention
-----------------------------------------
Thumb azimuth is measured from the top of the grip. Positive is toward the
target side for the trail hand and away from the target for the lead hand,
mirrored for left-handed golfers via :meth:`GripSpec.resolved_thumb_azimuth_deg`.

Force and contact parameters are never literals on :class:`GripSpec`; they are
resolved through :class:`GripForceProvider` and :class:`ContactParameterProvider`
registries (default ``unspecified`` reproduces today's unlimited limits and pad
layout preload).

This package is engine-free. Legacy consumers continue to use
:class:`~src.shared.python.model_appearance.grip_pose.GripPose`; call
:meth:`GripSpec.to_grip_pose` for the compatibility projection.
"""

from src.shared.python.grip_spec.providers import (
    ContactParameterProvider,
    GripForceProvider,
    register_contact_parameter_provider,
    register_grip_force_provider,
    resolve_contact_parameters,
    resolve_grip_force_limits,
)
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize

__all__ = [
    "ContactParameterProvider",
    "GripForceProvider",
    "GripSpec",
    "HandGripSpec",
    "HandSize",
    "register_contact_parameter_provider",
    "register_grip_force_provider",
    "resolve_contact_parameters",
    "resolve_grip_force_limits",
]
