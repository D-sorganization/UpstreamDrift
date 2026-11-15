"""Project HRG-GB1 authoring specs onto legacy :class:`GripPose`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from src.shared.python.contracts import require
from src.shared.python.grip_contact.grip_force_limit import GripForceLimit
from src.shared.python.model_appearance.grip_pose import GripPose

if TYPE_CHECKING:
    from src.shared.python.grip_spec.types import GripSpec


def grip_pose_from_authoring_spec(
    spec: GripSpec,
    *,
    force_limits: Mapping[str, GripForceLimit],
) -> GripPose:
    """Build a :class:`GripPose` from ``spec`` without changing legacy defaults."""
    require(
        "lead" in force_limits and "trail" in force_limits, "force_limits incomplete"
    )
    lead_v = spec.lead.v_azimuth_deg
    trail_v = spec.trail.v_azimuth_deg
    return GripPose(
        style=spec.style,
        trigger_finger=spec.trigger_finger,
        lead_below_butt_m=spec.lead_below_butt_m,
        hand_spacing_m=spec.hand_spacing_m,
        lead_v_azimuth_deg=lead_v,
        trail_v_offset_deg=trail_v - lead_v,
        lead_thumb_offset_deg=abs(spec.resolved_thumb_azimuth_deg("lead")),
        palm_diagonal_deg=spec.lead.club_cross_angle_deg,
        handedness=spec.handedness,
        lead_grip_force_limit_n=force_limits["lead"],
        trail_grip_force_limit_n=force_limits["trail"],
    )


__all__ = ["grip_pose_from_authoring_spec"]
