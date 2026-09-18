"""What the lab's three-camera rig can and cannot measure (issue #9543).

Stated as data rather than prose so the qualification report renders it
beside the verdicts and a reader cannot mistake a quantity the rig cannot
resolve for one that was merely not measured yet. Read off the measured
constraints in ``docs/motion_capture/usb_camera_rig_bringup.md`` and the
ledger's rig specs (:mod:`bunkershot3d.vandv.roadmap`). No camera purchase and
no experiment is assumed here; a quantity marked unmeasurable stays
unavailable in the stroke record.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "THREE_CAMERA_RIG_CAPABILITY",
    "RigCapability",
    "rig_capability_markdown",
]


@dataclass(frozen=True, slots=True)
class RigCapability:
    """What one quantity the program needs can or cannot be measured on the rig.

    Attributes:
        quantity: The quantity.
        measurable: Whether the rig can resolve it to a usable uncertainty.
        reason: Why, in terms of the rig's measured constraints.
    """

    quantity: str
    measurable: bool
    reason: str


THREE_CAMERA_RIG_CAPABILITY: tuple[RigCapability, ...] = (
    RigCapability(
        "ball launch speed and direction",
        True,
        "three AR0234 global-shutter cameras at 1920x1200 and 60 fps over a "
        "calibrated volume resolve the ball's first 0.5 m of flight to a few "
        "frames; the sub-frame launch instant is not resolved, so the stroke "
        "record must carry the expanded uncertainty that leaves",
    ),
    RigCapability(
        "divot geometry after the shot",
        True,
        "static photogrammetry of the cavity; a cast is still needed for a "
        "volume the ledger's divot spec accepts",
    ),
    RigCapability(
        "club entry and exit pose and speed during contact",
        False,
        "a 25 m/s head moves 0.4 m per frame at 60 fps and the engagement lasts "
        "about five milliseconds, under one frame; entry and exit states must "
        "come from a synchronised delivery measurement, not from this rig",
    ),
    RigCapability(
        "ball spin",
        False,
        "60 fps cannot resolve the rotation of an unmarked ball; spin needs a "
        "marked ball and the frame rate the ledger's video spec names",
    ),
    RigCapability(
        "ejecta speed and sand motion",
        False,
        "the ledger's ejecta spec needs 5000 fps or faster; at 60 fps the sand "
        "sheet is a blur, and no ejecta reference record can be formed",
    ),
    RigCapability(
        "force and impulse on the head",
        False,
        "the rig carries no force instrumentation; impulse stays a solver "
        "output and cannot be a reference record without a strain-gauged shaft",
    ),
)
"""What the lab's three-camera rig can and cannot measure for this program.

Read off the measured constraints in ``docs/motion_capture/usb_camera_rig_bringup.md``
and the ledger's rig specs. No camera purchase and no experiment is assumed
here; a quantity marked unmeasurable stays unavailable in the stroke record.
"""


def rig_capability_markdown(
    capabilities: tuple[RigCapability, ...] = THREE_CAMERA_RIG_CAPABILITY,
) -> str:
    """Render the rig capability register as a Markdown table."""
    lines = ["| Quantity | Measurable | Why |", "| --- | --- | --- |"]
    for item in capabilities:
        flag = "yes" if item.measurable else "no"
        lines.append(f"| {item.quantity} | {flag} | {item.reason} |")
    return "\n".join(lines)
