"""One collision-free humanoid for the gravity-only engine comparison.

This fixture qualifies free fall, not ground contact or bilateral grip closure.
Both engines consume the canonical URDF's inertias, joints and right-hand anchor.
"""

from __future__ import annotations

from pathlib import Path
from defusedxml.ElementTree import parse, tostring

from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
    CANONICAL_URDF,
)


def write_free_fall_urdf(directory: Path) -> Path:
    """Remove contact geometry from the shared gravity-only fixture."""
    root = parse(CANONICAL_URDF).getroot()
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)
    path = directory / "free_fall.urdf"
    path.write_text(tostring(root, encoding="unicode"), encoding="utf-8")
    return path


def write_free_fall_mjcf(urdf: Path) -> tuple[Path, int]:
    """Translate the URDF floating root and measurement frames to MuJoCo."""
    import mujoco

    root = parse(urdf).getroot()
    floating = root.find("./joint[@name='pelvis_floating']")
    assert floating is not None and floating.attrib["type"] == "floating"
    root.remove(floating)  # MuJoCo needs a freejoint, not a dangling world link.
    spec = mujoco.MjSpec.from_string(tostring(root, encoding="unicode"))
    spec.compiler.fusestatic = False
    spec.body("pelvis").add_freejoint(name="pelvis_floating")
    spec.body("right_hand").add_site(name="mid_hands")
    spec.body("club_shaft").add_site(name="clubhead")
    for joint in list(spec.joints):
        if joint.type != mujoco.mjtJoint.mjJNT_FREE:
            spec.add_actuator(
                name=joint.name + "_motor",
                target=joint.name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
            )
    model = spec.compile()
    path = urdf.with_suffix(".xml")
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path, int(model.nu)
