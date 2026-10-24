"""One collision-free humanoid for the gravity-only engine comparison.

This fixture qualifies free fall, not ground contact or bilateral grip closure.
Both engines consume the canonical URDF's inertias, joints and right-hand anchor.
"""

from __future__ import annotations

from pathlib import Path
from defusedxml.ElementTree import parse, tostring

_REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_URDF: Path = (
    _REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "drake"
    / "models"
    / "generated"
    / "golfer.urdf"
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


def _grip_anchor_body_name(root) -> str:
    """Return the link name that anchors the mid-hands grip measurement."""
    for name in ("mid_hands", "right_hand"):
        if root.find(f".//link[@name='{name}']") is not None:
            return name
    raise ValueError(
        "free-fall URDF must expose mid_hands or right_hand for grip tracking"
    )


def write_free_fall_mjcf(urdf: Path) -> tuple[Path, int]:
    """Translate the URDF floating root and measurement frames to MuJoCo."""
    import mujoco

    root = parse(urdf).getroot()
    floating = root.find("./joint[@name='pelvis_floating']")
    if floating is not None:
        joint_type = floating.attrib.get("type")
        if joint_type != "floating":
            raise ValueError(
                f"pelvis_floating must be type floating, got {joint_type!r}"
            )
        root.remove(floating)  # MuJoCo needs a freejoint, not a dangling world link.
    spec = mujoco.MjSpec.from_string(tostring(root, encoding="unicode"))
    spec.compiler.fusestatic = False
    pelvis = spec.body("pelvis")
    if pelvis is None:
        raise ValueError("free-fall URDF must contain a pelvis link")
    pelvis.add_freejoint(name="pelvis_floating")
    grip_body = _grip_anchor_body_name(root)
    if spec.site("mid_hands") is None and spec.body(grip_body) is not None:
        spec.body(grip_body).add_site(name="mid_hands")
    club_body = spec.body("club_shaft") or spec.body("club_head")
    if club_body is not None and spec.site("clubhead") is None:
        club_body.add_site(name="clubhead")
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
