"""``read_osim`` resolves inline and body-owned joint frames (#11756)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.execution.spec_builder import read_osim

pytestmark = pytest.mark.unit

_BODY = """
<Body name="{name}">
  <mass>1</mass><mass_center>0 0 0</mass_center><inertia>1 1 1 0 0 0</inertia>
  {components}
</Body>"""

_OWNED = """
<components>
  <PhysicalOffsetFrame name="{name}">
    <socket_parent>..</socket_parent>
    <translation>{t}</translation><orientation>{o}</orientation>
  </PhysicalOffsetFrame>
</components>"""

_INLINE_JOINT = """
<PinJoint name="hand_joint">
  <socket_parent_frame>arm_offset</socket_parent_frame>
  <socket_child_frame>hand_offset</socket_child_frame>
  <frames>
    <PhysicalOffsetFrame name="arm_offset">
      <socket_parent>/bodyset/arm</socket_parent>
      <translation>0 -0.3 0</translation><orientation>0 0 0.1</orientation>
    </PhysicalOffsetFrame>
    <PhysicalOffsetFrame name="hand_offset">
      <socket_parent>/bodyset/hand</socket_parent>
      <translation>0 0 0</translation><orientation>0 0 0</orientation>
    </PhysicalOffsetFrame>
  </frames>
</PinJoint>"""

_PATH_JOINT = """
<WeldJoint name="hand_to_club">
  <socket_parent_frame>/bodyset/hand/hand_grip</socket_parent_frame>
  <socket_child_frame>/bodyset/club/{child}</socket_child_frame>
</WeldJoint>"""


def _write(tmp_path: Path, club_frame: str = "club_grip") -> Path:
    bodies = "".join(
        [
            _BODY.format(name="arm", components=""),
            _BODY.format(
                name="hand",
                components=_OWNED.format(name="hand_grip", t="0 -0.08 0", o="0 0 0"),
            ),
            _BODY.format(
                name="club",
                components=_OWNED.format(name="club_grip", t="0 -1.1 0", o="0.2 0 0"),
            ),
        ]
    )
    joints = _INLINE_JOINT + _PATH_JOINT.format(child=club_frame)
    path = tmp_path / "model.osim"
    path.write_text(
        "<OpenSimDocument><Model>"
        f"<BodySet><objects>{bodies}</objects></BodySet>"
        f"<JointSet><objects>{joints}</objects></JointSet>"
        "</Model></OpenSimDocument>"
    )
    return path


def test_inline_frames_read_as_before(tmp_path: Path) -> None:
    _, joints = read_osim(_write(tmp_path))
    joint = joints["hand_joint"]
    assert (joint["parent_body"], joint["child_body"]) == ("arm", "hand")
    np.testing.assert_allclose(joint["parent_translation"], [0, -0.3, 0])
    np.testing.assert_allclose(joint["parent_orientation"], [0, 0, 0.1])


def test_body_owned_frames_resolve_by_path(tmp_path: Path) -> None:
    _, joints = read_osim(_write(tmp_path))
    joint = joints["hand_to_club"]
    assert (joint["parent_body"], joint["child_body"]) == ("hand", "club")
    np.testing.assert_allclose(joint["parent_translation"], [0, -0.08, 0])
    np.testing.assert_allclose(joint["child_translation"], [0, -1.1, 0])
    np.testing.assert_allclose(joint["child_orientation"], [0.2, 0, 0])


def test_unresolvable_socket_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hand_to_club"):
        read_osim(_write(tmp_path, club_frame="missing"))


def test_shipped_golf_humanoid_reads() -> None:
    root = Path(__file__).resolve().parents[3]
    osim = root / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
    _, joints = read_osim(osim)
    assert joints["hand_l_to_club"]["child_body"] == "Club"
    assert {"hip_r", "hip_l"} <= set(joints)
