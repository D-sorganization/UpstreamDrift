"""URDF-level invariants for the Pinocchio golfer model (issue #4112).

These checks parse ``golfer.urdf`` and ``golfer_ik.urdf`` as plain XML and
do not require the pinocchio runtime. They guard the cross-engine
parity-spec §2.6 contract that the club is welded to a virtual
``mid_hands`` frame.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[4]
GOLFER_URDF = (
    REPO_ROOT / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)
GOLFER_IK_URDF = (
    REPO_ROOT / "src/engines/physics_engines/pinocchio/models/generated/golfer_ik.urdf"
)


@pytest.fixture(scope="module")
def golfer_root() -> ET.Element:
    if not GOLFER_URDF.exists():
        pytest.skip(f"{GOLFER_URDF} not found")
    return ET.parse(GOLFER_URDF).getroot()


@pytest.fixture(scope="module")
def golfer_ik_root() -> ET.Element:
    if not GOLFER_IK_URDF.exists():
        pytest.skip(f"{GOLFER_IK_URDF} not found")
    return ET.parse(GOLFER_IK_URDF).getroot()


def _link_names(root: ET.Element) -> set[str]:
    return {el.attrib["name"] for el in root.findall("link")}


def _joint_by_name(root: ET.Element, name: str) -> ET.Element | None:
    for j in root.findall("joint"):
        if j.attrib.get("name") == name:
            return j
    return None


class TestMidHandsLinkPresent:
    def test_mid_hands_link_exists(self, golfer_root) -> None:
        assert "mid_hands" in _link_names(
            golfer_root
        ), "`mid_hands` virtual link missing from golfer.urdf"

    def test_mid_hands_parent_is_thorax3(self, golfer_root) -> None:
        joint = _joint_by_name(golfer_root, "thorax3_to_mid_hands")
        assert joint is not None, "thorax3_to_mid_hands joint missing"
        assert joint.attrib["type"] == "fixed"
        assert joint.find("parent").attrib["link"] == "thorax3"
        assert joint.find("child").attrib["link"] == "mid_hands"

    def test_mid_hands_origin_at_grip_centre(self, golfer_root) -> None:
        """The mid_hands link sits on the body midline below thorax3, at the
        midpoint of the two hand-tip frames in the address pose."""
        joint = _joint_by_name(golfer_root, "thorax3_to_mid_hands")
        origin = joint.find("origin").attrib["xyz"].split()
        x, y, z = (float(v) for v in origin)
        # x and y must be on the sagittal/midsagittal plane (== 0)
        assert abs(x) <= 1e-9
        assert abs(y) <= 1e-9
        # z places mid_hands at the height of the hand tips relative to
        # thorax3 (~ -0.17 m) — see header comment for derivation.
        assert -0.20 <= z <= -0.14


class TestClubWeldedToMidHands:
    def test_mid_hands_to_club_shaft_joint(self, golfer_root) -> None:
        joint = _joint_by_name(golfer_root, "mid_hands_to_club_shaft")
        assert (
            joint is not None
        ), "Club must be welded to mid_hands; mid_hands_to_club_shaft missing"
        assert joint.attrib["type"] == "fixed"
        assert joint.find("parent").attrib["link"] == "mid_hands"
        assert joint.find("child").attrib["link"] == "club_shaft"

    def test_old_hand_left_to_club_joint_removed(self, golfer_root) -> None:
        assert _joint_by_name(golfer_root, "hand_left_to_club_shaft") is None, (
            "Stale hand_left_to_club_shaft joint should be replaced by "
            "mid_hands_to_club_shaft"
        )


class TestPelvisIsRoot:
    """Pelvis must remain the URDF root link so callers can attach the
    free-flyer joint at load time via ``pin.JointModelFreeFlyer()``."""

    def test_pelvis_has_no_parent(self, golfer_root) -> None:
        children = {
            j.find("child").attrib["link"] for j in golfer_root.findall("joint")
        }
        assert "pelvis" not in children, "pelvis must be the root link of golfer.urdf"

    def test_pelvis_root_in_ik(self, golfer_ik_root) -> None:
        children = {
            j.find("child").attrib["link"] for j in golfer_ik_root.findall("joint")
        }
        assert "pelvis" not in children


class TestHeaderDocumentation:
    """Acceptance criterion 4: header comment block declares scope."""

    def test_golfer_urdf_has_scope_header(self) -> None:
        text = GOLFER_URDF.read_text(encoding="utf-8")
        assert "forward-simulation" in text.lower()
        assert "mid_hands" in text
        assert "floating base" in text.lower() or "free-flyer" in text.lower()

    def test_golfer_ik_urdf_has_scope_header(self) -> None:
        text = GOLFER_IK_URDF.read_text(encoding="utf-8")
        assert "ik" in text.lower()
        assert "no club" in text.lower() or "club is tracked externally" in text.lower()
