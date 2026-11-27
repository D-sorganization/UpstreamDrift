"""SHOW-A14 (#14739): the Simscape model states the SHOW-A7 grip zero (#14731).

Acceptance pinned here:
1. The Simscape grip docs state the A7 convention (0 = back of the wrist on top
   of the shaft, + = counterclockwise, lead 15 deg, trail 195 deg) and point at
   ``grip_roll.py`` as the single source instead of restating a formula.
2. The gs3dx grip topology hand frames are club-attached: re-zeroing the wrist
   joints (the A7 spec change) leaves them unchanged.
3. A MATLAB unit test reads the topology through the thin reader, with no copy
   of the roll formula, and the R2025b run is cataloged as deferred.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.simscape_grip_topology import (
    DEFAULT_FULL_BODY_SPEC_PATH,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MATLAB_ROOT = (
    REPO_ROOT / "src" / "engines" / "Simscape_Multibody_Models" / "3D_Golf_Model"
) / "matlab"
SHARED_DIR = MATLAB_ROOT / "motion_matching" / "shared"
GS3DX_DIR = MATLAB_ROOT / "exploratory_gs3dx"
PLAYBOOK_PATH = SHARED_DIR / "GRIP_FIT_PLAYBOOK.md"
CLUB_IK_SPEC_PATH = SHARED_DIR / "CLUB_IK_SPEC.md"
HUMAN_DOC_PATH = GS3DX_DIR / "docs" / "HUMAN.md"
MATLAB_TEST_PATH = GS3DX_DIR / "tests" / "test_gs3dx_grip_roll_convention.m"
DEFERRED_CATALOG_PATH = (
    REPO_ROOT / "docs" / "development" / "deferred-catalog-bundle.json"
)
CONVENTION_HEADING = "## Grip Roll Convention (SHOW-A7)"
WRIST_JOINT_SUFFIX = "Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint"


def _section(text: str, heading: str) -> str:
    """Return the markdown section that starts at ``heading``."""
    assert heading in text, f"missing section {heading!r}"
    body = text.split(heading, 1)[1]
    return body.split("\n## ", 1)[0]


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_playbook_states_the_a7_grip_zero() -> None:
    section = _section(PLAYBOOK_PATH.read_text(encoding="utf-8"), CONVENTION_HEADING)
    for needle in (
        "back of the wrist",
        "counterclockwise",
        "15°",
        "195°",
        "src/shared/python/motion_matching/grip_roll.py",
        "#14731",
        "GripStrength",
    ):
        assert needle in section, f"playbook convention section lacks {needle!r}"
    assert "atan2" not in section, (
        "state the convention; the formula lives in grip_roll.py"
    )


def test_club_ik_spec_and_human_doc_point_at_the_convention() -> None:
    club_ik = CLUB_IK_SPEC_PATH.read_text(encoding="utf-8")
    assert "GRIP_FIT_PLAYBOOK.md#grip-roll-convention-show-a7" in club_ik
    assert "grip_quat" in club_ik
    human = HUMAN_DOC_PATH.read_text(encoding="utf-8")
    assert "GRIP_FIT_PLAYBOOK.md#grip-roll-convention-show-a7" in human
    assert "not the A7 hand roll" in human


def test_topology_hand_frames_ignore_a_wrist_rezero() -> None:
    spec = json.loads(DEFAULT_FULL_BODY_SPEC_PATH.read_text(encoding="utf-8"))
    rezeroed = copy.deepcopy(spec)
    wrists = [j for j in rezeroed["joints"] if j["name"].endswith(WRIST_JOINT_SUFFIX)]
    assert len(wrists) == 2, "expected the left and right wrist universal joints"
    for joint in wrists:
        follower = np.asarray(joint["child_to_follower"], dtype=float)
        follower[:3, :3] = _rot_z(75.0) @ follower[:3, :3]
        joint["child_to_follower"] = follower.tolist()

    before = GripInterface.from_spec(spec)
    after = GripInterface.from_spec(rezeroed)
    for side in ("L", "R"):
        old, new = before.frame(side), after.frame(side)
        np.testing.assert_allclose(new.rotation, old.rotation, atol=1e-12)
        np.testing.assert_allclose(new.position_m, old.position_m, atol=1e-12)


def test_matlab_convention_test_is_a_thin_reader_and_is_cataloged() -> None:
    m_code = MATLAB_TEST_PATH.read_text(encoding="utf-8")
    assert "matlab.unittest.TestCase" in m_code
    assert "gs3dx_read_grip_topology" in m_code
    assert "shaft_axis" in m_code
    assert "atan2" not in m_code, "no MATLAB copy of the grip_roll formula"

    catalog = json.loads(DEFERRED_CATALOG_PATH.read_text(encoding="utf-8"))
    items = [i for i in catalog["deferred_items"] if i.get("issue") == 14739]
    assert len(items) == 1, "expected exactly one SHOW-A14 (#14739) deferred item"
    entry = items[0]
    assert entry["status"] == "deferred"
    assert "R2025b" in entry["host_requirement"]
    assert (
        entry["matlab_unittest"] == MATLAB_TEST_PATH.relative_to(REPO_ROOT).as_posix()
    )
