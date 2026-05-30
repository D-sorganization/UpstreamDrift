"""Unit tests for the diagnostic forward-kinematics evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "shared" / "python"))

from src.shared.python.motion_matching.diagnostics.forward_kinematics import (  # noqa: E402
    SegmentLengths,
    forward_kinematics,
)

pytestmark = pytest.mark.unit


def test_fk_evaluator_zero_angles_returns_t_pose() -> None:
    """All zeros = T-pose: shoulders horizontal, arms outstretched."""
    pose = forward_kinematics({})
    # Pelvis at origin.
    np.testing.assert_allclose(pose["pelvis"], [0, 0, 0], atol=1e-9)
    # Left shoulder above pelvis on +Y; symmetric on -Y.
    assert pose["l_shoulder"][1] > 0
    assert pose["r_shoulder"][1] < 0
    np.testing.assert_allclose(pose["l_shoulder"][1], -pose["r_shoulder"][1], atol=1e-9)
    # Hands further out laterally than shoulders.
    assert pose["l_hand"][1] > pose["l_shoulder"][1]
    assert pose["r_hand"][1] < pose["r_shoulder"][1]


def test_fk_evaluator_segment_lengths_respected() -> None:
    """Distance shoulder->elbow == upper_arm length when arm is straight."""
    lengths = SegmentLengths(upper_arm=0.42, forearm=0.31)
    pose = forward_kinematics({}, lengths=lengths)
    d_la = np.linalg.norm(pose["l_elbow"] - pose["l_shoulder"])
    d_lf = np.linalg.norm(pose["l_wrist"] - pose["l_elbow"])
    assert d_la == pytest.approx(0.42, abs=1e-9)
    assert d_lf == pytest.approx(0.31, abs=1e-9)


def test_fk_evaluator_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        forward_kinematics([1, 2, 3])  # type: ignore[arg-type]


def test_fk_evaluator_clubhead_distance_matches_shaft_length() -> None:
    lengths = SegmentLengths(club_shaft=1.05)
    pose = forward_kinematics({}, lengths=lengths)
    d = np.linalg.norm(pose["clubhead"] - pose["butt"])
    assert d == pytest.approx(1.05, abs=1e-9)
