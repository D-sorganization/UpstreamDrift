"""Tests for the shared swing model provider (epic #8390, B1/#8396)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import ClubModel, GolferModel
from src.shared.python.optimization.model_provider import (
    build_swing_rig,
    swing_joint_limits,
    swing_urdf,
)

pytestmark = pytest.mark.unit


def test_rig_has_one_dof_per_swing_joint() -> None:
    rig = build_swing_rig()
    assert rig.num_dofs == len(JOINTS) == 7
    assert list(rig.joints.keys()) == JOINTS


def test_rig_limits_come_from_golfer_rom() -> None:
    golfer = GolferModel(hip_rom=(-0.5, 0.6), trunk_rotation_rom=(-1.0, 1.1))
    rig = build_swing_rig(golfer)
    assert rig.joints["hip_rotation"].limits[0].lower == pytest.approx(-0.5)
    assert rig.joints["hip_rotation"].limits[0].upper == pytest.approx(0.6)
    assert rig.joints["trunk_rotation"].limits[0].upper == pytest.approx(1.1)
    limits = swing_joint_limits(golfer)
    assert set(limits) == set(JOINTS)


def test_urdf_parses_with_seven_revolute_joints() -> None:
    root = ET.fromstring(swing_urdf())
    revolute = [j for j in root.findall("joint") if j.get("type") == "revolute"]
    assert len(revolute) == 7
    names = [j.get("name") for j in revolute]
    assert names == [f"{j}_dof0" for j in JOINTS]


def test_urdf_scales_with_anthropometrics() -> None:
    tall = swing_urdf(GolferModel(height=2.0, trunk_length=0.6))
    short = swing_urdf(GolferModel(height=1.5, trunk_length=0.4))
    assert tall != short


def test_club_length_extends_terminal_offset() -> None:
    long_club = build_swing_rig(club=ClubModel(total_length=1.3))
    short_club = build_swing_rig(club=ClubModel(total_length=0.9))
    assert (
        long_club.joints["wrist_rotation"].tpose_offset[2]
        < short_club.joints["wrist_rotation"].tpose_offset[2]
    )


def test_mujoco_model_when_available() -> None:
    pytest.importorskip("mujoco")
    from src.shared.python.optimization.model_provider import build_mujoco_model

    model = build_mujoco_model()
    assert model.nq == 7


def test_pinocchio_builder_degrades_with_hint() -> None:
    from unittest.mock import patch

    from src.shared.python.optimization import model_provider

    with patch.object(model_provider, "_module_available", return_value=False):
        with pytest.raises(RuntimeError, match="pinocchio"):
            model_provider.build_pinocchio_model()
        with pytest.raises(RuntimeError, match="drake"):
            model_provider.build_drake_plant()
        with pytest.raises(RuntimeError, match="mujoco"):
            model_provider.build_mujoco_model()
