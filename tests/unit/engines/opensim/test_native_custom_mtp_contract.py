"""Portable exact-profile and dependency checks for CustomJoint MTP."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.native_custom_mtp_reduction import (
    ZeroCustomMtpReductionRequest,
    _PROFILE,
)
from src.engines.physics_engines.opensim.python.native_subtalar_reduction import (
    _reject_removed_coordinate_references,
)

pytestmark = pytest.mark.unit


def test_wrong_target_or_mutable_declaration_is_rejected(tmp_path: Path) -> None:
    for targets in (
        (("mtp_angle_r", 0.0), ("mtp_angle_l", 0.1)),
        [("mtp_angle_r", 0.0), ("mtp_angle_l", 0.0)],
    ):
        with pytest.raises(ValueError, match="zero-MTP"):
            ZeroCustomMtpReductionRequest(
                tmp_path / "source.osim",
                "a" * 64,
                tmp_path / "derived.osim",
                targets,
            )


def test_unknown_mtp_coordinate_consumer_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.osim"
    source.write_text(
        "<OpenSimDocument><CustomJoint name='mtp_r'/>"
        "<CustomJoint name='mtp_l'/>"
        "<UnknownComponent>mtp_angle_l</UnknownComponent></OpenSimDocument>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dependency|reference"):
        _reject_removed_coordinate_references(source, _PROFILE)


def test_external_entity_does_not_enter_profile_scan(tmp_path: Path) -> None:
    source = tmp_path / "source.osim"
    source.write_text(
        '<!DOCTYPE OpenSimDocument [<!ENTITY injected SYSTEM "file:///private">]>'
        "<OpenSimDocument><CustomJoint name='mtp_r'/>"
        "<CustomJoint name='mtp_l'/><note>&injected;</note></OpenSimDocument>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _reject_removed_coordinate_references(source, _PROFILE)
