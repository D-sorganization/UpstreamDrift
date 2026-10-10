"""Portable fail-closed contracts for the exact zero-subtalar preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.native_subtalar_reduction import (
    ZeroSubtalarReductionRequest,
    _reject_removed_coordinate_references,
)

pytestmark = pytest.mark.unit


def test_nonzero_declared_lock_target_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zero-target"):
        ZeroSubtalarReductionRequest(
            tmp_path / "source.osim",
            "a" * 64,
            tmp_path / "derived.osim",
            (("subtalar_angle_r", 0.0), ("subtalar_angle_l", 0.2)),
        )


def test_unknown_coordinate_consumer_is_not_silently_removed(tmp_path: Path) -> None:
    source = tmp_path / "source.osim"
    source.write_text(
        "<OpenSimDocument><CustomJoint name='subtalar_r'/>"
        "<CustomJoint name='subtalar_l'/>"
        "<unknown_consumer>subtalar_angle_l</unknown_consumer>"
        "</OpenSimDocument>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dependency|reference"):
        _reject_removed_coordinate_references(source)


def test_external_xml_entity_cannot_enter_dependency_scan(tmp_path: Path) -> None:
    source = tmp_path / "source.osim"
    source.write_text(
        '<!DOCTYPE OpenSimDocument [<!ENTITY outside SYSTEM "file:///private">]>'
        "<OpenSimDocument><CustomJoint name='subtalar_r'/>"
        "<CustomJoint name='subtalar_l'/><note>&outside;</note></OpenSimDocument>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _reject_removed_coordinate_references(source)
