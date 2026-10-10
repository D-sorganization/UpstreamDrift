"""Source-contract checks that run without an OpenSim SDK installation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.native_mtp_reduction import (
    ZeroMtpReductionRequest,
    _reject_removed_coordinate_references,
)

pytestmark = pytest.mark.unit


def test_request_rejects_nonzero_target_before_native_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zero-target"):
        ZeroMtpReductionRequest(
            source_model_path=tmp_path / "source.osim",
            source_sha256="a" * 64,
            derived_model_path=tmp_path / "output.osim",
            declared_target_rad=(("mtp_angle_r", 0.1), ("mtp_angle_l", 0.0)),
        )


def test_external_entity_cannot_enter_xml_dependency_scan(tmp_path: Path) -> None:
    source = tmp_path / "source.osim"
    source.write_text(
        '<!DOCTYPE OpenSimDocument [<!ENTITY outside SYSTEM "file:///private">]>'
        "<OpenSimDocument><PinJoint name='mtp_r'/><PinJoint name='mtp_l'/>"
        "<note>&outside;</note></OpenSimDocument>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _reject_removed_coordinate_references(source)
