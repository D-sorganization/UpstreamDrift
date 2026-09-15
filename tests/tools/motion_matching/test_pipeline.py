"""Tests for the Motion Matching tool's command construction and summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.motion_matching import pipeline as module

pytestmark = pytest.mark.unit


def test_request_validation_and_names() -> None:
    request = module.MatchRequest(capture="iron", club="iron7")
    assert request.document_name == "full_body_spec_anthro_iron7"
    assert request.run_name == "anthro_iron"
    assert request.output_dir.name == "anthro_iron"
    with pytest.raises(ValueError):
        module.MatchRequest(capture="wedge", club="iron7")
    with pytest.raises(ValueError):
        module.MatchRequest(capture="driver", club="putter")
    with pytest.raises(ValueError):
        module.MatchRequest(capture="driver", club="driver", mass_kg=0.0)


def test_commands_point_at_the_pipeline_scripts() -> None:
    request = module.MatchRequest(capture="driver", club="driver", stature_m=1.8)
    build = module.build_command(request)
    match = module.match_command(request)
    assert build[1] == str(module.BUILDER) and "--club" in build
    assert build[build.index("--stature") + 1] == "1.8"
    assert match[1] == str(module.DRIVER_SCRIPT)
    assert match[match.index("--capture") + 1] == "driver"
    assert match[match.index("--spec") + 1] == str(request.document_path)
    assert module.BUILDER.exists() and module.DRIVER_SCRIPT.exists()


def test_receipt_summary_reads_the_headline_numbers(tmp_path: Path) -> None:
    receipt = {
        "capture": "driver",
        "club": {"name": "driver"},
        "address": {
            "calibrated": {
                "marker_rms_m": 0.0071,
                "centre_of_mass": {"inside_support_polygon": True},
            }
        },
        "ik": {"marker_rms_m": 0.0248, "range_of_motion_flags": {"knee_angle_r": {}}},
        "dynamics": {
            "root_tracking_rms_m": 0.03,
            "inside_support_polygon_fraction": 0.88,
            "backswing_to_1s": {"root_error_max_m": 0.006},
            "range_of_motion_flags": {},
        },
    }
    summary = module.summarise_receipt(receipt)
    assert summary["address_marker_rms_mm"] == 7.1
    assert summary["full_capture_ik_rms_mm"] == 24.8
    assert summary["backswing_root_error_max_mm"] == 6.0
    assert summary["com_inside_polygon_at_address"] is True
    assert summary["range_of_motion_flags_ik"] == ["knee_angle_r"]
    with pytest.raises(ValueError):
        module.summarise_receipt({"address": {}})
    with pytest.raises(ValueError):
        module.read_summary(tmp_path)
    assert module.artefacts(tmp_path) == ()
