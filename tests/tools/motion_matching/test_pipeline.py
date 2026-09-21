"""Tests for the Motion Matching tool's command construction and summaries."""

from __future__ import annotations

import json
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


def test_match_request_extended_options_and_validation() -> None:
    req = module.MatchRequest(
        capture="driver",
        club="driver",
        bound_wrists=True,
        fit_closure=True,
        zmp_filter=True,
        shooting_fit=4,
        shooting_gain=0.7,
        cutoff_hz=12.0,
    )
    assert req.bound_wrists is True
    assert req.fit_closure is True
    assert req.zmp_filter is True
    assert req.shooting_fit == 4
    assert req.shooting_gain == 0.7
    assert req.cutoff_hz == 12.0

    # Mutual exclusion between free_wrists and bound_wrists
    with pytest.raises(ValueError, match="wrists"):
        module.MatchRequest(
            capture="driver", club="driver", free_wrists=True, bound_wrists=True
        )

    # Shooting iterations must be >= 0
    with pytest.raises(ValueError, match="iterations"):
        module.MatchRequest(capture="driver", club="driver", shooting_fit=-1)

    # Shooting gain must be in [0, 1]
    with pytest.raises(ValueError, match="gain"):
        module.MatchRequest(capture="driver", club="driver", shooting_gain=-0.1)
    with pytest.raises(ValueError, match="gain"):
        module.MatchRequest(capture="driver", club="driver", shooting_gain=1.5)

    # Cutoff must be positive and below Nyquist (180 Hz)
    with pytest.raises(ValueError, match="[Cc]utoff"):
        module.MatchRequest(capture="driver", club="driver", cutoff_hz=0.0)
    with pytest.raises(ValueError, match="[Cc]utoff"):
        module.MatchRequest(capture="driver", club="driver", cutoff_hz=200.0)


def test_match_command_builds_expected_cli_flags() -> None:
    # Baseline
    base_req = module.MatchRequest(capture="driver", club="driver")
    base_cmd = module.match_command(base_req)
    assert "--free-wrists" not in base_cmd
    assert "--bound-wrists" not in base_cmd
    assert "--shooting-fit" not in base_cmd

    # Free wrists
    free_req = module.MatchRequest(capture="driver", club="driver", free_wrists=True)
    free_cmd = module.match_command(free_req)
    assert "--free-wrists" in free_cmd

    # Driver, bound wrists, shooting fit 4
    fit_req = module.MatchRequest(
        capture="driver",
        club="driver",
        bound_wrists=True,
        fit_closure=True,
        zmp_filter=True,
        shooting_fit=4,
        shooting_gain=0.7,
    )
    fit_cmd = module.match_command(fit_req)
    assert "--bound-wrists" in fit_cmd
    assert "--fit-closure" in fit_cmd
    assert "--zmp-filter" in fit_cmd
    assert "--shooting-fit" in fit_cmd
    assert fit_cmd[fit_cmd.index("--shooting-fit") + 1] == "4"
    assert "--shooting-gain" in fit_cmd
    assert fit_cmd[fit_cmd.index("--shooting-gain") + 1] == "0.7"


def test_experiment_request_and_command(tmp_path: Path) -> None:
    exp = module.ExperimentRequest(
        run=tmp_path,
        name="test_exp",
        cutoff_hz=14.0,
        omega=110.0,
        zeta=1.2,
        feedforward=0.9,
        legs_omega=45.0,
        balance=False,
        root_regulation=(100.0, 20.0),
        transition_velocity=0.05,
        friction=(0.8, 0.6),
        stiffness=45000.0,
        dissipation=1.5,
        reference=tmp_path / "ref.npz",
        dt=5e-5,
        duration=1.4,
    )
    cmd = module.experiment_command(exp)
    assert cmd[1] == str(module.DOWNSWING_SCRIPT)
    assert cmd[cmd.index("--run") + 1] == str(tmp_path)
    assert cmd[cmd.index("--name") + 1] == "test_exp"
    assert cmd[cmd.index("--cutoff-hz") + 1] == "14.0"
    assert cmd[cmd.index("--omega") + 1] == "110.0"
    assert cmd[cmd.index("--zeta") + 1] == "1.2"
    assert cmd[cmd.index("--feedforward") + 1] == "0.9"
    assert cmd[cmd.index("--legs-omega") + 1] == "45.0"
    assert "--no-balance" in cmd
    assert cmd[
        cmd.index("--root-regulation") + 1 : cmd.index("--root-regulation") + 3
    ] == ["100.0", "20.0"]
    assert cmd[cmd.index("--transition-velocity") + 1] == "0.05"
    assert cmd[cmd.index("--friction") + 1 : cmd.index("--friction") + 3] == [
        "0.8",
        "0.6",
    ]
    assert cmd[cmd.index("--stiffness") + 1] == "45000.0"
    assert cmd[cmd.index("--dissipation") + 1] == "1.5"
    assert cmd[cmd.index("--reference") + 1] == str(tmp_path / "ref.npz")
    assert cmd[cmd.index("--dt") + 1] == "5e-05"
    assert cmd[cmd.index("--duration") + 1] == "1.4"

    # Validation errors
    with pytest.raises(ValueError, match="name"):
        module.ExperimentRequest(run=tmp_path, name="")
    with pytest.raises(ValueError, match="[Cc]utoff"):
        module.ExperimentRequest(run=tmp_path, name="e", cutoff_hz=0.0)
    with pytest.raises(ValueError, match="[Cc]utoff"):
        module.ExperimentRequest(run=tmp_path, name="e", cutoff_hz=190.0)
    with pytest.raises(ValueError, match="gain"):
        module.ExperimentRequest(run=tmp_path, name="e", feedforward=-0.1)
    with pytest.raises(ValueError, match="gain"):
        module.ExperimentRequest(run=tmp_path, name="e", feedforward=1.1)
    with pytest.raises(ValueError, match="Omega"):
        module.ExperimentRequest(run=tmp_path, name="e", omega=-1.0)


def test_export_mjx_and_validate_reference_commands(tmp_path: Path) -> None:
    export_cmd = module.export_mjx_command(tmp_path)
    assert export_cmd[1] == str(module.EXPORT_MJX_SCRIPT)
    assert export_cmd[export_cmd.index("--run") + 1] == str(tmp_path)
    assert module.EXPORT_MJX_SCRIPT.exists()

    npz_path = tmp_path / "opt.npz"
    validate_cmd = module.validate_reference_command(tmp_path, npz_path)
    assert validate_cmd[1] == str(module.DOWNSWING_SCRIPT)
    assert validate_cmd[validate_cmd.index("--run") + 1] == str(tmp_path)
    assert validate_cmd[validate_cmd.index("--reference") + 1] == str(npz_path)
    assert validate_cmd[validate_cmd.index("--name") + 1] == "mjx_validation"
    assert module.DOWNSWING_SCRIPT.exists()


def test_read_experiment_summary(tmp_path: Path) -> None:
    exp_receipt = {
        "root_error_max_m": 0.045,
        "root_error_timeline_m": {"0.5": 0.005, "1.0": 0.015, "1.5": 0.040},
        "marker_rms_to_1_5s_m": 0.038,
        "marker_rms_m": 0.052,
        "inside_support_polygon_fraction": 0.95,
        "peak_joint_torque_n_m": 120.5,
    }
    json_path = tmp_path / "downswing_trial_1.json"
    json_path.write_text(json.dumps(exp_receipt), encoding="utf-8")

    summary = module.read_experiment_summary(tmp_path, "trial_1")
    assert summary["name"] == "trial_1"
    assert summary["root_error_max_mm"] == 45.0
    assert summary["root_error_timeline_m"] == {
        "0.5": 0.005,
        "1.0": 0.015,
        "1.5": 0.040,
    }
    assert summary["marker_rms_to_1_5s_m"] == 0.038
    assert summary["marker_rms_to_1_5s_mm"] == 38.0
    assert summary["marker_rms_mm"] == 52.0
    assert summary["inside_support_polygon_fraction"] == 0.95
    assert summary["peak_joint_torque_n_m"] == 120.5

    with pytest.raises(ValueError, match="No experiment receipt"):
        module.read_experiment_summary(tmp_path, "nonexistent")
