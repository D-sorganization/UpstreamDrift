"""Standalone module invocation routes to the desktop or headless surface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import __main__ as entrypoint
from src.tools.shot_pattern_analysis.__main__ import resolve_club_inputs

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def test_no_arguments_open_the_optional_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entrypoint, "launch_gui", lambda: 17)
    assert entrypoint.main([]) == 17


def test_headless_mode_requires_an_output_folder() -> None:
    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main(["--shots", "20"])
    assert exc_info.value.code == 2


def test_cli_exposes_delivery_coupling_controls() -> None:
    parser = entrypoint.build_parser()
    args = parser.parse_args(
        [
            "--output",
            "results",
            "--delivery-mode",
            "shaft_rotation",
            "--lie",
            "58",
            "--shaft-lean",
            "10",
            "--attack-angle-deg",
            "-2",
            "--clubhead-mass-kg",
            "0.27",
        ]
    )
    assert args.delivery_mode == "shaft_rotation"
    assert args.lie_deg == 58.0
    assert args.shaft_lean_deg == 10.0
    assert args.attack_angle_deg == -2.0
    assert args.clubhead_mass_kg == 0.27


def test_cli_preset_values_are_shared_and_numeric_overrides_win() -> None:
    args = entrypoint.build_parser().parse_args(
        ["--output", "results", "--club-preset", "seven_iron", "--loft-deg", "25"]
    )
    inputs = resolve_club_inputs(args)

    assert inputs["club_id"] == "seven_iron"
    assert inputs["club_speed_mps"] == 36.0
    assert inputs["loft_deg"] == 25.0
    assert inputs["attack_angle_deg"] == -4.0
    assert inputs["lie_deg"] == 63.0
    assert inputs["clubhead_mass_kg"] == 0.272


def test_custom_preset_preserves_explicit_control_baseline_values() -> None:
    args = entrypoint.build_parser().parse_args(
        [
            "--output",
            "results",
            "--club-preset",
            "custom",
            "--club-speed-mps",
            "45",
            "--loft-deg",
            "10.9",
            "--lie-deg",
            "58",
        ]
    )
    inputs = resolve_club_inputs(args)

    assert inputs["club_id"] == "custom"
    assert inputs["club_speed_mps"] == 45.0
    assert inputs["loft_deg"] == 10.9
    assert inputs["lie_deg"] == 58.0


def test_launcher_script_path_supports_headless_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/tools/shot_pattern_analysis/__main__.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--curve-scale" in completed.stdout
