"""Evidence contracts for the local double-pendulum rank audit."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.proximal_distal_energy.run_local_linear_diagnostics import (
    CANONICAL_SIGNIFICANT_DIGITS,
    build_report,
    validate_report,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/local_linear_diagnostics.json"
)
RELEASE_MANIFEST = (
    ROOT / "docs/research/proximal_distal_energy_transfer/release_manifest.json"
)


def test_registered_report_is_reproducible_and_canonical() -> None:
    registered = json.loads(REPORT.read_text(encoding="utf-8"))

    assert registered == build_report()
    assert REPORT.read_text(encoding="utf-8").endswith("\n")
    assert validate_report(registered) == {
        "actuator_countermodel_count": 16,
        "measurement_countermodel_count": 16,
        "operating_point_count": 4,
        "rank_decision_count": 24,
        "scale_sensitivity_count": 12,
        "step_multiplier_count": 3,
    }


def test_published_floats_use_registered_cross_platform_precision() -> None:
    def visit(value: object) -> None:
        if isinstance(value, float):
            assert value == float(f"{value:.{CANONICAL_SIGNIFICANT_DIGITS}g}")
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    report = build_report()
    assert report["numeric_representation_contract"] == {
        "canonical_significant_digits": CANONICAL_SIGNIFICANT_DIGITS,
        "rank_decision_precision": "full_precision_before_publication_rounding",
        "scope": "published_json_floats_only",
    }
    visit(report)


def test_documented_cli_validates_without_inherited_pythonpath() -> None:
    """A clean-checkout reviewer must not need pytest's import setup."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research.proximal_distal_energy.run_local_linear_diagnostics",
            "validate",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"rank_decision_count": 24' in completed.stdout


def test_release_preset_exposes_deterministic_report_generation() -> None:
    """The open-resource manifest must advertise the executed diagnostic."""
    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["presets"]["local_linear_diagnostics"] == {
        "command": (
            "python -m scripts.research.proximal_distal_energy."
            "run_local_linear_diagnostics write"
        ),
        "tier": "analytical_double_pendulum_local_first_order_rank",
    }


def test_report_retains_local_only_inference_boundary() -> None:
    report = build_report()

    assert report["classification"] == "local_first_order_numerical_rank"
    assert report["structural_identifiability_status"] == "not_evaluated"
    assert report["practical_identifiability_status"] == "not_evaluated"
    assert report["global_nonlinear_status"] == "not_evaluated"
    assert "not establish structural identifiability" in report["inference_boundary"]
    assert report["conditioning_contract"] == {
        "interpreted_basis": "nondimensional_matrices",
        "raw_dimensional_conditioning_status": "not_interpreted",
        "unit_invariance_fixture": "equivalent_length_units",
    }


def test_report_declares_machine_readable_scale_coordinates_units_and_rationale() -> (
    None
):
    report = build_report()
    contract = report["nondimensional_scale_contract"]

    assert report["issue"].endswith("/9092")
    assert report["parent_issue"].endswith("/9027")
    assert contract["state_coordinates"] == [
        "shoulder_angle_rad",
        "wrist_relative_angle_rad",
        "shoulder_rate_rad_s",
        "wrist_relative_rate_rad_s",
    ]
    assert contract["state_units"] == ["rad", "rad", "rad/s", "rad/s"]
    assert contract["control_coordinates"] == [
        "shoulder_generalized_torque_nm",
        "wrist_relative_generalized_torque_nm",
    ]
    assert contract["control_units"] == ["N*m", "N*m"]
    assert contract["output_coordinates"] == [
        "shoulder_angle_rad",
        "wrist_relative_angle_rad",
    ]
    assert contract["output_units"] == ["rad", "rad"]
    assert contract["state_scale_derivation"]["angles"] == "pi rad"
    assert "registered-rollout rate" in contract["state_scale_derivation"]["rates"]
    assert "registered-rollout torque" in contract["control_scale_derivation"]
    assert "delivery_event_time_s" in contract["characteristic_time_derivation"]


def test_all_trace_derived_points_pass_declared_step_rank_stability_gate() -> None:
    report = build_report()

    assert report["summary"]["all_rank_decisions_stable"] is True
    for point in report["operating_points"]:
        assert point["source"] == "registered_synthetic_reference_rollout"
        assert point["rank_stable_across_steps"] is True
        assert len(point["step_trials"]) == 3
        for trial in point["step_trials"]:
            assert trial["rank_basis"] == "nondimensional"
            assert trial["raw_dimensional_conditioning_status"] == "not_interpreted"
            assert trial["scales"]["characteristic_time_s"] > 0.0
            assert trial["observability"]["rank"] == 4
            assert trial["controllability"]["rank"] == 4


def test_report_retains_scaling_sensitivity_and_countermodels() -> None:
    report = build_report()

    assert report["summary"]["scale_scenario_count"] == 3
    assert report["summary"]["measurement_countermodel_count"] == 16
    assert report["summary"]["actuator_countermodel_count"] == 16
    for point in report["operating_points"]:
        assert [trial["name"] for trial in point["scale_trials"]] == [
            "short_time_high_rate",
            "nominal",
            "long_time_low_rate",
        ]
        assert all(
            trial["rank_basis"] == "nondimensional" for trial in point["scale_trials"]
        )
        measurements = {
            item["name"]: item for item in point["measurement_countermodels"]
        }
        actuators = {item["name"]: item for item in point["actuator_countermodels"]}
        assert set(measurements) == {
            "both_joint_angles",
            "shoulder_angle_only",
            "wrist_relative_angle_only",
            "zero_output_killswitch",
        }
        assert set(actuators) == {
            "both_generalized_torques",
            "shoulder_torque_only",
            "wrist_torque_only",
            "zero_input_killswitch",
        }
        assert measurements["zero_output_killswitch"]["observability"]["rank"] == 0
        assert actuators["zero_input_killswitch"]["controllability"]["rank"] == 0


def test_validation_rejects_promoted_or_unstable_evidence() -> None:
    promoted = deepcopy(build_report())
    promoted["structural_identifiability_status"] = "established"
    with pytest.raises(ValueError, match="structural identifiability"):
        validate_report(promoted)

    unstable = deepcopy(build_report())
    unstable["operating_points"][0]["step_trials"][0]["controllability"]["rank"] = 3
    with pytest.raises(ValueError, match="rank stability"):
        validate_report(unstable)
