"""Evidence contracts for analytical double-pendulum identifiability."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.proximal_distal_energy.run_double_pendulum_identifiability import (
    CANONICAL_SIGNIFICANT_DIGITS,
    build_report,
    validate_report,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "double_pendulum_identifiability.json"
)


def test_registered_identifiability_report_is_deterministic() -> None:
    registered = json.loads(REPORT.read_text(encoding="utf-8"))

    assert registered == build_report()
    assert validate_report(registered)["physical_parameter_nullity"] == 4


def test_documented_cli_and_release_preset_are_portable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research.proximal_distal_energy.run_double_pendulum_identifiability",
            "validate",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["physical_parameter_nullity"] == 4

    manifest = json.loads(
        (
            ROOT / "docs/research/proximal_distal_energy_transfer/release_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["presets"]["double_pendulum_identifiability"] == {
        "command": (
            "python -m scripts.research.proximal_distal_energy."
            "run_double_pendulum_identifiability write"
        ),
        "tier": (
            "analytical_double_pendulum_exact_map_and_dimensionless_finite_record"
        ),
    }


def test_reference_record_identifies_base_coefficients_but_not_physical_parameters() -> (
    None
):
    report = build_report()

    assert report["physical_parameter_map"]["rank"] == 7
    assert report["physical_parameter_map"]["nullity"] == 4
    assert report["physical_parameter_map"]["rank_basis"] == ("analytic_nonzero_minor")
    assert (
        report["physical_parameter_map"]["structural_rank_witness"]["determinant"] > 0.0
    )
    assert report["finite_record_regressor"]["rank"] == 7
    assert report["finite_record_regressor"]["full_base_coefficient_rank"] is True
    assert report["zero_motion_killswitch"]["rank"] == 0
    assert len(report["exact_counterexamples"]) == 3
    screen = report["noise_aware_lower_bound_screen"]
    assert screen["classification"] == "oracle_kinematics_fisher_lower_bound"
    assert screen["confidence_level"] == 0.95
    assert [case["torque_noise_sd_nm"] for case in screen["full_record_cases"]] == [
        0.1,
        0.5,
        1.0,
        2.0,
    ]
    assert all(case["full_rank"] for case in screen["full_record_cases"])
    assert report["practical_identifiability_status"] == (
        "not_established_oracle_kinematics_lower_bound_only"
    )


def test_finite_record_rank_is_nondimensional_unit_and_scale_audited() -> None:
    report = build_report()
    record = report["finite_record_regressor"]

    assert record["rank_basis"] == "nondimensional_base_coefficient_regressor"
    assert record["raw_dimensional_conditioning_status"] == "not_interpreted"
    assert record["unit_invariance"]["coefficient_coordinate_conversion_factors"] == [
        1000.0,
        1000.0,
        1000.0,
        100.0,
        100.0,
        60.0,
        60.0,
    ]
    assert record["unit_invariance"]["fixture"] == "equivalent_coefficient_units"
    assert (
        record["unit_invariance"]["max_abs_dimensionless_regressor_difference"] <= 1e-12
    )
    assert record["unit_invariance"]["rank_after_conversion"] == 7
    assert record["unit_invariance"]["rank_before_conversion"] == 7
    assert {trial["name"] for trial in record["scale_sensitivity"]} == {
        "registered",
        "alternating_half_double",
        "alternating_double_half",
    }
    assert [trial["rank"] for trial in record["scale_sensitivity"]] == [7, 7, 7]
    assert report["numeric_representation_contract"] == {
        "canonical_significant_digits": CANONICAL_SIGNIFICANT_DIGITS,
        "decision_precision": "full_precision_before_publication_rounding",
        "scope": "published_json_floats_only",
    }


def test_noise_aware_screen_retains_conditioning_and_noise_adverse_cases() -> None:
    screen = build_report()["noise_aware_lower_bound_screen"]
    full_cases = screen["full_record_cases"]
    worst_relative_bounds = [
        case["worst_ci95_relative_half_width"] for case in full_cases
    ]

    assert worst_relative_bounds == sorted(worst_relative_bounds)
    assert worst_relative_bounds[-1] == pytest.approx(
        20.0 * worst_relative_bounds[0], rel=1e-5
    )
    early = screen["window_cases"][0]
    full = screen["window_cases"][-1]
    assert early["fraction"] == 0.1
    assert full["fraction"] == 1.0
    assert (
        early["worst_ci95_relative_half_width"]
        > (full["worst_ci95_relative_half_width"])
    )


def test_validation_rejects_parameter_promotion_and_lost_counterexample() -> None:
    promoted = deepcopy(build_report())
    promoted["physical_parameter_status"] = "identified"
    with pytest.raises(ValueError, match="physical parameters"):
        validate_report(promoted)

    incomplete = deepcopy(build_report())
    incomplete["exact_counterexamples"].pop()
    with pytest.raises(ValueError, match="counterexamples"):
        validate_report(incomplete)

    practical_promotion = deepcopy(build_report())
    practical_promotion["practical_identifiability_status"] = "identified"
    with pytest.raises(ValueError, match="practical identifiability"):
        validate_report(practical_promotion)

    promoted_bound = deepcopy(build_report())
    promoted_bound["noise_aware_lower_bound_screen"]["status"] = (
        "practical_identifiability_established"
    )
    with pytest.raises(ValueError, match="conditional lower bound"):
        validate_report(promoted_bound)

    dimensional = deepcopy(build_report())
    dimensional["finite_record_regressor"]["rank_basis"] = "raw_dimensional"
    with pytest.raises(ValueError, match="nondimensional basis"):
        validate_report(dimensional)

    broken_units = deepcopy(build_report())
    broken_units["finite_record_regressor"]["unit_invariance"][
        "rank_after_conversion"
    ] = 6
    with pytest.raises(ValueError, match="equivalent coefficient units"):
        validate_report(broken_units)
