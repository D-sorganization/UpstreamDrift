"""Governed evidence contracts for feasible closed-loop singular margins."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.proximal_distal_energy.run_closed_loop_singularity_margin import (
    build_report,
    validate_report,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "closed_loop_singularity_margin.json"
)


def test_registered_report_is_deterministic_and_portable() -> None:
    registered = json.loads(REPORT.read_text(encoding="utf-8"))

    assert registered == build_report()
    assert validate_report(registered) == {
        "near_boundary_case_count": 5,
        "nominal_orbit_sample_count": 362,
        "phase_resolution_case_count": 3,
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research.proximal_distal_energy.run_closed_loop_singularity_margin",
            "validate",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_report_retains_all_declared_sensitivity_and_adverse_controls() -> None:
    report = build_report()

    nominal = report["nominal_orbit"]
    assert nominal["minimum_rank"] == nominal["maximum_rank"] == 4
    assert nominal["minimum_nullity"] == nominal["maximum_nullity"] == 1
    assert nominal["maximum_closure_residual_m"] < 1e-12
    assert nominal["maximum_scaled_singular_value_spread_m"] < 1e-12
    assert nominal["maximum_scaled_singular_value_spread_m"] == 2e-15
    assert report["roundoff_diagnostic_significant_digits"] == 1
    exact = report["exact_triangle_degeneracies"]
    assert exact["lower_rank_audit"]["rank"] == 3
    assert exact["upper_rank_audit"]["rank"] == 3
    assert [
        case["phase_sample_count_per_branch"]
        for case in report["phase_resolution_controls"]
    ] == [17, 61, 181]
    assert len(report["scale_controls"]) == 3
    assert len(report["geometry_controls"]) == 3
    assert all(case["rejected"] for case in report["impossible_geometry_controls"])
    assert report["manufactured_matrix_killswitch"] == {
        "adverse_fixture": "fourth row replaced by third row",
        "adverse_nullity": 2,
        "adverse_rank": 3,
        "regular_nullity": 1,
        "regular_rank": 4,
    }


def test_near_boundary_sweep_retains_observed_tolerance_dependence() -> None:
    cases = build_report()["near_lower_boundary_sweep"]

    assert [case["distance_to_lower_degeneracy_m"] for case in cases] == [
        1e-4,
        1e-6,
        1e-8,
        1e-10,
        1e-12,
    ]
    rank_rows = [[trial["rank"] for trial in case["tolerance_cases"]] for case in cases]
    assert rank_rows == [
        [4, 4, 4, 4, 4],
        [4, 4, 4, 4, 3],
        [4, 4, 4, 4, 3],
        [4, 4, 4, 3, 3],
        [4, 4, 4, 3, 3],
    ]
    assert all(case["closure_residual_m"] < 1e-12 for case in cases)


def test_validation_killswitches_reject_corrupted_rank_and_inference() -> None:
    lost_rank = deepcopy(build_report())
    lost_rank["nominal_orbit"]["minimum_rank"] = 3
    with pytest.raises(ValueError, match="rank four"):
        validate_report(lost_rank)

    lost_degeneracy = deepcopy(build_report())
    lost_degeneracy["exact_triangle_degeneracies"]["lower_rank_audit"]["rank"] = 4
    with pytest.raises(ValueError, match="lower degeneracy"):
        validate_report(lost_degeneracy)

    promoted = deepcopy(build_report())
    promoted["inference_boundary"] = "Human strategy is established."
    with pytest.raises(ValueError, match="inference boundary"):
        validate_report(promoted)

    disabled = deepcopy(build_report())
    disabled["manufactured_matrix_killswitch"]["adverse_rank"] = 4
    with pytest.raises(ValueError, match="killswitch"):
        validate_report(disabled)
