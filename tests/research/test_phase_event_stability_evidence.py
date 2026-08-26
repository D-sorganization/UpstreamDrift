"""Governed evidence contracts for phase/event finite-time stability."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_phase_event_stability import (
    ARRAY_PATH,
    DIRECT_TRANSITION_RESIDUAL_GATE,
    DIRECT_TRANSITION_RESIDUAL_SIGNIFICANT_DIGITS,
    EQUIVALENT_UNIT_RESIDUAL_GATE,
    REFINEMENT_RESIDUAL_GATE,
    REFINEMENT_RESIDUAL_SIGNIFICANT_DIGITS,
    REPORT_PATH,
    build_report,
    canonicalize_direct_transition_residual,
    canonicalize_equivalent_unit_residual,
    canonicalize_refinement_residual,
    validate_report,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]


def test_refinement_residual_projection_is_portable_across_registered_runtimes() -> (
    None
):
    """Published diagnostics must not claim digits that vary by Python runtime."""

    assert REFINEMENT_RESIDUAL_SIGNIFICANT_DIGITS == 2
    assert canonicalize_refinement_residual(6.85575e-6) == 6.9e-6
    assert canonicalize_refinement_residual(6.87737e-6) == 6.9e-6
    assert canonicalize_refinement_residual(8.78627e-7) == 8.8e-7
    assert canonicalize_refinement_residual(8.79523e-7) == 8.8e-7
    with pytest.raises(ValueError, match="raw step-refinement residual"):
        canonicalize_refinement_residual(REFINEMENT_RESIDUAL_GATE)


def test_direct_transition_residual_is_a_portable_conservative_upper_bound() -> None:
    """Direct-rollout diagnostics must remain true on every qualified runtime."""

    assert DIRECT_TRANSITION_RESIDUAL_SIGNIFICANT_DIGITS == 1
    assert canonicalize_direct_transition_residual(3.43946e-7) == 4e-7
    assert canonicalize_direct_transition_residual(3.46646e-7) == 4e-7
    with pytest.raises(ValueError, match="raw direct-transition residual"):
        canonicalize_direct_transition_residual(DIRECT_TRANSITION_RESIDUAL_GATE)


def test_equivalent_unit_residual_is_a_portable_decade_upper_bound() -> None:
    """Machine epsilon differences must not become release-facing digits."""

    assert canonicalize_equivalent_unit_residual(2.22045e-16) == 1e-15
    assert canonicalize_equivalent_unit_residual(8.88178e-16) == 1e-15
    with pytest.raises(ValueError, match="raw equivalent-unit residual"):
        canonicalize_equivalent_unit_residual(EQUIVALENT_UNIT_RESIDUAL_GATE)


@pytest.fixture(scope="module")
def registered_report() -> dict[str, Any]:
    """Load the immutable report once for read-only evidence assertions."""

    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_registered_report_is_deterministic_and_portable(
    registered_report: dict[str, Any],
) -> None:
    registered = registered_report

    assert registered == build_report()
    assert validate_report(registered) == {
        "direct_event_trial_count": 3,
        "direct_transition_trial_count": 3,
        "phase_checkpoint_count": 5,
        "step_trial_count": 3,
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research.proximal_distal_energy.run_phase_event_stability",
            "validate",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_byte_governed_phase_outputs_are_excluded_from_prettier() -> None:
    """Keep canonical generator bytes under scientific validators, not Prettier."""

    precommit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "claim_numeric_contracts" in precommit
    assert "phase_event_stability" in precommit


def test_report_retains_local_and_adverse_boundaries(
    registered_report: dict[str, Any],
) -> None:
    report = registered_report

    assert report["reference_event"]["crossing_count"] == 1
    assert report["event_time_sensitivity"]["implicit"]["status"] == "transverse"
    assert all(
        trial["status"] == "available_transverse_candidates"
        for trial in report["event_time_sensitivity"]["direct_trials"]
    )
    assert (
        report["event_time_sensitivity"]["constructed_near_grazing_control"]["status"]
        == "near_grazing"
    )
    assert report["periodicity_gate"]["periodic"] is False
    assert report["periodicity_gate"]["floquet_multipliers"] is None
    assert (
        report["saltation_controls"]["time_guard_identity_reset_max_abs_residual"]
        == 0.0
    )
    assert report["saltation_controls"][
        "corrupted_reset_max_abs_deviation_from_identity"
    ] == pytest.approx(0.05)
    assert (
        report["equivalent_unit_controls"][
            "radian_to_degree_scaled_transition_max_abs_residual"
        ]
        < 1e-12
    )
    assert (
        report["equivalent_unit_controls"][
            "second_to_millisecond_exponent_roundtrip_max_abs_residual_per_s"
        ]
        < 1e-12
    )


def test_variational_predictions_converge_and_match_direct_rollouts(
    registered_report: dict[str, Any],
) -> None:
    report = registered_report

    assert (
        report["registration"]["refinement_residual_reporting_significant_digits"]
        == REFINEMENT_RESIDUAL_SIGNIFICANT_DIGITS
    )
    assert (
        report["registration"]["refinement_residual_gate"] == REFINEMENT_RESIDUAL_GATE
    )
    assert report["registration"]["direct_transition_residual_gate"] == (
        DIRECT_TRANSITION_RESIDUAL_GATE
    )
    assert (
        report["registration"][
            "direct_transition_residual_reporting_significant_digits"
        ]
        == DIRECT_TRANSITION_RESIDUAL_SIGNIFICANT_DIGITS
    )

    assert (
        max(
            trial["event_transition_max_abs_residual_from_nominal"]
            for trial in report["step_refinement"]
        )
        < 1e-5
    )
    assert (
        max(
            trial["maximum_abs_residual"]
            for trial in report["direct_transition_controls"]
        )
        < 1e-4
    )
    assert (
        max(
            trial["maximum_abs_residual_from_implicit_s"]
            for trial in report["event_time_sensitivity"]["direct_trials"]
        )
        < 1e-4
    )


def test_npz_retains_full_transition_and_adverse_control_arrays() -> None:
    with np.load(ARRAY_PATH) as arrays:
        assert arrays["state"].shape[1] == 4
        assert arrays["physical_transition"].shape[1:] == (4, 4)
        assert arrays["scaled_transition"].shape == arrays["physical_transition"].shape
        assert arrays["singular_values"].shape[1] == 4
        assert (
            arrays["finite_time_exponents_per_s"].shape
            == arrays["singular_values"].shape
        )
        assert arrays["direct_transition_matrices"].shape == (3, 4, 4)
        assert arrays["direct_event_derivatives_s_per_scaled_state"].shape == (3, 4)


def test_validation_rejects_floquet_promotion_and_lost_killswitches(
    registered_report: dict[str, Any],
) -> None:
    report = registered_report

    promoted = deepcopy(report)
    promoted["periodicity_gate"]["floquet_eligible"] = True
    with pytest.raises(ValueError, match="periodicity"):
        validate_report(promoted)

    lost_grazing = deepcopy(report)
    lost_grazing["event_time_sensitivity"]["constructed_near_grazing_control"][
        "status"
    ] = "transverse"
    with pytest.raises(ValueError, match="near-grazing"):
        validate_report(lost_grazing)

    lost_reset = deepcopy(report)
    lost_reset["saltation_controls"][
        "corrupted_reset_max_abs_deviation_from_identity"
    ] = 0.0
    with pytest.raises(ValueError, match="corrupted reset"):
        validate_report(lost_reset)
