"""Tests for canonical double-pendulum plant transport (#9126)."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.nonlinear_controller_plant_transport import (
    evaluate_plant_transport,
    validate_plant_transport,
)

pytestmark = pytest.mark.scientific


def test_plant_transport_evaluates_exact_step_matches() -> None:
    summary = evaluate_plant_transport()

    assert summary.plant_identity == "canonical_analytical_double_pendulum_rk4"
    assert summary.all_step_sizes_passed is True
    assert summary.double_pendulum_evaluation_count == 0
    assert summary.ranking_eligible_method_count == 0

    for check in summary.step_checks:
        assert check.is_exact_match is True
        assert check.max_state_discrepancy < 1e-12
        assert check.control_semantics_preserved is True


def test_plant_transport_validation_runner() -> None:
    evidence = validate_plant_transport()

    assert evidence["status"] == "PASSED"
    assert evidence["double_pendulum_evaluation_count"] == 0
    assert evidence["ranking_eligible_method_count"] == 0
    assert (
        "scientific" in evidence["inference_boundary"].lower()
        or "analytical" in evidence["inference_boundary"].lower()
    )
