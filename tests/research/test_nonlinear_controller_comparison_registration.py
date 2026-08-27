"""Tests for nonlinear controller comparison registration (#9126)."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.nonlinear_controller_registration import (
    ProspectiveControllerRegistration,
    validate_registration,
)

pytestmark = pytest.mark.scientific


def test_prospective_registration_structure_and_digest() -> None:
    reg = ProspectiveControllerRegistration.create()

    assert reg.plant_identity == "analytical_double_pendulum_rk4"
    assert reg.state_ordering == ("theta1", "theta2", "omega1", "omega2")
    assert reg.control_ordering == ("tau1", "tau2")
    assert reg.integration_step_s == 0.002
    assert reg.planning_horizon_steps == 60
    assert reg.double_pendulum_evaluation_count == 0
    assert reg.ranking_eligible_method_count == 0
    assert "bounded_nmpc_collocation" not in reg.candidate_solvers
    assert "bounded_nmpc_collocation" in reg.unavailable_solvers
    assert len(reg.source_digest) == 64


def test_registration_validation_runner() -> None:
    evidence = validate_registration()

    assert evidence["status"] == "PASSED"
    assert evidence["double_pendulum_evaluation_count"] == 0
    assert evidence["ranking_eligible_method_count"] == 0
    assert (
        "scientific" in evidence["inference_boundary"].lower()
        or "analytical" in evidence["inference_boundary"].lower()
    )
