"""DbC and mutation tests for fail-closed impact-interval audit & export (#9547)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from src.shared.python.physics.impact_interval_audit import (
    ImpactAuditError,
    audit_and_export_impact_run,
    audit_impact_interval,
    qualified_post_impact_state,
    recompute_energy_residual,
)

pytestmark = pytest.mark.unit

_POST_IMPACT = {"state": "valid_post_impact_state"}


class _TrackedResult:
    """Mock impact solve result tracking whether to_post_impact_state was invoked."""

    def __init__(
        self,
        audit: SimpleNamespace,
        *,
        termination: str = "SEPARATED",
        contact_completed: bool = True,
    ) -> None:
        self.audit = audit
        self.termination = SimpleNamespace(name=termination)
        self.contact_completed = contact_completed
        self.to_post_impact_state_calls = 0

    def to_post_impact_state(self) -> Any:
        self.to_post_impact_state_calls += 1
        return _POST_IMPACT


def _build_audit(**overrides: float) -> SimpleNamespace:
    """Construct a closed energy ledger mirroring Tools ImpactIntervalResult."""
    terms: dict[str, float] = {
        "initial_kinetic_energy_j": 250.0,
        "final_kinetic_energy_j": 235.0,
        "dashpot_and_friction_dissipation_j": 13.0,
        "torsional_damping_dissipation_j": 0.5,
        "dissipated_energy_j": 13.5,
        "unilateral_release_energy_j": 0.4,
        "boundary_stored_energy_j": 1.0,
        "stored_contact_energy_initial_j": 0.0,
        "stored_contact_energy_final_j": 0.0,
        "integrated_normal_impulse_n_s": 3.4,
        "integrated_friction_impulse_n_s": 0.0,
        "linear_momentum_residual_n_s": 1.0e-14,
        "supported_momentum_residual_n_m_s": math.nan,
    }
    terms.update(overrides)
    terms.setdefault(
        "energy_residual_j", recompute_energy_residual(SimpleNamespace(**terms))
    )
    return SimpleNamespace(**terms)


def test_valid_separated_solve_passes_and_exports_separated() -> None:
    audit = _build_audit()
    result = _TrackedResult(audit, termination="SEPARATED", contact_completed=True)

    verdict = audit_impact_interval(result)
    assert verdict.qualified is True
    assert verdict.failures == ()

    post_state = qualified_post_impact_state(result)
    assert post_state is _POST_IMPACT
    assert result.to_post_impact_state_calls == 1

    exported = audit_and_export_impact_run(result)
    assert exported["qualified"] is True
    assert exported["contact_completed"] is True
    assert exported["termination"] == "SEPARATED"
    assert exported["completion_status"] == "separated"
    assert exported["audit"] == verdict.to_report()
    assert exported["post_impact_state"] is _POST_IMPACT
    assert result.to_post_impact_state_calls == 2


def test_mutation_slack_variable_zeroing_detected() -> None:
    # Mutate energy_residual_j to 0.0 while leaving a 5.777 J gap in the ledger terms
    audit = _build_audit(unilateral_release_energy_j=5.777, energy_residual_j=0.0)
    result = _TrackedResult(audit, termination="SEPARATED", contact_completed=True)

    verdict = audit_impact_interval(result)
    assert verdict.qualified is False
    assert any("does not equal the residual recomputed" in f for f in verdict.failures)

    with pytest.raises(
        ImpactAuditError, match="does not equal the residual recomputed"
    ) as exc_info:
        qualified_post_impact_state(result)
    assert exc_info.value.verdict.qualified is False
    assert exc_info.value.verdict.failures == verdict.failures

    exported = audit_and_export_impact_run(result)
    assert exported["qualified"] is False
    assert exported["completion_status"] == "separated"
    assert exported["post_impact_state"] is None
    # Ensure to_post_impact_state() was NEVER called because qualified is False
    assert result.to_post_impact_state_calls == 0


def test_mutation_truncated_contact_time_limit_rejected() -> None:
    # Truncated contact result (#9547): contact_completed=False, termination="TIME_LIMIT", stored_contact_energy_final_j=5.777
    audit = _build_audit(
        final_kinetic_energy_j=200.0,
        stored_contact_energy_final_j=5.777,
    )
    result = _TrackedResult(
        audit,
        termination="TIME_LIMIT",
        contact_completed=False,
    )

    with pytest.raises(ImpactAuditError, match="did not separate") as exc_info:
        qualified_post_impact_state(result)
    assert exc_info.value.verdict.termination == "TIME_LIMIT"
    assert exc_info.value.verdict.stored_contact_energy_final_j == 5.777

    exported = audit_and_export_impact_run(result)
    assert exported["qualified"] is False
    assert exported["contact_completed"] is False
    assert exported["termination"] == "TIME_LIMIT"
    assert exported["completion_status"] == "time_limit"
    assert exported["post_impact_state"] is None
    # Must NOT invoke to_post_impact_state() on an unqualified or truncated contact run
    assert result.to_post_impact_state_calls == 0


def test_mutation_physical_miss_no_contact() -> None:
    # Physical miss: contact_completed=False, termination="NO_CONTACT"
    audit = _build_audit()
    result = _TrackedResult(
        audit,
        termination="NO_CONTACT",
        contact_completed=False,
    )

    exported = audit_and_export_impact_run(result)
    assert exported["qualified"] is False
    assert exported["contact_completed"] is False
    assert exported["termination"] == "NO_CONTACT"
    assert exported["completion_status"] == "no_contact"
    assert exported["post_impact_state"] is None
    assert result.to_post_impact_state_calls == 0

    with pytest.raises(ImpactAuditError):
        qualified_post_impact_state(result)


def test_completion_status_incomplete_fallback() -> None:
    audit = _build_audit()
    result = _TrackedResult(
        audit,
        termination="STEP_LIMIT",
        contact_completed=False,
    )

    exported = audit_and_export_impact_run(result)
    assert exported["qualified"] is False
    assert exported["contact_completed"] is False
    assert exported["termination"] == "STEP_LIMIT"
    assert exported["completion_status"] == "incomplete"
    assert exported["post_impact_state"] is None
    assert result.to_post_impact_state_calls == 0
