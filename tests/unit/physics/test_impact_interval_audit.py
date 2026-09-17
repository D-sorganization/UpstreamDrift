"""Contract tests for the fail-closed impact-interval audit gate (#9548)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.shared.python.physics.impact_interval_audit import (
    IMPACT_AUDIT_LIMITATIONS,
    ImpactAuditError,
    ImpactAuditTolerances,
    ImpactAuditVerdict,
    audit_impact_interval,
    qualified_post_impact_state,
    recompute_energy_residual,
)

pytestmark = pytest.mark.unit

_POST_IMPACT = object()


def _audit(**overrides: float) -> SimpleNamespace:
    """A closed ledger mirroring the pinned Tools audit record."""
    terms = {
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


def _result(audit: SimpleNamespace, *, separated: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        audit=audit,
        termination=SimpleNamespace(name="SEPARATED" if separated else "TIME_LIMIT"),
        contact_completed=separated,
        to_post_impact_state=lambda: _POST_IMPACT,
    )


def test_closed_ledger_qualifies_and_reports_limitations() -> None:
    verdict = audit_impact_interval(_result(_audit()))
    assert verdict.qualified
    assert verdict.failures == ()
    assert verdict.energy_residual_j == pytest.approx(0.1)
    assert verdict.recomputed_energy_residual_j == verdict.energy_residual_j
    assert verdict.limitations == IMPACT_AUDIT_LIMITATIONS
    report = verdict.to_report()
    assert report["supported_momentum_residual_n_m_s"] is None
    assert report["limitations"] == list(IMPACT_AUDIT_LIMITATIONS)
    json.dumps(report)
    assert qualified_post_impact_state(_result(_audit())) is _POST_IMPACT


def test_unseparated_contact_with_stored_energy_is_blocked() -> None:
    audit = _audit(final_kinetic_energy_j=210.0, stored_contact_energy_final_j=32.5)
    with pytest.raises(ImpactAuditError, match="did not separate") as info:
        qualified_post_impact_state(_result(audit, separated=False))
    verdict = info.value.verdict
    assert not verdict.qualified
    assert verdict.termination == "TIME_LIMIT"
    assert verdict.stored_contact_energy_final_j == 32.5
    assert verdict.unilateral_release_energy_j == 0.4


def test_zeroed_residual_that_ignores_the_ledger_is_detected() -> None:
    # A provider that reports a zero residual while its own terms leave 0.1 J
    # unaccounted: the reported residual must equal the ledger identity.
    audit = _audit(energy_residual_j=0.0)
    verdict = audit_impact_interval(_result(audit))
    assert not verdict.qualified
    assert any("does not equal the residual recomputed" in f for f in verdict.failures)


def test_dissipation_must_be_the_sum_of_its_channels() -> None:
    audit = _audit(dissipated_energy_j=14.0)
    verdict = audit_impact_interval(_result(audit))
    assert any("sum of dashpot/friction" in f for f in verdict.failures)


def test_visible_residual_beyond_tolerance_blocks_qualified_output() -> None:
    audit = _audit(final_kinetic_energy_j=233.0)
    assert audit.energy_residual_j == pytest.approx(2.1)
    with pytest.raises(ImpactAuditError, match="exceeds 0.15 J"):
        qualified_post_impact_state(_result(audit))
    loose = ImpactAuditTolerances(energy_residual_j=5.0)
    assert audit_impact_interval(_result(audit), loose).qualified


def test_momentum_diagnostics_are_checked_separately() -> None:
    free = audit_impact_interval(_result(_audit(linear_momentum_residual_n_s=1e-6)))
    assert [f for f in free.failures if "free linear-momentum" in f]
    supported = audit_impact_interval(
        _result(_audit(supported_momentum_residual_n_m_s=0.01))
    )
    assert [f for f in supported.failures if "supported angular-momentum" in f]
    assert supported.to_report()["supported_momentum_residual_n_m_s"] == 0.01
    # A fixed attachment carries reaction impulse: the free linear balance
    # does not apply to supported boundaries and must not fail them.
    pinned = audit_impact_interval(
        _result(
            _audit(
                linear_momentum_residual_n_s=0.035,
                supported_momentum_residual_n_m_s=1e-5,
            )
        )
    )
    assert pinned.qualified, pinned.failures


def test_preconditions_are_enforced() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ImpactAuditTolerances(energy_residual_j=0.0)
    with pytest.raises(TypeError, match="tolerances"):
        audit_impact_interval(_result(_audit()), tolerances=0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="audit"):
        audit_impact_interval(SimpleNamespace())
    with pytest.raises(TypeError, match="ledger term"):
        audit_impact_interval(_result(SimpleNamespace(initial_kinetic_energy_j=1.0)))
    with pytest.raises(ValueError, match="must be finite"):
        audit_impact_interval(_result(_audit(energy_residual_j=math.inf)))
    verdict = audit_impact_interval(_result(_audit()))
    with pytest.raises(ValueError, match="cannot carry failures"):
        replace(verdict, failures=("x",))
    with pytest.raises(ValueError, match="must name a failure"):
        ImpactAuditVerdict(
            **{**verdict.__dict__, "qualified": False, "failures": ()},
        )
