"""Contracts for the independent original-results reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.audit_results_chapter import build_audit

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
AUDIT = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/results_chapter_audit.json"
)


def test_committed_results_audit_is_fresh() -> None:
    assert json.loads(AUDIT.read_text(encoding="utf-8")) == build_audit()


def test_original_results_reconcile_without_promoting_scope() -> None:
    audit = build_audit()
    assert audit["attempted_programs"] == 92
    assert audit["impact_status_counts"] == {
        "accepted_registered_delivery_zone": 63,
        "crossing_outside_registered_delivery_zone": 29,
    }
    assert audit["summary_representatives_match"] is True
    assert audit["energy_balance_residual_max_w"] < 1e-10
    assert audit["parameter_case_count"] == 13
    assert audit["all_parameter_cases_preserve_ordering"] is True
    assert audit["impact_family_ordering_preserved"] is True
    assert audit["smooth_all_orderings_preserved"] is True
    assert "not biological validation" in audit["interpretation"]
