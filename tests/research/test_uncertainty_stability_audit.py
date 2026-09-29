"""Contracts for uncertainty and control stability perturbations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.audit_uncertainty_stability import (
    build_audit,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
AUDIT = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/uncertainty_control_stability_audit.json"
)


def test_committed_stability_audit_is_fresh() -> None:
    assert json.loads(AUDIT.read_text(encoding="utf-8")) == build_audit()


def test_small_sample_conclusions_expose_perturbation_stability() -> None:
    audit = build_audit()
    prcc = audit["prcc_leave_one_out"]
    pareto = audit["held_out_pareto_jackknife"]
    ranks = audit["identifiability_threshold_sensitivity"]

    assert prcc["replicates"] == 24
    assert all(sum(counts.values()) == 24 for counts in prcc["leader_counts"].values())
    assert pareto["replicates"] == 6
    assert set(pareto["membership_counts"]) == set(pareto["point_members"])
    assert ranks["rank_by_fraction_of_largest"]["0.05"] == 6
    assert "cannot quantify population uncertainty" in audit["interpretation"]
