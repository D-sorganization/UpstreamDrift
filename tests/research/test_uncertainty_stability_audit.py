"""Contracts for uncertainty and control stability perturbations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy import audit_uncertainty_stability
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


def test_stability_audit_rejects_changed_study_array_with_unchanged_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A near-identical matrix must not inherit the released study's values."""
    for name in ("uncertainty_control_study.json", "uncertainty_control_study.npz"):
        (tmp_path / name).write_bytes(
            (audit_uncertainty_stability.DATA / name).read_bytes()
        )
    array_path = tmp_path / "uncertainty_control_study.npz"
    with np.load(array_path) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["standardized_sensitivity"] = arrays["standardized_sensitivity"].copy()
    arrays["standardized_sensitivity"][0, 0] = np.nextafter(
        arrays["standardized_sensitivity"][0, 0], np.inf
    )
    np.savez_compressed(array_path, **arrays)
    monkeypatch.setattr(audit_uncertainty_stability, "DATA", tmp_path)

    with pytest.raises(ValueError, match="input.*(identity|hash)|hash.*mismatch"):
        build_audit()


def test_stability_audit_rejects_changed_primary_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("uncertainty_control_study.json", "uncertainty_control_study.npz"):
        (tmp_path / name).write_bytes(
            (audit_uncertainty_stability.DATA / name).read_bytes()
        )
    record_path = tmp_path / "uncertainty_control_study.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["identifiability"]["coupled_parameter_screen"][
        "standardized_sensitivity_singular_values"
    ][2] += 1e-10
    record_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(audit_uncertainty_stability, "DATA", tmp_path)

    with pytest.raises(ValueError, match="input.*(identity|hash)|hash.*mismatch"):
        build_audit()
