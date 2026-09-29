"""Regression controls for the cross-repository terminology contract."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/research/proximal_distal_energy_transfer"
pytestmark = pytest.mark.scientific


def test_publication_contract_declares_normative_cross_repository_authority() -> None:
    text = (PAPER / "CONTROL_AFFINE_TERMINOLOGY_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    assert "AffineDrift/blob/main/NOTATION.md" in text
    assert "complete autonomous vector field" in text
    assert "zero-velocity control-preserved" in text
    assert "realized drift-to-input ratio (DIR)" in text


def test_canonical_counterfactual_chapters_do_not_restore_old_zvcf_meaning() -> None:
    paths = (
        PAPER / "chapters/_ch01_introduction.qmd",
        PAPER / "chapters/_ch03b_hand_path_attribution.qmd",
        PAPER / "chapters/_ch03c_ground_reaction_drift.qmd",
        PAPER / "chapters/_ch04_counterfactuals.qmd",
        PAPER / "HAND_PATH_ATTRIBUTION_CONTRACT.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = (
        "ZVCF preserves the applied control",
        "ZVCF preserves applied control",
        "same configuration and applied inputs with velocity set to zero",
        "ZVCF is the pure control",
    )
    for phrase in forbidden:
        assert phrase.casefold() not in combined.casefold()


def test_schema_v3_keeps_canonical_and_control_preserved_arrays_separate() -> None:
    source = (ROOT / "src/shared/python/simulation_backends/ztcf_zvcf.py").read_text(
        encoding="utf-8"
    )
    assert '_ANALYSIS_SCHEMA_VERSION = "3.0.0"' in source
    assert '"zvcf_acceleration"' in source
    assert '"zero_velocity_control_preserved_acceleration"' in source


def test_realized_input_ratio_is_not_published_as_dcr() -> None:
    analyzer = (ROOT / "src/tools/drift_control/analyzer.py").read_text(
        encoding="utf-8"
    )
    education = (ROOT / "src/shared/python/ai/education.py").read_text(encoding="utf-8")
    assert "realized drift-to-input (DIR)" in analyzer
    assert "A realized-input denominator defines DIR, not DCR" in education
    assert "DCR = ||control|| / (||drift|| + ||control||)" not in education
