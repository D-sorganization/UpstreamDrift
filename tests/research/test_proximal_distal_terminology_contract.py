"""Publication controls for the proximal--distal terminology contract."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
pytestmark = pytest.mark.unit


def test_normative_terminology_contract_covers_high_risk_terms() -> None:
    text = (ARTICLE / "TERMINOLOGY_AND_CONVENTIONS.md").read_text(encoding="utf-8")
    defined_terms = {
        line.split("|", maxsplit=2)[1].strip()
        for line in text.splitlines()
        if line.startswith("|") and line.count("|") >= 2
    }
    for term in (
        "Proximal-to-distal sequence",
        "Energy transfer",
        "Interaction force",
        "Drift",
        "Control contribution",
        "Negative torque",
        "Negative power",
        "Preload",
        "Slack",
        "Passive after killswitch",
        "Model support",
    ):
        assert term in defined_terms


def test_advanced_chapter_declares_adapter_and_biology_boundaries() -> None:
    text = (ARTICLE / "chapters/_ch07b_frames_biology_engines.qmd").read_text(
        encoding="utf-8"
    )
    assert "representation" in text
    assert "does not execute five dynamics engines" in text
    assert "does not identify a unique muscle activation vector" in text


def test_falsification_matrix_registers_new_hypotheses() -> None:
    text = (ARTICLE / "MODEL_COMPLETION_FALSIFICATION_MATRIX.md").read_text(
        encoding="utf-8"
    )
    assert "H7: Frame and Adapter Consistency" in text
    assert "H8: Biological Redundancy and History" in text
