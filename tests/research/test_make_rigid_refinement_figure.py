"""Deterministic figure checks for the rigid-refinement extension."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.research.proximal_distal_energy import make_rigid_refinement_figure


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "articulated_rigid_refinement/summary.json"
)


def test_load_summary_retains_three_state_dependent_failures() -> None:
    summary = make_rigid_refinement_figure.load_summary(SUMMARY)
    groups = make_rigid_refinement_figure.completed_groups(summary)

    assert len(groups) == 36
    assert sum(not group["passes"] for group in groups) == 3


def test_render_figure_writes_pdf_and_svg(tmp_path: Path) -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    output = tmp_path / "rigid-refinement"

    make_rigid_refinement_figure.render_figure(summary, output)

    assert output.with_suffix(".pdf").stat().st_size > 1_000
    svg = output.with_suffix(".svg").read_text(encoding="utf-8")
    assert "Rigid Refinement Across Screening States" in svg
    assert svg.endswith("\n")
    plt.close("all")
