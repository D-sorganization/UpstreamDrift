from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.research.proximal_distal_energy import (
    make_stateful_distributed_figure as figure,
)


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data"
    / "articulated_stateful_distributed_smoke/summary.json"
)


def test_load_summary_preserves_failed_refinement_and_unavailable_parity() -> None:
    summary = figure.load_summary(SUMMARY)

    assert summary["counts"] == {
        "completed": 27,
        "failed": 0,
        "registered": 54,
        "unavailable": 27,
    }
    assert summary["promotion"]["eligible"] is False
    assert "cross_engine_parity_unavailable" in summary["promotion"]["failure_codes"]
    failed = figure.completed_groups(summary, passed=False)
    assert [group["variant"] for group in failed] == [
        "frictionless_killswitch",
        "low_friction_slip_probe",
    ]


def test_render_figure_writes_pdf_and_svg(tmp_path: Path) -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    output = tmp_path / "stateful"

    figure.render_figure(summary, output)

    assert output.with_suffix(".pdf").stat().st_size > 1_000
    svg = output.with_suffix(".svg").read_text(encoding="utf-8")
    assert "Stateful Distributed-Grip Falsification" in svg
    assert svg.endswith("\n")
    plt.close("all")
