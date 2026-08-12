"""Publication-boundary tests for the transmission-robustness package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"


def test_generated_evidence_and_figures_are_complete() -> None:
    record = json.loads(
        (ARTICLE / "data/transmission_robustness_study.json").read_text()
    )
    with np.load(
        ARTICLE / "data" / record["array_artifact"], allow_pickle=False
    ) as data:
        assert set(data.files) >= {
            "training_perturbations",
            "held_out_perturbations",
            "training_outcomes",
            "held_out_outcomes",
            "local_outcome_jacobian",
        }
    for stem in (
        "fig_transmission_pathway_framework",
        "fig_robust_speed_variability_pareto",
        "fig_clock_vs_state_perturbation_response",
        "fig_task_null_variability_map",
    ):
        for suffix in ("pdf", "svg"):
            path = ARTICLE / "figures" / f"{stem}.{suffix}"
            assert path.is_file() and path.stat().st_size > 1_000


def test_manuscript_preserves_adversarial_claim_boundaries() -> None:
    chapter = (ARTICLE / "chapters/_ch07c_transmission_robustness.qmd").read_text(
        encoding="utf-8"
    )
    audit = (ARTICLE / "ADVERSARIAL_TRANSMISSION_REVIEW.md").read_text(encoding="utf-8")
    combined = chapter + audit
    for required in (
        "torque sign does not",
        "pointwise drift",
        "nominal speed",
        "human self-stabilization",
        "Pareto",
        "task-null",
        "participant-held-out",
    ):
        assert required.casefold() in combined.casefold()


def test_every_gap_is_falsifiable_and_has_a_forward_action() -> None:
    record = json.loads(
        (ARTICLE / "data/transmission_robustness_study.json").read_text()
    )
    for gap in record["adversarial_gap_register"]:
        assert gap["severity"] in {"critical", "high", "medium", "low"}
        assert gap["counterexample"]
        assert gap["falsifier"]
        assert gap["path_forward"]


pytestmark = pytest.mark.scientific
