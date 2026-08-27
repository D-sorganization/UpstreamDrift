"""Evidence and reproduction contract tests for #9125."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.run_event_topology_robustness import (
    run_qualification,
)

pytestmark = pytest.mark.scientific


def test_event_topology_robustness_evidence_passes_all_gates() -> None:
    evidence = run_qualification()

    assert evidence["status"] == "PASSED"
    assert evidence["zero_perturbation_reproduces_nominal"] is True
    assert evidence["step_refinement_stable"] is True
    assert evidence["channel_coverage_passed"] is True
    assert evidence["noise_robustness_retained_unique_fraction"] >= 0.75
    assert (
        "scientific" in evidence["inference_boundary"].lower()
        or "analytical" in evidence["inference_boundary"].lower()
    )
