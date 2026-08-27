"""Evidence and reproduction contract tests for #9124."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.run_bounded_event_reachability import (
    run_qualification,
)

pytestmark = pytest.mark.scientific


def test_bounded_event_reachability_evidence_passes_all_gates() -> None:
    evidence = run_qualification()

    assert evidence["status"] == "PASSED"
    assert evidence["small_amplitude_max_discrepancy"] < 0.20
    assert evidence["finite_amplitude_saturation_detected"] is True
    assert evidence["both_channels_feasible_count"] >= 1
    assert (
        "scientific" in evidence["inference_boundary"].lower()
        or "analytical" in evidence["inference_boundary"].lower()
    )
