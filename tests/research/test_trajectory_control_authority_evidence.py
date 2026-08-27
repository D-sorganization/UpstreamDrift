"""Evidence and reproduction contract tests for #9123."""

from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.run_trajectory_control_authority import (
    run_qualification,
)

pytestmark = pytest.mark.scientific


def test_trajectory_control_authority_evidence_passes_all_gates() -> None:
    evidence = run_qualification()

    assert evidence["status"] == "PASSED"
    assert evidence["is_transverse"] is True
    assert evidence["full_rank_both"] == 4
    assert evidence["tangent_rank_both"] == 3
    assert evidence["full_rank_zero"] == 0
    assert evidence["tangent_rank_zero"] == 0
    assert evidence["additivity_residual_norm"] < 1e-10
    assert evidence["pulse_agreement_relative_error"] < 0.05
    assert (
        "scientific" in evidence["inference_boundary"].lower()
        or "analytical" in evidence["inference_boundary"].lower()
    )
