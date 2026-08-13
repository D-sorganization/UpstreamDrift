"""Evidence-schema contracts for the trajectory-level strategy study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_shoulder_velocity_strategy_study import (
    run_study,
)

pytestmark = pytest.mark.scientific


def test_strategy_study_keeps_every_attempt_and_pareto_tradeoff() -> None:
    record, arrays = run_study()

    assert record["schema_version"] == "shoulder-velocity-strategy-evidence-v1"
    assert len(record["programs"]) == 60
    assert record["valid_impact_count"] > 0
    assert record["pareto_program_indices"]
    assert arrays["impact_speed_m_s"].shape == (60,)
    assert np.all(np.isfinite(arrays["transfer_work_closure_residual_j"]))
    assert np.max(np.abs(arrays["transfer_work_closure_residual_j"])) < 1e-8


def test_strategy_study_preserves_claim_boundaries() -> None:
    record, _ = run_study()

    assert record["claim_status"]["proximal_link_velocity"] == "tested_model_coordinate"
    assert record["claim_status"]["anatomical_shoulder_velocity"] == "not_tested"
    assert record["claim_status"]["torso_rotation_strategy"] == "not_tested"
    assert record["claim_status"]["universal_coaching_strategy"] == "unsupported"
    assert "association" in record["analysis_boundary"].lower()
