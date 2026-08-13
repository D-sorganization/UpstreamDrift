"""Evidence-schema tests for the shoulder-velocity transfer atlas."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_shoulder_velocity_transfer_study import (
    run_study,
)

pytestmark = pytest.mark.scientific


def test_study_covers_all_declared_phases_and_velocity_constraints() -> None:
    record, arrays = run_study()

    assert record["schema_version"] == "shoulder-velocity-transfer-evidence-v1"
    assert record["model_tier"] == "exact_planar_double_pendulum"
    assert set(record["phase_labels"]) == {
        "Transition",
        "Early Downswing",
        "Mid-Downswing",
        "Delivery and Release",
        "Pre-Impact",
    }
    assert set(record["velocity_constraints"]) == {
        "preserve_relative_club_rate",
        "preserve_absolute_club_rate",
    }
    assert arrays["proximal_velocity_rad_s"].size == 90
    assert np.max(np.abs(arrays["acceleration_closure_residual_rad_s2"])) < 1e-10
    assert np.max(np.abs(arrays["force_closure_residual_n"])) < 1e-10


def test_study_keeps_torso_and_coaching_claims_out_of_scope() -> None:
    record, _ = run_study()

    assert record["claim_status"]["anatomical_shoulder_strategy"] == "untested"
    assert record["claim_status"]["torso_rotation_strategy"] == "untested"
    assert record["claim_status"]["universal_coaching_instruction"] == "unsupported"
    assert "pointwise" in record["counterfactual_kind"]
    assert len(record["falsification_tests"]) >= 5
