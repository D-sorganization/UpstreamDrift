"""Evidence-schema contracts for the trajectory-level strategy study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_shoulder_velocity_strategy_study import (
    run_study,
    write_outputs,
)

pytestmark = pytest.mark.scientific


def test_strategy_study_keeps_every_attempt_and_pareto_tradeoff() -> None:
    record, arrays = run_study()

    assert record["schema_version"] == "shoulder-velocity-strategy-evidence-v2"
    assert len(record["programs"]) == 60
    assert record["valid_impact_count"] > 0
    assert record["pareto_program_indices"]
    assert arrays["impact_speed_m_s"].shape == (60,)
    assert np.all(np.isfinite(arrays["transfer_work_closure_residual_j"]))
    assert np.max(np.abs(arrays["transfer_work_closure_residual_j"])) < 1e-8
    assert record["pareto_diagnostics"]["valid_program_count"] == 26
    assert record["pareto_diagnostics"]["nondominated_program_count"] == 21
    assert record["pareto_diagnostics"]["nondominated_fraction"] == pytest.approx(
        21 / 26
    )
    sensitivity = record["fixed_program_timestep_sensitivity"]
    assert sensitivity["reference_dt_s"] == pytest.approx(0.001)
    assert sensitivity["refined_dt_s"] == pytest.approx(0.0005)
    assert sensitivity["same_valid_impact_classification"] is True
    assert sensitivity["absolute_impact_speed_difference_m_s"] < 0.01
    assert sensitivity["absolute_total_grip_work_difference_j"] < 0.1


def test_strategy_study_preserves_claim_boundaries() -> None:
    record, _ = run_study()

    assert record["claim_status"]["proximal_link_velocity"] == "tested_model_coordinate"
    assert record["claim_status"]["anatomical_shoulder_velocity"] == "not_tested"
    assert record["claim_status"]["torso_rotation_strategy"] == "not_tested"
    assert record["claim_status"]["universal_coaching_strategy"] == "unsupported"
    assert "association" in record["analysis_boundary"].lower()
    assert "selection" in record["analysis_boundary"].lower()


def test_strategy_association_records_design_adequacy() -> None:
    record, _ = run_study()

    association = record["associations_valid_impact_only"]
    design = association["standardized_speed_regression_diagnostics"]
    assert design["observation_count"] == 26
    assert design["parameter_count_including_intercept"] == 5
    assert design["matrix_rank"] == 5
    assert design["condition_number"] < 10.0
    assert design["selection_rule"] == "valid_impact_only"
    assert association["release_velocity_vs_peak_force_pearson_r"] == pytest.approx(
        -0.6085210777
    )


def test_strategy_study_hashes_every_declared_computational_dependency() -> None:
    record, _ = run_study()

    required = {
        "scripts/research/proximal_distal_energy/double_pendulum_attribution.py",
        "scripts/research/proximal_distal_energy/interaction_forces.py",
        "scripts/research/proximal_distal_energy/run_experiments.py",
        "scripts/research/proximal_distal_energy/run_shoulder_velocity_strategy_study.py",
        "scripts/research/proximal_distal_energy/shoulder_velocity_strategy_search.py",
        "scripts/research/proximal_distal_energy/swing_model.py",
        "src/shared/python/biomechanics/drift_control_transfer.py",
        "src/shared/python/simulation_backends/model_params.py",
    }
    assert required <= set(record["source_sha256"])


def test_strategy_artifacts_are_byte_deterministic(tmp_path) -> None:
    first = write_outputs(tmp_path / "first")
    second = write_outputs(tmp_path / "second")

    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()
