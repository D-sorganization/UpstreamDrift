"""Evidence-schema tests for the shoulder-velocity transfer atlas."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_shoulder_velocity_transfer_study import (
    run_study,
    write_outputs,
)

pytestmark = pytest.mark.scientific


def test_study_covers_all_declared_phases_and_velocity_constraints() -> None:
    record, arrays = run_study()

    assert record["schema_version"] == "shoulder-velocity-transfer-evidence-v2"
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

    sweep = record["velocity_sweep_contract"]
    assert sweep["reference_state_included"] is True
    assert sweep["offset_fraction_of_scale"] == pytest.approx(0.75)
    assert sweep["minimum_scale_rad_s"] == pytest.approx(4.0)

    for summary in record["phase_summaries"]:
        selected = [
            row
            for row in record["rows"]
            if row["phase"] == summary["phase"]
            and row["velocity_constraint"] == summary["velocity_constraint"]
        ]
        proximal_rates = np.asarray(
            [row["proximal_velocity_rad_s"] for row in selected]
        )
        assert np.any(
            np.isclose(proximal_rates, summary["reference_proximal_rate_rad_s"])
        )
        assert summary["reference_state_in_grid"] is True
        assert 0.0 <= summary["drift_power_linear_r_squared"] <= 1.0
        assert np.isfinite(summary["drift_power_endpoint_delta_w"])
        assert np.isfinite(summary["reference_centered_drift_power_slope_w_per_rad_s"])


def test_study_keeps_torso_and_coaching_claims_out_of_scope() -> None:
    record, _ = run_study()

    assert record["claim_status"]["anatomical_shoulder_strategy"] == "untested"
    assert record["claim_status"]["torso_rotation_strategy"] == "untested"
    assert record["claim_status"]["universal_coaching_instruction"] == "unsupported"
    assert "pointwise" in record["counterfactual_kind"]
    assert len(record["falsification_tests"]) >= 5


def test_study_hashes_every_declared_computational_dependency() -> None:
    record, _ = run_study()

    required = {
        "scripts/research/proximal_distal_energy/interaction_forces.py",
        "scripts/research/proximal_distal_energy/run_experiments.py",
        "scripts/research/proximal_distal_energy/run_shoulder_velocity_transfer_study.py",
        "scripts/research/proximal_distal_energy/shoulder_velocity_transfer.py",
        "scripts/research/proximal_distal_energy/swing_model.py",
        "scripts/research/proximal_distal_energy/torque_programs.py",
        "src/shared/python/simulation_backends/model_params.py",
    }
    assert required <= set(record["source_sha256"])


def test_study_artifacts_are_byte_deterministic(tmp_path) -> None:
    first = write_outputs(tmp_path / "first")
    second = write_outputs(tmp_path / "second")

    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()
