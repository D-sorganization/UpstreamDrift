"""Contracts for the delayed-observer and recovery experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _evidence() -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads((DATA / "observer_recovery_study.json").read_text())
    with np.load(DATA / record["array_artifact"], allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    return record, arrays


def test_design_has_common_held_out_disturbances_and_observer_uncertainty() -> None:
    record, arrays = _evidence()

    assert record["registered_before_preferred_result"] is True
    assert record["design"]["paired_common_random_numbers"] is True
    assert record["design"]["held_out_perturbations"] >= 12
    assert record["observer_conditions"]["delayed_noisy"]["delay_s"] > 0.0
    assert record["observer_conditions"]["delayed_noisy"]["angle_noise_sd_rad"] > 0.0
    assert arrays["normalized_error_trajectories"].ndim == 3
    assert np.all(np.isfinite(arrays["normalized_error_trajectories"]))


def test_recovery_is_trajectory_level_and_kept_separate_from_sensitivity() -> None:
    record, arrays = _evidence()

    required = {
        "initial_normalized_error",
        "terminal_error_ratio",
        "minimum_error_ratio",
        "recovery_time_s",
        "returned_to_viable_set",
        "terminal_delivery_speed_m_s",
        "peak_hand_force_n",
        "effort_proxy_nms",
    }
    assert required <= set(record["metric_names"])
    assert record["recovery_definition"]["requires_sustained_error_decay"] is True
    assert record["claim_status"]["low_sensitivity_is_recovery"] == "rejected"
    assert arrays["metrics"].shape[-1] == len(record["metric_names"])


def test_claims_remain_model_bounded_and_report_adverse_costs() -> None:
    record, _ = _evidence()

    assert record["claim_status"]["human_self_correction"] == "untested"
    assert record["claim_status"]["universal_timing_advantage"] == "unsupported"
    assert record["adverse_costs"] == ["peak_hand_force_n", "effort_proxy_nms"]
    assert record["limitations"]
    assert all(
        summary["recovery_fraction"] >= 0.0 for summary in record["summaries"].values()
    )
    source = (
        ROOT / "scripts/research/proximal_distal_energy/run_observer_recovery_study.py"
    )
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


pytestmark = pytest.mark.scientific
