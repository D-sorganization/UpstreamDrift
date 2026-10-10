"""F04 evidence cannot promote failed or contradictory native experiments."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _properties() -> tuple[dict[str, object], dict[str, object]]:
    digest = "a" * 64
    tuning: dict[str, object] = {
        **dict.fromkeys(
            (
                "source_model_sha256",
                "initial_state_sha256",
                "policy_sha256",
                "time_grid_sha256",
                "teacher_input_sha256",
            ),
            digest,
        ),
        "initial_objective": 0.012,
        "final_objective": 0.008,
        "parameters": [1.0, 1.2],
        "holdout_phase_group_losses": [[0.02, 0.003], [0.007, 0.001]],
        "holdout_initial_rmse_rad": 0.12,
        "holdout_final_rmse_rad": 0.08,
        "cross_jacobian_norms": [[0.002, 0.0004], [0.0002, 0.001]],
        "singular_values": [0.002, 0.0008],
        "rank_deficient": False,
        "uncertainty_status": "unavailable_no_resampling_or_capture_noise_model",
        "train_evaluations": 120,
        "holdout_evaluations": 3,
        "distinct_applied_inputs": 100,
        "checkpoint_reasons": ["accepted", "accepted", "budget_exhausted"],
        "max_full_state_replay_error": 0.0,
        "total_wall_s": 6.0,
        "total_cpu_s": 5.0,
    }
    intervention: dict[str, object] = {
        "response_rad_per_nm": [[0.16, -0.05], [-0.06, 0.31]],
        "channel_ids": ["hip_torque", "knee_torque"],
        "response_joint_ids": ["hip", "knee"],
        "applied_input_sha256": [char * 64 for char in "bcde"],
        "initial_state_sha256": digest,
        "policy_sha256": digest,
        "time_grid_sha256": digest,
        "interpretation": "synthetic_plant_input_intervention_not_human_control",
        "intervention_wall_s": 0.1,
    }
    return tuning, intervention


def _junit(
    path: Path,
    tuning: dict[str, object],
    intervention: dict[str, object],
    *,
    skipped: bool = False,
) -> None:
    skip = "<skipped/>" if skipped else ""
    cases = (
        (
            "test_native_tuning_uses_train_only_and_independent_holdout",
            "f04_native_tuning_evidence",
            tuning,
        ),
        (
            "test_paired_native_motor_intervention_is_distinct_from_residual_covariance",
            "f04_native_intervention_evidence",
            intervention,
        ),
    )
    content = (
        "<testsuite>"
        + "".join(
            f'<testcase name="{name}">{skip if index == 0 else ""}'
            f'<properties><property name="{property_name}" value="{escape(json.dumps(value), quote=True)}"/>'
            "</properties></testcase>"
            for index, (name, property_name, value) in enumerate(cases)
        )
        + "</testsuite>"
    )
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    ("target", "change", "message"),
    [
        ("tuning", {"final_objective": 0.02}, "improvement"),
        ("tuning", {"holdout_final_rmse_rad": 0.13}, "held-out"),
        ("tuning", {"rank_deficient": "unknown"}, "rank"),
        ("tuning", {"uncertainty_status": "known"}, "uncertainty"),
        ("tuning", {"max_full_state_replay_error": 0.01}, "replay"),
        ("intervention", {"policy_sha256": "f" * 64}, "policy"),
        ("intervention", {"response_rad_per_nm": [[1, 0], [0, 1]]}, "cross"),
    ],
)
def test_receipt_rejects_invalid_coupling_claim(
    tmp_path: Path, target: str, change: dict[str, object], message: str
) -> None:
    from scripts.f04_native_coupling_receipt import extract_native_evidence

    tuning, intervention = _properties()
    (tuning if target == "tuning" else intervention).update(change)
    path = tmp_path / "native.xml"
    _junit(path, tuning, intervention)
    with pytest.raises(ValueError, match=message):
        extract_native_evidence(path)


def test_receipt_requires_two_passed_native_tests(tmp_path: Path) -> None:
    from scripts.f04_native_coupling_receipt import extract_native_evidence

    tuning, intervention = _properties()
    path = tmp_path / "native.xml"
    _junit(path, tuning, intervention, skipped=True)
    with pytest.raises(ValueError, match="passed"):
        extract_native_evidence(path)
