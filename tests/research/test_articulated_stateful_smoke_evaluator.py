"""Governed JSON evaluator tests for the stateful distributed countermodel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    build_registered_cases,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_study import (
    ForwardAttributionStudyPlan,
)
from scripts.research.proximal_distal_energy.articulated_stateful_smoke_evaluator import (
    evaluate_stateful_smoke_case,
    run_registered_stateful_smoke,
)

SOURCE_DATA_SHA256 = "9fa4364571ba5535995c63226289c0711ee1ebf37c58b7a3b4e4d14a98561779"


def _manifest() -> dict[str, object]:
    manifest = ForwardAttributionStudyPlan(
        source_revision="a" * 40,
        source_data_sha256=SOURCE_DATA_SHA256,
    ).to_manifest()
    design = manifest["design"]
    assert isinstance(design, dict)
    design["stateful_contact_law"] = {
        "name": "distributed_elastic_perfectly_plastic_coulomb",
        "station_count_per_hand": 3,
        "station_width_m": 0.03,
        "friction_coefficient": 0.4,
        "tangential_stiffness_n_m": 600.0,
        "slack_distance_m": 0.0,
        "initial_preload_vector_m": [0.0, 0.0, 0.0],
        "static_stick_modeled": True,
    }
    tolerances = manifest["tolerances"]
    assert isinstance(tolerances, dict)
    tolerances.update(
        {
            "coupling_work_relative": 0.01,
            "virtual_power_w": 1.0e-10,
            "constitutive_ledger_j": 1.0e-12,
        }
    )
    return manifest


def _nominal_mujoco_case(manifest: dict[str, object]):
    return next(
        case
        for case in build_registered_cases(manifest)
        if case.engine == "mujoco"
        and case.variant == "nominal"
        and case.time_step_s == 0.0005
    )


def test_stateful_evaluator_retains_json_safe_full_histories() -> None:
    manifest = _manifest()
    result = evaluate_stateful_smoke_case(_nominal_mujoco_case(manifest), manifest)

    assert result["estimand"] == "stateful_contact_counterfactual_trajectory"
    assert result["contact_model"]["static_stick_modeled"] is True
    assert result["closure"]["constitutive_ledger_residual_j"] <= 1.0e-12
    assert result["closure"]["failure_codes"] == []
    assert result["outcomes"]["clubhead_speed_m_s"] > 0.0
    histories = result["histories"]
    assert len(histories["node_time_s"]) == 11
    assert len(histories["interval_time_start_s"]) == 10
    assert np.asarray(histories["node_elastic_displacement_m"]).shape == (
        11,
        2,
        3,
        3,
    )
    assert result["claim_boundary"]["human_or_anatomical_inference"] is False
    json.dumps(result, allow_nan=False)


def test_open_contact_releases_registered_preload_without_human_inference() -> None:
    manifest = _manifest()
    design = manifest["design"]
    assert isinstance(design, dict)
    law = design["stateful_contact_law"]
    assert isinstance(law, dict)
    law["slack_distance_m"] = 0.01
    law["initial_preload_vector_m"] = [0.001, 0.0, 0.0]

    result = evaluate_stateful_smoke_case(_nominal_mujoco_case(manifest), manifest)

    assert result["regimes"] == {"open": 60}
    assert result["outcomes"]["total_release_dissipation_j"] > 0.0
    assert result["claim_boundary"]["human_or_coaching_inference"] is False


def test_source_hash_mismatch_fails_closed() -> None:
    manifest = _manifest()
    identity = manifest["identity"]
    assert isinstance(identity, dict)
    identity["source_data_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source-data SHA-256"):
        evaluate_stateful_smoke_case(_nominal_mujoco_case(manifest), manifest)


def test_stateful_runner_uses_atomic_registered_case_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        "scripts.research.proximal_distal_energy."
        "articulated_stateful_smoke_evaluator.evaluate_stateful_smoke_case",
        lambda case, _manifest: {"case_key": case.case_key},
    )

    checkpoints = run_registered_stateful_smoke(
        manifest=manifest,
        execution_revision="c" * 40,
        checkpoint_dir=tmp_path,
    )

    assert len(checkpoints) == 42
    assert all(checkpoint.status == "completed" for checkpoint in checkpoints)
