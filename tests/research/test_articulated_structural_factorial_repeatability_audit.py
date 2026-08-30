"""Exact repeatability classification for one registered structural case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_evidence import (
    EVIDENCE_SIDECAR_SCHEMA,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_repeatability_audit import (
    AUDIT_SCHEMA,
    audit_repeatability_probe,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    StructuralCase,
    StructuralEvaluation,
    build_launch_manifest,
    run_serial_cases,
)

pytestmark = pytest.mark.scientific
HASHES = {
    "closed_state_npz": "1" * 64,
    "shaft_structural_basis_json": "2" * 64,
    "shaft_structural_basis_npz": "3" * 64,
    "shaft_atlas_json": "4" * 64,
    "shaft_atlas_npz": "5" * 64,
    "ground_atlas_json": "6" * 64,
    "ground_atlas_npz": "7" * 64,
}


def _plan() -> dict[str, object]:
    plan = StructuralFactorialPlan(
        design_authority_revision="a" * 40, authority_sha256=HASHES
    ).to_manifest()
    design = dict(plan["design"])  # type: ignore[arg-type]
    design["states"] = design["states"][:1]
    design["velocity_factors"] = design["velocity_factors"][:1]
    design["engines"] = ["mujoco"]
    design["time_steps_s"] = design["time_steps_s"][:1]
    design["registered_engine_attempt_count"] = 16
    design["expected_native_attempt_count"] = 16
    plan["design"] = design
    return plan


def _evaluation(metric: float, q_value: float) -> StructuralEvaluation:
    arrays = {
        "time_s": np.array([0.0, 0.1]),
        "q": np.full((2, 20), q_value),
        "qd": np.full((2, 20), q_value),
        "elastic_coordinates": np.zeros((2, 0)),
        "base_coordinates": np.zeros((2, 0)),
        "net_club_force_n": np.full((2, 3), q_value),
        "maximum_station_force_n": np.full(2, q_value),
        "active_station_count": np.zeros(2, dtype=int),
        "ground_force_n": np.full((2, 3), q_value),
    }
    return StructuralEvaluation(
        result={
            "metric": metric,
            "evidence_sidecar_schema": EVIDENCE_SIDECAR_SCHEMA,
        },
        parity_arrays=arrays,
    )


def _run(
    *,
    plan: dict[str, object],
    directory: Path,
    revision: str,
    metric: float,
    q_value: float,
    single_case: bool = False,
) -> tuple[dict[str, object], Path]:
    launch = build_launch_manifest(plan=plan, execution_revision=revision)

    def evaluator(_case: StructuralCase) -> StructuralEvaluation:
        return _evaluation(metric, q_value)

    run_serial_cases(
        plan=plan,
        launch=launch,
        checkpoint_dir=directory,
        evaluator=evaluator,
        case_stop=1 if single_case else None,
    )
    return launch, directory


def _audit(
    tmp_path: Path, *, repeat_metric: float, repeat_q: float
) -> dict[str, object]:
    plan = _plan()
    legacy_launch, legacy_dir = _run(
        plan=plan,
        directory=tmp_path / "legacy",
        revision="b" * 40,
        metric=1.0,
        q_value=0.0,
    )
    enriched_launch, enriched_dir = _run(
        plan=plan,
        directory=tmp_path / "enriched",
        revision="c" * 40,
        metric=2.0,
        q_value=1.0,
    )
    repeat_launch, repeat_dir = _run(
        plan=plan,
        directory=tmp_path / "repeat",
        revision="c" * 40,
        metric=repeat_metric,
        q_value=repeat_q,
        single_case=True,
    )
    return audit_repeatability_probe(
        legacy_plan=plan,
        legacy_launch=legacy_launch,
        legacy_checkpoint_dir=legacy_dir,
        enriched_plan=plan,
        enriched_launch=enriched_launch,
        enriched_checkpoint_dir=enriched_dir,
        repeat_plan=plan,
        repeat_launch=repeat_launch,
        repeat_checkpoint_dir=repeat_dir,
        case_index=0,
    )


def test_repeatability_probe_classifies_enriched_only_match(tmp_path: Path) -> None:
    audit = _audit(tmp_path, repeat_metric=2.0, repeat_q=1.0)

    assert audit["schema_version"] == AUDIT_SCHEMA
    assert audit["classification"] == "deterministic_source_delta_supported"
    assert audit["matches"]["enriched"] is True
    assert audit["matches"]["legacy"] is False
    assert audit["claim_boundary"]["outcome_values_reported"] is False
    assert audit["claim_boundary"]["campaign_promotion_authorized"] is False


def test_repeatability_probe_classifies_legacy_only_match(tmp_path: Path) -> None:
    audit = _audit(tmp_path, repeat_metric=1.0, repeat_q=0.0)

    assert audit["classification"] == "first_enriched_replay_anomaly_supported"
    assert audit["matches"] == {"enriched": False, "legacy": True}


def test_repeatability_probe_classifies_match_to_neither(tmp_path: Path) -> None:
    audit = _audit(tmp_path, repeat_metric=3.0, repeat_q=2.0)

    assert audit["classification"] == "cross_run_nonrepeatability_demonstrated"
    assert audit["matches"] == {"enriched": False, "legacy": False}


def test_repeatability_probe_retains_nondiscriminating_authorities(
    tmp_path: Path,
) -> None:
    plan = _plan()
    launch, legacy_dir = _run(
        plan=plan,
        directory=tmp_path / "legacy",
        revision="b" * 40,
        metric=1.0,
        q_value=0.0,
    )
    _, enriched_dir = _run(
        plan=plan,
        directory=tmp_path / "enriched",
        revision="c" * 40,
        metric=1.0,
        q_value=0.0,
    )
    repeat_launch, repeat_dir = _run(
        plan=plan,
        directory=tmp_path / "repeat",
        revision="c" * 40,
        metric=1.0,
        q_value=0.0,
        single_case=True,
    )

    audit = audit_repeatability_probe(
        legacy_plan=plan,
        legacy_launch=launch,
        legacy_checkpoint_dir=legacy_dir,
        enriched_plan=plan,
        enriched_launch=repeat_launch,
        enriched_checkpoint_dir=enriched_dir,
        repeat_plan=plan,
        repeat_launch=repeat_launch,
        repeat_checkpoint_dir=repeat_dir,
        case_index=0,
    )

    assert audit["classification"] == "authorities_not_discriminating"
    assert audit["matches"] == {"enriched": True, "legacy": True}


def test_repeatability_probe_rejects_different_repeat_execution(tmp_path: Path) -> None:
    plan = _plan()
    legacy_launch, legacy_dir = _run(
        plan=plan,
        directory=tmp_path / "legacy",
        revision="b" * 40,
        metric=1.0,
        q_value=0.0,
    )
    enriched_launch, enriched_dir = _run(
        plan=plan,
        directory=tmp_path / "enriched",
        revision="c" * 40,
        metric=2.0,
        q_value=1.0,
    )
    repeat_launch, repeat_dir = _run(
        plan=plan,
        directory=tmp_path / "repeat",
        revision="d" * 40,
        metric=2.0,
        q_value=1.0,
        single_case=True,
    )

    with pytest.raises(ValueError, match="same execution revision"):
        audit_repeatability_probe(
            legacy_plan=plan,
            legacy_launch=legacy_launch,
            legacy_checkpoint_dir=legacy_dir,
            enriched_plan=plan,
            enriched_launch=enriched_launch,
            enriched_checkpoint_dir=enriched_dir,
            repeat_plan=plan,
            repeat_launch=repeat_launch,
            repeat_checkpoint_dir=repeat_dir,
            case_index=0,
        )
