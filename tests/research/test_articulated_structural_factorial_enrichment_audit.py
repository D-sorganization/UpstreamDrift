"""Exact-replay gates for structural evidence enrichment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_enrichment_audit import (
    AUDIT_SCHEMA,
    audit_enrichment_replay,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_evidence import (
    EVIDENCE_SIDECAR_SCHEMA,
)
from scripts.research.proximal_distal_energy.articulated_structural_factorial_plan import (
    StructuralFactorialPlan,
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


def _legacy_evaluation(_case: StructuralCase) -> StructuralEvaluation:
    arrays = {
        "time_s": np.array([0.0, 0.1]),
        "q": np.zeros((2, 20)),
        "qd": np.zeros((2, 20)),
        "elastic_coordinates": np.zeros((2, 0)),
        "base_coordinates": np.zeros((2, 0)),
        "net_club_force_n": np.zeros((2, 3)),
        "maximum_station_force_n": np.zeros(2),
        "active_station_count": np.zeros(2, dtype=int),
        "ground_force_n": np.zeros((2, 3)),
    }
    return StructuralEvaluation(result={"metric": 1.0}, parity_arrays=arrays)


def _enriched_evaluation(case: StructuralCase) -> StructuralEvaluation:
    legacy = _legacy_evaluation(case)
    arrays = dict(legacy.parity_arrays)
    arrays.update(
        {
            "elastic_velocities": np.zeros((2, 0)),
            "base_velocities": np.zeros((2, 0)),
            "station_force_on_club_n": np.zeros((2, 2, 1, 3)),
            "active_station": np.zeros((2, 2, 1), dtype=bool),
            "active_set_transition": np.zeros(2, dtype=bool),
            "contact_power_w": np.zeros(2),
            "cumulative_contact_impulse_n_s": np.zeros((2, 3)),
            "cumulative_contact_work_j": np.zeros(2),
            "force_couple_vector_nm": np.zeros((2, 3)),
            "grip_strain_energy_j": np.zeros(2),
            "grip_dissipation_power_w": np.zeros(2),
            "virtual_power_residual_w": np.zeros(2),
            "shaft_strain_energy_j": np.zeros(2),
            "shaft_damping_power_w": np.zeros(2),
            "shaft_power_residual_w": np.zeros(2),
            "ground_intrinsic_free_moment_nm": np.zeros(2),
            "ground_transported_moment_nm": np.zeros(2),
            "ground_strain_energy_j": np.zeros(2),
            "ground_damping_power_w": np.zeros(2),
            "ground_power_residual_w": np.zeros(2),
            "total_mechanical_energy_j": np.zeros(2),
            "total_energy_j": np.zeros(2),
            "cumulative_dissipation_j": np.zeros(2),
            "work_energy_residual_j": np.zeros(2),
            "tip_bending_m": np.zeros((2, 2)),
            "twist_angle_rad": np.zeros(2),
            "base_translation_m": np.zeros((2, 2)),
            "base_pitch_rad": np.zeros(2),
        }
    )
    return StructuralEvaluation(
        result={
            "metric": 1.0,
            "evidence_sidecar_schema": EVIDENCE_SIDECAR_SCHEMA,
        },
        parity_arrays=arrays,
    )


def _run_pair(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    plan = _plan()
    legacy_launch = build_launch_manifest(plan=plan, execution_revision="b" * 40)
    enriched_launch = build_launch_manifest(plan=plan, execution_revision="c" * 40)
    legacy_dir, enriched_dir = tmp_path / "legacy", tmp_path / "enriched"
    run_serial_cases(
        plan=plan,
        launch=legacy_launch,
        checkpoint_dir=legacy_dir,
        evaluator=_legacy_evaluation,
    )
    run_serial_cases(
        plan=plan,
        launch=enriched_launch,
        checkpoint_dir=enriched_dir,
        evaluator=_enriched_evaluation,
    )
    return legacy_launch, enriched_launch, legacy_dir, enriched_dir


def test_enrichment_audit_requires_exact_legacy_reproduction(tmp_path: Path) -> None:
    plan = _plan()
    legacy_launch, enriched_launch, legacy_dir, enriched_dir = _run_pair(tmp_path)

    audit = audit_enrichment_replay(
        legacy_plan=plan,
        legacy_launch=legacy_launch,
        legacy_checkpoint_dir=legacy_dir,
        enriched_plan=plan,
        enriched_launch=enriched_launch,
        enriched_checkpoint_dir=enriched_dir,
    )

    assert audit["schema_version"] == AUDIT_SCHEMA
    assert audit["legacy_prefix"]["checkpoint_count"] == 16
    assert audit["legacy_prefix"]["compared_array_count"] == 16 * 9
    assert audit["gates"]["passes"] is True
    assert audit["claim_boundary"]["legacy_prefix_promotable_by_itself"] is False


def test_enrichment_audit_rejects_changed_legacy_arrays(tmp_path: Path) -> None:
    plan = _plan()
    legacy_launch, enriched_launch, legacy_dir, enriched_dir = _run_pair(tmp_path)
    first = sorted(enriched_dir.glob("case-*.npz"))[0]
    with np.load(first, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    arrays = deepcopy(arrays)
    arrays["q"][0, 0] = 1.0
    np.savez_compressed(first, **arrays)
    checkpoint = first.with_suffix(".json")
    payload = __import__("json").loads(checkpoint.read_text(encoding="utf-8"))
    payload["outcome"]["parity_sidecar"]["sha256"] = (
        __import__("hashlib").sha256(first.read_bytes()).hexdigest()
    )
    checkpoint.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="changes legacy sidecar array: q"):
        audit_enrichment_replay(
            legacy_plan=plan,
            legacy_launch=legacy_launch,
            legacy_checkpoint_dir=legacy_dir,
            enriched_plan=plan,
            enriched_launch=enriched_launch,
            enriched_checkpoint_dir=enriched_dir,
        )
