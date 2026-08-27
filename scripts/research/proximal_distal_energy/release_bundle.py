"""Deterministic release manifest and qualification for the open resource."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

ARTICLE_REL = Path("docs/research/proximal_distal_energy_transfer")
_EXCLUDED = frozenset(
    {
        "release_manifest.json",
        "claim_evidence_manifest.json",
        "CHECKSUMS.sha256",
    }
)
_CANONICAL_TEXT_SUFFIXES = frozenset(
    {".bib", ".cff", ".csv", ".json", ".md", ".py", ".qmd", ".svg"}
)


def canonical_artifact_bytes(path: Path) -> bytes:
    """Return platform-stable bytes for a release artifact.

    Git normalizes the declared text formats to LF in repository objects, but
    an existing Windows worktree can retain CRLF bytes after attribute changes.
    Hashing the canonical representation keeps release qualification identical
    across clean POSIX and Windows checkouts without altering binary evidence.
    """
    content = path.read_bytes()
    if path.suffix.lower() in _CANONICAL_TEXT_SUFFIXES:
        return content.replace(b"\r\n", b"\n")
    return content


def artifact_sha256(path: Path) -> str:
    """Return the SHA-256 digest of the canonical artifact representation."""
    return hashlib.sha256(canonical_artifact_bytes(path)).hexdigest()


def artifact_size(path: Path) -> int:
    """Return the byte length of the canonical artifact representation."""
    return len(canonical_artifact_bytes(path))


def _artifact_paths(root: Path) -> tuple[Path, ...]:
    article = root / ARTICLE_REL
    selected: set[Path] = set()
    for pattern in (
        "*.md",
        "*.qmd",
        "CITATION.cff",
        "references.bib",
        "proximal_distal_energy_transfer.pdf",
        "chapters/*.qmd",
        "data/**/*.json",
        "data/**/*.npz",
        "data/**/*.csv",
        "data/**/*.pdf",
        "figures/*.pdf",
        "figures/*.svg",
    ):
        selected.update(path for path in article.glob(pattern) if path.is_file())
    scripts = root / "scripts/research/proximal_distal_energy"
    selected.update(path for path in scripts.glob("*.py") if path.is_file())
    selected.add(root / "src/shared/python/biomechanics/interaction_evidence.py")
    return tuple(
        sorted(
            (path for path in selected if path.name not in _EXCLUDED),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


_RELEASE_METADATA: dict[str, Any] = {
    "schema_version": "proximal-distal-open-release-v1",
    "release_id": "proximal-distal-model-ladder-2026-08",
    "resource_framing": "neutral_open_research_resource",
    "integrity_authorities": {
        "claim_evidence_manifest": (
            "deterministic_self_excluded_authority_to_avoid_recursive_hashing"
        ),
        "external_source_review": (
            "offline_url_complete_work_and_claim_adjudication_embedded_in_artifacts"
        ),
        "publication_quality": (
            "runtime_revision_and_manifest_digest_bound_every_page_pdf_inspection"
        ),
    },
    "presets": {
        "double_pendulum": {
            "command": "python -m scripts.research.proximal_distal_energy.run_experiments",
            "tier": "planar_open_chain",
        },
        "pendulum_force_source_optimization": {
            "command": "python -m scripts.research.proximal_distal_energy.run_force_source_optimization",
            "tier": "planar_open_chain_coordinate_explicit_force_attribution",
        },
        "local_linear_diagnostics": {
            "command": "python -m scripts.research.proximal_distal_energy.run_local_linear_diagnostics write",
            "tier": "analytical_double_pendulum_local_first_order_rank",
        },
        "constraint_internal_force_diagnostics": {
            "command": "python -m scripts.research.proximal_distal_energy.run_constraint_internal_force_diagnostics write",
            "tier": "scaled_cross_tier_constraint_and_wrench_map_rank",
        },
        "closed_loop_singularity_margin": {
            "command": "python -m scripts.research.proximal_distal_energy.run_closed_loop_singularity_margin write",
            "tier": "analytical_planar_exact_position_closure",
        },
        "phase_event_stability": {
            "command": "python -m scripts.research.proximal_distal_energy.run_phase_event_stability write",
            "tier": "analytical_double_pendulum_local_finite_time_event_sensitivity",
        },
        "trajectory_control_authority": {
            "command": "python -m scripts.research.proximal_distal_energy.run_trajectory_control_authority write",
            "tier": (
                "analytical_double_pendulum_trajectory_varying_"
                "event_conditioned_authority"
            ),
        },
        "double_pendulum_identifiability": {
            "command": "python -m scripts.research.proximal_distal_energy.run_double_pendulum_identifiability write",
            "tier": "analytical_double_pendulum_exact_map_and_dimensionless_finite_record",
        },
        "forward_two_hand": {
            "command": "python -m scripts.research.proximal_distal_energy.run_forward_two_arm_study",
            "tier": "planar_constrained_forward",
        },
        "moving_base_flexible_club": {
            "command": "python -m scripts.research.proximal_distal_energy.run_moving_base_flexible_study",
            "tier": "planar_coupled_base_flex",
        },
        "forward_modal_shaft": {
            "command": "python -m scripts.research.proximal_distal_energy.run_moving_base_modal_shaft_study",
            "tier": "planar_coupled_base_distributed_modal_shaft",
        },
        "shaft_beam_reference": {
            "command": "python -m scripts.research.proximal_distal_energy.run_shaft_beam_reference",
            "tier": "synthetic_distributed_shaft_comparison",
        },
        "torque_allocation_preload": {
            "command": "python -m scripts.research.proximal_distal_energy.run_torque_allocation_preload_study",
            "tier": "matched_task_allocation_and_phenomenological_transmission",
        },
        "spatial_common_state": {
            "command": "python -m scripts.research.proximal_distal_energy.run_spatial_full_body_study",
            "tier": "reduced_full_body_common_state",
        },
        "subject_scaled_spatial_geometry": {
            "command": "python -m scripts.research.proximal_distal_energy.run_subject_scaled_spatial_geometry",
            "tier": "prescribed_subject_scaled_contact_closure_audit",
        },
        "subject_scaled_closed_contact": {
            "command": "python -m scripts.research.proximal_distal_energy.run_subject_scaled_closed_contact",
            "tier": "subject_scaled_bounded_closed_contact_inverse_kinematics",
        },
        "closed_state_forward_bridge": {
            "command": "python -m scripts.research.proximal_distal_energy.run_closed_state_forward_bridge",
            "tier": "subject_scaled_closed_state_reduced_forward_initialization",
        },
        "forward_contact_validity_horizon": {
            "command": "python -m scripts.research.proximal_distal_energy.run_forward_contact_validity_horizon",
            "tier": "subject_scaled_closed_state_reduced_forward_horizon_map",
        },
        "articulated_inertia_cross_engine": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_inertia_cross_engine",
            "tier": "subject_scaled_closed_state_articulated_common_state_dynamics",
        },
        "articulated_native_constraint_discrepancy": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_native_constraint_discrepancy",
            "tier": "native_equality_integrator_versus_projected_contact_formulation_discrepancy",
        },
        "articulated_contact_projection": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_contact_projection",
            "tier": "subject_scaled_articulated_contact_initial_acceleration",
        },
        "articulated_forward_contact": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_forward_contact",
            "tier": "bounded_subject_scaled_articulated_bilateral_attachment_forward_dynamics",
        },
        "articulated_slack_atlas": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_slack_atlas",
            "tier": "bounded_subject_scaled_typed_unilateral_attachment_forward_dynamics",
        },
        "articulated_distributed_grip_atlas": {
            "command": "python -m scripts.research.proximal_distal_energy.run_distributed_grip_atlas",
            "tier": "bounded_subject_scaled_distributed_unilateral_grip_forward_dynamics",
        },
        "articulated_shaft_structural_basis": {
            "command": "python -m scripts.research.proximal_distal_energy.generate_articulated_shaft_structural_basis",
            "tier": "frozen_first_mode_finite_element_structural_authority",
        },
        "articulated_shaft_time_step_diagnostic": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_shaft_time_step_diagnostic",
            "tier": "limiting_torsion_cell_three_level_refinement",
        },
        "articulated_shaft_atlas": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_shaft_atlas",
            "tier": "bounded_subject_scaled_distributed_grip_passive_shaft_forward_dynamics",
        },
        "articulated_ground_diagnostic": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_ground_diagnostic",
            "tier": "finite_base_initialization_and_three_level_refinement",
        },
        "articulated_ground_atlas": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_ground_atlas",
            "tier": "bounded_subject_scaled_finite_ground_and_intrinsic_free_moment",
        },
        "articulated_ground_posthoc_sensitivity": {
            "command": "python -m scripts.research.proximal_distal_energy.run_articulated_ground_posthoc_sensitivity",
            "tier": "explicit_post_hoc_primary_match_failure_sensitivity",
        },
        "scapulothoracic_contact_screen": {
            "command": "python -m scripts.research.proximal_distal_energy.run_scapulothoracic_contact_screen",
            "tier": "paired_arm_only_scapula_on_ellipsoid_geometry_screen",
        },
        "spatial_forward_contact": {
            "command": "python -m scripts.research.proximal_distal_energy.run_spatial_forward_contact_study",
            "tier": "reduced_two_engine_forward_contact",
        },
        "uncertainty_control": {
            "command": "python -m scripts.research.proximal_distal_energy.run_uncertainty_control_study",
            "tier": "coupled_uncertainty_control",
        },
        "experimental_readiness": {
            "command": "python -m scripts.research.proximal_distal_energy.run_experimental_protocol_dry_run",
            "tier": "synthetic_protocol_qualification_only",
        },
        "advanced_biological_bridge": {
            "command": "python -m scripts.research.proximal_distal_energy.run_advanced_biological_bridge",
            "tier": "frame_invariance_and_reduced_hill_type_mechanism",
        },
        "transmission_robustness": {
            "command": "python -m scripts.research.proximal_distal_energy.run_transmission_robustness_study",
            "tier": "paired_state_trigger_and_task_robustness",
        },
        "timing_viability_adverse_load": {
            "command": "python -m scripts.research.proximal_distal_energy.run_timing_viability_study",
            "tier": "common_phase_paired_adverse_load_recovery",
        },
        "typed_slack_dynamic_audit": {
            "command": "python -m scripts.research.proximal_distal_energy.run_typed_slack_dynamic_study",
            "tier": "synthetic_scalar_dynamic_constitutive_screen",
        },
        "shoulder_velocity_pointwise": {
            "command": "python -m scripts.research.proximal_distal_energy.run_shoulder_velocity_transfer_study",
            "tier": "planar_fixed_hub_pointwise_phase_sensitivity",
        },
        "shoulder_velocity_strategy": {
            "command": "python -m scripts.research.proximal_distal_energy.run_shoulder_velocity_strategy_study",
            "tier": "planar_fixed_hub_control_program_search",
        },
        "joint_matched_proximal_rate": {
            "command": "python -m scripts.research.proximal_distal_energy.run_joint_matched_proximal_rate_study",
            "tier": "planar_fixed_hub_joint_work_load_matching_screen",
        },
        "rotating_base_torso_velocity": {
            "command": "python -m scripts.research.proximal_distal_energy.run_rotating_base_torso_velocity_study",
            "tier": "planar_rotating_base_two_hand_compliant_club",
        },
        "bilateral_wrench_identifiability": {
            "command": "python -m scripts.research.proximal_distal_energy.run_bilateral_wrench_identifiability_study",
            "tier": "instantaneous_linear_structural_identifiability",
        },
        "bilateral_wrench_sensor_qualification": {
            "command": "python -m scripts.research.proximal_distal_energy.run_bilateral_wrench_sensor_qualification",
            "tier": "synthetic_trajectory_point_force_sensor_qualification",
        },
    },
    "claims": {
        "interaction_dynamics_planar": "supported_at_declared_model_tier",
        "feasible_closed_loop_singularity_margin": (
            "supported_for_declared_exact_planar_kinematic_triangle"
        ),
        "phase_event_finite_time_stability": (
            "supported_for_declared_local_nonperiodic_analytical_trajectory"
        ),
        "trajectory_varying_event_control_authority": (
            "supported_for_declared_local_first_order_analytical_trajectory"
        ),
        "double_pendulum_base_coefficient_excitation": (
            "full_rank_for_registered_synthetic_record"
        ),
        "double_pendulum_physical_parameter_identifiability": (
            "structurally_non_identifiable_under_declared_model"
        ),
        "double_pendulum_practical_identifiability": (
            "not_established_oracle_kinematics_lower_bound_only"
        ),
        "coordinate_force_source_attribution": (
            "supported_at_declared_planar_model_and_coordinate_tier"
        ),
        "geometry_transfer_spatial_common_state": "supported_at_declared_model_tier",
        "distributed_shaft_modal_reduction": "supported_on_synthetic_structural_case",
        "distributed_modal_shaft_coupled_forward": (
            "supported_at_declared_planar_mechanism_tier"
        ),
        "arm_wrist_allocation_equivalence": (
            "supported_for_the_declared_same_state_club_task"
        ),
        "preload_continuity_advantage": (
            "conditional_on_the_declared_dead_zone_transmission_family"
        ),
        "scapular_or_muscle_strategy_identification": "unsupported",
        "passive_negative_couple_spatial_forward": (
            "supported_at_declared_reduced_contact_tier"
        ),
        "universal_control_strategy": "unsupported",
        "human_experimental": "untested",
        "reference_frame_power_invariance": "supported_to_declared_numerical_tolerance",
        "muscle_redundancy_same_moment": "supported_at_reduced_hill_type_tier",
        "canonical_pose_adapter_round_trip": (
            "supported_for_coordinate_representation_only"
        ),
        "drake_opensim_myosuite_human_validation": "unexecuted",
        "state_triggered_model_robustness": "conditional_with_force_tradeoff",
        "state_triggered_larger_timing_region": (
            "falsified_in_registered_moving_base_planar_screen"
        ),
        "registered_model_sustained_recovery": "not_observed_in_60_cases",
        "human_self_stabilization": "untested",
        "high_proximal_velocity_universally_beneficial": (
            "falsified_at_declared_planar_tiers"
        ),
        "shoulder_velocity_control_strategy": (
            "conditional_on_phase_geometry_wrist_state_and_objective"
        ),
        "rotating_base_torso_velocity_transfer": (
            "supported_conditionally_at_declared_reduced_model_tier"
        ),
        "human_torso_velocity_strategy": "untested",
        "global_slack_benefit": "unsupported",
        "single_channel_slack_class_identification": "not_established",
        "individual_hand_allocation_from_net_wrench": ("structurally_unidentifiable"),
        "bilateral_human_wrench_validation": "untested",
        "synthetic_bilateral_point_force_sensor_qualification": (
            "qualified_for_declared_synthetic_cases"
        ),
        "physical_bilateral_six_axis_device_validation": "untested",
        "subject_scaled_spatial_contact_feasibility": (
            "prescribed_states_rejected_closed_contact_forward_test_open"
        ),
        "subject_scaled_closed_contact_feasibility": (
            "reduced_tree_closed_contact_screen_and_short_forward_initialization_passed"
        ),
        "closed_state_forward_initialization": (
            "supported_for_234_mappings_and_54_short_inertia_bias_transport_cases"
        ),
        "closed_state_forward_validity_horizon": (
            "no_failure_observed_through_registered_50_ms_reduced_model_interval"
        ),
        "subject_scaled_articulated_inertia": (
            "native_common_state_mass_bias_and_inverse_dynamics_qualified"
        ),
        "articulated_manufactured_solution": (
            "independent_numerical_controls_qualified"
        ),
        "native_constraint_formulation_discrepancy": (
            "native_branch_executed_nonzero_discrepancy_retained"
        ),
        "subject_scaled_articulated_contact_projection": (
            "same_state_bilateral_contact_projection_and_initial_acceleration_qualified"
        ),
        "bounded_articulated_forward_contact": (
            "five_millisecond_bilateral_attachment_forward_gate_qualified"
        ),
        "typed_articulated_slack": (
            "five_millisecond_typed_attachment_event_gate_qualified"
        ),
        "distributed_grip_discretization": (
            "fifty_millisecond_distributed_fiber_gate_qualified"
        ),
        "articulated_shaft_bending_torsion": (
            "fifty_millisecond_passive_shaft_gate_qualified_with_mixed_matched_outcomes"
        ),
        "articulated_ground_free_moment": (
            "fifty_millisecond_finite_ground_gate_qualified_primary_match_empty"
        ),
        "scapulothoracic_contact_geometry": (
            "partial_reachability_with_high_allocation_nullity_forward_test_open"
        ),
    },
    "known_open_gates": [
        "longer-horizon three-dimensional articulated contact with calibrated unilateral foot support, friction, tissue, and force-plate comparison",
        "equipment-calibrated distributed beam and grip coupled into a subject-scaled forward solve",
        "measured tissue-level preload and slack identification",
        "governed held-out human experimental evaluation",
        "external archive deposit and persistent identifier",
    ],
    "archive": {
        "persistent_identifier_status": "pending_external_archive",
        "reason": "Archive deposition is an external publication action and has not been executed.",
    },
}


def build_release_manifest(root: str | Path) -> dict[str, Any]:
    """Build the current deterministic release qualification record."""
    root_path = Path(root).resolve()
    artifacts = {
        path.relative_to(root_path).as_posix(): {
            "sha256": artifact_sha256(path),
            "bytes": artifact_size(path),
        }
        for path in _artifact_paths(root_path)
    }
    return {**_RELEASE_METADATA, "artifacts": artifacts}


def validate_release_manifest(
    root: str | Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed on missing, changed, unsafe, or unsupported artifacts."""
    root_path = Path(root).resolve()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("release manifest validation failed: artifacts are missing")
    mismatches: list[str] = []
    for relative, expected in artifacts.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            mismatches.append(f"unsafe path: {relative}")
            continue
        path = root_path / relative_path
        if not path.is_file():
            mismatches.append(f"missing: {relative}")
            continue
        if not isinstance(expected, dict):
            mismatches.append(f"invalid record: {relative}")
            continue
        if artifact_sha256(path) != expected.get("sha256"):
            mismatches.append(f"hash mismatch: {relative}")
        if artifact_size(path) != expected.get("bytes"):
            mismatches.append(f"size mismatch: {relative}")
    if mismatches:
        raise ValueError(
            "release manifest validation failed: " + "; ".join(mismatches[:8])
        )
    return {"valid": True, "artifact_count": len(artifacts), "mismatches": []}


def checksum_lines(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return sorted sha256sum-compatible records."""
    artifacts = manifest["artifacts"]
    return tuple(f"{artifacts[path]['sha256']}  {path}" for path in sorted(artifacts))
