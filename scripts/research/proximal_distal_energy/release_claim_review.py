"""Build and validate the release-level proximal-distal claim review.

This module distinguishes completion of a traceable review from scientific
closure.  Every release claim is linked to atomic claims with executable or
documented evidence, controls, falsifiers, and an explicit remaining gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
)
REPORT_PATH = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/release_claim_review.json"
)

OPEN_AUDIT_STATES = frozenset({"pending", "in_progress"})


def _spec(
    audit_state: str,
    disposition: str,
    claim_ids: tuple[str, ...],
    remaining_gate: str,
) -> dict[str, Any]:
    return {
        "audit_state": audit_state,
        "scientific_disposition": disposition,
        "supporting_claim_ids": claim_ids,
        "remaining_scientific_gate": remaining_gate,
    }


REVIEW_SPECS: dict[str, dict[str, Any]] = {
    "trajectory_varying_event_control_authority": _spec(
        "reviewed_as_local_trajectory_varying_event_conditioned_authority",
        "supported_for_declared_local_first_order_analytical_trajectory",
        ("PD-CLAIM-317", "PD-CLAIM-318"),
        (
            "Independent nonlinear bounded-control reachability, registered "
            "delay/noise and actuator-limit studies, then participant-held-out "
            "prediction before any human feasibility or strategy interpretation."
        ),
    ),
    "phase_event_finite_time_stability": _spec(
        "reviewed_as_local_finite_window_and_transverse_event_qualification",
        "supported_for_declared_local_nonperiodic_analytical_trajectory",
        ("PD-CLAIM-315", "PD-CLAIM-316"),
        (
            "Independent variational implementation, registered delay/noise and "
            "failure-region studies, then participant-held-out event and outcome "
            "predictions before any human robustness interpretation."
        ),
    ),
    "feasible_closed_loop_singularity_margin": _spec(
        "reviewed_as_exact_planar_kinematic_qualification",
        "supported_for_declared_exact_planar_kinematic_triangle",
        ("PD-CLAIM-313", "PD-CLAIM-314"),
        (
            "Calibrated spatial anatomical closure and compliant forward contact, "
            "followed by comparison with governed participant trajectories."
        ),
    ),
    "double_pendulum_base_coefficient_excitation": _spec(
        "reviewed_as_dimensionless_finite_record_excitation",
        "full_rank_for_registered_synthetic_record",
        ("PD-CLAIM-309",),
        "Independent trajectories, derivative pipelines, noise models, and held-out prediction under the same declared scale contract.",
    ),
    "double_pendulum_physical_parameter_identifiability": _spec(
        "reviewed_as_exact_structural_nonidentifiability",
        "structurally_non_identifiable_under_declared_model",
        ("PD-CLAIM-308",),
        "A richer independently justified model or measurement set that breaks the registered exact invariance families.",
    ),
    "double_pendulum_practical_identifiability": _spec(
        "reviewed_as_oracle_lower_bound_only",
        "not_established_oracle_kinematics_lower_bound_only",
        ("PD-CLAIM-310",),
        "Governed repeated-participant data with synchronized loads and kinematics, calibrated error models, and participant-held-out prediction.",
    ),
    "coordinate_force_source_attribution": _spec(
        "reviewed_as_bounded_coordinate_explicit_model_result",
        "supported_at_declared_planar_model_and_coordinate_tier",
        ("PD-CLAIM-305", "PD-CLAIM-306", "PD-CLAIM-307"),
        "Independent coordinate implementation, measured bilateral grip wrenches, and participant-held-out validation.",
    ),
    "interaction_dynamics_planar": _spec(
        "reviewed_at_declared_model_tier",
        "mechanism_supported_at_declared_model_tier",
        ("PD-CLAIM-002", "PD-CLAIM-003"),
        "Independent forward-model and experimental replication with measured bilateral loads.",
    ),
    "geometry_transfer_spatial_common_state": _spec(
        "reviewed_at_declared_model_tier",
        "geometry_dependence_supported_at_declared_model_tier",
        ("PD-CLAIM-004",),
        "Subject-scaled spatial replication over measured trajectories and uncertainty.",
    ),
    "distributed_shaft_modal_reduction": _spec(
        "reviewed_as_synthetic_structural_case",
        "supported_on_synthetic_structural_case",
        ("PD-CLAIM-224", "PD-CLAIM-227", "PD-CLAIM-239", "PD-CLAIM-240"),
        "Experimental modal identification and convergence on a measured club-shaft assembly.",
    ),
    "distributed_modal_shaft_coupled_forward": _spec(
        "reviewed_as_model_conditional_with_failed_small_deflection_screen",
        "mechanism_supported_but_quantitative_screen_failed",
        (
            "PD-CLAIM-223",
            "PD-CLAIM-224",
            "PD-CLAIM-225",
            "PD-CLAIM-226",
            "PD-CLAIM-228",
            "PD-CLAIM-229",
        ),
        "Geometrically nonlinear forward shaft validation within measured deformation and load bounds.",
    ),
    "arm_wrist_allocation_equivalence": _spec(
        "reviewed",
        "task_level_equivalence_with_internal_allocation_nonuniqueness",
        ("PD-CLAIM-200", "PD-CLAIM-201", "PD-CLAIM-202"),
        "Synchronized bilateral wrench, kinematic, and neuromuscular observations.",
    ),
    "preload_continuity_advantage": _spec(
        "reviewed_as_model_conditional",
        "conditional_on_declared_transmission_family",
        ("PD-CLAIM-203", "PD-CLAIM-204", "PD-CLAIM-205"),
        "Measured contact compliance, activation history, and participant-held-out outcome tests.",
    ),
    "scapular_or_muscle_strategy_identification": _spec(
        "reviewed_as_unsupported",
        "unsupported_from_net_club_wrench_or_reduced_geometry",
        ("PD-CLAIM-200", "PD-CLAIM-206"),
        "Anatomical shoulder model plus synchronized EMG, pressure, kinematics, and bilateral wrench data.",
    ),
    "passive_negative_couple_spatial_forward": _spec(
        "reviewed_at_declared_reduced_contact_tier",
        "supported_at_declared_reduced_contact_tier",
        ("PD-CLAIM-005",),
        "Independent spatial forward replication with measured contact and shaft parameters.",
    ),
    "universal_control_strategy": _spec(
        "reviewed_as_unsupported",
        "unsupported",
        ("PD-CLAIM-205", "PD-CLAIM-248"),
        "A preregistered multi-participant comparison across objectives, perturbations, and alternatives.",
    ),
    "human_experimental": _spec(
        "reviewed_as_untested",
        "untested",
        ("PD-CLAIM-199", "PD-CLAIM-206", "PD-CLAIM-248"),
        "A qualifying governed human dataset and participant-held-out validation execution.",
    ),
    "reference_frame_power_invariance": _spec(
        "reviewed",
        "supported_to_declared_numerical_tolerance",
        ("PD-CLAIM-194",),
        "Independent implementation and measured-data coordinate audit.",
    ),
    "muscle_redundancy_same_moment": _spec(
        "reviewed_at_reduced_model_tier",
        "supported_at_reduced_muscle_model_tier",
        ("PD-CLAIM-195", "PD-CLAIM-196"),
        "Subject-specific muscle geometry, activation dynamics, and experimental force validation.",
    ),
    "canonical_pose_adapter_round_trip": _spec(
        "reviewed_as_representation_only",
        "supported_for_coordinate_representation_only",
        ("PD-CLAIM-197",),
        "Dynamic inertia-and-bias transport, force, constraint, and energy agreement under shared contact and integration on common cases.",
    ),
    "drake_opensim_myosuite_human_validation": _spec(
        "reviewed_as_unexecuted",
        "unexecuted",
        ("PD-CLAIM-197", "PD-CLAIM-198", "PD-CLAIM-199"),
        "Execute the registered engines against governed subject-specific human observations.",
    ),
    "state_triggered_model_robustness": _spec(
        "reviewed_as_model_conditional_with_load_tradeoff",
        "model_conditional_with_load_tradeoff",
        ("PD-CLAIM-230", "PD-CLAIM-231", "PD-CLAIM-232", "PD-CLAIM-233"),
        "Broader out-of-distribution perturbations and participant-held-out experimental tests.",
    ),
    "state_triggered_larger_timing_region": _spec(
        "reviewed_as_falsified_at_declared_model_tier",
        "falsified_at_declared_model_tier",
        ("PD-CLAIM-252",),
        "Test whether another registered model family or human cohort reverses the adverse result.",
    ),
    "registered_model_sustained_recovery": _spec(
        "reviewed_as_not_observed_at_declared_model_tier",
        "not_observed_at_declared_model_tier",
        ("PD-CLAIM-252",),
        "Prospective recovery tests with disturbances, state estimation, and explicit recovery criteria.",
    ),
    "human_self_stabilization": _spec(
        "reviewed_as_untested",
        "untested",
        ("PD-CLAIM-234", "PD-CLAIM-245", "PD-CLAIM-252"),
        "Participant-held-out perturbation study with measured impedance and recovery outcomes.",
    ),
    "high_proximal_velocity_universally_beneficial": _spec(
        "reviewed_as_falsified_at_declared_model_tiers",
        "general_rule_rejected_at_declared_model_tiers",
        ("PD-CLAIM-086", "PD-CLAIM-247", "PD-CLAIM-250"),
        "Objective-specific human and higher-fidelity tests, without restoring a universal claim.",
    ),
    "shoulder_velocity_control_strategy": _spec(
        "reviewed_as_model_conditional",
        "conditional_on_phase_geometry_wrist_state_and_objective",
        tuple(f"PD-CLAIM-{number:03d}" for number in range(81, 93))
        + ("PD-CLAIM-249", "PD-CLAIM-250"),
        "Subject-scaled causal interventions with matched state, work, load, and timing controls.",
    ),
    "rotating_base_torso_velocity_transfer": _spec(
        "reviewed_as_model_conditional",
        "supported_conditionally_at_declared_reduced_model_tier",
        ("PD-CLAIM-235", "PD-CLAIM-236", "PD-CLAIM-237", "PD-CLAIM-238"),
        "Measured rotating-base trajectories and participant-held-out adverse-load validation.",
    ),
    "human_torso_velocity_strategy": _spec(
        "reviewed_as_untested",
        "untested",
        ("PD-CLAIM-090", "PD-CLAIM-091", "PD-CLAIM-248"),
        "Causal human intervention with matched geometry, energy, and delivery objectives.",
    ),
    "global_slack_benefit": _spec(
        "reviewed_as_unsupported",
        "unsupported_as_a_global_rule",
        ("PD-CLAIM-247", "PD-CLAIM-253"),
        "Operationally separate slack classes and test each with measured transmission states.",
    ),
    "single_channel_slack_class_identification": _spec(
        "reviewed_as_not_identified",
        "not_identified",
        ("PD-CLAIM-253",),
        "Add independent contact, tissue, tendon, shaft, and grip observations sufficient for class identification.",
    ),
    "individual_hand_allocation_from_net_wrench": _spec(
        "reviewed_as_structurally_unidentifiable",
        "structurally_unidentifiable",
        ("PD-CLAIM-254", "PD-CLAIM-255"),
        "Measure both hand wrenches or impose independently justified allocation constraints.",
    ),
    "bilateral_human_wrench_validation": _spec(
        "reviewed_as_untested",
        "untested",
        ("PD-CLAIM-199", "PD-CLAIM-256", "PD-CLAIM-259"),
        "Acquire governed synchronized bilateral six-axis human grip-wrench data.",
    ),
    "synthetic_bilateral_point_force_sensor_qualification": _spec(
        "reviewed_as_model_conditional",
        "qualified_for_declared_synthetic_cases",
        ("PD-CLAIM-257", "PD-CLAIM-258"),
        "Bench calibration across realistic contact centers, cross-talk, bandwidth, and uncertainty.",
    ),
    "physical_bilateral_six_axis_device_validation": _spec(
        "reviewed_as_untested",
        "untested",
        ("PD-CLAIM-259",),
        "Build and calibrate a bilateral six-axis device, then validate it dynamically and in humans.",
    ),
    "subject_scaled_spatial_contact_feasibility": _spec(
        "reviewed_as_adverse_model_structure_result",
        "adverse_prescribed_state_contact_closure_result",
        ("PD-CLAIM-260", "PD-CLAIM-261", "PD-CLAIM-262"),
        "Calibrated compliant forward contact with subject-measured geometry and trajectories.",
    ),
    "subject_scaled_closed_contact_feasibility": _spec(
        "reviewed_as_necessary_condition_result",
        "necessary_condition_and_short_forward_initialization_supported_not_sufficient",
        ("PD-CLAIM-262", "PD-CLAIM-263", "PD-CLAIM-264"),
        "Full-horizon articulated forward dynamics with calibrated contact, loads, and delivery outcomes.",
    ),
    "closed_state_forward_initialization": _spec(
        "reviewed_as_short_horizon_reduced_model_result",
        "supported_for_declared_mapping_and_spanning_subset",
        ("PD-CLAIM-268", "PD-CLAIM-269", "PD-CLAIM-270"),
        "Replace hand carriages with articulated arms and run full-horizon calibrated contact controls.",
    ),
    "closed_state_forward_validity_horizon": _spec(
        "reviewed_as_right_censored_reduced_model_result",
        "no_failure_observed_through_registered_50_ms_interval",
        ("PD-CLAIM-271", "PD-CLAIM-272", "PD-CLAIM-273"),
        "Replace hand carriages with articulated subject-scaled arms and repeat the horizon, adverse-load, contact-loss, and energy controls with calibrated grip and shaft properties.",
    ),
    "subject_scaled_articulated_inertia": _spec(
        "reviewed_as_common_state_articulated_dynamics_result",
        "native_articulated_inertia_operators_agree_at_declared_closed_states",
        ("PD-CLAIM-274", "PD-CLAIM-275", "PD-CLAIM-276"),
        "Apply bilateral compliant contact to the qualified articulated tree and repeat horizon, adverse-load, contact-loss, refinement, power, and energy controls.",
    ),
    "articulated_manufactured_solution": _spec(
        "reviewed_as_independent_synthetic_numerical_control",
        "operator_conservation_and_first_order_controls_qualified",
        tuple(f"PD-CLAIM-{number}" for number in range(297, 302)),
        "Repeat across additional states and integrators, then retain governed human-data boundaries for any biomechanical inference.",
    ),
    "native_constraint_formulation_discrepancy": _spec(
        "reviewed_as_synthetic_formulation_discrepancy_control",
        "native_constraint_and_integrator_executed_with_nonzero_discrepancy",
        ("PD-CLAIM-302", "PD-CLAIM-303", "PD-CLAIM-304"),
        "Repeat with calibrated distributed grip contact, additional states, independent native formulations, and governed human data before physical or biomechanical inference.",
    ),
    "subject_scaled_articulated_contact_projection": _spec(
        "reviewed_as_same_state_articulated_contact_projection_result",
        "bilateral_contact_projection_and_native_initial_acceleration_qualified",
        ("PD-CLAIM-277", "PD-CLAIM-278", "PD-CLAIM-279"),
        "Integrate a bounded articulated bilateral-contact horizon and repeat contact-loss, adverse-load, refinement, power, and work-energy controls.",
    ),
    "bounded_articulated_forward_contact": _spec(
        "reviewed_as_right_censored_synthetic_forward_result",
        "five_millisecond_bilateral_attachment_forward_gate_qualified",
        ("PD-CLAIM-280", "PD-CLAIM-281", "PD-CLAIM-282"),
        "Extend the right-censored horizon with typed unilateral slack, calibrated distributed grip and shaft compliance, ground coupling, and governed bilateral human wrenches.",
    ),
    "typed_articulated_slack": _spec(
        "reviewed_as_right_censored_synthetic_contact_result",
        "five_millisecond_typed_attachment_event_gate_qualified",
        ("PD-CLAIM-283", "PD-CLAIM-284", "PD-CLAIM-285"),
        "Extend the right-censored typed laws through calibrated distributed grip and shaft contact, longer matched-work/load delivery, ground coupling, and governed bilateral human wrenches.",
    ),
    "distributed_grip_discretization": _spec(
        "reviewed_as_right_censored_synthetic_discretization_result",
        "fifty_millisecond_distributed_fiber_gate_qualified",
        ("PD-CLAIM-286", "PD-CLAIM-287", "PD-CLAIM-288"),
        "Calibrate grip friction and pressure, couple the qualified fibers to shaft bending and torsion, add ground pathways, and test longer matched-work/load delivery against governed bilateral human wrenches.",
    ),
    "articulated_shaft_bending_torsion": _spec(
        "reviewed_as_synthetic_first_mode_articulated_result",
        "passive_shaft_pathway_qualified_with_mixed_matched_outcomes",
        ("PD-CLAIM-289", "PD-CLAIM-290", "PD-CLAIM-291", "PD-CLAIM-292"),
        "Calibrate distributed grip and shaft properties, add higher-mode fast-load and finite-ground pathways, then test governed delivery, impact, and human outcomes.",
    ),
    "articulated_ground_free_moment": _spec(
        "reviewed_as_synthetic_finite_base_result_with_adverse_primary_match",
        "fifty_millisecond_finite_ground_gate_qualified_primary_match_empty",
        ("PD-CLAIM-293", "PD-CLAIM-294", "PD-CLAIM-295", "PD-CLAIM-296"),
        "Replace bilateral linear support with calibrated unilateral three-dimensional foot contact and force-plate observations, then repeat matched delivery and governed human tests.",
    ),
    "scapulothoracic_contact_geometry": _spec(
        "reviewed_as_paired_geometry_screen_with_explicit_boundaries",
        "paired_geometry_screen_partial_with_explicit_boundaries",
        ("PD-CLAIM-262", "PD-CLAIM-265", "PD-CLAIM-266", "PD-CLAIM-267"),
        "Subject-specific articulated shoulder and scapulothoracic forward-contact validation.",
    ),
}


def _stable_union(claims: list[dict[str, Any]], field: str) -> list[Any]:
    result: list[Any] = []
    for claim in claims:
        for value in claim[field]:
            if value not in result:
                result.append(value)
    return result


def build_release_claim_review(
    registry: dict[str, Any], *, root: Path = ROOT
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an updated registry and complete release-review report."""
    inventory = registry.get("release_claim_inventory", [])
    inventory_keys = [item.get("release_claim_key") for item in inventory]
    if len(inventory_keys) != len(set(inventory_keys)):
        raise ValueError("release claim keys contain duplicates")
    if set(inventory_keys) != set(REVIEW_SPECS):
        raise ValueError("release claim keys do not match review specifications")

    claims_by_id = {claim["claim_id"]: claim for claim in registry.get("claims", [])}
    if len(claims_by_id) != len(registry.get("claims", [])):
        raise ValueError("atomic claim identifiers contain duplicates")

    reviews: list[dict[str, Any]] = []
    for item in inventory:
        key = item["release_claim_key"]
        spec = REVIEW_SPECS[key]
        supporting: list[dict[str, Any]] = []
        for claim_id in spec["supporting_claim_ids"]:
            if claim_id not in claims_by_id:
                raise ValueError(f"unknown supporting atomic claim {claim_id}")
            claim = claims_by_id[claim_id]
            for field in (
                "evidence_artifacts",
                "source_locations",
                "negative_controls",
                "falsifier",
                "uncertainty_boundary",
            ):
                if not claim.get(field):
                    raise ValueError(f"{claim_id} lacks {field}")
            for artifact in claim["evidence_artifacts"]:
                if (
                    not str(artifact).startswith(("http://", "https://"))
                    and not (root / artifact).exists()
                ):
                    raise ValueError(
                        f"{claim_id} evidence artifact is missing: {artifact}"
                    )
            supporting.append(claim)

        item["audit_state"] = spec["audit_state"]
        reviews.append(
            {
                "release_claim_key": key,
                "published_status": item["published_status"],
                "audit_state": spec["audit_state"],
                "scientific_disposition": spec["scientific_disposition"],
                "supporting_claim_ids": list(spec["supporting_claim_ids"]),
                "evidence_artifacts": _stable_union(supporting, "evidence_artifacts"),
                "source_locations": _stable_union(supporting, "source_locations"),
                "negative_controls": _stable_union(supporting, "negative_controls"),
                "falsifiers": [claim["falsifier"] for claim in supporting],
                "uncertainty_boundaries": [
                    claim["uncertainty_boundary"] for claim in supporting
                ],
                "remaining_scientific_gate": spec["remaining_scientific_gate"],
            }
        )

    registry["audit_scope"]["release_review_completion_status"] = "complete"
    open_count = sum(review["audit_state"] in OPEN_AUDIT_STATES for review in reviews)
    report = {
        "schema_version": "1.0.0",
        "authority": "D-sorganization/UpstreamDrift#8557",
        "interpretation": (
            "Review completion means every release claim has traceable atomic "
            "support, controls, falsifiers, boundaries, and a remaining gate. "
            "It does not mean every scientific claim is validated."
        ),
        "summary": {
            "release_claim_count": len(reviews),
            "reviewed_release_claim_count": len(reviews) - open_count,
            "open_release_review_count": open_count,
            "scientifically_open_gate_count": sum(
                bool(review["remaining_scientific_gate"]) for review in reviews
            ),
            "atomic_claim_count": len(claims_by_id),
        },
        "release_claim_reviews": reviews,
    }
    return registry, report


def _serialized(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2) + "\n"


def write_review() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    updated, report = build_release_claim_review(registry)
    REGISTRY_PATH.write_text(_serialized(updated), encoding="utf-8")
    REPORT_PATH.write_text(_serialized(report), encoding="utf-8")


def validate_review() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    updated, report = build_release_claim_review(registry)
    if _serialized(updated) != REGISTRY_PATH.read_text(encoding="utf-8"):
        raise SystemExit("claim registry release review is stale; run with 'write'")
    if not REPORT_PATH.exists() or report != json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    ):
        raise SystemExit("release claim review report is stale; run with 'write'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate"))
    args = parser.parse_args()
    if args.action == "write":
        write_review()
    else:
        validate_review()


if __name__ == "__main__":
    main()
