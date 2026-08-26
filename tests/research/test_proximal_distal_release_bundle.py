from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.release_bundle import (
    build_release_manifest,
    validate_release_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
pytestmark = pytest.mark.unit


def test_release_manifest_has_model_ladder_presets_and_neutral_boundaries() -> None:
    manifest = build_release_manifest(ROOT)
    assert list(manifest["presets"]) == [
        "double_pendulum",
        "pendulum_force_source_optimization",
        "forward_two_hand",
        "moving_base_flexible_club",
        "forward_modal_shaft",
        "shaft_beam_reference",
        "torque_allocation_preload",
        "spatial_common_state",
        "subject_scaled_spatial_geometry",
        "subject_scaled_closed_contact",
        "closed_state_forward_bridge",
        "forward_contact_validity_horizon",
        "articulated_inertia_cross_engine",
        "articulated_native_constraint_discrepancy",
        "articulated_contact_projection",
        "articulated_forward_contact",
        "articulated_slack_atlas",
        "articulated_distributed_grip_atlas",
        "articulated_shaft_structural_basis",
        "articulated_shaft_time_step_diagnostic",
        "articulated_shaft_atlas",
        "articulated_ground_diagnostic",
        "articulated_ground_atlas",
        "articulated_ground_posthoc_sensitivity",
        "scapulothoracic_contact_screen",
        "spatial_forward_contact",
        "uncertainty_control",
        "experimental_readiness",
        "advanced_biological_bridge",
        "transmission_robustness",
        "timing_viability_adverse_load",
        "typed_slack_dynamic_audit",
        "shoulder_velocity_pointwise",
        "shoulder_velocity_strategy",
        "joint_matched_proximal_rate",
        "rotating_base_torso_velocity",
        "bilateral_wrench_identifiability",
        "bilateral_wrench_sensor_qualification",
    ]
    assert manifest["claims"]["human_experimental"] == "untested"
    assert manifest["claims"]["coordinate_force_source_attribution"] == (
        "supported_at_declared_planar_model_and_coordinate_tier"
    )
    assert manifest["claims"]["high_proximal_velocity_universally_beneficial"] == (
        "falsified_at_declared_planar_tiers"
    )
    assert manifest["claims"]["state_triggered_larger_timing_region"] == (
        "falsified_in_registered_moving_base_planar_screen"
    )
    assert manifest["claims"]["registered_model_sustained_recovery"] == (
        "not_observed_in_60_cases"
    )
    assert manifest["claims"]["global_slack_benefit"] == "unsupported"
    assert manifest["claims"]["single_channel_slack_class_identification"] == (
        "not_established"
    )
    assert manifest["claims"]["typed_articulated_slack"] == (
        "five_millisecond_typed_attachment_event_gate_qualified"
    )
    assert manifest["claims"]["distributed_grip_discretization"] == (
        "fifty_millisecond_distributed_fiber_gate_qualified"
    )
    assert manifest["claims"]["articulated_shaft_bending_torsion"] == (
        "fifty_millisecond_passive_shaft_gate_qualified_with_mixed_matched_outcomes"
    )
    assert manifest["claims"]["articulated_ground_free_moment"] == (
        "fifty_millisecond_finite_ground_gate_qualified_primary_match_empty"
    )
    assert (
        manifest["claims"]["synthetic_bilateral_point_force_sensor_qualification"]
        == "qualified_for_declared_synthetic_cases"
    )
    assert (
        manifest["claims"]["physical_bilateral_six_axis_device_validation"]
        == "untested"
    )
    assert (
        manifest["claims"]["distributed_shaft_modal_reduction"]
        == "supported_on_synthetic_structural_case"
    )
    assert (
        manifest["claims"]["passive_negative_couple_spatial_forward"]
        == "supported_at_declared_reduced_contact_tier"
    )
    assert (
        manifest["claims"]["arm_wrist_allocation_equivalence"]
        == "supported_for_the_declared_same_state_club_task"
    )
    assert (
        manifest["claims"]["preload_continuity_advantage"]
        == "conditional_on_the_declared_dead_zone_transmission_family"
    )
    assert manifest["claims"]["scapular_or_muscle_strategy_identification"] == (
        "unsupported"
    )
    assert manifest["claims"]["scapulothoracic_contact_geometry"] == (
        "partial_reachability_with_high_allocation_nullity_forward_test_open"
    )
    assert manifest["claims"]["canonical_pose_adapter_round_trip"] == (
        "supported_for_coordinate_representation_only"
    )
    assert manifest["presets"]["forward_modal_shaft"]["command"].endswith(
        "run_moving_base_modal_shaft_study"
    )
    allocation_command = manifest["presets"]["torque_allocation_preload"]["command"]
    assert allocation_command.endswith("run_torque_allocation_preload_study")
    assert (
        "measured tissue-level preload and slack identification"
        in manifest["known_open_gates"]
    )
    assert (
        manifest["archive"]["persistent_identifier_status"]
        == "pending_external_archive"
    )
    assert manifest["resource_framing"] == "neutral_open_research_resource"
    assert manifest["integrity_authorities"]["claim_evidence_manifest"].startswith(
        "deterministic_self_excluded"
    )
    assert manifest["integrity_authorities"]["external_source_review"].startswith(
        "offline_url_complete"
    )
    assert manifest["integrity_authorities"]["publication_quality"].startswith(
        "runtime_revision_and_manifest_digest_bound"
    )
    assert not any(
        path.endswith("claim_evidence_manifest.json") for path in manifest["artifacts"]
    )
    assert any(
        path.endswith("external_source_review.json") for path in manifest["artifacts"]
    )
    assert any(
        path.endswith("fig_shoulder_velocity_strategy_pareto.pdf")
        for path in manifest["artifacts"]
    )
    assert any(
        path.endswith("publication_quality.py") for path in manifest["artifacts"]
    )


def test_committed_release_manifest_matches_builder_and_validates() -> None:
    committed = json.loads((ARTICLE / "release_manifest.json").read_text())
    assert committed == build_release_manifest(ROOT)
    report = validate_release_manifest(ROOT, committed)
    assert report["valid"]
    assert report["mismatches"] == []


def test_validator_fails_closed_on_tampered_file(tmp_path: Path) -> None:
    manifest = build_release_manifest(ROOT)
    copied = tmp_path / "copy.json"
    copied.write_text("{}\n", encoding="utf-8")
    first = next(iter(manifest["artifacts"]))
    manifest["artifacts"] = {str(copied): manifest["artifacts"][first]}
    with pytest.raises(ValueError, match="release manifest validation failed"):
        validate_release_manifest(ROOT, manifest)


def test_validator_normalizes_checked_out_text_line_endings(tmp_path: Path) -> None:
    """A release manifest must validate identically on Windows and POSIX."""
    artifact = tmp_path / "paper.md"
    artifact.write_bytes(b"first\r\nsecond\r\n")
    canonical = b"first\nsecond\n"
    manifest = {
        "artifacts": {
            "paper.md": {
                "sha256": hashlib.sha256(canonical).hexdigest(),
                "bytes": len(canonical),
            }
        }
    }

    report = validate_release_manifest(tmp_path, manifest)

    assert report == {"valid": True, "artifact_count": 1, "mismatches": []}


def test_checksum_file_is_sorted_and_covers_every_artifact() -> None:
    manifest = build_release_manifest(ROOT)
    lines = (ARTICLE / "CHECKSUMS.sha256").read_text().splitlines()
    assert lines == sorted(lines, key=lambda item: item.split("  ", 1)[1])
    assert len(lines) == len(manifest["artifacts"])
    assert {line.split("  ", 1)[1] for line in lines} == set(manifest["artifacts"])
