"""Deterministic release manifest and qualification for the open resource."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


ARTICLE_REL = Path("docs/research/proximal_distal_energy_transfer")
_EXCLUDED = frozenset({"release_manifest.json", "CHECKSUMS.sha256"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def build_release_manifest(root: str | Path) -> dict[str, Any]:
    """Build the current deterministic release qualification record."""
    root_path = Path(root).resolve()
    artifacts = {
        path.relative_to(root_path).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in _artifact_paths(root_path)
    }
    return {
        "schema_version": "proximal-distal-open-release-v1",
        "release_id": "proximal-distal-model-ladder-2026-08",
        "resource_framing": "neutral_open_research_resource",
        "presets": {
            "double_pendulum": {
                "command": "python -m scripts.research.proximal_distal_energy.run_experiments",
                "tier": "planar_open_chain",
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
        },
        "claims": {
            "interaction_dynamics_planar": "supported_at_declared_model_tier",
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
            "human_self_stabilization": "untested",
        },
        "known_open_gates": [
            "subject-scaled articulated spatial contact with calibrated grip and distributed shaft",
            "equipment-calibrated distributed beam and grip coupled into a subject-scaled forward solve",
            "measured tissue-level preload and slack identification",
            "governed held-out human experimental evaluation",
            "external archive deposit and persistent identifier",
        ],
        "archive": {
            "persistent_identifier_status": "pending_external_archive",
            "reason": "Archive deposition is an external publication action and has not been executed.",
        },
        "artifacts": artifacts,
    }


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
        if _sha256(path) != expected.get("sha256"):
            mismatches.append(f"hash mismatch: {relative}")
        if path.stat().st_size != expected.get("bytes"):
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
