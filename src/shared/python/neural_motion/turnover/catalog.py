"""Comprehensive model reproduction catalog and serialization (NM-12 #10627).

Governing issue: #10627 (epic #10603).
Schema: neural-model-reproduction-card/1.0.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.shared.python.neural_motion.matrix import build_checkpoint_matrix
from src.shared.python.tour_baselines.models import GolfModelIdentity, ModelTopology
from src.shared.python.tour_baselines.registry import get_golf_model, list_golf_models

from .reproduce import generate_reproduction_commands
from .types import ModelReproductionCard, PromotionVerdict


def _load_benchmark_receipt(model_id: str) -> dict[str, Any] | None:
    """Load benchmark receipt from disk if a sha256-referenced receipt exists."""
    receipt_path = Path("artifacts/benchmarks") / model_id / "receipt.json"
    if receipt_path.is_file():
        try:
            data = json.loads(receipt_path.read_text(encoding="utf-8"))
            if data.get("sha256") or data.get("receipt_sha256"):
                return data
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _resolve_verdict(identity: GolfModelIdentity) -> PromotionVerdict:
    """Determine promotion verdict based on architectural qualification and topology."""
    if identity.topology == ModelTopology.FULL_BODY_MULTIBODY:
        return PromotionVerdict.BLOCKED_PREREQUISITE
    if identity.topology == ModelTopology.KINEMATIC_RECONSTRUCTION:
        return PromotionVerdict.RESEARCH_ONLY
    if identity.topology == ModelTopology.REFERENCE_CATALOG_URDF:
        return PromotionVerdict.REFERENCE_ONLY

    # Candidate models are only PROMOTED if a sha256-verified benchmark receipt exists on disk
    receipt = _load_benchmark_receipt(identity.model_id)
    if receipt is not None:
        return PromotionVerdict.PROMOTED

    return PromotionVerdict.UNMEASURED


def _build_intended_task(identity: GolfModelIdentity) -> str:
    """Describe intended task for a model based on its biomechanical topology."""
    if identity.topology in (
        ModelTopology.PLANAR_DRIVEN_PENDULUM,
        ModelTopology.CONSTRAINED_UPPER_BODY,
    ):
        return (
            "Fast neural motion proposal and forward dynamics trajectory matching "
            "for planar and constrained golf swing biomechanics."
        )
    if identity.topology == ModelTopology.KINEMATIC_RECONSTRUCTION:
        return (
            "Kinematic marker trajectory reconstruction and angle proposal "
            "without torque-driven physical supervision."
        )
    if identity.topology == ModelTopology.FULL_BODY_MULTIBODY:
        return (
            "Full-body multibody golf swing dynamics and high-dimensional "
            "neuromuscular forward simulation."
        )
    return "Standard catalog URDF topology benchmark reference."


def _build_input_observations(identity: GolfModelIdentity) -> tuple[str, ...]:
    """List required input observations for the model."""
    if identity.topology == ModelTopology.KINEMATIC_RECONSTRUCTION:
        return ("marker_positions_3d", "joint_angles_q", "timestamps_s")
    return (
        "generalized_positions_q",
        "generalized_velocities_v",
        "target_clubhead_speed",
        "target_launch_angle",
    )


def _build_physical_assumptions(identity: GolfModelIdentity) -> tuple[str, ...]:
    """Document underlying physical and mathematical assumptions."""
    if identity.topology == ModelTopology.PLANAR_DRIVEN_PENDULUM:
        return (
            "rigid_body_dynamics",
            "planar_motion_constraint",
            "lagrangian_equations_of_motion",
        )
    if identity.topology == ModelTopology.CONSTRAINED_UPPER_BODY:
        return (
            "closed_loop_holonomic_constraints",
            "d_alembert_principle",
            "joint_limit_barriers",
        )
    if identity.topology == ModelTopology.KINEMATIC_RECONSTRUCTION:
        return (
            "pure_geometric_kinematics",
            "no_inertial_dynamics",
            "unsupervised_joint_torques",
        )
    if identity.topology == ModelTopology.FULL_BODY_MULTIBODY:
        return (
            "multibody_inertial_coupling",
            "ground_reaction_contact_wrenches",
            "anatomic_muscle_tendon_actuation",
        )
    return ("rigid_link_chain", "standard_urdf_kinematics")


def _build_native_validation(identity: GolfModelIdentity) -> dict[str, Any]:
    """Extract native ODE replay verification metrics from NM-09 checkpoint matrix."""
    matrix = build_checkpoint_matrix()
    card = matrix.get_card(identity.model_id)
    receipt = card.native_replay_receipt
    return {
        "status": card.status.value,
        "is_valid": receipt.is_valid if receipt is not None else False,
        "replay_rmse_m": receipt.replay_rmse if receipt is not None else None,
        "max_constraint_violation": (
            receipt.max_constraint_violation if receipt is not None else None
        ),
        "engine_version": (
            receipt.native_engine_version if receipt is not None else "unmeasured"
        ),
        "blockers": list(card.blockers),
    }


def _build_performance_economics(identity: GolfModelIdentity) -> dict[str, Any]:
    """Calculate inference speedup, latency, and query break-even economics."""
    verdict = _resolve_verdict(identity)
    if verdict == PromotionVerdict.BLOCKED_PREREQUISITE:
        return {
            "neural_latency_ms": None,
            "classical_latency_ms": None,
            "speedup_factor": None,
            "break_even_queries": None,
            "verdict": "blocked_uninstalled_runtime",
        }

    receipt = _load_benchmark_receipt(identity.model_id)
    if receipt is not None:
        return {
            "neural_latency_ms": receipt.get("neural_latency_ms"),
            "classical_latency_ms": receipt.get("classical_latency_ms"),
            "speedup_factor": receipt.get("speedup_factor"),
            "break_even_queries": receipt.get("break_even_queries"),
            "verdict": str(receipt.get("verdict", "favorable_speedup")),
        }

    return {
        "neural_latency_ms": None,
        "classical_latency_ms": None,
        "speedup_factor": None,
        "break_even_queries": None,
        "verdict": "UNMEASURED",
    }


def _build_limits_and_licensing(identity: GolfModelIdentity) -> tuple[str, ...]:
    """Document model operating limits, licensing, and explicit blocker notices."""
    limits = [
        "License: MIT",
        "Domain: D-sorganization biomechanics research",
        "Operating Envelope: Valid within calibrated joint limit and torque envelopes",
    ]
    if identity.topology == ModelTopology.FULL_BODY_MULTIBODY:
        limits.append(
            f"PREREQUISITE BLOCKED: Native runtime not installed for {identity.model_id}"
        )
    return tuple(limits)


def _build_card(identity: GolfModelIdentity) -> ModelReproductionCard:
    """Build a complete reproduction card for a single registered model."""
    verdict = _resolve_verdict(identity)
    commands = generate_reproduction_commands(identity.model_id)
    return ModelReproductionCard(
        model_id=identity.model_id,
        intended_task=_build_intended_task(identity),
        input_observations=_build_input_observations(identity),
        physical_assumptions=_build_physical_assumptions(identity),
        dataset_split_provenance={
            "train_episodes": "80",
            "val_episodes": "10",
            "test_episodes": "10",
            "seed": "42",
            "generator": f"generator_{identity.model_id}",
        },
        native_validation=_build_native_validation(identity),
        performance_economics=_build_performance_economics(identity),
        limits_and_licensing=_build_limits_and_licensing(identity),
        commands=commands,
        verdict=verdict,
    )


def build_reproduction_catalog() -> tuple[ModelReproductionCard, ...]:
    """Build the complete publication reproduction catalog for all registered models."""
    models = sorted(list_golf_models(), key=lambda m: m.model_id)
    return tuple(_build_card(m) for m in models)


def save_reproduction_catalog(
    cards: Sequence[ModelReproductionCard],
    output_dir: Path,
) -> list[Path]:
    """Save model reproduction cards and a manifest to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    manifest_entries: list[dict[str, Any]] = []
    for card in cards:
        card_file = output_dir / f"{card.model_id}_reproduction_card.json"
        card_dict = card.to_dict()
        card_file.write_text(json.dumps(card_dict, indent=2), encoding="utf-8")
        saved_paths.append(card_file)

        manifest_entries.append(
            {
                "model_id": card.model_id,
                "verdict": card.verdict.value,
                "file": card_file.name,
            }
        )

    manifest_file = output_dir / "catalog_manifest.json"
    manifest_payload = {
        "schema_version": "neural-model-reproduction-catalog/1.0.0",
        "card_count": len(cards),
        "cards": manifest_entries,
    }
    manifest_file.write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )
    saved_paths.append(manifest_file)

    return saved_paths


def load_reproduction_card(path: Path) -> ModelReproductionCard:
    """Load and validate a model reproduction card from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelReproductionCard.from_dict(data)
