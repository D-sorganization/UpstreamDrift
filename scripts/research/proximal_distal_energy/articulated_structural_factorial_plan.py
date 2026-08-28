"""Prospective full-factorial structural pathway plan for UpstreamDrift #9153."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations, product
import re
from typing import Any

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AUTHORITY_KEYS = (
    "closed_state_npz",
    "shaft_structural_basis_json",
    "shaft_structural_basis_npz",
    "shaft_atlas_json",
    "shaft_atlas_npz",
    "ground_atlas_json",
    "ground_atlas_npz",
)
_FACTORS = (
    "shaft_bending",
    "shaft_torsion",
    "ground_translation",
    "ground_free_moment",
)
_CASE_INDICES = (0, 8, 9, 17)
_SAMPLE_INDICES = (0, 6, 12)
_SAMPLE_TIMES_S = {0: 0.0, 6: 0.12, 12: 0.24}
_VELOCITY_FACTORS = (1.0, -1.0)
_TIME_STEPS_S = (0.0002, 0.0001, 0.00005)
_HORIZONS_S = (0.004, 0.01, 0.025, 0.05)
_ENGINES = ("mujoco", "pinocchio")


def _activation_name(first: int, second: int, *, domain: str) -> str:
    if domain == "shaft":
        return {
            (0, 0): "rigid",
            (1, 0): "bending",
            (0, 1): "torsion",
            (1, 1): "coupled",
        }[(first, second)]
    return {
        (0, 0): "fixed",
        (1, 0): "translation",
        (0, 1): "free_moment",
        (1, 1): "coupled",
    }[(first, second)]


def _factorial_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for levels in product((0, 1), repeat=len(_FACTORS)):
        cell_id = "".join(str(level) for level in levels)
        cells.append(
            {
                "cell_id": cell_id,
                "levels": list(levels),
                "shaft_activation": _activation_name(*levels[:2], domain="shaft"),
                "ground_activation": _activation_name(*levels[2:], domain="ground"),
            }
        )
    return cells


def _contrast(factors: tuple[str, ...]) -> dict[str, Any]:
    return {
        "contrast_id": "x".join(factors),
        "factors": list(factors),
        "estimator": (
            "within-block Walsh contrast: mean(outcome * product(2*level-1)) "
            "over all sixteen pathway cells"
        ),
    }


@dataclass(frozen=True, slots=True)
class StructuralFactorialPlan:
    """Freeze pathway toggles and estimands before implementing the runner."""

    design_authority_revision: str
    authority_sha256: Mapping[str, str]
    worker_count: int = 1

    def __post_init__(self) -> None:
        if not _SHA40.fullmatch(self.design_authority_revision):
            raise ValueError(
                "design_authority_revision must be a lowercase 40-character SHA"
            )
        if set(self.authority_sha256) != set(_AUTHORITY_KEYS):
            raise ValueError(
                f"authority_sha256 keys must be exactly {_AUTHORITY_KEYS!r}"
            )
        if any(
            not _SHA256.fullmatch(value) for value in self.authority_sha256.values()
        ):
            raise ValueError("every authority_sha256 value must be a lowercase SHA-256")
        if self.worker_count != 1:
            raise ValueError("worker_count must remain one on DeskComputer")

    def to_manifest(self) -> dict[str, Any]:
        """Return the deterministic outcome-blind design manifest."""

        primary = [
            _contrast(factors)
            for order in (1, 2)
            for factors in combinations(_FACTORS, order)
        ]
        exploratory = [
            _contrast(factors)
            for order in (3, 4)
            for factors in combinations(_FACTORS, order)
        ]
        state_count = len(_CASE_INDICES) * len(_SAMPLE_INDICES)
        attempt_count = (
            state_count
            * 2 ** len(_FACTORS)
            * len(_VELOCITY_FACTORS)
            * len(_TIME_STEPS_S)
            * len(_ENGINES)
        )
        return {
            "schema_version": "articulated-structural-factorial-plan/1.1.0",
            "study_id": "prospective-rigid-shaft-ground-pathway-factorial",
            "identity": {
                "design_authority_revision": self.design_authority_revision,
                "authority_sha256": dict(sorted(self.authority_sha256.items())),
                "issue": 9153,
            },
            "preregistration": {
                "timing": "before_combined_runner_implementation_or_execution",
                "status": "design_frozen_execution_not_started",
                "amendment": (
                    "v1.1 adds the byte-identical regenerated shaft-basis NPZ and its "
                    "updated formatting-only source provenance after the disclosed "
                    "timing probe failed closed; no outcome was produced"
                ),
                "reason": (
                    "the shaft and ground atlases used post-registered outcome matching; "
                    "this design instead identifies pathway contrasts by exact within-state "
                    "intervention"
                ),
                "prior_results_retained": {
                    "shaft_post_registered_matched_cells": 126,
                    "shaft_total_cells": 384,
                    "ground_primary_matched_cells": 0,
                    "ground_total_cells": 384,
                    "rigid_refinement_failed_groups": 3,
                    "pinocchio_current_status": "typed_unavailable",
                },
            },
            "design": {
                "factors": list(_FACTORS),
                "factor_meanings": {
                    "shaft_bending": "toggle the registered first bending-mode coordinates",
                    "shaft_torsion": "toggle the registered first torsion-mode coordinate",
                    "ground_translation": "toggle registered planar base translation",
                    "ground_free_moment": "toggle the registered base pitch/free-moment pathway",
                },
                "factorial_cells": _factorial_cells(),
                "states": [
                    {
                        "source_case_index": case_index,
                        "source_sample_index": sample_index,
                        "source_time_s": _SAMPLE_TIMES_S[sample_index],
                    }
                    for case_index in _CASE_INDICES
                    for sample_index in _SAMPLE_INDICES
                ],
                "velocity_factors": list(_VELOCITY_FACTORS),
                "time_steps_s": list(_TIME_STEPS_S),
                "horizons_s": list(_HORIZONS_S),
                "engines": list(_ENGINES),
                "initialization": (
                    "identical closed rigid state, signed club perturbation, natural-zero "
                    "shaft coordinates, and natural-zero base coordinates within each block"
                ),
                "active_driver_or_joint_torque": "none; motion is an initial condition",
                "registered_engine_attempt_count": attempt_count,
                "expected_native_attempt_count": attempt_count // len(_ENGINES),
            },
            "analysis": {
                "blocking_key": [
                    "source_case_index",
                    "source_sample_index",
                    "velocity_factor",
                    "time_step_s",
                    "engine",
                    "horizon_s",
                ],
                "primary_outcomes": [
                    "final_club_translation_speed_m_s",
                    "club_linear_momentum_change_kg_m_s",
                    "signed_contact_impulse_n_s",
                    "signed_contact_work_j",
                    "terminal_total_dissipation_j",
                ],
                "primary_contrasts": primary,
                "exploratory_higher_order_contrasts": exploratory,
                "outcome_matching": "prohibited",
                "missing_cell_policy": (
                    "retain typed failure or unavailable status; suppress the affected "
                    "within-block contrast rather than impute"
                ),
                "mediators_not_eligibility_filters": [
                    "peak_grip_force_n",
                    "terminal_contact_dissipation_j",
                    "terminal_shaft_dissipation_j",
                    "terminal_ground_dissipation_j",
                ],
                "aggregation": (
                    "publish every paired block, sign distribution, range, median, and "
                    "all-state heatmap; deterministic model outputs receive no p-values"
                ),
            },
            "gates": {
                "normalized_work_energy_residual_max": 0.05,
                "three_step_refinement_ratio_max": 0.8,
                "initial_energy_relative_range_max": 1e-12,
                "cross_engine_trajectory_relative_error_max": 1e-7,
                "cross_engine_force_relative_error_max": 1e-7,
                "active_set_parity_required": True,
                "shaft_small_deflection_and_twist_limits": "use frozen model limits",
                "base_translation_max_m": 0.05,
                "base_pitch_max_rad": 0.17453292519943295,
            },
            "falsification": {
                "sign_reversal_suppresses_universal_benefit": True,
                "failed_refinement_suppresses_affected_contrast": True,
                "missing_cross_engine_parity_suppresses_promotion": True,
                "pathway_attribution_requires_corresponding_mediator_response": True,
                "zero_effect_is_retained_not_reinterpreted_as_success": True,
                "scope": "causal only within the declared synthetic model interventions",
            },
            "execution": {
                "worker_count": self.worker_count,
                "maximum_python_process_count": 1,
                "maximum_logical_cpu_count": 2,
                "checkpoint_policy": "one_atomic_checkpoint_per_attempt",
                "resume_policy": "verify_identity_and_hash_then_skip_complete_checkpoint",
                "runtime_calibration": (
                    "run a disclosed non-evidence timing probe before launch; do not use "
                    "its outcomes to alter factors, states, estimands, or gates"
                ),
                "resource_contract": (
                    "DeskComputer serial CPU only; no ControlTower, WSL, or VHDX access"
                ),
                "launch_status": "blocked_pending_immutable_runner_revision",
            },
            "promotion": {
                "eligible": False,
                "requirements": [
                    "immutable combined-runner revision and launch manifest",
                    "all registered checkpoints accounted for",
                    "all numerical gates pass for every promoted contrast",
                    "MuJoCo-Pinocchio parity available and passing",
                    "paper, claims, evidence hashes, and handoff regenerated",
                    "protected review and merge",
                ],
                "human_or_coaching_claims": False,
                "equipment_optimization_claims": False,
            },
        }


__all__ = ["StructuralFactorialPlan"]
