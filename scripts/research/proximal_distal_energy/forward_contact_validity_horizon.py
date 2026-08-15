"""Cross-engine validity-horizon and adverse-load study for closed states."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.closed_state_forward_bridge import (
    canonical_state_from_vector,
    compare_bridge_traces,
    run_bridge_trace,
)
from scripts.research.proximal_distal_energy.spatial_forward_contract import (
    SpatialContactParameters,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"


@dataclass(frozen=True, slots=True)
class HorizonStudyConfig:
    """Frozen horizons, factor levels, and numerical closure gate."""

    horizons_s: tuple[float, ...] = (0.004, 0.010, 0.025, 0.050)
    low_factor: float = 0.5
    high_factor: float = 2.0
    energy_closure_limit: float = 0.08

    def __post_init__(self) -> None:
        horizons = np.asarray(self.horizons_s, dtype=float)
        if (
            horizons.ndim != 1
            or horizons.size == 0
            or np.any(~np.isfinite(horizons))
            or np.any(horizons <= 0.0)
            or np.any(np.diff(horizons) <= 0.0)
        ):
            raise ValueError("horizons_s must be finite, positive, and increasing")
        if not 0.0 < self.low_factor < 1.0 < self.high_factor:
            raise ValueError("factor levels must bracket one")
        if not np.isfinite(self.energy_closure_limit) or not (
            0.0 < self.energy_closure_limit < 1.0
        ):
            raise ValueError("energy_closure_limit must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class HorizonVariant:
    """One-factor-at-a-time perturbation from the nominal bridge."""

    variant_id: str
    stiffness_factor: float = 1.0
    damping_factor: float = 1.0
    hand_mass_factor: float = 1.0
    timestep_factor: float = 1.0
    driver_enabled: bool = True


def registered_variants(config: HorizonStudyConfig) -> tuple[HorizonVariant, ...]:
    """Return the preregistered nominal, adverse, and null branches."""

    lo, hi = config.low_factor, config.high_factor
    return (
        HorizonVariant("nominal"),
        HorizonVariant("contact_stiffness_low", stiffness_factor=lo),
        HorizonVariant("contact_stiffness_high", stiffness_factor=hi),
        HorizonVariant("contact_damping_low", damping_factor=lo),
        HorizonVariant("contact_damping_high", damping_factor=hi),
        HorizonVariant("hand_mass_low", hand_mass_factor=lo),
        HorizonVariant("hand_mass_high", hand_mass_factor=hi),
        HorizonVariant("timestep_half", timestep_factor=lo),
        HorizonVariant("timestep_double", timestep_factor=hi),
        HorizonVariant("driver_off", driver_enabled=False),
    )


def _parameters(
    state_values: FloatArray,
    profile_index: int,
    span_m: float,
    variant: HorizonVariant,
) -> tuple[Any, SpatialContactParameters]:
    profiles = default_synthetic_profiles()
    state = canonical_state_from_vector(state_values)
    _, metadata = build_subject_scaled_model(profiles[profile_index])
    base_hand_mass = float(metadata["represented_body_masses_kg"]["hand"])
    offsets = ((0.0, span_m / 2.0, -0.03), (0.0, -span_m / 2.0, -0.03))
    params = SpatialContactParameters(
        hand_mass=base_hand_mass * variant.hand_mass_factor,
        lead_grip_offset=offsets[0],
        trail_grip_offset=offsets[1],
        club_initial_position=tuple(state.club_position),
        contact_stiffness=1800.0 * variant.stiffness_factor,
        contact_damping=18.0 * variant.damping_factor,
        time_step=0.00025 * variant.timestep_factor,
    )
    return state, params


def _prefix(trace: dict[str, Any], count: int) -> dict[str, Any]:
    return {
        key: value[:count] if isinstance(value, np.ndarray) else value
        for key, value in trace.items()
    }


def _energy_closure(trace: dict[str, Any]) -> float:
    scale = max(1.0, float(np.ptp(trace["total_energy"])))
    return float(np.max(np.abs(trace["energy_balance_residual"])) / scale)


def _evaluate_horizon(
    traces: dict[str, dict[str, Any]],
    horizon_s: float,
    time_step_s: float,
    closure_limit: float,
) -> dict[str, Any]:
    count = int(round(horizon_s / time_step_s)) + 1
    prefixes = {engine: _prefix(trace, count) for engine, trace in traces.items()}
    comparison = compare_bridge_traces(prefixes["mujoco"], prefixes["pinocchio"])
    closures = {engine: _energy_closure(trace) for engine, trace in prefixes.items()}
    closure_pass = all(value <= closure_limit for value in closures.values())
    return {
        "horizon_s": horizon_s,
        **comparison,
        "energy_closure_by_engine": closures,
        "energy_closure_gate_passed": closure_pass,
        "all_gates_passed": bool(
            comparison["trajectory_gate_passed"]
            and comparison["wrench_gate_passed"]
            and comparison["energy_gate_passed"]
            and closure_pass
        ),
    }


def _summarize(
    rows: list[dict[str, Any]], variants: tuple[HorizonVariant, ...]
) -> dict[str, Any]:
    summaries = []
    for variant in variants:
        selected = [row for row in rows if row["variant_id"] == variant.variant_id]
        by_horizon = []
        for horizon in sorted({row["horizon_s"] for row in selected}):
            cohort = [row for row in selected if row["horizon_s"] == horizon]
            by_horizon.append(
                {
                    "horizon_s": horizon,
                    "case_count": len(cohort),
                    "pass_fraction": float(
                        np.mean([row["all_gates_passed"] for row in cohort])
                    ),
                    "worst_position_max_m": max(
                        row["observed_metrics"]["club_position_max_m"] for row in cohort
                    ),
                    "worst_wrench_relative_rms": max(
                        row["observed_metrics"]["contact_wrench_relative_rms"]
                        for row in cohort
                    ),
                    "worst_energy_discrepancy": max(
                        row["observed_metrics"]["normalized_energy_discrepancy"]
                        for row in cohort
                    ),
                    "worst_energy_closure": max(
                        max(row["energy_closure_by_engine"].values()) for row in cohort
                    ),
                }
            )
        first_failure = next(
            (item["horizon_s"] for item in by_horizon if item["pass_fraction"] < 1.0),
            None,
        )
        summaries.append(
            {
                "variant_id": variant.variant_id,
                "first_incomplete_pass_horizon_s": first_failure,
                "by_horizon": by_horizon,
            }
        )
    return {"variants": summaries}


def run_validity_horizon_study(
    config: HorizonStudyConfig = HorizonStudyConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Execute the complete paired horizon and one-factor adverse-load matrix."""

    with np.load(DATA_DIR / "closed_state_forward_bridge.npz") as archive:
        source = {name: archive[name].copy() for name in archive.files}
    phase_indices = (0, source["time_s"].size // 2, source["time_s"].size - 1)
    variants = registered_variants(config)
    rows: list[dict[str, Any]] = []
    result_matrix = np.zeros((len(variants), 18, 3, len(config.horizons_s), 8))
    profiles = default_synthetic_profiles()
    for variant_index, variant in enumerate(variants):
        for case in range(source["case_profile_index"].size):
            profile_index = int(source["case_profile_index"][case])
            span = float(source["case_grip_span_m"][case])
            for phase_slot, phase_index in enumerate(phase_indices):
                state, params = _parameters(
                    source["canonical_state"][case, phase_index],
                    profile_index,
                    span,
                    variant,
                )
                traces = {
                    engine: run_bridge_trace(
                        engine,
                        params,
                        state,
                        duration_s=max(config.horizons_s),
                        driver_enabled=variant.driver_enabled,
                    )
                    for engine in ("mujoco", "pinocchio")
                }
                for horizon_index, horizon in enumerate(config.horizons_s):
                    outcome = _evaluate_horizon(
                        traces, horizon, params.time_step, config.energy_closure_limit
                    )
                    row = {
                        "variant_id": variant.variant_id,
                        "case_index": case,
                        "profile_id": profiles[profile_index].profile_id,
                        "grip_span_m": span,
                        "phase_index": phase_index,
                        **outcome,
                    }
                    rows.append(row)
                    metrics = outcome["observed_metrics"]
                    result_matrix[variant_index, case, phase_slot, horizon_index] = (
                        metrics["club_position_rms_m"],
                        metrics["club_position_max_m"],
                        metrics["club_orientation_max_rad"],
                        metrics["contact_wrench_relative_rms"],
                        metrics["normalized_energy_discrepancy"],
                        outcome["energy_closure_by_engine"]["mujoco"],
                        outcome["energy_closure_by_engine"]["pinocchio"],
                        float(outcome["all_gates_passed"]),
                    )
    record = {
        "schema_version": "forward-contact-validity-horizon/v1",
        "study_id": "closed-state-cross-engine-validity-horizon",
        "issue": "https://github.com/D-sorganization/UpstreamDrift/issues/8666",
        "design": {
            "configuration": asdict(config),
            "variants": [asdict(variant) for variant in variants],
            "phase_indices": list(phase_indices),
            "case_count": 18,
            "evaluated_trace_count": len(variants) * 18 * 3 * 2,
            "evaluated_horizon_case_count": len(rows),
            "metric_order": [
                "club_position_rms_m",
                "club_position_max_m",
                "club_orientation_max_rad",
                "contact_wrench_relative_rms",
                "normalized_energy_discrepancy",
                "mujoco_energy_closure",
                "pinocchio_energy_closure",
                "all_gates_passed",
            ],
        },
        "results": _summarize(rows, variants),
        "cases": rows,
        "controls": {
            "initial_zero_preload": "inherited_exactly_from_closed-state-forward-bridge/v1",
            "action_reaction": "equal-and-opposite force pair by constitutive construction",
            "coincident_moment_arm": "zero couple by transport identity",
            "reversed_moment_arm": "couple sign reversal by transport identity",
            "driver_null": "driver_off variant applies zero driver force from the first integration step",
        },
        "claim_boundary": {
            "reduced_cross_engine_persistence": "evaluated",
            "articulated_anatomy": "not_established",
            "calibrated_equipment": "not_established",
            "full_delivery_or_impact": "not_established",
            "human_or_coaching_strategy": "not_established",
        },
    }
    arrays = {
        "result_matrix": result_matrix,
        "horizons_s": np.asarray(config.horizons_s),
        "phase_indices": np.asarray(phase_indices),
        "variant_ids": np.asarray([variant.variant_id for variant in variants]),
    }
    return record, arrays


__all__ = [
    "HorizonStudyConfig",
    "HorizonVariant",
    "registered_variants",
    "run_validity_horizon_study",
]
