"""Quantify timing-region viability and recovery under declared adverse loads."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from scripts.research.proximal_distal_energy.moving_base_flexible_club import (
    MovingBaseFlexibleConfig,
    MovingBaseFlexibleParams,
    MovingBaseFlexibleTrace,
    initial_state,
    rollout,
)
from scripts.research.proximal_distal_energy.timing_viability import (
    ViabilityLimits,
    summarize_timing_viability,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import TwoArmControl
from scripts.research.proximal_distal_energy.uncertainty_control import (
    ActuatorLimits,
    ControlProgram,
    delayed_control_law,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"
FIGURE_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/figures"
JSON_PATH = DATA_DIR / "timing_viability_study.json"
NPZ_PATH = DATA_DIR / "timing_viability_study.npz"
FIGURE_PATH = FIGURE_DIR / "fig_timing_viability_adverse_load.pdf"
SCHEMA_VERSION = "timing-viability-adverse-load/v1"
ISSUE = 8623
PARENT_EPIC = 8557
DURATION_S = 0.24
STEP_S = 0.004
NOMINAL_EVENT_TIME_S = 0.14
PHASE_OFFSETS_S = np.array([-0.030, -0.015, 0.0, 0.015, 0.030])
POLICY_NAMES = ("clock", "state_triggered")
METRIC_NAMES = (
    "delivery_speed_m_s",
    "face_path_error_deg",
    "peak_hand_force_n",
    "effort_proxy_nms",
    "returned_to_viable_set",
    "normalized_energy_residual",
    "realized_event_time_s",
)
STATE_SCALES = np.array(
    [0.05, 0.05, 0.05, 0.05, 0.07, 0.02, 0.5, 0.5, 0.5, 0.5, 0.7, 0.2]
)
BASE_PERTURBATION = np.array([0.022, -0.018, 0.22, -0.17])


@dataclass(frozen=True, slots=True)
class LoadCase:
    """One declared nuisance/load cohort applied to both policies."""

    name: str
    distal_mass_scale: float = 1.0
    shaft_stiffness_scale: float = 1.0
    actuator_delay_add_s: float = 0.0
    perturbation_scale: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.distal_mass_scale,
                self.shaft_stiffness_scale,
                self.actuator_delay_add_s,
                self.perturbation_scale,
            ]
        )
        if not self.name.strip() or not np.all(np.isfinite(values)):
            raise ValueError("load-case name and values must be finite")
        if self.distal_mass_scale <= 0.0 or self.shaft_stiffness_scale <= 0.0:
            raise ValueError("mass and stiffness scales must be positive")
        if self.actuator_delay_add_s < 0.0 or self.perturbation_scale <= 0.0:
            raise ValueError("delay must be nonnegative and perturbation positive")


LOAD_CASES = (
    LoadCase("nominal"),
    LoadCase("heavier_club", distal_mass_scale=1.15),
    LoadCase("lower_shaft_stiffness", shaft_stiffness_scale=0.80),
    LoadCase("longer_actuator_delay", actuator_delay_add_s=0.012),
    LoadCase("larger_initial_perturbation", perturbation_scale=1.50),
    LoadCase(
        "combined_adverse",
        distal_mass_scale=1.10,
        shaft_stiffness_scale=0.85,
        actuator_delay_add_s=0.010,
        perturbation_scale=1.30,
    ),
)

LIMITS = {
    "strict": ViabilityLimits(
        speed_fraction_min=0.98,
        face_error_allowance_deg=1.0,
        peak_force_ratio_max=1.05,
        effort_ratio_max=1.05,
    ),
    "primary": ViabilityLimits(),
    "lenient": ViabilityLimits(
        speed_fraction_min=0.90,
        face_error_allowance_deg=4.0,
        peak_force_ratio_max=1.20,
        effort_ratio_max=1.20,
    ),
}


def _program(name: str, onset_s: float) -> ControlProgram:
    return ControlProgram(
        name=name,
        wrist_onset_s=onset_s,
        early_wrist_nm=-2.0,
        late_wrist_nm=5.0,
        shoulder_scale=1.0,
        elbow_scale=1.0,
        impedance_nms_rad=0.08,
    )


def _parameters(load: LoadCase) -> MovingBaseFlexibleParams:
    base = MovingBaseFlexibleParams.publication_default()
    return replace(
        base,
        distal_club_mass_kg=base.distal_club_mass_kg * load.distal_mass_scale,
        distal_club_inertia_kg_m2=(
            base.distal_club_inertia_kg_m2 * load.distal_mass_scale
        ),
        shaft_stiffness_nm_rad=(
            base.shaft_stiffness_nm_rad * load.shaft_stiffness_scale
        ),
    )


def _clock_law(offset_s: float, limits: ActuatorLimits, *, step_s: float):
    return delayed_control_law(
        _program("clock", NOMINAL_EVENT_TIME_S + offset_s),
        limits,
        duration_s=DURATION_S,
        step_s=step_s,
    )


def _state_law(
    threshold_rad: float,
    limits: ActuatorLimits,
    *,
    noise_phase: float,
    step_s: float,
):
    early = delayed_control_law(
        _program("state_early", DURATION_S),
        limits,
        duration_s=DURATION_S,
        step_s=step_s,
    )
    late = delayed_control_law(
        _program("state_late", 0.0),
        limits,
        duration_s=DURATION_S,
        step_s=step_s,
    )
    observer_delay_s = 0.016
    noise_sd_rad = 0.006

    def law(time_s: float, q: np.ndarray, qdot: np.ndarray) -> TwoArmControl:
        observed = float(q[0] - observer_delay_s * qdot[0])
        observed += noise_sd_rad * math.sin(2.0 * math.pi * 31.0 * time_s + noise_phase)
        selected = late if observed >= threshold_rad else early
        return selected(time_s, q, qdot)

    return law


def _perturb_state(
    q: np.ndarray, qdot: np.ndarray, load: LoadCase
) -> tuple[np.ndarray, np.ndarray]:
    q_value = q.copy()
    qdot_value = qdot.copy()
    perturbation = BASE_PERTURBATION * load.perturbation_scale
    q_value[[0, 2, 8]] += perturbation[0]
    q_value[[1, 3]] += perturbation[1]
    qdot_value[[0, 2, 8]] += perturbation[2]
    qdot_value[[1, 3]] += perturbation[3]
    return q_value, qdot_value


def _run(
    policy: str,
    offset_s: float,
    threshold_rad: float,
    load: LoadCase,
    *,
    perturb: bool,
    noise_phase: float,
    step_s: float = STEP_S,
) -> MovingBaseFlexibleTrace:
    params = _parameters(load)
    q0, qdot0 = initial_state(params)
    if perturb:
        q0, qdot0 = _perturb_state(q0, qdot0, load)
    limits = ActuatorLimits(delay_s=0.025 + load.actuator_delay_add_s)
    law = (
        _clock_law(offset_s, limits, step_s=step_s)
        if policy == "clock"
        else _state_law(threshold_rad, limits, noise_phase=noise_phase, step_s=step_s)
    )
    return rollout(
        q0,
        qdot0,
        law,
        params,
        MovingBaseFlexibleConfig(duration_s=DURATION_S, step_s=step_s),
    )


def _state_error(
    perturbed: MovingBaseFlexibleTrace, reference: MovingBaseFlexibleTrace
) -> np.ndarray:
    coordinates = (0, 1, 2, 3, 8, 9)
    state = np.column_stack(
        (
            perturbed.q[:, coordinates] - reference.q[:, coordinates],
            perturbed.qdot[:, coordinates] - reference.qdot[:, coordinates],
        )
    )
    return np.sqrt(np.mean((state / STATE_SCALES) ** 2, axis=1))


def _sustained_recovery(error: np.ndarray) -> bool:
    initial = max(float(error[0]), 1e-12)
    ratio = error / initial
    return any(np.all(ratio[index:] <= 0.5) for index in range(5, len(ratio) - 2))


def _realized_event_time(
    policy: str,
    offset_s: float,
    threshold_rad: float,
    trace: MovingBaseFlexibleTrace,
    *,
    noise_phase: float,
) -> float:
    if policy == "clock":
        return NOMINAL_EVENT_TIME_S + offset_s
    observed = trace.q[:, 0] - 0.016 * trace.qdot[:, 0]
    observed += 0.006 * np.sin(2.0 * math.pi * 31.0 * trace.time + noise_phase)
    crossings = np.flatnonzero(observed >= threshold_rad)
    return DURATION_S if crossings.size == 0 else float(trace.time[int(crossings[0])])


def _outcomes(
    trace: MovingBaseFlexibleTrace,
    error: np.ndarray,
    event_time_s: float,
) -> np.ndarray:
    delivery_velocity = trace.clubhead_velocity_m_s[-1]
    path_angle = math.atan2(float(delivery_velocity[1]), float(delivery_velocity[0]))
    face_angle = float(trace.q[-1, 8] + trace.q[-1, 9])
    face_error = abs(
        math.degrees(
            math.atan2(
                math.sin(face_angle - path_angle), math.cos(face_angle - path_angle)
            )
        )
    )
    controls = np.asarray(
        [list(asdict(control).values()) for control in trace.controls], dtype=float
    )
    work_control = float(np.trapezoid(trace.applied_control_power_w, x=trace.time))
    work_dissipation = float(np.trapezoid(trace.dissipation_power_w, x=trace.time))
    projection_work = float(np.sum(trace.projection_energy_change_j))
    energy_change = float(trace.mechanical_energy_j[-1] - trace.mechanical_energy_j[0])
    closure = work_control + work_dissipation + projection_work - energy_change
    scale = max(abs(work_control), abs(work_dissipation), abs(energy_change), 1e-12)
    return np.array(
        [
            np.linalg.norm(delivery_velocity),
            face_error,
            np.max(np.linalg.norm(trace.contact_force_on_club_n, axis=2)),
            np.trapezoid(np.sum(controls**2, axis=1), x=trace.time),
            float(_sustained_recovery(error)),
            abs(closure) / scale,
            event_time_s,
        ],
        dtype=np.float64,
    )


def _reference_outcomes(
    trace: MovingBaseFlexibleTrace, event_time_s: float
) -> np.ndarray:
    error = np.zeros_like(trace.time)
    result = _outcomes(trace, error, event_time_s)
    result[METRIC_NAMES.index("returned_to_viable_set")] = 1.0
    return result


def _common_phase_mapping() -> tuple[np.ndarray, MovingBaseFlexibleTrace]:
    load = LOAD_CASES[0]
    nominal = _run("clock", 0.0, 0.0, load, perturb=False, noise_phase=0.0)
    target_times = NOMINAL_EVENT_TIME_S + PHASE_OFFSETS_S
    thresholds = np.interp(target_times, nominal.time, nominal.q[:, 0])
    if np.any(np.diff(thresholds) <= 0.0):
        raise ValueError("common nominal phase mapping is not one-to-one")
    return thresholds, nominal


def _policy_ordering(sensitivity: dict[str, Any]) -> str:
    directions: list[int] = []
    for item in sensitivity.values():
        clock = item["policies"]["clock"]
        state = item["policies"]["state_triggered"]
        clock_score = (
            clock["robust_viable_fraction"],
            clock["robust_contiguous_width_s"],
        )
        state_score = (
            state["robust_viable_fraction"],
            state["robust_contiguous_width_s"],
        )
        directions.append((state_score > clock_score) - (state_score < clock_score))
    if len(set(directions)) > 1:
        return "threshold_sensitive"
    if directions[0] > 0:
        return "state_larger"
    if directions[0] < 0:
        return "clock_larger"
    return "no_separation"


def _numerical_sensitivity(threshold_rad: float) -> dict[str, Any]:
    """Refine the timestep for one adverse representative from each policy."""

    load = LOAD_CASES[-1]
    result: dict[str, Any] = {}
    for policy in POLICY_NAMES:
        metrics: dict[str, np.ndarray] = {}
        for step in (STEP_S, STEP_S / 2.0):
            trace = _run(
                policy,
                0.0,
                threshold_rad,
                load,
                perturb=True,
                noise_phase=math.pi,
                step_s=step,
            )
            event_time = _realized_event_time(
                policy,
                0.0,
                threshold_rad,
                trace,
                noise_phase=math.pi,
            )
            metrics[str(step)] = _reference_outcomes(trace, event_time)
        coarse = metrics[str(STEP_S)]
        refined = metrics[str(STEP_S / 2.0)]
        result[policy] = {
            "coarse_step_s": STEP_S,
            "refined_step_s": STEP_S / 2.0,
            "delivery_speed_absolute_difference_m_s": float(
                abs(coarse[0] - refined[0])
            ),
            "peak_hand_force_relative_difference": float(
                abs(coarse[2] - refined[2]) / max(abs(refined[2]), 1e-12)
            ),
            "coarse_normalized_energy_residual": float(coarse[5]),
            "refined_normalized_energy_residual": float(refined[5]),
        }
    return result


def run_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute the preregistered common-phase, paired-perturbation study."""

    thresholds, nominal_mapping_trace = _common_phase_mapping()
    shape = (
        len(POLICY_NAMES),
        len(LOAD_CASES),
        len(PHASE_OFFSETS_S),
        len(METRIC_NAMES),
    )
    outcomes = np.empty(shape, dtype=np.float64)
    references = np.empty(shape, dtype=np.float64)
    errors = np.empty(shape[:-1] + (nominal_mapping_trace.time.size,), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(POLICY_NAMES):
        for load_index, load in enumerate(LOAD_CASES):
            noise_phase = 2.0 * math.pi * (load_index + 0.5) / len(LOAD_CASES)
            for phase_index, (offset, threshold) in enumerate(
                zip(PHASE_OFFSETS_S, thresholds, strict=True)
            ):
                reference = _run(
                    policy,
                    float(offset),
                    float(threshold),
                    load,
                    perturb=False,
                    noise_phase=noise_phase,
                )
                perturbed = _run(
                    policy,
                    float(offset),
                    float(threshold),
                    load,
                    perturb=True,
                    noise_phase=noise_phase,
                )
                error = _state_error(perturbed, reference)
                event_time = _realized_event_time(
                    policy,
                    float(offset),
                    float(threshold),
                    perturbed,
                    noise_phase=noise_phase,
                )
                outcomes[policy_index, load_index, phase_index] = _outcomes(
                    perturbed, error, event_time
                )
                references[policy_index, load_index, phase_index] = _reference_outcomes(
                    reference, event_time
                )
                errors[policy_index, load_index, phase_index] = error
                rows.append(
                    {
                        "policy": policy,
                        "load_case": load.name,
                        "phase_offset_s": float(offset),
                        "state_angle_threshold_rad": float(threshold),
                        "status": "valid",
                        "outcomes": dict(
                            zip(
                                METRIC_NAMES,
                                outcomes[
                                    policy_index, load_index, phase_index
                                ].tolist(),
                                strict=True,
                            )
                        ),
                    }
                )

    zero_phase_index = int(np.flatnonzero(PHASE_OFFSETS_S == 0.0)[0])
    common_baselines = outcomes[0, :, zero_phase_index]
    sensitivity: dict[str, Any] = {}
    recovery_sensitivity: dict[str, Any] = {}
    for name, limits in LIMITS.items():
        sensitivity[name] = {
            "policies": {
                policy: summarize_timing_viability(
                    PHASE_OFFSETS_S,
                    outcomes[index],
                    common_baselines,
                    load_names=tuple(item.name for item in LOAD_CASES),
                    metric_names=METRIC_NAMES,
                    limits=replace(limits, require_sustained_recovery=False),
                )
                for index, policy in enumerate(POLICY_NAMES)
            }
        }
        recovery_sensitivity[name] = {
            "policies": {
                policy: summarize_timing_viability(
                    PHASE_OFFSETS_S,
                    outcomes[index],
                    common_baselines,
                    load_names=tuple(item.name for item in LOAD_CASES),
                    metric_names=METRIC_NAMES,
                    limits=limits,
                )
                for index, policy in enumerate(POLICY_NAMES)
            }
        }

    recovery_index = METRIC_NAMES.index("returned_to_viable_set")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "common-phase-timing-viability-adverse-load-recovery",
        "parent_epic": PARENT_EPIC,
        "issue": ISSUE,
        "registered_before_preferred_result": True,
        "model_tier": "moving_base_two_hand_compliant_club",
        "policies": list(POLICY_NAMES),
        "phase_offsets_s": PHASE_OFFSETS_S.tolist(),
        "nominal_event_time_s": NOMINAL_EVENT_TIME_S,
        "common_phase_mapping": {
            "source": "common_nominal_clock_trajectory",
            "target_event_times_s": (NOMINAL_EVENT_TIME_S + PHASE_OFFSETS_S).tolist(),
            "state_angle_thresholds_rad": thresholds.tolist(),
            "mapping_coordinate": "right_shoulder_angle_rad",
        },
        "load_cases": [asdict(item) for item in LOAD_CASES],
        "case_count": int(np.prod(shape[:-1])),
        "metric_names": list(METRIC_NAMES),
        "common_baseline_rule": "clock policy at zero phase offset within each load cohort",
        "numerical_guard_basis": "A five-percent normalized residual ceiling is paired with explicit half-step sensitivity because the constrained projection integrator's registered coarse-step residual is percent-scale.",
        "recovery_definition": {
            "state": "six declared angles and six rates normalized by engineering scales",
            "criterion": "error at or below 50 percent of initial perturbation for every remaining delivery sample",
            "interpretation": "trajectory recovery in this model, not neural correction",
        },
        "rows": rows,
        "viability_sensitivity": sensitivity,
        "recovery_qualified_viability_sensitivity": recovery_sensitivity,
        "numerical_sensitivity": _numerical_sensitivity(
            float(thresholds[zero_phase_index])
        ),
        "recovery_fraction_by_policy_and_load": {
            policy: {
                load.name: float(np.mean(outcomes[p_index, l_index, :, recovery_index]))
                for l_index, load in enumerate(LOAD_CASES)
            }
            for p_index, policy in enumerate(POLICY_NAMES)
        },
        "claim_status": {
            "model_policy_ordering": _policy_ordering(sensitivity),
            "recovery_policy_ordering": _policy_ordering(recovery_sensitivity),
            "state_triggered_larger_timing_region": "not_supported",
            "registered_sustained_recovery": "not_observed_in_any_case",
            "human_timing_demand": "untested",
            "human_self_correction": "untested",
            "coaching_prescription": "unsupported",
        },
        "limitations": [
            "The terminal state is a delivery proxy, not a modeled ball-impact collision.",
            "The declared engineering load cases are not a golfer population or fitted probability distribution.",
            "The state observer is a delayed noisy engineering surrogate, not an identified sensory estimator.",
            "Sustained return toward a matched trajectory is model recovery, not motor learning, reflex correction, or coaching evidence.",
            "The common phase map is local to one nominal trajectory and must be rebuilt for another model tier or subject.",
        ],
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("timing_viability.py"),
                Path(__file__).with_name("moving_base_flexible_club.py"),
                Path(__file__).with_name("uncertainty_control.py"),
            )
        },
        "array_artifact": NPZ_PATH.name,
        "figure_artifact": FIGURE_PATH.name,
    }
    return record, {
        "phase_offsets_s": PHASE_OFFSETS_S,
        "state_angle_thresholds_rad": thresholds,
        "outcomes": outcomes,
        "reference_outcomes": references,
        "normalized_error_trajectories": errors,
        "common_baselines": common_baselines,
    }


def _write_figure(record: dict[str, Any], outcomes: np.ndarray) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), constrained_layout=True)
    colors = {True: "#2A9D8F", False: "#D9D9D9"}
    primary = record["viability_sensitivity"]["primary"]["policies"]
    for policy_index, policy in enumerate(POLICY_NAMES):
        masks = np.asarray(
            [
                primary[policy]["per_load"][load.name]["viable_mask"]
                for load in LOAD_CASES
            ],
            dtype=bool,
        )
        y_base = policy_index * (len(LOAD_CASES) + 1)
        for load_index, _load in enumerate(LOAD_CASES):
            for phase_index, offset in enumerate(PHASE_OFFSETS_S):
                axes[0].scatter(
                    1000.0 * offset,
                    y_base + load_index,
                    marker="s",
                    s=155,
                    color=colors[bool(masks[load_index, phase_index])],
                    edgecolor="#333333",
                    linewidth=0.4,
                )
    labels = [
        f"{policy.replace('_', ' ').title()}: {load.name.replace('_', ' ').title()}"
        for policy in POLICY_NAMES
        for load in LOAD_CASES
    ]
    positions = [
        p_index * (len(LOAD_CASES) + 1) + l_index
        for p_index in range(len(POLICY_NAMES))
        for l_index in range(len(LOAD_CASES))
    ]
    axes[0].set_yticks(positions, labels, fontsize=7)
    axes[0].set(
        xlabel="Nominal Release-Phase Offset (ms)",
        title="Primary Task Viability by Policy and Load",
    )
    event_index = METRIC_NAMES.index("realized_event_time_s")
    for policy_index, policy in enumerate(POLICY_NAMES):
        event = outcomes[policy_index, :, :, event_index]
        axes[1].plot(
            1000.0 * PHASE_OFFSETS_S,
            1000.0 * np.median(event, axis=0),
            marker="o",
            label=policy.replace("_", " ").title(),
        )
        axes[1].fill_between(
            1000.0 * PHASE_OFFSETS_S,
            1000.0 * np.min(event, axis=0),
            1000.0 * np.max(event, axis=0),
            alpha=0.15,
        )
    axes[1].plot(
        1000.0 * PHASE_OFFSETS_S,
        1000.0 * (NOMINAL_EVENT_TIME_S + PHASE_OFFSETS_S),
        color="#555555",
        linestyle="--",
        label="Target Event Time",
    )
    axes[1].set(
        xlabel="Nominal Release-Phase Offset (ms)",
        ylabel="Realized Trigger Time (ms)",
        title="Realized Event Timing Across Declared Loads",
    )
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)
    return FIGURE_PATH


def write_outputs() -> tuple[Path, Path, Path]:
    """Write deterministic JSON, numerical arrays, and reader-facing figure."""

    record, arrays = run_study()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cast(Any, np.savez_compressed)(NPZ_PATH, **arrays)
    return JSON_PATH, NPZ_PATH, _write_figure(record, arrays["outcomes"])


if __name__ == "__main__":
    for output in write_outputs():
        print(output)
