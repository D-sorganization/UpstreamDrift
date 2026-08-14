"""Test delayed/noisy state triggering and trajectory-level error recovery.

The experiment deliberately distinguishes three concepts: terminal outcome
dispersion, decay of a state error after a matched perturbation, and evidence
about human correction.  Only the second is called recovery here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt

from scripts.research.proximal_distal_energy.moving_base_flexible_club import (
    MovingBaseFlexibleConfig,
    MovingBaseFlexibleParams,
    MovingBaseFlexibleTrace,
    initial_state,
    rollout,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import TwoArmControl
from scripts.research.proximal_distal_energy.uncertainty_control import (
    ActuatorLimits,
    ControlProgram,
    delayed_control_law,
    latin_hypercube,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
JSON_PATH = DATA_DIR / "observer_recovery_study.json"
NPZ_PATH = DATA_DIR / "observer_recovery_study.npz"
FIGURE_PATH = (
    REPO_ROOT
    / "docs/research/proximal_distal_energy_transfer/figures/fig_observer_recovery.pdf"
)
SCHEMA_VERSION = "proximal-distal-observer-recovery-v1"
DURATION_S = 0.24
STEP_S = 0.004
SAMPLE_COUNT = 15
SEED = 8595
STATE_SCALES = np.array(
    [0.05, 0.05, 0.05, 0.05, 0.07, 0.02, 0.5, 0.5, 0.5, 0.5, 0.7, 0.2]
)
METRIC_NAMES = (
    "initial_normalized_error",
    "terminal_error_ratio",
    "minimum_error_ratio",
    "recovery_time_s",
    "returned_to_viable_set",
    "terminal_delivery_speed_m_s",
    "peak_hand_force_n",
    "effort_proxy_nms",
)


@dataclass(frozen=True, slots=True)
class ObserverCondition:
    """Declared state-access condition for one policy."""

    name: str
    trigger: str
    delay_s: float
    angle_noise_sd_rad: float
    impedance_nms_rad: float


CONDITIONS = (
    ObserverCondition("clock", "clock", 0.0, 0.0, 0.08),
    ObserverCondition("perfect_state", "state", 0.0, 0.0, 0.08),
    ObserverCondition("delayed_noisy", "state", 0.032, 0.012, 0.08),
    ObserverCondition("delayed_noisy_higher_impedance", "state", 0.032, 0.012, 0.24),
)


def _design() -> np.ndarray:
    unit = latin_hypercube(SAMPLE_COUNT, 8, seed=SEED)
    half_ranges = np.array([0.035, 0.045, 0.35, 0.45, 0.08, 0.012, 0.18, 0.12])
    return (2.0 * unit - 1.0) * half_ranges


def _initial_perturbation(
    q: np.ndarray, qdot: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    perturbed_q, perturbed_qdot = q.copy(), qdot.copy()
    perturbed_q[[0, 2, 8]] += value[0]
    perturbed_q[[1, 3]] += value[1]
    perturbed_qdot[[0, 2, 8]] += value[2]
    perturbed_qdot[[1, 3]] += value[3]
    return perturbed_q, perturbed_qdot


def _law(
    condition: ObserverCondition,
    value: np.ndarray,
    limits: ActuatorLimits,
    *,
    noise_phase: float,
):
    scale = 1.0 + float(value[4])
    program = ControlProgram(
        name=condition.name,
        wrist_onset_s=0.14,
        early_wrist_nm=-2.0 * scale,
        late_wrist_nm=5.0 * scale,
        shoulder_scale=scale,
        elbow_scale=scale,
        impedance_nms_rad=condition.impedance_nms_rad,
    )
    if condition.trigger == "clock":
        return delayed_control_law(
            program, limits, duration_s=DURATION_S, step_s=STEP_S
        )

    early = delayed_control_law(
        replace(program, wrist_onset_s=DURATION_S),
        limits,
        duration_s=DURATION_S,
        step_s=STEP_S,
    )
    late = delayed_control_law(
        replace(program, wrist_onset_s=0.0),
        limits,
        duration_s=DURATION_S,
        step_s=STEP_S,
    )

    def state_law(time_s: float, q: np.ndarray, qdot: np.ndarray) -> TwoArmControl:
        # First-order back-extrapolation is an explicit, deliberately simple
        # delayed observer.  The deterministic two-frequency signal makes the
        # sensor uncertainty reproducible and independent of integrator calls.
        observed_angle = float(q[0] - condition.delay_s * qdot[0])
        observed_angle += (
            condition.angle_noise_sd_rad
            * (
                math.sin(2.0 * math.pi * 31.0 * time_s + noise_phase)
                + 0.5 * math.sin(2.0 * math.pi * 47.0 * time_s + 0.7 * noise_phase)
            )
            / math.sqrt(1.25)
        )
        selected = late if observed_angle >= -0.42 else early
        return selected(time_s, q, qdot)

    return state_law


def _rollout(
    condition: ObserverCondition,
    value: np.ndarray,
    *,
    perturb_initial_state: bool,
    noise_phase: float,
) -> MovingBaseFlexibleTrace:
    base = MovingBaseFlexibleParams.publication_default()
    params = replace(
        base,
        shaft_stiffness_nm_rad=base.shaft_stiffness_nm_rad * (1.0 + value[6]),
        right_grip_offset_m=base.right_grip_offset_m * (1.0 + value[7]),
        left_grip_offset_m=base.left_grip_offset_m * (1.0 + value[7]),
    )
    q0, qdot0 = initial_state(params)
    if perturb_initial_state:
        q0, qdot0 = _initial_perturbation(q0, qdot0, value)
    limits = ActuatorLimits(delay_s=0.025 + float(value[5]))
    return rollout(
        q0,
        qdot0,
        _law(condition, value, limits, noise_phase=noise_phase),
        params,
        MovingBaseFlexibleConfig(duration_s=DURATION_S, step_s=STEP_S),
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


def _metrics(trace: MovingBaseFlexibleTrace, error: np.ndarray) -> np.ndarray:
    initial = max(float(error[0]), 1e-12)
    ratio = error / initial
    threshold = 0.5
    recovery_index = len(error) - 1
    recovered = False
    for index in range(5, len(error) - 2):
        if np.all(ratio[index:] <= threshold):
            recovery_index = index
            recovered = True
            break
    controls = np.asarray(
        [list(asdict(control).values()) for control in trace.controls], dtype=float
    )
    return np.array(
        [
            initial,
            ratio[-1],
            np.min(ratio[5:]),
            trace.time[recovery_index] if recovered else DURATION_S,
            float(recovered),
            np.linalg.norm(trace.clubhead_velocity_m_s[-1]),
            np.max(np.linalg.norm(trace.contact_force_on_club_n, axis=2)),
            np.trapezoid(np.sum(controls**2, axis=1), x=trace.time),
        ]
    )


def run_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute paired reference/perturbed rollouts for every observer policy."""

    design = _design()
    errors = np.empty((len(CONDITIONS), SAMPLE_COUNT, int(DURATION_S / STEP_S) + 1))
    metrics = np.empty((len(CONDITIONS), SAMPLE_COUNT, len(METRIC_NAMES)))
    for condition_index, condition in enumerate(CONDITIONS):
        for sample_index, value in enumerate(design):
            phase = 2.0 * math.pi * (sample_index + 0.5) / SAMPLE_COUNT
            reference = _rollout(
                condition, value, perturb_initial_state=False, noise_phase=phase
            )
            perturbed = _rollout(
                condition, value, perturb_initial_state=True, noise_phase=phase
            )
            errors[condition_index, sample_index] = _state_error(perturbed, reference)
            metrics[condition_index, sample_index] = _metrics(
                perturbed, errors[condition_index, sample_index]
            )

    recovery_index = METRIC_NAMES.index("returned_to_viable_set")
    summaries = {
        condition.name: {
            "recovery_fraction": float(np.mean(metrics[index, :, recovery_index])),
            "median_terminal_error_ratio": float(np.median(metrics[index, :, 1])),
            "median_recovery_time_s": float(np.median(metrics[index, :, 3])),
            "q90_peak_hand_force_n": float(np.quantile(metrics[index, :, 6], 0.9)),
            "mean_effort_proxy_nms": float(np.mean(metrics[index, :, 7])),
        }
        for index, condition in enumerate(CONDITIONS)
    }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "delayed-observer-trajectory-recovery-v1",
        "registered_before_preferred_result": True,
        "model_tier": "moving_base_two_hand_compliant_club",
        "design": {
            "paired_common_random_numbers": True,
            "held_out_perturbations": SAMPLE_COUNT,
            "seed": SEED,
            "duration_s": DURATION_S,
            "step_s": STEP_S,
            "matched_reference_rule": "same nuisance parameters and sensor realization; initial state perturbation removed",
        },
        "observer_conditions": {item.name: asdict(item) for item in CONDITIONS},
        "metric_names": list(METRIC_NAMES),
        "recovery_definition": {
            "error_state": "six angles and six rates normalized by declared engineering scales",
            "threshold": "error at or below 50 percent of initial perturbation",
            "sustained_samples": "all remaining samples through delivery",
            "requires_sustained_error_decay": True,
            "nonrecovery_time_s": DURATION_S,
        },
        "summaries": summaries,
        "adverse_costs": ["peak_hand_force_n", "effort_proxy_nms"],
        "claim_status": {
            "low_sensitivity_is_recovery": "rejected",
            "model_policy_recovery": "conditional_on_reported_policy_and_envelope",
            "human_self_correction": "untested",
            "universal_timing_advantage": "unsupported",
        },
        "limitations": [
            "The observer is a first-order delayed approximation, not an identified human sensory estimator.",
            "The perturbation envelope is an engineering design, not a fitted golfer population.",
            "Recovery to a matched model trajectory is not proof of motor learning, reflex correction, or coaching benefit.",
            "The terminal sample is a delivery proxy rather than a ball-impact collision.",
        ],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "array_artifact": NPZ_PATH.name,
    }
    return record, {
        "perturbations": design,
        "normalized_error_trajectories": errors,
        "metrics": metrics,
    }


def _write_figure(errors: np.ndarray, metrics: np.ndarray) -> Path:
    """Write the reader-facing trajectory and adverse-cost comparison."""

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(errors.shape[-1]) * STEP_S
    colors = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for index, (condition, color) in enumerate(zip(CONDITIONS, colors, strict=True)):
        normalized = errors[index] / np.maximum(errors[index, :, :1], 1e-12)
        median = np.median(normalized, axis=0)
        axes[0].plot(time, median, color=color, label=condition.name.replace("_", " "))
        axes[0].fill_between(
            time,
            np.quantile(normalized, 0.1, axis=0),
            np.quantile(normalized, 0.9, axis=0),
            color=color,
            alpha=0.12,
        )
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Normalized State Error",
        title="Median Error and 10th–90th Percentile Band",
    )
    axes[0].legend(frameon=False, fontsize=8)
    recovery = np.mean(metrics[:, :, 4], axis=1)
    force = np.quantile(metrics[:, :, 6], 0.9, axis=1)
    labels = [
        item.name.replace("delayed_noisy", "delayed\nnoisy").replace("_", " ")
        for item in CONDITIONS
    ]
    bars = axes[1].bar(labels, recovery, color=colors)
    axes[1].set(
        ylim=(0.0, 1.0),
        ylabel="Sustained Recovery Fraction",
        title="Recovery With 90th-Percentile Hand Force",
    )
    axes[1].tick_params(axis="x", labelsize=8)
    for bar, value in zip(bars, force, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{value:.0f} N",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)
    return FIGURE_PATH


def write_outputs() -> tuple[Path, Path, Path]:
    """Write deterministic human-readable and numeric evidence."""

    record, arrays = run_study()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cast(Any, np.savez_compressed)(NPZ_PATH, **arrays)
    return (
        JSON_PATH,
        NPZ_PATH,
        _write_figure(arrays["normalized_error_trajectories"], arrays["metrics"]),
    )


if __name__ == "__main__":
    for output in write_outputs():
        print(output)
