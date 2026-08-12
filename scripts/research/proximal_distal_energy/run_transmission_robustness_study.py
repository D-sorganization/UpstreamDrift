"""Run paired transmission-pathway and task-robustness experiments.

The study uses the existing moving-base, two-hand, compliant-club tier.  It
compares time-triggered and state-triggered wrist transitions under common
perturbations.  "Self-stabilizing" is intentionally restricted to lower local
input--outcome amplification or recovery of a declared model error; it is not
a statement about reflexes, learning, or a human technique.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts.research.proximal_distal_energy.moving_base_flexible_club import (
    MovingBaseFlexibleConfig,
    MovingBaseFlexibleParams,
    initial_state,
    rollout,
)
from scripts.research.proximal_distal_energy.transmission_robustness import (
    PerturbationEnsemble,
    finite_difference_outcome_jacobian,
    nondominated_indices,
    perturbation_summary,
    task_variance_partition,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import TwoArmControl
from scripts.research.proximal_distal_energy.uncertainty_control import (
    ActuatorLimits,
    ControlProgram,
    delayed_control_law,
    latin_hypercube,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs/research/proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
JSON_PATH = DATA_DIR / "transmission_robustness_study.json"
NPZ_PATH = DATA_DIR / "transmission_robustness_study.npz"
SCHEMA_VERSION = "proximal-distal-transmission-robustness-v1"
DURATION_S = 0.24
STEP_S = 0.004
TRAINING_SAMPLE_COUNT = 10
HELD_OUT_SAMPLE_COUNT = 15
TRAINING_SEED = 8507
HELD_OUT_SEED = 8508
PERTURBATION_NAMES = (
    "initial_arm_angle_rad",
    "initial_wrist_angle_rad",
    "initial_arm_speed_rad_s",
    "initial_wrist_speed_rad_s",
    "command_scale",
    "activation_delay_s",
    "shaft_stiffness_scale",
    "grip_separation_scale",
)
PERTURBATION_HALF_RANGES = np.array(
    [0.035, 0.045, 0.35, 0.45, 0.08, 0.012, 0.18, 0.12], dtype=np.float64
)
OUTCOME_NAMES = (
    "delivery_speed_m_s",
    "face_path_error_deg",
    "peak_hand_force_n",
    "effort_proxy_nms",
    "event_time_s",
    "late_contact_work_j",
    "late_direct_wrist_work_j",
    "shaft_energy_release_j",
)
PROGRAM_NAMES = (
    "clock_restrain_then_drive",
    "state_triggered_handoff",
    "state_triggered_higher_impedance",
    "early_drive",
)


@dataclass(frozen=True, slots=True)
class ProgramSpec:
    """One declared timing/control strategy."""

    name: str
    trigger: str
    onset_s: float
    arm_angle_threshold_rad: float
    early_wrist_nm: float
    late_wrist_nm: float
    impedance_nms_rad: float


PROGRAMS = (
    ProgramSpec("clock_restrain_then_drive", "clock", 0.14, -0.42, -2.0, 5.0, 0.08),
    ProgramSpec("state_triggered_handoff", "arm_angle", 0.14, -0.42, -2.0, 5.0, 0.08),
    ProgramSpec(
        "state_triggered_higher_impedance", "arm_angle", 0.14, -0.42, -2.0, 5.0, 0.24
    ),
    ProgramSpec("early_drive", "clock", 0.0, -0.42, 4.0, 4.0, 0.08),
)


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/research/proximal_distal_energy/transmission_robustness.py",
        "scripts/research/proximal_distal_energy/run_transmission_robustness_study.py",
        "scripts/research/proximal_distal_energy/moving_base_flexible_club.py",
        "scripts/research/proximal_distal_energy/uncertainty_control.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _physical_perturbations(count: int, *, seed: int) -> np.ndarray:
    unit = latin_hypercube(count, len(PERTURBATION_NAMES), seed=seed)
    return (2.0 * unit - 1.0) * PERTURBATION_HALF_RANGES


def _apply_initial_perturbation(
    q: np.ndarray, qdot: np.ndarray, perturbation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    perturbed_q = q.copy()
    perturbed_v = qdot.copy()
    # Common-mode changes preserve the exact two-hand position and velocity
    # constraints while perturbing the proximal and distal relative states.
    perturbed_q[[0, 2, 8]] += perturbation[0]
    perturbed_q[[1, 3]] += perturbation[1]
    perturbed_v[[0, 2, 8]] += perturbation[2]
    perturbed_v[[1, 3]] += perturbation[3]
    return perturbed_q, perturbed_v


def _control_law(
    spec: ProgramSpec,
    perturbation: np.ndarray,
    limits: ActuatorLimits,
):
    scale = 1.0 + float(perturbation[4])
    base_program = ControlProgram(
        name=spec.name,
        wrist_onset_s=spec.onset_s,
        early_wrist_nm=scale * spec.early_wrist_nm,
        late_wrist_nm=scale * spec.late_wrist_nm,
        shoulder_scale=scale,
        elbow_scale=scale,
        impedance_nms_rad=spec.impedance_nms_rad,
    )
    if spec.trigger == "clock":
        return delayed_control_law(
            base_program, limits, duration_s=DURATION_S, step_s=STEP_S
        )

    early_program = replace(base_program, wrist_onset_s=DURATION_S)
    late_program = replace(base_program, wrist_onset_s=0.0)
    early_law = delayed_control_law(
        early_program, limits, duration_s=DURATION_S, step_s=STEP_S
    )
    late_law = delayed_control_law(
        late_program, limits, duration_s=DURATION_S, step_s=STEP_S
    )

    def law(time_s: float, q: np.ndarray, qdot: np.ndarray) -> TwoArmControl:
        selected = late_law if q[0] >= spec.arm_angle_threshold_rad else early_law
        return selected(time_s, q, qdot)

    return law


def _angle_difference(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


def _evaluate(
    spec: ProgramSpec, perturbation: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    base = MovingBaseFlexibleParams.publication_default()
    params = replace(
        base,
        shaft_stiffness_nm_rad=base.shaft_stiffness_nm_rad * (1.0 + perturbation[6]),
        right_grip_offset_m=base.right_grip_offset_m * (1.0 + perturbation[7]),
        left_grip_offset_m=base.left_grip_offset_m * (1.0 + perturbation[7]),
    )
    q0, qdot0 = initial_state(params)
    q0, qdot0 = _apply_initial_perturbation(q0, qdot0, perturbation)
    limits = ActuatorLimits(delay_s=0.025 + float(perturbation[5]))
    trace = rollout(
        q0,
        qdot0,
        _control_law(spec, perturbation, limits),
        params,
        MovingBaseFlexibleConfig(duration_s=DURATION_S, step_s=STEP_S),
    )
    speed = np.linalg.norm(trace.clubhead_velocity_m_s, axis=1)
    delivery_velocity = trace.clubhead_velocity_m_s[-1]
    path_angle = math.atan2(float(delivery_velocity[1]), float(delivery_velocity[0]))
    face_angle = float(trace.q[-1, 8] + trace.q[-1, 9])
    controls = np.asarray(
        [
            [
                value.right_shoulder_nm,
                value.right_elbow_nm,
                value.right_wrist_nm,
                value.left_shoulder_nm,
                value.left_elbow_nm,
                value.left_wrist_nm,
            ]
            for value in trace.controls
        ],
        dtype=np.float64,
    )
    effort = float(np.trapezoid(np.sum(controls**2, axis=1), x=trace.time))
    trigger_mask = trace.q[:, 0] >= spec.arm_angle_threshold_rad
    event_index = (
        int(np.argmax(trigger_mask)) if np.any(trigger_mask) else len(trace.time) - 1
    )
    event_time = (
        spec.onset_s if spec.trigger == "clock" else float(trace.time[event_index])
    )
    late_mask = trace.time >= event_time
    wrist_power = trace.direct_wrist_torque_nm * trace.qdot[:, 8]
    shaft_release = float(
        trace.shaft_strain_energy_j[event_index] - trace.shaft_strain_energy_j[-1]
    )
    outcomes = np.array(
        [
            speed[-1],
            abs(math.degrees(_angle_difference(face_angle, path_angle))),
            np.max(np.linalg.norm(trace.contact_force_on_club_n, axis=2)),
            effort,
            event_time,
            np.trapezoid(trace.contact_power_w[late_mask], x=trace.time[late_mask]),
            np.trapezoid(wrist_power[late_mask], x=trace.time[late_mask]),
            shaft_release,
        ],
        dtype=np.float64,
    )
    # Whole-system energy: external actuation minus modeled dissipation and
    # projection correction must equal the mechanical-energy change.
    work_control = float(np.trapezoid(trace.applied_control_power_w, x=trace.time))
    work_dissipation = float(np.trapezoid(trace.dissipation_power_w, x=trace.time))
    projection_work = float(np.sum(trace.projection_energy_change_j))
    energy_change = float(trace.mechanical_energy_j[-1] - trace.mechanical_energy_j[0])
    closure = work_control + work_dissipation + projection_work - energy_change
    closure_scale = max(
        abs(work_control), abs(work_dissipation), abs(energy_change), 1e-12
    )
    return outcomes, {
        "pathway_residual_j": abs(closure),
        "normalized_pathway_residual": abs(closure) / closure_scale,
        "maximum_contact_power_residual_w": float(
            np.max(np.abs(trace.contact_power_identity_residual_w))
        ),
        "maximum_position_constraint_m": float(
            np.max(trace.position_constraint_norm_m)
        ),
        "maximum_velocity_constraint_m_s": float(
            np.max(trace.velocity_constraint_norm_m_s)
        ),
    }


def _evaluate_design(
    spec: ProgramSpec, design: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    outcomes = np.empty((design.shape[0], len(OUTCOME_NAMES)), dtype=np.float64)
    maxima = {
        "pathway_residual_j": 0.0,
        "normalized_pathway_residual": 0.0,
        "maximum_contact_power_residual_w": 0.0,
        "maximum_position_constraint_m": 0.0,
        "maximum_velocity_constraint_m_s": 0.0,
    }
    for index, perturbation in enumerate(design):
        outcomes[index], closure = _evaluate(spec, perturbation)
        for name, value in closure.items():
            maxima[name] = max(maxima[name], value)
    return outcomes, maxima


def _summary(outcomes: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean(outcomes[:, index])),
            "q10": float(np.quantile(outcomes[:, index], 0.10)),
            "q90": float(np.quantile(outcomes[:, index], 0.90)),
            "std": float(np.std(outcomes[:, index], ddof=1)),
        }
        for index, name in enumerate(OUTCOME_NAMES)
    }


def _objective_matrix(outputs: np.ndarray) -> np.ndarray:
    # Maximize lower-tail speed; minimize speed/face dispersion, force and effort.
    return np.column_stack(
        (
            -np.quantile(outputs[:, :, 0], 0.10, axis=1),
            np.std(outputs[:, :, 0], axis=1, ddof=1),
            np.mean(outputs[:, :, 1], axis=1),
            np.quantile(outputs[:, :, 2], 0.90, axis=1),
            np.mean(outputs[:, :, 3], axis=1),
        )
    )


def _gap_register() -> list[dict[str, str]]:
    return [
        {
            "id": "G1",
            "severity": "critical",
            "status": "confirmed",
            "gap": "Nominal clubhead speed was sometimes discussed near robustness language.",
            "counterexample": "A faster nominal program can have larger speed or face-path dispersion.",
            "falsifier": "Paired perturbations must show lower amplification and acceptable lower-tail outcomes.",
            "path_forward": "Retain separate nominal, q10, dispersion, load, and effort objectives.",
        },
        {
            "id": "G2",
            "severity": "critical",
            "status": "confirmed",
            "gap": "Negative torque can be mistaken for energy removal or biological braking.",
            "counterexample": "Negative torque with negative angular velocity produces positive power.",
            "falsifier": "Report torque, angular velocity, power, and integrated work together.",
            "path_forward": "Require sign-quadrant and interface-power plots in human validation.",
        },
        {
            "id": "G3",
            "severity": "high",
            "status": "confirmed",
            "gap": "A kinematic sequence does not uniquely identify an energy pathway.",
            "counterexample": "Identical peak ordering can arise with different control and constraint-force work.",
            "falsifier": "Pathway-resolved work must distinguish matched kinematics.",
            "path_forward": "Fit full inverse-dynamics and bilateral wrench alternatives, not peak order alone.",
        },
        {
            "id": "G4",
            "severity": "high",
            "status": "confirmed",
            "gap": "Instantaneous drift is not a forward future or causal mediator.",
            "counterexample": "Control changes the subsequent state, so integrated pointwise ZTCF is not a killed-control trajectory.",
            "falsifier": "Same-state forward killswitches and causal intervention models must agree on the bounded claim.",
            "path_forward": "Keep pointwise, forward-counterfactual, and statistical mediation labels separate.",
        },
        {
            "id": "G5",
            "severity": "high",
            "status": "open",
            "gap": "Fixed terminal time is not ball impact and may reorder programs.",
            "counterexample": "A program can be fast at 240 ms but miss the impact manifold.",
            "falsifier": "Event-aligned ball/contact outcomes preserve the Pareto ordering.",
            "path_forward": "Add 3-D face/path/contact and impact-location events in spatial engines.",
        },
        {
            "id": "G6",
            "severity": "high",
            "status": "open",
            "gap": "Engineering parameter envelopes are not population distributions.",
            "counterexample": "Uniform/LHS weighting can favor regions uncommon in golfers.",
            "falsifier": "Participant-held-out posterior predictive checks reproduce distributions and rankings.",
            "path_forward": "Estimate hierarchical subject/equipment distributions from synchronized data.",
        },
        {
            "id": "G7",
            "severity": "high",
            "status": "open",
            "gap": "Impedance and coactivation proxies do not establish biological stability.",
            "counterexample": "Higher coactivation can reduce displacement while increasing force, effort, and coupled instability.",
            "falsifier": "Measured perturbation recovery improves without unacceptable load or accuracy cost.",
            "path_forward": "Identify time-varying endpoint impedance with EMG and controlled perturbations.",
        },
        {
            "id": "G8",
            "severity": "medium",
            "status": "confirmed",
            "gap": "Less joint variability is not necessarily better performance.",
            "counterexample": "Task-null variability can coexist with stable club state and support adaptation.",
            "falsifier": "UCM/task-Jacobian analysis separates null from outcome-relevant variance.",
            "path_forward": "Report covariance structure, not scalar trajectory dispersion alone.",
        },
        {
            "id": "G9",
            "severity": "medium",
            "status": "open",
            "gap": "A local linear null space may fail under large perturbations.",
            "counterexample": "Curvature moves a nominally null direction into task-relevant error.",
            "falsifier": "Nonlinear re-evaluation bounds prediction error across perturbation amplitude.",
            "path_forward": "Add second-order and trajectory-level UCM/manifold tests.",
        },
        {
            "id": "G10",
            "severity": "high",
            "status": "open",
            "gap": "Distance and accuracy are not reducible to clubhead speed and face/path alone.",
            "counterexample": "Off-center impact, dynamic loft, attack angle, and strike location can reverse carry rankings.",
            "falsifier": "Ball-flight and impact models preserve strategy tradeoffs on carry and dispersion.",
            "path_forward": "Propagate full impact state through calibrated launch and aerodynamic models.",
        },
        {
            "id": "G11",
            "severity": "medium",
            "status": "open",
            "gap": "Open-loop and instantaneous state feedback omit sensory delay and estimation error.",
            "counterexample": "A stable perfect-state controller may destabilize under delayed noisy feedback.",
            "falsifier": "Delay/noise sweeps and observer models retain bounded task error.",
            "path_forward": "Add delayed state estimation and phase-response identification.",
        },
        {
            "id": "G12",
            "severity": "critical",
            "status": "open",
            "gap": "No current model result establishes a universal human technique or injury benefit.",
            "counterexample": "Anthropometry, strength, equipment, intent, and injury tolerance can change the optimum.",
            "falsifier": "Prospective participant-held-out intervention improves prespecified outcomes and safety endpoints.",
            "path_forward": "Treat model results as experiment generators and personalize only after validation.",
        },
    ]


@lru_cache(maxsize=1)
def run_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute the registered paired training and held-out comparisons."""

    training = _physical_perturbations(TRAINING_SAMPLE_COUNT, seed=TRAINING_SEED)
    held_out = _physical_perturbations(HELD_OUT_SAMPLE_COUNT, seed=HELD_OUT_SEED)
    train_outputs = np.empty((len(PROGRAMS), TRAINING_SAMPLE_COUNT, len(OUTCOME_NAMES)))
    held_outputs = np.empty((len(PROGRAMS), HELD_OUT_SAMPLE_COUNT, len(OUTCOME_NAMES)))
    closure_maxima = {
        "pathway_residual_j": 0.0,
        "normalized_pathway_residual": 0.0,
        "maximum_contact_power_residual_w": 0.0,
        "maximum_position_constraint_m": 0.0,
        "maximum_velocity_constraint_m_s": 0.0,
    }
    for index, program in enumerate(PROGRAMS):
        train_outputs[index], train_closure = _evaluate_design(program, training)
        held_outputs[index], held_closure = _evaluate_design(program, held_out)
        for closure in (train_closure, held_closure):
            for name, value in closure.items():
                closure_maxima[name] = max(closure_maxima[name], value)

    training_objectives = _objective_matrix(train_outputs)
    held_objectives = _objective_matrix(held_outputs)
    training_pareto = nondominated_indices(training_objectives)
    held_pareto = nondominated_indices(held_objectives)
    clock_index = PROGRAM_NAMES.index("clock_restrain_then_drive")
    state_index = PROGRAM_NAMES.index("state_triggered_handoff")
    paired = PerturbationEnsemble(
        perturbations=held_out,
        baseline_outcomes=held_outputs[clock_index],
        candidate_outcomes=held_outputs[state_index],
        outcome_names=OUTCOME_NAMES,
    )

    local_center = np.zeros(len(PERTURBATION_NAMES), dtype=np.float64)
    local_steps = np.array([0.008, 0.010, 0.08, 0.10, 0.02, 0.003, 0.04, 0.03])
    local = finite_difference_outcome_jacobian(
        lambda value: _evaluate(PROGRAMS[state_index], value)[0][:3],
        center=local_center,
        steps=local_steps,
        input_names=PERTURBATION_NAMES,
        outcome_names=OUTCOME_NAMES[:3],
    )
    variance = task_variance_partition(local.jacobian, held_out)
    nominal = np.asarray([_evaluate(program, local_center)[0] for program in PROGRAMS])
    nominal_fastest = int(np.argmax(nominal[:, 0]))
    least_speed_dispersion = int(
        np.argmin(np.std(held_outputs[:, :, 0], axis=1, ddof=1))
    )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "paired-transmission-pathway-task-robustness-v1",
        "registered_before_preferred_result": True,
        "model_tier": "moving_base_two_hand_compliant_club",
        "design": {
            "paired_common_random_numbers": True,
            "training_perturbations": TRAINING_SAMPLE_COUNT,
            "held_out_perturbations": HELD_OUT_SAMPLE_COUNT,
            "training_seed": TRAINING_SEED,
            "held_out_seed": HELD_OUT_SEED,
            "duration_s": DURATION_S,
            "step_s": STEP_S,
            "perturbation_half_ranges": dict(
                zip(PERTURBATION_NAMES, PERTURBATION_HALF_RANGES.tolist(), strict=True)
            ),
        },
        "programs": list(PROGRAM_NAMES),
        "program_specs": [asdict(program) for program in PROGRAMS],
        "perturbation_names": list(PERTURBATION_NAMES),
        "outcome_names": list(OUTCOME_NAMES),
        "program_summaries": {
            program.name: {
                "nominal": dict(
                    zip(OUTCOME_NAMES, nominal[index].tolist(), strict=True)
                ),
                "training": _summary(train_outputs[index]),
                "held_out": _summary(held_outputs[index]),
            }
            for index, program in enumerate(PROGRAMS)
        },
        "clock_vs_state_paired_held_out": perturbation_summary(paired),
        "objectives": [
            "maximize q10 delivery speed",
            "minimize delivery-speed dispersion",
            "minimize mean face-path error",
            "minimize q90 individual-hand force",
            "minimize mean squared-torque effort proxy",
        ],
        "training_pareto_programs": [PROGRAM_NAMES[index] for index in training_pareto],
        "held_out_pareto_programs": [PROGRAM_NAMES[index] for index in held_pareto],
        "local_task_map": {
            "input_names": list(local.input_names),
            "outcome_names": list(local.outcome_names),
            "steps": local.steps.tolist(),
            "task_rank": variance.task_rank,
            "nullity": variance.nullity,
            "null_variance": variance.null_variance,
            "task_relevant_variance": variance.task_relevant_variance,
            "synergy_index": variance.synergy_index,
            "interpretation": "local_model_input_outcome_partition_not_neural_synergy",
        },
        "closure": {
            "maximum_pathway_residual_j": closure_maxima["pathway_residual_j"],
            "maximum_normalized_pathway_residual": closure_maxima[
                "normalized_pathway_residual"
            ],
            "maximum_contact_power_residual_w": closure_maxima[
                "maximum_contact_power_residual_w"
            ],
            "maximum_position_constraint_m": closure_maxima[
                "maximum_position_constraint_m"
            ],
            "maximum_velocity_constraint_m_s": closure_maxima[
                "maximum_velocity_constraint_m_s"
            ],
        },
        "adversarial_gap_register": _gap_register(),
        "claim_status": {
            "universal_optimum": "rejected_by_tradeoffs",
            "nominal_speed_implies_repeatability": (
                "rejected"
                if nominal_fastest != least_speed_dispersion
                else "not_supported"
            ),
            "state_trigger_is_self_stabilizing": "model_conditional",
            "human_self_stabilization": "untested",
            "causal_biological_pathway": "untested",
            "coaching_prescription": "unsupported",
        },
        "limitations": [
            "The terminal sample is a delivery proxy, not a ball-impact collision.",
            "Perturbations are declared engineering envelopes, not a fitted golfer population.",
            "State feedback has perfect state access and omits sensory delay beyond actuator delay.",
            "The squared-torque effort metric is not metabolic cost or muscle activation.",
            "The planar face-path proxy omits 3-D face orientation, strike location, and ball flight.",
            "Local task-null directions are linearized model properties, not measured neural synergies.",
        ],
        "source_sha256": _source_hashes(),
        "array_artifact": NPZ_PATH.name,
    }
    arrays = {
        "training_perturbations": training,
        "held_out_perturbations": held_out,
        "training_outcomes": train_outputs,
        "held_out_outcomes": held_outputs,
        "training_objectives": training_objectives,
        "held_out_objectives": held_objectives,
        "local_outcome_jacobian": local.jacobian,
        "local_steps": local.steps,
        "nominal_outcomes": nominal,
    }
    return record, arrays


def write_outputs() -> tuple[Path, Path]:
    """Write deterministic JSON and compressed numerical evidence."""

    record, arrays = run_study()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cast(Any, np.savez_compressed)(NPZ_PATH, **arrays)
    return JSON_PATH, NPZ_PATH


def main() -> None:
    json_path, npz_path = write_outputs()
    print(json_path)
    print(npz_path)


if __name__ == "__main__":
    main()
