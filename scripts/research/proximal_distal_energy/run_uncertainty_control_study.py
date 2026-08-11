"""Run coupled uncertainty, identifiability, and bounded-control experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts.research.proximal_distal_energy.moving_base_flexible_club import (
    MovingBaseFlexibleConfig,
    MovingBaseFlexibleParams,
    initial_state,
    rollout,
)
from scripts.research.proximal_distal_energy.uncertainty_control import (
    ActuatorLimits,
    ControlProgram,
    delayed_control_law,
    latin_hypercube,
    nondominated_indices,
    partial_rank_correlations,
    planar_two_hand_wrench_map,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"
JSON_PATH = DATA_DIR / "uncertainty_control_study.json"
NPZ_PATH = DATA_DIR / "uncertainty_control_study.npz"
SCHEMA_VERSION = "proximal-distal-uncertainty-control-v1"
STUDY_ID = "coupled-uncertainty-identifiability-control-v1"
DURATION_S = 0.24
STEP_S = 0.004
GLOBAL_SAMPLE_COUNT = 24
TRAINING_SAMPLE_COUNT = 6
HELD_OUT_SAMPLE_COUNT = 6

PARAMETER_RANGES: tuple[tuple[str, float, float], ...] = (
    ("anthropometric_scale", 0.95, 1.05),
    ("limb_mass_scale", 0.90, 1.10),
    ("inertia_distribution_scale", 0.85, 1.15),
    ("base_mass_scale", 0.85, 1.15),
    ("base_stiffness_scale", 0.75, 1.25),
    ("joint_damping_scale", 0.60, 1.40),
    ("grip_separation_scale", 0.85, 1.15),
    ("shaft_stiffness_scale", 0.70, 1.30),
    ("shaft_damping_scale", 0.60, 1.40),
    ("activation_delay_s", 0.015, 0.055),
    ("activation_time_constant_s", 0.020, 0.055),
    ("impedance_scale", 0.70, 1.30),
)

METRIC_NAMES = (
    "delivery_speed_m_s",
    "face_path_error_deg",
    "peak_individual_hand_force_n",
    "effort_proxy_nms",
    "minimum_force_generated_couple_nm",
    "peak_shaft_flex_deg",
)


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/research/proximal_distal_energy/uncertainty_control.py",
        "scripts/research/proximal_distal_energy/run_uncertainty_control_study.py",
        "scripts/research/proximal_distal_energy/moving_base_flexible_club.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _physical_design(unit_design: np.ndarray) -> np.ndarray:
    lower = np.asarray([entry[1] for entry in PARAMETER_RANGES])
    upper = np.asarray([entry[2] for entry in PARAMETER_RANGES])
    return lower + unit_design * (upper - lower)


def _apply_parameters(
    values: np.ndarray,
    program: ControlProgram,
) -> tuple[MovingBaseFlexibleParams, ActuatorLimits, ControlProgram]:
    sample = dict(zip((entry[0] for entry in PARAMETER_RANGES), values, strict=True))
    base = MovingBaseFlexibleParams.publication_default()
    length = sample["anthropometric_scale"]
    mass = sample["limb_mass_scale"]
    inertia = sample["inertia_distribution_scale"]
    upper_length = base.upper_length_m * length
    forearm_length = base.forearm_length_m * length
    upper_mass = base.upper_mass_kg * mass
    forearm_mass = base.forearm_mass_kg * mass
    params = replace(
        base,
        right_shoulder_offset_m=(base.right_shoulder_offset_m[0] * length, 0.0),
        left_shoulder_offset_m=(base.left_shoulder_offset_m[0] * length, 0.0),
        upper_length_m=upper_length,
        forearm_length_m=forearm_length,
        upper_mass_kg=upper_mass,
        forearm_mass_kg=forearm_mass,
        upper_inertia_kg_m2=upper_mass * upper_length**2 / 12.0 * inertia,
        forearm_inertia_kg_m2=forearm_mass * forearm_length**2 / 12.0 * inertia,
        base_mass_kg=base.base_mass_kg * sample["base_mass_scale"],
        base_stiffness_n_m=(base.base_stiffness_n_m * sample["base_stiffness_scale"]),
        joint_damping_nms_rad=(
            base.joint_damping_nms_rad * sample["joint_damping_scale"]
        ),
        right_grip_offset_m=(
            base.right_grip_offset_m * sample["grip_separation_scale"]
        ),
        left_grip_offset_m=(base.left_grip_offset_m * sample["grip_separation_scale"]),
        shaft_stiffness_nm_rad=(
            base.shaft_stiffness_nm_rad * sample["shaft_stiffness_scale"]
        ),
        shaft_damping_nms_rad=(
            base.shaft_damping_nms_rad * sample["shaft_damping_scale"]
        ),
    )
    limits = ActuatorLimits(
        delay_s=float(sample["activation_delay_s"]),
        time_constant_s=float(sample["activation_time_constant_s"]),
    )
    perturbed_program = replace(
        program,
        impedance_nms_rad=(program.impedance_nms_rad * sample["impedance_scale"]),
    )
    return params, limits, perturbed_program


def _programs() -> tuple[ControlProgram, ...]:
    return (
        ControlProgram("passive_wrist", 0.14, 0.0, 0.0, 1.0, 1.0, 0.08),
        ControlProgram("early_drive", 0.00, 4.0, 4.0, 1.0, 1.0, 0.08),
        ControlProgram("late_drive", 0.14, 0.0, 5.0, 1.0, 1.0, 0.08),
        ControlProgram("restrain_then_drive", 0.14, -2.0, 5.0, 1.0, 1.0, 0.08),
        ControlProgram("higher_impedance", 0.14, -2.0, 5.0, 1.0, 1.0, 0.24),
        ControlProgram("later_release", 0.18, -2.0, 5.0, 1.0, 1.0, 0.08),
        ControlProgram("lower_drive", 0.14, -1.0, 3.0, 1.0, 1.0, 0.08),
        ControlProgram("early_restrain", 0.10, -2.0, 4.0, 1.0, 1.0, 0.08),
    )


def _angle_difference(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


def _evaluate(
    values: np.ndarray, program: ControlProgram
) -> tuple[np.ndarray, dict[str, float]]:
    params, limits, perturbed_program = _apply_parameters(values, program)
    anthropometric_scale = float(values[0])
    q0, qdot0 = initial_state(params, grip_center_m=(0.0, -0.5 * anthropometric_scale))
    law = delayed_control_law(
        perturbed_program,
        limits,
        duration_s=DURATION_S,
        step_s=STEP_S,
    )
    trace = rollout(
        q0,
        qdot0,
        law,
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
        ]
    )
    effort_proxy = float(np.trapezoid(np.sum(controls**2, axis=1), x=trace.time))
    metrics = np.array(
        [
            speed[-1],
            abs(math.degrees(_angle_difference(face_angle, path_angle))),
            np.max(np.linalg.norm(trace.contact_force_on_club_n, axis=2)),
            effort_proxy,
            np.min(trace.force_generated_couple_nm),
            np.max(np.abs(np.rad2deg(trace.q[:, 9]))),
        ],
        dtype=np.float64,
    )
    closure = {
        "maximum_position_constraint_m": float(
            np.max(trace.position_constraint_norm_m)
        ),
        "maximum_velocity_constraint_m_s": float(
            np.max(trace.velocity_constraint_norm_m_s)
        ),
        "maximum_kkt_residual": float(np.max(trace.kkt_residual_norm)),
        "maximum_contact_power_residual_w": float(
            np.max(np.abs(trace.contact_power_identity_residual_w))
        ),
    }
    if not np.all(np.isfinite(metrics)):
        raise RuntimeError("uncertainty/control rollout produced nonfinite metrics")
    return metrics, closure


def _evaluate_design(
    design: np.ndarray, program: ControlProgram
) -> tuple[np.ndarray, dict[str, float]]:
    outputs = np.empty((design.shape[0], len(METRIC_NAMES)))
    closure_maxima = {
        "maximum_position_constraint_m": 0.0,
        "maximum_velocity_constraint_m_s": 0.0,
        "maximum_kkt_residual": 0.0,
        "maximum_contact_power_residual_w": 0.0,
    }
    for index, values in enumerate(design):
        outputs[index], closure = _evaluate(values, program)
        for key, value in closure.items():
            closure_maxima[key] = max(closure_maxima[key], value)
    return outputs, closure_maxima


def _intervals(outputs: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {
            "q05": float(np.quantile(outputs[:, index], 0.05)),
            "median": float(np.median(outputs[:, index])),
            "q95": float(np.quantile(outputs[:, index], 0.95)),
        }
        for index, name in enumerate(METRIC_NAMES)
    }


def _standardized_sensitivity(design: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    x = (design - np.mean(design, axis=0)) / np.std(design, axis=0, ddof=1)
    y = (outputs - np.mean(outputs, axis=0)) / np.std(outputs, axis=0, ddof=1)
    return np.linalg.lstsq(x, y, rcond=None)[0].T


def _candidate_summary(
    program: ControlProgram,
    training: np.ndarray,
    held_out: np.ndarray,
) -> dict[str, Any]:
    def summary(outputs: np.ndarray) -> dict[str, float]:
        return {
            "delivery_speed_mean_m_s": float(np.mean(outputs[:, 0])),
            "delivery_speed_q10_m_s": float(np.quantile(outputs[:, 0], 0.10)),
            "delivery_speed_std_m_s": float(np.std(outputs[:, 0], ddof=1)),
            "face_path_error_mean_deg": float(np.mean(outputs[:, 1])),
            "peak_hand_force_q90_n": float(np.quantile(outputs[:, 2], 0.90)),
            "effort_proxy_mean_nms": float(np.mean(outputs[:, 3])),
            "negative_couple_median_nm": float(np.median(outputs[:, 4])),
        }

    return {
        "name": program.name,
        "program": asdict(program),
        "training": summary(training),
        "held_out": summary(held_out),
    }


def _objective_matrix(candidates: list[dict[str, Any]], split: str) -> np.ndarray:
    return np.asarray(
        [
            [
                -candidate[split]["delivery_speed_q10_m_s"],
                candidate[split]["face_path_error_mean_deg"],
                candidate[split]["peak_hand_force_q90_n"],
                candidate[split]["effort_proxy_mean_nms"],
                candidate[split]["delivery_speed_std_m_s"],
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    )


def _balanced_index(objectives: np.ndarray) -> int:
    lower = np.min(objectives, axis=0)
    span = np.maximum(np.max(objectives, axis=0) - lower, np.finfo(float).eps)
    normalized = (objectives - lower) / span
    return int(np.argmin(np.max(normalized, axis=1)))


def run_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute all registered uncertainty, identifiability, and control tests."""

    programs = _programs()
    reference_program = next(
        program for program in programs if program.name == "restrain_then_drive"
    )
    global_unit = latin_hypercube(GLOBAL_SAMPLE_COUNT, len(PARAMETER_RANGES), seed=8426)
    training_unit = latin_hypercube(
        TRAINING_SAMPLE_COUNT, len(PARAMETER_RANGES), seed=8449
    )
    held_out_unit = latin_hypercube(
        HELD_OUT_SAMPLE_COUNT, len(PARAMETER_RANGES), seed=8450
    )
    global_design = _physical_design(global_unit)
    training_design = _physical_design(training_unit)
    held_out_design = _physical_design(held_out_unit)
    global_outputs, global_closure = _evaluate_design(global_design, reference_program)
    prcc = np.column_stack(
        [
            partial_rank_correlations(global_design, global_outputs[:, index])
            for index in range(global_outputs.shape[1])
        ]
    )
    sensitivity = _standardized_sensitivity(global_design, global_outputs)
    singular_values = np.linalg.svd(sensitivity, compute_uv=False)
    threshold = 0.05 * float(singular_values[0])
    practical_rank = int(np.sum(singular_values > threshold))
    structural_map = planar_two_hand_wrench_map(0.065, -0.065)

    candidates: list[dict[str, Any]] = []
    training_outputs: list[np.ndarray] = []
    held_out_outputs: list[np.ndarray] = []
    candidate_closure = dict(global_closure)
    for program in programs:
        train, train_closure = _evaluate_design(training_design, program)
        held, held_closure = _evaluate_design(held_out_design, program)
        training_outputs.append(train)
        held_out_outputs.append(held)
        candidates.append(_candidate_summary(program, train, held))
        for closure in (train_closure, held_closure):
            for key, value in closure.items():
                candidate_closure[key] = max(candidate_closure[key], value)

    training_objectives = _objective_matrix(candidates, "training")
    held_out_objectives = _objective_matrix(candidates, "held_out")
    pareto = nondominated_indices(training_objectives)
    balanced = _balanced_index(training_objectives)
    speed_priority = int(np.argmin(training_objectives[:, 0]))
    load_priority = int(np.argmin(training_objectives[:, 2]))
    held_out_pareto = nondominated_indices(held_out_objectives)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "model_tier": "moving_base_two_hand_flexible_club",
        "registered_before_preferred_result": True,
        "design": {
            "method": "deterministic_latin_hypercube_global_screening",
            "global_samples": GLOBAL_SAMPLE_COUNT,
            "training_samples": TRAINING_SAMPLE_COUNT,
            "held_out_samples": HELD_OUT_SAMPLE_COUNT,
            "training_seed": 8449,
            "held_out_seed": 8450,
            "duration_s": DURATION_S,
            "step_s": STEP_S,
            "parameters": [
                {"name": name, "lower": lower, "upper": upper}
                for name, lower, upper in PARAMETER_RANGES
            ],
        },
        "actuator_contract": {
            "components": [
                "pure_delay",
                "first_order_activation",
                "torque_rate_limit",
                "asymmetric_torque_velocity_limit",
                "joint_impedance_proxy",
            ],
            "effort_definition": "time integral of squared delivered joint torques",
            "physiological_interpretation": "unsupported",
        },
        "uncertainty_intervals": _intervals(global_outputs),
        "global_sensitivity": {
            "method": "partial_rank_correlation_on_latin_hypercube",
            "parameter_names": [entry[0] for entry in PARAMETER_RANGES],
            "metric_names": list(METRIC_NAMES),
            "coefficients": prcc.tolist(),
            "strongest_parameter_by_metric": {
                name: PARAMETER_RANGES[int(np.argmax(np.abs(prcc[:, index])))][0]
                for index, name in enumerate(METRIC_NAMES)
            },
        },
        "identifiability": {
            "individual_hand_force_from_net_planar_wrench": {
                "mapping_rank": int(np.linalg.matrix_rank(structural_map)),
                "unknown_force_components": int(structural_map.shape[1]),
                "nullity": int(
                    structural_map.shape[1] - np.linalg.matrix_rank(structural_map)
                ),
                "status": "structurally_nonidentifiable_without_additional_measurement",
            },
            "coupled_parameter_screen": {
                "parameter_count": len(PARAMETER_RANGES),
                "observable_count": len(METRIC_NAMES),
                "standardized_sensitivity_singular_values": singular_values.tolist(),
                "effective_rank_at_five_percent": practical_rank,
                "nullity_lower_bound": len(PARAMETER_RANGES) - practical_rank,
                "status": "practically_nonidentifiable_as_a_full_parameter_vector",
            },
        },
        "control_comparison": {
            "objectives": [
                "maximize held-ensemble q10 delivery speed",
                "minimize mean face-path error",
                "minimize q90 individual-hand force",
                "minimize squared-torque effort proxy",
                "minimize delivery-speed dispersion",
            ],
            "candidates": candidates,
            "training_pareto_programs": [programs[index].name for index in pareto],
            "held_out_pareto_programs": [
                programs[index].name for index in held_out_pareto
            ],
            "objective_specific_selections": {
                "balanced_equal_range_chebyshev": programs[balanced].name,
                "speed_priority": programs[speed_priority].name,
                "load_priority": programs[load_priority].name,
            },
            "universal_optimum_claim": "unsupported",
        },
        "closure": candidate_closure,
        "claim_status": {
            "H4_preactivation_under_delay": (
                "tested_as_model_strategy_not_human_neural_evidence"
            ),
            "individual_hand_force_identification": "falsified_from_net_wrench_alone",
            "robust_universal_strategy": "unsupported_tradeoffs_remain",
            "human_or_coaching_inference": "unsupported",
        },
        "limitations": [
            "Parameter ranges are declared engineering envelopes, not fitted population distributions.",
            "The delivery event is a fixed model time and not a ball-impact collision.",
            "The effort metric is a squared-torque proxy and is not metabolic or muscular cost.",
            "The actuator model has no muscles, tendons, reflexes, or measured activation data.",
            "The planar model cannot resolve three-dimensional face orientation or ground pathways.",
        ],
        "source_sha256": _source_hashes(),
        "array_artifact": NPZ_PATH.name,
    }
    arrays = {
        "global_unit_design": global_unit,
        "global_physical_design": global_design,
        "global_outputs": global_outputs,
        "training_unit_design": training_unit,
        "training_physical_design": training_design,
        "held_out_unit_design": held_out_unit,
        "held_out_physical_design": held_out_design,
        "training_outputs": np.asarray(training_outputs),
        "held_out_outputs": np.asarray(held_out_outputs),
        "training_objectives": training_objectives,
        "held_out_objectives": held_out_objectives,
        "prcc": prcc,
        "standardized_sensitivity": sensitivity,
        "individual_hand_wrench_map": structural_map,
    }
    return record, arrays


def write_outputs() -> tuple[Path, Path]:
    """Write deterministic metadata and numerical arrays."""

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
