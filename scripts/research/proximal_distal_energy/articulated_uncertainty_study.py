"""Articulated parameter-uncertainty study and sensitivity analysis (#8752).

Performs Latin Hypercube Sampling sweeps across joint limits, anthropometrics,
grip stiffness/damping, shaft modes, and ground parameters. Generates PRCC sensitivity
maps and failure maps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_forward import (
    DistributedForwardConfig,
    DistributedIntegrationCase,
    integrate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    BodySpec,
    JointSpec,
    SpatialModel,
    prescribed_state,
)
from scripts.research.proximal_distal_energy.subject_scaled_closed_contact import (
    ClosedContactConfig,
    ClosedContactSolution,
    solve_closed_contact_configuration,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"


def latin_hypercube(samples: int, dimensions: int, *, seed: int) -> FloatArray:
    """Return a deterministic centered-jitter Latin hypercube in (0, 1)."""
    if samples < 2 or dimensions < 1:
        raise ValueError("samples must be >= 2 and dimensions must be >= 1")
    rng = np.random.default_rng(seed)
    design = np.empty((samples, dimensions), dtype=np.float64)
    for column in range(dimensions):
        permutation = rng.permutation(samples)
        jitter = rng.uniform(0.15, 0.85, size=samples)
        design[:, column] = (permutation + jitter) / samples
    return design


def _rank(values: NDArray[np.float64]) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def partial_rank_correlations(
    design: NDArray[np.float64], output: NDArray[np.float64]
) -> FloatArray:
    """Compute partial rank correlations by residualizing other inputs."""
    matrix = np.asarray(design, dtype=np.float64)
    response = np.asarray(output, dtype=np.float64)
    ranked_x = np.column_stack(
        [_rank(matrix[:, index]) for index in range(matrix.shape[1])]
    )
    ranked_y = _rank(response)
    result = np.empty(matrix.shape[1], dtype=np.float64)
    for index in range(matrix.shape[1]):
        others = np.delete(ranked_x, index, axis=1)
        regressors = np.column_stack([np.ones(matrix.shape[0]), others])
        x_residual = (
            ranked_x[:, index]
            - regressors
            @ np.linalg.lstsq(regressors, ranked_x[:, index], rcond=None)[0]
        )
        y_residual = (
            ranked_y - regressors @ np.linalg.lstsq(regressors, ranked_y, rcond=None)[0]
        )
        denominator = float(np.linalg.norm(x_residual) * np.linalg.norm(y_residual))
        result[index] = (
            0.0
            if denominator <= np.finfo(float).eps
            else x_residual @ y_residual / denominator
        )
    return result


UNCERTAINTY_PARAMETERS = (
    "arm_length_scale",
    "body_mass_scale",
    "grip_stiffness_n_m",
    "grip_damping_n_s_m",
    "friction_coefficient",
    "club_mass_kg",
    "club_moi_scale",
    "initial_velocity_m_s",
)

OUTPUT_METRICS = (
    "peak_station_force_n",
    "peak_force_couple_nm",
    "max_sliding_speed_m_s",
    "total_transition_count",
    "normalized_work_energy_residual",
)


@dataclass(frozen=True, slots=True)
class ArticulatedUncertaintyConfig:
    """Design bounds for articulated parameter uncertainty study."""

    sample_count: int = 40
    seed: int = 8752
    duration_s: float = 0.02
    time_step_s: float = 0.001
    station_count_per_hand: int = 3
    station_width_m: float = 0.03

    # Parameter bounds: (min, max)
    arm_length_range: tuple[float, float] = (0.90, 1.10)
    body_mass_range: tuple[float, float] = (0.85, 1.15)
    grip_stiffness_range: tuple[float, float] = (1000.0, 3000.0)
    grip_damping_range: tuple[float, float] = (10.0, 35.0)
    friction_range: tuple[float, float] = (0.15, 0.65)
    club_mass_range: tuple[float, float] = (0.28, 0.42)
    club_moi_range: tuple[float, float] = (0.80, 1.25)
    velocity_range: tuple[float, float] = (0.01, 0.20)


@dataclass(frozen=True, slots=True)
class UncertainClosedState:
    """A regenerated anthropometric state and its explicit domain status."""

    model: SpatialModel
    metadata: dict[str, Any]
    q: FloatArray
    qd: FloatArray
    feasible: bool
    failure_class: str
    maximum_closure_error_m: float
    minimum_joint_limit_margin_rad: float
    minimum_collision_clearance_m: float


def _failure_class(solution: ClosedContactSolution) -> str:
    if not solution.solver_converged:
        return "ik_nonconvergence"
    if not solution.contact_closed:
        return "bilateral_closure_failure"
    if not solution.joint_limits_satisfied:
        return "joint_limit_failure"
    if not solution.collision_free:
        return "collision_domain_failure"
    if solution.constraint_jacobian_rank != 6:
        return "constraint_rank_failure"
    return "feasible"


def resolve_uncertain_closed_state(
    *,
    profile_index: int,
    grip_span_m: float,
    sample_time_s: float,
    height_scale: float,
    mass_scale: float,
    joint_limit_scale: float,
    difference_step_s: float = 1.0e-4,
) -> UncertainClosedState:
    """Regenerate bilateral closure after anthropometric/limit perturbation."""

    profiles = default_synthetic_profiles()
    if not isinstance(profile_index, int) or not 0 <= profile_index < len(profiles):
        raise ValueError("profile_index is outside the synthetic design")
    for name, value in (
        ("grip_span_m", grip_span_m),
        ("height_scale", height_scale),
        ("mass_scale", mass_scale),
        ("difference_step_s", difference_step_s),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(sample_time_s) or sample_time_s < 0.0:
        raise ValueError("sample_time_s must be finite and nonnegative")
    base = profiles[profile_index]
    profile = replace(
        base,
        profile_id=f"{base.profile_id}-u",
        height_m=base.height_m * height_scale,
        mass_kg=base.mass_kg * mass_scale,
    )
    model, metadata = build_subject_scaled_model(profile)
    times = np.array(
        [
            max(0.0, sample_time_s - difference_step_s),
            sample_time_s,
            sample_time_s + difference_step_s,
        ]
    )
    solutions: list[ClosedContactSolution] = []
    seed: FloatArray | None = None
    contact_config = ClosedContactConfig(joint_limit_scale=joint_limit_scale)
    for time_s in times:
        reference, _, _ = prescribed_state(model, float(time_s))
        solution = solve_closed_contact_configuration(
            model,
            q_reference=reference,
            grip_span_m=grip_span_m,
            hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
            q_seed=seed,
            config=contact_config,
        )
        solutions.append(solution)
        seed = solution.q if solution.contact_closed else None
    center = solutions[1]
    qd = (solutions[2].q - solutions[0].q) / (times[2] - times[0])
    return UncertainClosedState(
        model=model,
        metadata=metadata,
        q=center.q,
        qd=qd,
        feasible=all(solution.feasible for solution in solutions),
        failure_class=_failure_class(center),
        maximum_closure_error_m=max(
            float(np.max(solution.hand_to_grip_distance_m)) for solution in solutions
        ),
        minimum_joint_limit_margin_rad=min(
            solution.minimum_joint_limit_margin_rad for solution in solutions
        ),
        minimum_collision_clearance_m=min(
            solution.minimum_collision_clearance_m for solution in solutions
        ),
    )


def _scale_model(
    base_model: SpatialModel,
    arm_scale: float,
    mass_scale: float,
    club_mass: float,
    moi_scale: float,
) -> SpatialModel:
    """Build parameter-perturbed SpatialModel."""
    new_joints = []
    for j in base_model.joints:
        offset = j.offset_m.copy()
        if "arm" in j.region:
            offset *= arm_scale
        new_joints.append(
            JointSpec(j.name, j.parent, j.kind, j.axis.copy(), offset, j.region)
        )

    new_bodies = []
    for b in base_model.bodies:
        m = b.mass_kg * mass_scale if b.region != "club" else club_mass
        r = b.radius_m * moi_scale if b.region == "club" else b.radius_m
        new_bodies.append(
            BodySpec(b.name, b.joint, m, r, b.com_offset_m.copy(), b.region)
        )

    return SpatialModel(
        joints=tuple(new_joints),
        bodies=tuple(new_bodies),
        club_dof_indices=base_model.club_dof_indices.copy(),
        lead_hand_joint=base_model.lead_hand_joint,
        trail_hand_joint=base_model.trail_hand_joint,
        club_frame_joint=base_model.club_frame_joint,
    )


def _sample_parameters(
    config: ArticulatedUncertaintyConfig,
) -> FloatArray:
    dim = len(UNCERTAINTY_PARAMETERS)
    lhs = latin_hypercube(config.sample_count, dim, seed=config.seed)
    ranges = [
        config.arm_length_range,
        config.body_mass_range,
        config.grip_stiffness_range,
        config.grip_damping_range,
        config.friction_range,
        config.club_mass_range,
        config.club_moi_range,
        config.velocity_range,
    ]
    param_matrix = np.zeros_like(lhs)
    for i, (low, high) in enumerate(ranges):
        param_matrix[:, i] = low + lhs[:, i] * (high - low)
    return param_matrix


def _evaluate_uncertainty_trajectory(
    sample_params: FloatArray,
    base_model: SpatialModel,
    q_base: FloatArray,
    grip_span_m: float,
    hand_contact_x: float,
    config: ArticulatedUncertaintyConfig,
    forward_cfg: DistributedForwardConfig,
) -> tuple[float, float, float, int, float, str]:
    arm_scale, mass_scale, k_grip, c_grip, mu, m_club, moi_scale, v_init = (
        float(x) for x in sample_params
    )
    model_s = _scale_model(base_model, arm_scale, mass_scale, m_club, moi_scale)
    grip_cfg = DistributedGripConfig(
        station_count_per_hand=config.station_count_per_hand,
        station_width_m=config.station_width_m,
        total_stiffness_n_m=k_grip,
        total_damping_n_s_m=c_grip,
        tangential_damping_n_s_m=c_grip,
        friction_coefficient=mu,
    )
    case = DistributedIntegrationCase(
        q=q_base,
        qd=np.zeros(base_model.nq),
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_x * arm_scale,
        time_step_s=config.time_step_s,
        initial_club_displacement_m=0.001,
        initial_club_velocity_m_s=v_init,
        engine="mujoco",
        grip=grip_cfg,
    )
    res = integrate_distributed_grip(model_s, case, forward_cfg)
    p_force = float(np.max(res["maximum_station_force_n"]))
    couples = np.asarray(res["force_couple_vector_nm"], dtype=float)
    p_couple = float(np.max(np.linalg.norm(couples, axis=1)))
    sl_speed = float(np.max(res["maximum_sliding_speed_m_s"]))
    tr_count = int(res["total_transition_count"])
    tot_e = np.asarray(res["total_energy_j"], dtype=float)
    res_e = np.asarray(res["work_energy_residual_j"], dtype=float)
    e_res = float(np.max(np.abs(res_e))) / max(1.0, float(np.ptp(tot_e)))
    f_class = str(res["first_failure_class"])
    return p_force, p_couple, sl_speed, tr_count, e_res, f_class


def _build_uncertainty_study_record(
    config: ArticulatedUncertaintyConfig,
    param_matrix: FloatArray,
    peak_forces: FloatArray,
    peak_couples: FloatArray,
    sliding_speeds: FloatArray,
    transition_counts: NDArray[np.int_],
    energy_residuals: FloatArray,
    failure_classes: list[str],
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    response_matrix = np.column_stack(
        [
            peak_forces,
            peak_couples,
            sliding_speeds,
            transition_counts.astype(float),
            energy_residuals,
        ]
    )
    dim = param_matrix.shape[1]
    prcc_matrix = np.zeros((len(OUTPUT_METRICS), dim))
    for m in range(len(OUTPUT_METRICS)):
        prcc_matrix[m] = partial_rank_correlations(param_matrix, response_matrix[:, m])

    unique_classes, counts = np.unique(failure_classes, return_counts=True)
    failure_map = {str(k): int(v) for k, v in zip(unique_classes, counts, strict=True)}

    arrays = {
        "parameter_samples": param_matrix,
        "parameter_names": np.asarray(UNCERTAINTY_PARAMETERS),
        "output_metric_names": np.asarray(OUTPUT_METRICS),
        "response_matrix": response_matrix,
        "prcc_sensitivity_matrix": prcc_matrix,
        "failure_classes": np.asarray(failure_classes),
    }
    record = {
        "schema_version": "articulated-uncertainty-study/v1",
        "study_id": "articulated-parameter-uncertainty-and-sensitivity",
        "configuration": asdict(config),
        "uncertainty_parameters": list(UNCERTAINTY_PARAMETERS),
        "output_metrics": list(OUTPUT_METRICS),
        "results": {
            "sample_count": config.sample_count,
            "failure_distribution": failure_map,
            "mean_peak_force_n": float(np.mean(peak_forces)),
            "mean_peak_couple_nm": float(np.mean(peak_couples)),
            "maximum_normalized_work_energy_residual": float(np.max(energy_residuals)),
            "all_simulations_energy_closed": bool(np.all(energy_residuals < 0.05)),
            "top_sensitivities": {
                metric: {
                    UNCERTAINTY_PARAMETERS[
                        int(np.argmax(np.abs(prcc_matrix[m])))
                    ]: float(prcc_matrix[m, int(np.argmax(np.abs(prcc_matrix[m])))])
                }
                for m, metric in enumerate(OUTPUT_METRICS)
            },
        },
    }
    return record, arrays


def run_articulated_uncertainty_study(
    config: ArticulatedUncertaintyConfig = ArticulatedUncertaintyConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Execute LHS parameter sweeps, PRCC sensitivity mapping, and failure classification."""
    base_model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        q_base = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    hand_contact_x = float(metadata["hand_contact_local_x_m"])

    param_matrix = _sample_parameters(config)
    peak_forces = np.zeros(config.sample_count)
    peak_couples = np.zeros(config.sample_count)
    sliding_speeds = np.zeros(config.sample_count)
    transition_counts = np.zeros(config.sample_count, dtype=int)
    energy_residuals = np.zeros(config.sample_count)
    failure_classes: list[str] = []

    forward_cfg = DistributedForwardConfig(
        duration_s=config.duration_s,
        time_steps_s=(0.001, 0.0005),
    )

    for s in range(config.sample_count):
        pf, pc, ss, tc, er, fc = _evaluate_uncertainty_trajectory(
            param_matrix[s],
            base_model,
            q_base,
            grip_span_m,
            hand_contact_x,
            config,
            forward_cfg,
        )
        peak_forces[s] = pf
        peak_couples[s] = pc
        sliding_speeds[s] = ss
        transition_counts[s] = tc
        energy_residuals[s] = er
        failure_classes.append(fc)

    return _build_uncertainty_study_record(
        config,
        param_matrix,
        peak_forces,
        peak_couples,
        sliding_speeds,
        transition_counts,
        energy_residuals,
        failure_classes,
    )


def main() -> None:
    record, arrays = run_articulated_uncertainty_study()
    json_path = DATA_DIR / "articulated_uncertainty_study.json"
    npz_path = DATA_DIR / "articulated_uncertainty_study.npz"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    print("Saved:", json_path)
    print("Saved:", npz_path)
    print("Results:", json.dumps(record["results"], indent=2))


if __name__ == "__main__":
    main()
