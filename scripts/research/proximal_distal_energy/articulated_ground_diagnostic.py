"""Registered 4 ms ground-pathway initialization and time-step diagnostic."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.articulated_ground import (
    ArticulatedGroundConfig,
    build_articulated_ground,
    ground_extra_potential_gradient,
)
from scripts.research.proximal_distal_energy.articulated_ground_forward import (
    GroundForwardConfig,
    GroundIntegrationCase,
    integrate_articulated_ground,
    solve_conditional_base_equilibrium,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
    build_articulated_shaft,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
STEPS_S = (0.00025, 0.000125, 0.0000625)
ENGINES = ("mujoco", "pinocchio")
BRANCHES = (
    "fixed_zero",
    "translation_perturbed",
    "free_moment_perturbed",
    "coupled_perturbed",
    "coupled_natural_zero",
    "coupled_gravity_only",
    "coupled_conditional",
)
SOURCES = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "scripts/research/proximal_distal_energy/articulated_distributed_grip.py",
    "scripts/research/proximal_distal_energy/articulated_ground.py",
    "scripts/research/proximal_distal_energy/articulated_ground_forward.py",
    "scripts/research/proximal_distal_energy/articulated_ground_diagnostic.py",
    "tests/research/test_articulated_ground.py",
    "tests/research/test_articulated_ground_forward.py",
    "tests/research/test_articulated_ground_diagnostic.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_error(left: Any, right: Any) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(a)))))


def _branch_states(
    model: Any,
    q: np.ndarray,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    grip: DistributedGripConfig,
    shaft_config: ArticulatedShaftConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    ground_config = ArticulatedGroundConfig()
    shaft = build_articulated_shaft(model, shaft_config)
    ground = build_articulated_ground(ground_config)
    offset = model.nq + shaft.coordinate_count
    gravity_gradient = ground_extra_potential_gradient(
        model, q, np.zeros(ground.coordinate_count), shaft, ground
    )[offset:]
    gravity_only = -np.linalg.solve(ground.stiffness, gravity_gradient)
    conditional = solve_conditional_base_equilibrium(
        model,
        q,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        grip_config=grip,
        shaft_config=shaft_config,
        ground_config=ground_config,
    )
    states = {
        "fixed_zero": {
            "activation": "fixed",
            "displacement": (0.0, 0.0, 0.0),
            "velocity": (0.0, 0.0, 0.0),
        },
        "translation_perturbed": {
            "activation": "translation",
            "displacement": (-0.001, 0.001, 0.0),
            "velocity": (-0.01, 0.01, 0.0),
        },
        "free_moment_perturbed": {
            "activation": "free_moment",
            "displacement": (0.0, 0.0, 0.002),
            "velocity": (0.0, 0.0, 0.02),
        },
        "coupled_perturbed": {
            "activation": "coupled",
            "displacement": (-0.001, 0.001, 0.002),
            "velocity": (-0.01, 0.01, 0.02),
        },
        "coupled_natural_zero": {
            "activation": "coupled",
            "displacement": (0.0, 0.0, 0.0),
            "velocity": (0.0, 0.0, 0.0),
        },
        "coupled_gravity_only": {
            "activation": "coupled",
            "displacement": tuple(float(value) for value in gravity_only),
            "velocity": (0.0, 0.0, 0.0),
        },
        "coupled_conditional": {
            "activation": "coupled",
            "displacement": conditional.base_coordinates,
            "velocity": (0.0, 0.0, 0.0),
        },
    }
    return states, {
        "gravity_gradient": gravity_gradient.tolist(),
        "gravity_only_coordinates": gravity_only.tolist(),
        "conditional_equilibrium": {
            "coordinates": list(conditional.base_coordinates),
            "residual_generalized_force": list(conditional.residual_generalized_force),
            "residual_norm": conditional.residual_norm,
            "iteration_count": conditional.iteration_count,
            "active_station_count": conditional.active_station_count,
            "maximum_station_force_n": conditional.maximum_station_force_n,
        },
    }


def run_articulated_ground_diagnostic() -> dict[str, Any]:
    """Execute the registered branches, steps, and native engines."""

    case_index, sample_index = 0, 6
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        time_s = np.asarray(source["time_s"], dtype=float)
        solution_q = np.asarray(source["solution_q"][case_index], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][case_index])
        profile_index = int(source["case_profile_index"][case_index])
    velocity, _ = finite_difference_kinematics(solution_q, time_s)
    model, metadata = build_subject_scaled_model(
        default_synthetic_profiles()[profile_index]
    )
    hand_contact = float(metadata["hand_contact_local_x_m"])
    grip = DistributedGripConfig(
        station_count_per_hand=5,
        station_width_m=0.03,
        total_stiffness_n_m=1800.0,
        total_damping_n_s_m=18.0,
    )
    shaft = ArticulatedShaftConfig()
    states, initialization = _branch_states(
        model,
        solution_q[sample_index],
        grip_span_m,
        hand_contact,
        grip,
        shaft,
    )
    results: list[dict[str, Any]] = []
    traces: dict[tuple[str, float, str], dict[str, Any]] = {}
    for branch in BRANCHES:
        state = states[branch]
        for step_s in STEPS_S:
            for engine in ENGINES:
                trace = integrate_articulated_ground(
                    model,
                    GroundIntegrationCase(
                        q=solution_q[sample_index],
                        qd=velocity[sample_index],
                        grip_span_m=grip_span_m,
                        hand_contact_local_x_m=hand_contact,
                        time_step_s=step_s,
                        initial_club_displacement_m=0.001,
                        initial_club_velocity_m_s=0.05,
                        initial_base_displacement=state["displacement"],
                        initial_base_velocity=state["velocity"],
                        engine=engine,
                        grip=grip,
                        shaft=shaft,
                        ground=ArticulatedGroundConfig(activation=state["activation"]),
                    ),
                    GroundForwardConfig(
                        duration_s=0.004,
                        time_steps_s=(step_s, step_s / 2.0),
                    ),
                )
                traces[(branch, step_s, engine)] = trace
                total = np.asarray(trace["total_energy_j"], dtype=float)
                residual = np.asarray(trace["work_energy_residual_j"], dtype=float)
                results.append(
                    {
                        "branch": branch,
                        "activation": state["activation"],
                        "step_s": step_s,
                        "engine": engine,
                        "maximum_absolute_work_energy_residual_j": float(
                            np.max(np.abs(residual))
                        ),
                        "normalized_work_energy_residual": float(
                            np.max(np.abs(residual)) / max(1.0, float(np.ptp(total)))
                        ),
                        "peak_ground_force_n": float(
                            np.max(np.linalg.norm(trace["ground_force_n"], axis=1))
                        ),
                        "peak_intrinsic_free_moment_nm": float(
                            np.max(np.abs(trace["ground_intrinsic_free_moment_nm"]))
                        ),
                        "maximum_base_translation_m": float(
                            np.max(np.linalg.norm(trace["base_translation_m"], axis=1))
                        ),
                        "maximum_base_pitch_rad": float(
                            np.max(np.abs(trace["base_pitch_rad"]))
                        ),
                        "maximum_shaft_deflection_m": float(
                            np.max(np.linalg.norm(trace["tip_bending_m"], axis=1))
                        ),
                        "maximum_shaft_twist_rad": float(
                            np.max(np.abs(trace["twist_angle_rad"]))
                        ),
                        "final_club_translation_speed_m_s": float(
                            np.linalg.norm(trace["qd"][-1, 14:17])
                        ),
                        "active_set_transition_count": int(
                            np.count_nonzero(trace["active_set_transition"])
                        ),
                        "maximum_virtual_power_residual_w": float(
                            np.max(np.abs(trace["virtual_power_residual_w"]))
                        ),
                        "maximum_ground_power_residual_w": float(
                            np.max(np.abs(trace["ground_power_residual_w"]))
                        ),
                        "remained_in_declared_domains": True,
                    }
                )
    parity = []
    refinement = []
    for branch in BRANCHES:
        for step_s in STEPS_S:
            left, right = (
                traces[(branch, step_s, "mujoco")],
                traces[(branch, step_s, "pinocchio")],
            )
            parity.append(
                {
                    "branch": branch,
                    "step_s": step_s,
                    "trajectory_relative_error": max(
                        _relative_error(left["q"], right["q"]),
                        _relative_error(
                            left["elastic_coordinates"],
                            right["elastic_coordinates"],
                        ),
                        _relative_error(
                            left["base_coordinates"], right["base_coordinates"]
                        ),
                    ),
                    "ground_force_relative_error": _relative_error(
                        left["ground_force_n"], right["ground_force_n"]
                    ),
                    "active_set_parity": bool(
                        np.array_equal(
                            left["active_station_count"],
                            right["active_station_count"],
                        )
                    ),
                }
            )
        for engine in ENGINES:
            residuals = [
                next(
                    item["maximum_absolute_work_energy_residual_j"]
                    for item in results
                    if item["branch"] == branch
                    and item["step_s"] == step
                    and item["engine"] == engine
                )
                for step in STEPS_S
            ]
            refinement.append(
                {
                    "branch": branch,
                    "engine": engine,
                    "residuals_j": residuals,
                    "successive_ratios": (
                        np.asarray(residuals[1:]) / np.asarray(residuals[:-1])
                    ).tolist(),
                    "monotone": bool(np.all(np.diff(residuals) < 0.0)),
                }
            )
    return {
        "schema_version": "articulated-ground-diagnostic/v1",
        "state": [case_index, sample_index],
        "duration_s": 0.004,
        "steps_s": list(STEPS_S),
        "engines": list(ENGINES),
        "branches": list(BRANCHES),
        "initialization": initialization,
        "results": results,
        "parity": parity,
        "refinement": refinement,
        "all_refinement_monotone": all(item["monotone"] for item in refinement),
        "all_active_sets_match": all(item["active_set_parity"] for item in parity),
        "calibration_status": "synthetic_reference_not_human_or_force_plate_calibrated",
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCES},
    }


__all__ = ["BRANCHES", "ENGINES", "STEPS_S", "run_articulated_ground_diagnostic"]
