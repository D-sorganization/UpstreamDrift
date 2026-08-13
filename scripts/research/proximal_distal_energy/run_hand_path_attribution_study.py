"""Generate deterministic hand-path drift/control attribution evidence.

The model ladder deliberately distinguishes pointwise drift/control attribution
from a zero-velocity control-preserved evaluation. The two-arm tier
uses prescribed, constraint-consistent kinematics; it is not a forward
simulation or a measured golfer trajectory.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.research.proximal_distal_energy.double_pendulum_attribution import (
    double_pendulum_joint_transfer_trajectory,
)
from scripts.research.proximal_distal_energy.one_arm_attribution import (
    one_arm_joint_transfer_trajectory,
)
from scripts.research.proximal_distal_energy.run_experiments import rollout_program
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import (
    TwoArmControl,
    TwoArmParams,
    constraint_jacobian,
    two_arm_joint_transfer_trajectory,
)
from src.shared.python.biomechanics.drift_control_transfer import (
    JointTransferTrajectory,
    SwingPhase,
    attribution_shares,
    build_phase_masks,
    compute_impulses,
    compute_path_frame,
    compute_path_weighted_mean_force,
    compute_power_and_work,
    project_forces_onto_path,
    summarize_phases,
)
from src.shared.python.pendulum_simulator.physics_triple import TriplePendulumParams
from src.shared.python.pendulum_simulator.simulation_triple import run_simulation
from src.shared.python.simulation_backends import GolfModelParams

SCHEMA_VERSION = "hand-path-attribution-evidence-v1"
ZERO_VELOCITY_CONTROL_PRESERVED_PROTOCOL = (
    "Same configuration and applied control; zero generalized velocity; "
    "gravity retained; velocity-dependent damping evaluates at zero."
)
FIGURE_STEMS = (
    "fig_hand_path_force_vectors",
    "fig_hand_path_components",
    "fig_hand_path_impulse_work",
    "fig_hand_path_phase_joint_shares",
    "fig_hand_path_power_shares",
    "fig_two_hand_common_differential",
    "fig_hand_path_sensitivity_closure",
)
_MODEL_ORDER = ("double_pendulum", "one_arm", "two_arm")
_MODEL_LABELS = {
    "double_pendulum": "Double Pendulum",
    "one_arm": "One-Arm Three-Link",
    "two_arm": "Two-Arm Closed Loop",
}
_SPLIT_COLORS = {
    "total": "#1f2937",
    "drift": "#0072B2",
    "control": "#D55E00",
    "zvcf": "#009E73",
}


def _split_label(split: str) -> str:
    return "Zero-Velocity Control-Preserved" if split == "zvcf" else split.title()


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_manifest() -> list[dict[str, str]]:
    root = _source_root()
    paths = (
        Path(__file__).resolve(),
        root / "scripts/research/proximal_distal_energy/double_pendulum_attribution.py",
        root / "scripts/research/proximal_distal_energy/one_arm_attribution.py",
        root / "scripts/research/proximal_distal_energy/two_arm_closed_loop.py",
        root / "src/shared/python/biomechanics/drift_control_transfer.py",
    )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]


def _double_pendulum() -> tuple[JointTransferTrajectory, JointTransferTrajectory]:
    params = GolfModelParams.default()
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    time, q, velocity, controls = rollout_program(params, program)
    impact = find_impact(time, q, velocity, PlanarInertials.from_params(params))
    if impact is None:
        raise ValueError("reference double-pendulum case has no valid first impact")
    sampled_time = np.linspace(float(time[0]), impact[0], 81)

    def interpolate(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.interp(sampled_time, time, values[:, column]) for column in range(2)]
        )

    q, velocity, controls = map(interpolate, (q, velocity, controls))
    time = sampled_time
    achieved = double_pendulum_joint_transfer_trajectory(
        time, q, velocity, controls, params
    )
    zvcf = double_pendulum_joint_transfer_trajectory(
        time, q, np.zeros_like(velocity), controls, params
    )
    return achieved, zvcf


def _one_arm() -> tuple[JointTransferTrajectory, JointTransferTrajectory]:
    params = TriplePendulumParams(
        m1=2.1,
        m2=1.4,
        m3=0.45,
        L1=0.32,
        L2=0.30,
        L3=1.05,
        b1=0.08,
        b2=0.05,
        b3=0.02,
    )
    analysis_end = 0.40
    integration_end = 0.405

    def torque(time_s: float) -> tuple[float, float, float]:
        fraction = float(np.clip(time_s / analysis_end, 0.0, 1.0))
        return (
            34.0 + 8.0 * fraction,
            13.0 - 7.0 * fraction,
            -7.0 + 15.0 * fraction,
        )

    result = run_simulation(
        params,
        np.array([-1.05, -0.85, -0.55, 0.0, 0.0, 0.0]),
        integration_end,
        torque,
        dt=0.005,
        rtol=1e-9,
        atol=1e-11,
    )
    time = result.t
    q = result.states[:, :3]
    velocity = result.states[:, 3:]
    retained = time <= analysis_end + 1e-12
    time, q, velocity = time[retained], q[retained], velocity[retained]
    if time.size != 81 or not np.isclose(time[-1], analysis_end):
        raise ValueError(
            "one-arm integration did not cover the declared analysis window"
        )
    controls = np.asarray([torque(sample) for sample in time])
    achieved = one_arm_joint_transfer_trajectory(time, q, velocity, controls, params)
    zvcf = one_arm_joint_transfer_trajectory(
        time, q, np.zeros_like(velocity), controls, params
    )
    return achieved, zvcf


def _two_arm() -> tuple[JointTransferTrajectory, JointTransferTrajectory]:
    params = TwoArmParams.publication_default()
    time = np.linspace(0.0, 0.40, 81)
    s = time / time[-1]
    centers = np.column_stack(
        (0.010 * np.sin(np.pi * s), -0.50 - 0.025 * s - 0.008 * np.sin(2 * np.pi * s))
    )
    angles = 0.16 + 0.10 * s + 0.025 * np.sin(np.pi * s)
    q = np.vstack(
        [
            params.consistent_configuration(center, angle)
            for center, angle in zip(centers, angles, strict=True)
        ]
    )
    raw_velocity = np.gradient(q, time, axis=0, edge_order=2)
    velocity = np.empty_like(raw_velocity)
    for index, (configuration, candidate) in enumerate(
        zip(q, raw_velocity, strict=True)
    ):
        jacobian = constraint_jacobian(configuration, params)
        gram = jacobian @ jacobian.T
        velocity[index] = candidate - jacobian.T @ np.linalg.solve(
            gram, jacobian @ candidate
        )
    controls = tuple(
        TwoArmControl(
            right_shoulder_nm=float(18.0 + 4.0 * value),
            right_elbow_nm=float(7.0 - 1.5 * value),
            right_wrist_nm=float(-3.0 + 2.0 * value),
            left_shoulder_nm=float(16.0 + 3.0 * value),
            left_elbow_nm=float(6.0 - value),
            left_wrist_nm=float(2.0 - value),
        )
        for value in s
    )
    achieved = two_arm_joint_transfer_trajectory(time, q, velocity, controls, params)
    zvcf = two_arm_joint_transfer_trajectory(
        time, q, np.zeros_like(velocity), controls, params
    )
    return achieved, zvcf


def _phases(time: np.ndarray) -> tuple[SwingPhase, ...]:
    names = tuple(f"Normalized Time Quartile {index}" for index in range(1, 5))
    indices = (0, 20, 40, 60, 80)
    return tuple(
        SwingPhase(name, float(time[start]), float(time[end]))
        for name, start, end in zip(names, indices[:-1], indices[1:], strict=True)
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _closure(
    trajectory: JointTransferTrajectory, power: Any, summaries: tuple[Any, ...]
) -> dict[str, float]:
    force = trajectory.force_total - trajectory.force_drift - trajectory.force_control
    couple = (
        trajectory.couple_total - trajectory.couple_drift - trajectory.couple_control
    )
    power_residual = (
        power.total_power_total - power.total_power_drift - power.total_power_control
    )
    work_residual = (
        power.total_work_total - power.total_work_drift - power.total_work_control
    )
    phase_total = np.sum([summary.total_work_total for summary in summaries], axis=0)
    phase_additivity = phase_total - power.total_work_total[-1]
    return {
        "force_max_abs": float(np.max(np.abs(force))),
        "couple_max_abs": float(np.max(np.abs(couple))),
        "power_max_abs": float(np.max(np.abs(power_residual))),
        "work_max_abs": float(np.max(np.abs(work_residual))),
        "phase_additivity_max_abs": float(np.max(np.abs(phase_additivity))),
    }


def _sampling_sensitivity(trajectory: JointTransferTrajectory) -> dict[str, Any]:
    distal = trajectory.joint_count - 1
    reference_force_work = float(
        np.trapezoid(
            np.einsum(
                "td,td->t",
                trajectory.force_total[:, distal],
                trajectory.velocity[:, distal],
            ),
            trajectory.time,
        )
    )
    reference_vector_impulse = np.trapezoid(
        trajectory.force_total[:, distal], trajectory.time, axis=0
    )
    rows = []
    for stride in (1, 2, 4):
        indices = np.arange(0, trajectory.sample_count, stride)
        if indices[-1] != trajectory.sample_count - 1:
            indices = np.append(indices, trajectory.sample_count - 1)
        work = float(
            np.trapezoid(
                np.einsum(
                    "td,td->t",
                    trajectory.force_total[indices, distal],
                    trajectory.velocity[indices, distal],
                ),
                trajectory.time[indices],
            )
        )
        impulse = np.trapezoid(
            trajectory.force_total[indices, distal], trajectory.time[indices], axis=0
        )
        rows.append(
            {
                "stride": stride,
                "sample_count": int(indices.size),
                "force_work_abs_delta_j": abs(work - reference_force_work),
                "vector_impulse_abs_delta_n_s": float(
                    np.linalg.norm(impulse - reference_vector_impulse)
                ),
            }
        )
    return {
        "type": "quadrature_sampling_sensitivity_not_parameter_uncertainty",
        "rows": rows,
    }


def _primary_estimand(
    key: str,
    trajectory: JointTransferTrajectory,
    zvcf: JointTransferTrajectory,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Return MacKenzie-compatible net force work per achieved path length."""
    if key == "two_arm":
        position = 0.5 * (trajectory.position[:, 2] + trajectory.position[:, 5])
        velocity = 0.5 * (trajectory.velocity[:, 2] + trajectory.velocity[:, 5])
        forces = {
            "total": trajectory.force_total[:, 2] + trajectory.force_total[:, 5],
            "drift": trajectory.force_drift[:, 2] + trajectory.force_drift[:, 5],
            "control": trajectory.force_control[:, 2] + trajectory.force_control[:, 5],
            "zvcf": zvcf.force_total[:, 2] + zvcf.force_total[:, 5],
        }
        reference_point = "mid_grip_equal_to_club_center_for_symmetric_grip_offsets"
    else:
        distal = trajectory.joint_count - 1
        position = trajectory.position[:, distal]
        velocity = trajectory.velocity[:, distal]
        forces = {
            "total": trajectory.force_total[:, distal],
            "drift": trajectory.force_drift[:, distal],
            "control": trajectory.force_control[:, distal],
            "zvcf": zvcf.force_total[:, distal],
        }
        reference_point = trajectory.joint_names[distal]
    speed = np.linalg.norm(velocity, axis=1)
    path_length = float(np.trapezoid(speed, trajectory.time))
    if path_length <= 1e-12:
        raise ValueError(f"{key} primary path length is undefined")
    tangent = np.zeros_like(velocity)
    valid = speed > 1e-12
    tangent[valid] = velocity[valid] / speed[valid, None]
    force_along = {
        split: np.einsum("td,td->t", force, tangent) for split, force in forces.items()
    }
    force_work = {
        split: float(
            np.trapezoid(np.einsum("td,td->t", force, velocity), trajectory.time)
        )
        for split, force in forces.items()
    }
    prefix = f"{key}__primary_"
    arrays[prefix + "position"] = position
    arrays[prefix + "velocity"] = velocity
    arrays[prefix + "path_tangent"] = tangent
    for split, force in forces.items():
        arrays[prefix + f"force_{split}"] = force
        arrays[prefix + f"force_along_{split}"] = force_along[split]
        cumulative_impulse = np.zeros_like(force)
        cumulative_impulse[1:] = np.cumsum(
            0.5 * (force[:-1] + force[1:]) * np.diff(trajectory.time)[:, None],
            axis=0,
        )
        arrays[prefix + f"vector_impulse_{split}"] = cumulative_impulse

    drift_power = np.einsum("td,td->t", forces["drift"], velocity)
    final_half_indices = slice(40, 81)
    final_half_along = force_along["drift"][final_half_indices]
    nonzero_along = final_half_along[np.abs(final_half_along) > 1e-10]
    direction_reversal = bool(
        nonzero_along.size > 1
        and np.min(nonzero_along) < 0.0
        and np.max(nonzero_along) > 0.0
    )
    quartile_3_work = float(np.trapezoid(drift_power[40:61], trajectory.time[40:61]))
    quartile_4_work = float(np.trapezoid(drift_power[60:81], trajectory.time[60:81]))
    work_reversal = bool(quartile_3_work * quartile_4_work < 0.0)
    return {
        "name": "net_golfer_on_club_force_work_per_hand_path_length",
        "definition": "W_linear/L = integral(F_net dot v_mid-grip dt) / integral(|v_mid-grip| dt)",
        "reference_point": reference_point,
        "force_direction": "golfer_on_club",
        "path_length_m": path_length,
        "force_work_j": force_work,
        "mean_force_n": {
            split: work / path_length for split, work in force_work.items()
        },
        "zvcf_projection_note": (
            "Zero-velocity control-preserved force is projected on the achieved "
            "mid-grip path only for this "
            "diagnostic; the zero-velocity evaluation itself has no traversed path."
        ),
        "final_half_drift_diagnostic": {
            "window": "Normalized Time Quartiles 3-4",
            "direction_reversal_present": direction_reversal,
            "quartile_3_force_work_j": quartile_3_work,
            "quartile_4_force_work_j": quartile_4_work,
            "phase_work_reversal_present": work_reversal,
            "conclusion": (
                "present_in_declared_case"
                if direction_reversal and work_reversal
                else "not_demonstrated_by_declared_case"
            ),
        },
    }


def _analyze(
    key: str,
    trajectory: JointTransferTrajectory,
    zvcf: JointTransferTrajectory,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    frame = compute_path_frame(trajectory.velocity, speed_epsilon=1e-10)
    projection = project_forces_onto_path(trajectory, frame)
    impulses = compute_impulses(trajectory, projection)
    power = compute_power_and_work(trajectory)
    mean_force = compute_path_weighted_mean_force(trajectory, frame)
    phases = _phases(trajectory.time)
    masks = build_phase_masks(trajectory.time, phases)
    summaries = summarize_phases(trajectory, impulses, power, phases)
    zvcf_vector_impulse = np.zeros_like(zvcf.force_total)
    dt = np.diff(zvcf.time)[:, None, None]
    zvcf_vector_impulse[1:] = np.cumsum(
        0.5 * (zvcf.force_total[:-1] + zvcf.force_total[1:]) * dt, axis=0
    )

    prefix = f"{key}__"
    arrays.update(
        {
            prefix + "time": trajectory.time,
            prefix + "position": trajectory.position,
            prefix + "velocity": trajectory.velocity,
            prefix + "force_total": trajectory.force_total,
            prefix + "force_drift": trajectory.force_drift,
            prefix + "force_control": trajectory.force_control,
            prefix + "force_zvcf": zvcf.force_total,
            prefix + "couple_total": trajectory.couple_total,
            prefix + "couple_drift": trajectory.couple_drift,
            prefix + "couple_control": trajectory.couple_control,
            prefix + "path_tangent": frame.tangent,
            prefix + "path_normal": frame.normal,
            prefix + "path_valid": frame.valid,
            prefix + "force_along_total": projection.total_along,
            prefix + "force_along_drift": projection.drift_along,
            prefix + "force_along_control": projection.control_along,
            prefix + "force_normal_total": projection.total_normal,
            prefix + "force_normal_drift": projection.drift_normal,
            prefix + "force_normal_control": projection.control_normal,
            prefix + "vector_impulse_total": impulses.vector_total,
            prefix + "vector_impulse_drift": impulses.vector_drift,
            prefix + "vector_impulse_control": impulses.vector_control,
            prefix + "vector_impulse_zvcf": zvcf_vector_impulse,
            prefix + "force_power_total": power.force_power_total,
            prefix + "force_power_drift": power.force_power_drift,
            prefix + "force_power_control": power.force_power_control,
            prefix + "couple_power_total": power.couple_power_total,
            prefix + "couple_power_drift": power.couple_power_drift,
            prefix + "couple_power_control": power.couple_power_control,
            prefix + "total_power_total": power.total_power_total,
            prefix + "total_power_drift": power.total_power_drift,
            prefix + "total_power_control": power.total_power_control,
            prefix + "force_work_total": power.force_work_total,
            prefix + "force_work_drift": power.force_work_drift,
            prefix + "force_work_control": power.force_work_control,
            prefix + "couple_work_total": power.couple_work_total,
            prefix + "couple_work_drift": power.couple_work_drift,
            prefix + "couple_work_control": power.couple_work_control,
            prefix + "total_work_total": power.total_work_total,
            prefix + "total_work_drift": power.total_work_drift,
            prefix + "total_work_control": power.total_work_control,
        }
    )
    for split in ("total", "drift", "control"):
        for measure in ("signed", "positive", "negative", "absolute"):
            arrays[prefix + f"tangent_impulse_{split}_{measure}"] = getattr(
                impulses, f"tangent_{split}_{measure}"
            )
    power_denominator = np.abs(power.total_power_drift) + np.abs(
        power.total_power_control
    )
    power_share_valid = power_denominator > 1e-10
    drift_power_magnitude_share = np.full(power_denominator.shape, np.nan)
    power_cancellation_index = np.full(power_denominator.shape, np.nan)
    drift_power_magnitude_share[power_share_valid] = (
        np.abs(power.total_power_drift[power_share_valid])
        / power_denominator[power_share_valid]
    )
    power_cancellation_index[power_share_valid] = np.clip(
        1.0
        - np.abs(power.total_power_total[power_share_valid])
        / power_denominator[power_share_valid],
        0.0,
        1.0,
    )
    arrays[prefix + "drift_power_magnitude_share"] = drift_power_magnitude_share
    arrays[prefix + "power_share_valid"] = power_share_valid
    arrays[prefix + "power_cancellation_index"] = power_cancellation_index
    arrays[prefix + "power_cancellation_flag"] = power_share_valid & (
        power_cancellation_index >= 0.25
    )
    phase_records = []
    for summary in summaries:
        tangent_shares = attribution_shares(
            summary.tangent_impulse_total,
            summary.tangent_impulse_drift,
            summary.tangent_impulse_control,
        )
        work_shares = attribution_shares(
            summary.total_work_total,
            summary.total_work_drift,
            summary.total_work_control,
        )
        item = asdict(summary)
        item["tangent_impulse_shares"] = asdict(tangent_shares)
        item["total_work_shares"] = asdict(work_shares)
        item["cancellation_flag"] = (
            np.nan_to_num(work_shares.cancellation_index, nan=0.0) >= 0.25
        )
        start, end = summary.start_index, summary.end_index
        item["zvcf_vector_impulse"] = (
            zvcf_vector_impulse[end] - zvcf_vector_impulse[start]
        )
        phase_records.append(_json_value(item))

    coverage = np.sum(np.stack(tuple(masks.values())), axis=0)
    result: dict[str, Any] = {
        "model_tier": trajectory.model_tier,
        "trajectory_kind": {
            "double_pendulum": "forward_simulation_truncated_at_first_valid_impact",
            "one_arm": "forward_simulation_on_declared_fixed_time_window",
            "two_arm": "prescribed_constraint_consistent_kinematics",
        }[key],
        "joint_names": list(trajectory.joint_names),
        "force_direction": trajectory.force_direction,
        "frame": trajectory.frame,
        "units": trajectory.units,
        "sample_count": trajectory.sample_count,
        "phase_coverage": {
            "exhaustive": bool(np.all(coverage == 1)),
            "overlap_count": int(np.sum(coverage > 1)),
        },
        "phase_semantics": (
            "Equal-duration normalized-time quartiles; names do not assert anatomical or "
            "event-defined swing phases."
        ),
        "integration_interpretation": (
            "Kinematic-sweep diagnostic on a prescribed constraint-consistent path; "
            "not realized forward-simulation work."
            if key == "two_arm"
            else "Integrated along a forward-simulated trajectory."
        ),
        "zero_velocity_control_preserved": {
            "status": "available",
            "protocol": ZERO_VELOCITY_CONTROL_PRESERVED_PROTOCOL,
            "interpretation": (
                "A same-configuration diagnostic that preserves applied control; "
                "not canonical ZVCF, not the control contribution, and not a "
                "forward counterfactual trajectory."
            ),
            "legacy_array_suffix": "zvcf",
        },
        "path_weighted_mean_force": _json_value(asdict(mean_force)),
        "instantaneous_power_share": {
            "definition": "abs(P_drift)/(abs(P_drift)+abs(P_control))",
            "power_scope": "total_wrench_force_plus_couple_power",
            "near_zero_denominator_threshold_w": 1e-10,
            "undefined_samples_are_masked": True,
            "cancellation_flag_threshold": 0.25,
        },
        "primary_estimand": _primary_estimand(key, trajectory, zvcf, arrays),
        "phase_summaries": phase_records,
        "closure": _closure(trajectory, power, summaries),
        "sensitivity": _sampling_sensitivity(trajectory),
    }
    if key == "two_arm":
        result["common_differential"] = {
            "convention": "common=right+left; differential=(right-left)/2",
            "force_on": "club_at_hand_contacts",
        }
        for split, values in (
            ("total", trajectory.force_total),
            ("drift", trajectory.force_drift),
            ("control", trajectory.force_control),
            ("zvcf", zvcf.force_total),
        ):
            right, left = values[:, 2], values[:, 5]
            arrays[prefix + f"common_{split}"] = right + left
            arrays[prefix + f"differential_{split}"] = 0.5 * (right - left)
    if key == "double_pendulum":
        result["terminal_event"] = {
            "name": "first_valid_club_vertical_impact",
            "time_s": float(trajectory.time[-1]),
            "source": "swing_model.find_impact",
        }
    return result


def build_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build deterministic metadata and dense numeric arrays in memory."""
    trajectories = {
        "double_pendulum": _double_pendulum(),
        "one_arm": _one_arm(),
        "two_arm": _two_arm(),
    }
    arrays: dict[str, np.ndarray] = {}
    models = {key: _analyze(key, *trajectories[key], arrays) for key in _MODEL_ORDER}
    vector_scales = {}
    for key in _MODEL_ORDER:
        joint_indices = (
            (2, 5) if key == "two_arm" else (len(models[key]["joint_names"]) - 1,)
        )
        vector_scales[key] = float(
            max(
                np.linalg.norm(
                    arrays[f"{key}__force_{split}"][:, joint_indices], axis=2
                ).max()
                for split in _SPLIT_COLORS
            )
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "study_scope": (
            "Deterministic planar mechanism evidence; not subject-specific inference, "
            "muscle-force identification, or a claim of optimal human control."
        ),
        "models": models,
        "provenance": {
            "generator": "scripts/research/proximal_distal_energy/run_hand_path_attribution_study.py",
            "source_files": _source_manifest(),
            "random_seed": None,
            "runtime_network_required": False,
        },
        "figure_protocols": {
            "force_vectors": {
                "scaling": (
                    "Local coordinates centered at the contact (single-hand tiers) or "
                    "mid-grip (two-arm tier); one fixed force scale and one square "
                    "viewport per model row across all quartiles, contacts, and splits."
                ),
                "force_arrow_local_length": "force_n/(2.5*row_max_force_n)",
                "path_tangent_local_length": 0.22,
                "row_max_force_n": vector_scales,
            }
        },
    }
    return record, arrays


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    svg_path = output / f"{stem}.svg"
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Creator": "Independent Open Research", "Date": None},
    )
    # Matplotlib leaves insignificant spaces at the end of multiline path
    # definitions. Normalize those bytes so generated SVGs satisfy the same
    # whitespace gate as hand-authored sources.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(
        output / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "Independent Open Research",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _make_figures(
    record: dict[str, Any], arrays: dict[str, np.ndarray], output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "figure.titlesize": 12,
            "svg.fonttype": "none",
            "svg.hashsalt": "hand-path-attribution-v1",
            # Reuse PDF core fonts instead of embedding seven duplicate subsets.
            # This keeps text as vectors while satisfying the lossless article
            # size guard without reducing figure content or resolution.
            "pdf.use14corefonts": True,
            "pdf.compression": 9,
        }
    )

    fig, axes = plt.subplots(3, 4, figsize=(11.0, 7.2), sharex=False, sharey=False)
    for row, key in enumerate(_MODEL_ORDER):
        joint_indices = (
            (2, 5)
            if key == "two_arm"
            else (len(record["models"][key]["joint_names"]) - 1,)
        )
        row_scale = record["figure_protocols"]["force_vectors"]["row_max_force_n"][key]
        panel_data = []
        row_extent_points = []
        for index in (20, 40, 60, 80):
            absolute_origins = arrays[f"{key}__position"][index, joint_indices]
            center = np.mean(absolute_origins, axis=0)
            local_origins = absolute_origins - center
            vectors = {
                split: arrays[f"{key}__force_{split}"][index, joint_indices]
                for split in _SPLIT_COLORS
            }
            tangent = arrays[f"{key}__primary_path_tangent"][index]
            points = [*local_origins, 0.22 * tangent]
            for local_origin, joint_offset in zip(
                local_origins, range(len(joint_indices)), strict=True
            ):
                for split in _SPLIT_COLORS:
                    points.append(
                        local_origin + vectors[split][joint_offset] / (2.5 * row_scale)
                    )
            row_extent_points.extend(points)
            panel_data.append((local_origins, vectors, tangent))
        half_extent = max(0.25, 1.15 * float(np.max(np.abs(row_extent_points))))

        for col, (local_origins, vectors, tangent) in enumerate(panel_data):
            ax = axes[row, col]
            if len(local_origins) == 2:
                ax.plot(
                    local_origins[:, 0],
                    local_origins[:, 1],
                    color="#9CA3AF",
                    linewidth=1.0,
                )
            for joint_offset, origin in enumerate(local_origins):
                ax.scatter(*origin, color="black", s=10, zorder=5)
                for split in _SPLIT_COLORS:
                    vector = vectors[split][joint_offset]
                    ax.quiver(
                        *origin,
                        *vector,
                        angles="xy",
                        scale_units="xy",
                        scale=2.5 * row_scale,
                        color=_SPLIT_COLORS[split],
                        label=_split_label(split) if joint_offset == 0 else None,
                    )
            tangent_end = 0.22 * tangent
            ax.annotate(
                "",
                xy=tangent_end,
                xytext=(0.0, 0.0),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#6B7280",
                    "linestyle": "--",
                    "linewidth": 1.0,
                },
            )
            ax.plot(
                [],
                [],
                color="#6B7280",
                linestyle="--",
                label="Achieved Path Tangent",
            )
            ax.axhline(0.0, color="#E5E7EB", linewidth=0.6, zorder=0)
            ax.axvline(0.0, color="#E5E7EB", linewidth=0.6, zorder=0)
            ax.set_xlim(-half_extent, half_extent)
            ax.set_ylim(-half_extent, half_extent)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Normalized Time Quartile {col + 1}")
            if col == 0:
                ax.set_ylabel(_MODEL_LABELS[key])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle("Force Vectors Across the Model Ladder")
    fig.text(
        0.5,
        0.055,
        "Local Contact Coordinates; Force-Arrow Length Uses One Fixed Scale per Model Row",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(bottom=0.13, top=0.90)
    _save_figure(fig, output, FIGURE_STEMS[0])

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.5), sharex=True)
    for ax, key in zip(axes, _MODEL_ORDER, strict=True):
        time = arrays[f"{key}__time"]
        normalized = (time - time[0]) / (time[-1] - time[0])
        for split in _SPLIT_COLORS:
            ax.plot(
                normalized,
                arrays[f"{key}__primary_force_along_{split}"],
                color=_SPLIT_COLORS[split],
                label=_split_label(split),
                linestyle="--" if split == "zvcf" else "-",
            )
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_ylabel("Force (N)")
        ax.set_title(
            "Two-Arm Net Mid-Grip Resultant"
            if key == "two_arm"
            else _MODEL_LABELS[key] + " Net Club Force"
        )
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Normalized Trajectory Time")
    axes[0].legend(ncol=3)
    fig.suptitle("Signed Force Along the Joint Path")
    fig.tight_layout()
    _save_figure(fig, output, FIGURE_STEMS[1])

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 8.0))
    x = np.arange(4)
    for row, key in enumerate(_MODEL_ORDER):
        impulse_values = [
            arrays[f"{key}__primary_vector_impulse_{split}"][-1, 0]
            for split in _SPLIT_COLORS
        ]
        work_values = [
            record["models"][key]["primary_estimand"]["force_work_j"][split]
            for split in _SPLIT_COLORS
        ]
        for column, (values, title, ylabel) in enumerate(
            (
                (impulse_values, "Target-Direction Vector Impulse", "Impulse (N s)"),
                (work_values, "Force Work", "Work (J)"),
            )
        ):
            ax = axes[row, column]
            bars = ax.bar(x, values, color=list(_SPLIT_COLORS.values()))
            ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)
            ax.set_xticks(x, [_split_label(split) for split in _SPLIT_COLORS])
            ax.set_title(f"{_MODEL_LABELS[key]}: {title}")
            ax.set_ylabel(ylabel)
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Signed Impulse and Force Work Attribution")
    fig.tight_layout()
    _save_figure(fig, output, FIGURE_STEMS[2])

    labels, values, cancellation_flags, cancellation_values = [], [], [], []
    for key in _MODEL_ORDER:
        model = record["models"][key]
        for phase in model["phase_summaries"]:
            shares = phase["total_work_shares"]
            for joint_index, joint in enumerate(model["joint_names"]):
                labels.append(f"{_MODEL_LABELS[key]} | {phase['phase_name']} | {joint}")
                values.append(shares["magnitude_drift_share"][joint_index])
                cancellation_flags.append(phase["cancellation_flag"][joint_index])
                cancellation_values.append(shares["cancellation_index"][joint_index])
    fig, ax = plt.subplots(figsize=(9.2, 9.8))
    y = np.arange(len(labels))
    ax.barh(y, np.nan_to_num(values), color="#0072B2")
    cancellation_indices = np.flatnonzero(cancellation_flags)
    ax.scatter(
        np.asarray(cancellation_values)[cancellation_indices],
        cancellation_indices,
        marker="x",
        color="#D55E00",
        s=16,
    )
    ax.set_yticks(y, labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(
        "Drift Magnitude Share; Cross Position Is the Cancellation Index When At Least 0.25"
    )
    ax.set_title("Phase-Resolved Drift Shares and Cancellation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, output, FIGURE_STEMS[3])

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.8), sharex=True, sharey=True)
    for ax, key in zip(axes, _MODEL_ORDER, strict=True):
        time = arrays[f"{key}__time"]
        normalized = (time - time[0]) / (time[-1] - time[0])
        shares = arrays[f"{key}__drift_power_magnitude_share"]
        flags = arrays[f"{key}__power_cancellation_flag"]
        for joint_index, joint_name in enumerate(record["models"][key]["joint_names"]):
            line = ax.plot(normalized, shares[:, joint_index], label=joint_name)[0]
            flagged = flags[:, joint_index]
            ax.scatter(
                normalized[flagged],
                shares[flagged, joint_index],
                marker="x",
                color=line.get_color(),
                s=13,
            )
        ax.set_title(_MODEL_LABELS[key])
        ax.set_ylabel("Drift Power Magnitude Share")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.legend(ncol=min(3, len(record["models"][key]["joint_names"])), fontsize=7)
    axes[-1].set_xlabel("Normalized Trajectory Time")
    fig.suptitle("Instantaneous Total-Wrench Power Shares and Cancellation")
    fig.text(
        0.5,
        0.01,
        "Crosses Mark Samples With Cancellation Index At Least 0.25; Near-Zero Denominators Are Masked",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    _save_figure(fig, output, FIGURE_STEMS[4])

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    time = arrays["two_arm__time"]
    normalized = (time - time[0]) / (time[-1] - time[0])
    for ax, mode in zip(axes, ("common", "differential"), strict=True):
        for split in _SPLIT_COLORS:
            magnitude = np.linalg.norm(arrays[f"two_arm__{mode}_{split}"], axis=1)
            ax.plot(
                normalized,
                magnitude,
                color=_SPLIT_COLORS[split],
                label=_split_label(split),
            )
        ax.set_title(mode.title() + " Force Mode")
        ax.set_ylabel("Magnitude (N)")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Normalized Trajectory Time")
    axes[0].legend(ncol=4)
    fig.suptitle("Two-Hand Common and Differential Force Modes")
    fig.tight_layout()
    _save_figure(fig, output, FIGURE_STEMS[5])

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.2))
    for key in _MODEL_ORDER:
        rows = record["models"][key]["sensitivity"]["rows"]
        axes[0].plot(
            [row["sample_count"] for row in rows],
            [row["force_work_abs_delta_j"] for row in rows],
            marker="o",
            label=_MODEL_LABELS[key],
        )
    closure_names = (
        "force_max_abs",
        "couple_max_abs",
        "power_max_abs",
        "work_max_abs",
        "phase_additivity_max_abs",
    )
    closure_values = [
        max(record["models"][key]["closure"][name] for key in _MODEL_ORDER)
        for name in closure_names
    ]
    axes[1].bar(
        np.arange(len(closure_names)),
        np.maximum(closure_values, 1e-18),
        color="#6B7280",
    )
    axes[1].set_xticks(
        np.arange(len(closure_names)),
        ("Force", "Couple", "Power", "Work", "Phase"),
        rotation=20,
    )
    axes[1].set_yscale("log")
    axes[0].set_title("Quadrature Sampling Sensitivity")
    axes[0].set_xlabel("Sample Count")
    axes[0].set_ylabel("Force-Work Absolute Delta (J)")
    axes[0].legend()
    axes[1].set_title("Maximum Closure Residual")
    axes[1].set_ylabel("Absolute Residual")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Sensitivity and Closure Diagnostics")
    fig.tight_layout()
    _save_figure(fig, output, FIGURE_STEMS[6])


def write_study(output_root: Path | str) -> dict[str, Path]:
    """Write JSON, compressed arrays, and paired SVG/PDF figures."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    record, arrays = build_study()
    json_path = output / "hand_path_attribution_study.json"
    npz_path = output / "hand_path_attribution_traces.npz"
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(npz_path, **arrays)
    _make_figures(record, arrays, output / "figures")
    return {"json": json_path, "npz": npz_path, "figures": output / "figures"}


def main() -> None:
    target = (
        _source_root()
        / "docs/research/proximal_distal_energy_transfer/data/hand_path_attribution"
    )
    write_study(target)


if __name__ == "__main__":
    main()


__all__ = ["FIGURE_STEMS", "build_study", "write_study"]
