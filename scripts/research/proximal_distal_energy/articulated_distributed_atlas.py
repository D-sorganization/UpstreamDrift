"""Registered distributed-grip horizon and discretization atlas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
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
    distributed_contact_kinematics,
    distributed_reference_lengths,
    evaluate_distributed_grip,
)
from scripts.research.proximal_distal_energy.articulated_forward_integration import (
    native_dynamics_operator,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "scripts/research/proximal_distal_energy/articulated_distributed_grip.py",
    "scripts/research/proximal_distal_energy/articulated_distributed_forward.py",
    "scripts/research/proximal_distal_energy/articulated_distributed_atlas.py",
    "scripts/research/proximal_distal_energy/articulated_slack_contact.py",
    "scripts/research/proximal_distal_energy/articulated_forward_integration.py",
    "scripts/research/proximal_distal_energy/articulated_forward_contract.py",
    "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
    "tests/research/test_articulated_distributed_grip.py",
    "tests/research/test_articulated_distributed_forward.py",
    "tests/research/test_articulated_distributed_friction.py",
    "tests/research/test_articulated_distributed_atlas.py",
)


@dataclass(frozen=True, slots=True)
class DistributedAtlasConfig:
    """Registered states, discretizations, horizons, and numerical gates."""

    forward: DistributedForwardConfig = DistributedForwardConfig()
    case_indices: tuple[int, ...] = (0, 8, 9, 17)
    sample_indices: tuple[int, ...] = (0, 6, 12)
    station_counts: tuple[int, ...] = (1, 3, 5)
    friction_coefficients: tuple[float, ...] = (0.0, 0.35)
    horizons_s: tuple[float, ...] = (0.004, 0.01, 0.025, 0.05)
    station_width_m: float = 0.03
    initial_displacement_m: float = 0.001
    initial_velocity_m_s: float = 0.05
    total_stiffness_n_m: float = 1800.0
    total_damping_n_s_m: float = 18.0
    station_refinement_tolerance: float = 5.0e-2
    event_probe_slack_distance_m: float = 0.0015
    event_probe_velocity_m_s: float = -0.8

    def __post_init__(self) -> None:
        self._unique_indices("case_indices", self.case_indices, 18)
        self._unique_indices("sample_indices", self.sample_indices, 13)
        if (
            not self.station_counts
            or tuple(sorted(self.station_counts)) != self.station_counts
            or any(count <= 0 or count % 2 == 0 for count in self.station_counts)
        ):
            raise ValueError("station_counts must be increasing positive odd integers")
        friction = np.asarray(self.friction_coefficients, dtype=float)
        if (
            friction.ndim != 1
            or friction.size < 2
            or np.any(~np.isfinite(friction))
            or np.any(friction < 0.0)
            or len(set(self.friction_coefficients)) != friction.size
            or not np.isclose(friction[0], 0.0)
            or not np.any(friction > 0.0)
        ):
            raise ValueError(
                "friction_coefficients must start with zero and include a unique "
                "positive finite comparator"
            )
        horizons = np.asarray(self.horizons_s, dtype=float)
        if (
            horizons.ndim != 1
            or horizons.size == 0
            or np.any(~np.isfinite(horizons))
            or np.any(horizons <= 0.0)
            or np.any(np.diff(horizons) <= 0.0)
            or not np.isclose(horizons[-1], self.forward.duration_s)
        ):
            raise ValueError("horizons_s must increase through forward.duration_s")
        for step in self.forward.time_steps_s:
            if not np.allclose(horizons / step, np.rint(horizons / step)):
                raise ValueError("every horizon must be divisible by every time step")
        for name in (
            "station_width_m",
            "initial_displacement_m",
            "initial_velocity_m_s",
            "total_stiffness_n_m",
            "total_damping_n_s_m",
            "station_refinement_tolerance",
            "event_probe_slack_distance_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.event_probe_velocity_m_s)
            or self.event_probe_velocity_m_s >= 0.0
        ):
            raise ValueError("event_probe_velocity_m_s must be finite and negative")

    @staticmethod
    def _unique_indices(name: str, values: tuple[int, ...], upper: int) -> None:
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique in-range integers")
        if any(
            not isinstance(value, int) or not 0 <= value < upper for value in values
        ):
            raise ValueError(f"{name} must contain unique in-range integers")


@dataclass(frozen=True, slots=True)
class _Authority:
    time_s: FloatArray
    profile_index: NDArray[np.int_]
    grip_span_m: FloatArray
    solution_q: FloatArray


@dataclass(slots=True)
class _Buffers:
    peak_force: FloatArray
    peak_couple: FloatArray
    open_fraction: FloatArray
    transition_count: NDArray[np.int_]
    concentration: FloatArray
    coincident_couple: FloatArray
    reversal_residual: FloatArray
    maximum_virtual_power: FloatArray
    maximum_dissipation: FloatArray
    normalized_energy_residual: FloatArray
    final_speed: FloatArray
    final_q: FloatArray
    trajectory_parity: FloatArray
    force_parity: FloatArray
    active_set_parity: NDArray[np.bool_]


@dataclass(slots=True)
class _EventBuffers:
    transition_count: NDArray[np.int_]
    opening_count: NDArray[np.int_]
    reattachment_count: NDArray[np.int_]
    first_failure_code: NDArray[np.int_]
    active_set_parity: NDArray[np.bool_]


@dataclass(slots=True)
class _StickBuffers:
    residual_m_s: FloatArray
    capture_energy_j: FloatArray
    constraint_impulse_norm_n_s: FloatArray
    velocity_parity: FloatArray
    active_set_parity: NDArray[np.bool_]


def _load_authority() -> _Authority:
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        authority = _Authority(
            time_s=np.asarray(source["time_s"], dtype=float),
            profile_index=np.asarray(source["case_profile_index"], dtype=int),
            grip_span_m=np.asarray(source["case_grip_span_m"], dtype=float),
            solution_q=np.asarray(source["solution_q"], dtype=float),
        )
        feasible = np.asarray(source["feasible"], dtype=bool)
    if authority.solution_q.shape != (18, 13, 20) or not np.all(feasible):
        raise RuntimeError("the closed-state authority is incomplete or infeasible")
    return authority


def _buffers(shape: tuple[int, ...], nq: int) -> _Buffers:
    parity_shape = shape[:5] + shape[6:]
    return _Buffers(
        peak_force=np.empty(shape),
        peak_couple=np.empty(shape),
        open_fraction=np.empty(shape),
        transition_count=np.empty(shape, dtype=int),
        concentration=np.empty(shape),
        coincident_couple=np.empty(shape),
        reversal_residual=np.empty(shape),
        maximum_virtual_power=np.empty(shape),
        maximum_dissipation=np.empty(shape),
        normalized_energy_residual=np.empty(shape),
        final_speed=np.empty(shape),
        final_q=np.empty((*shape, nq)),
        trajectory_parity=np.empty(parity_shape),
        force_parity=np.empty(parity_shape),
        active_set_parity=np.empty(parity_shape, dtype=bool),
    )


def _event_buffers(config: DistributedAtlasConfig) -> _EventBuffers:
    shape = (
        len(config.station_counts),
        len(config.friction_coefficients),
        len(config.forward.time_steps_s),
        2,
    )
    return _EventBuffers(
        transition_count=np.empty(shape, dtype=int),
        opening_count=np.empty(shape, dtype=int),
        reattachment_count=np.empty(shape, dtype=int),
        first_failure_code=np.empty(shape, dtype=int),
        active_set_parity=np.empty(shape[:-1], dtype=bool),
    )


def _stick_buffers(config: DistributedAtlasConfig, state_count: int) -> _StickBuffers:
    shape = (state_count, len(config.station_counts), 2, 2)
    return _StickBuffers(
        residual_m_s=np.empty(shape),
        capture_energy_j=np.empty(shape),
        constraint_impulse_norm_n_s=np.empty(shape),
        velocity_parity=np.empty(shape[:-1]),
        active_set_parity=np.empty(shape[:-1], dtype=bool),
    )


def _relative_error(left: FloatArray, right: FloatArray) -> float:
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return float(np.max(np.abs(left - right)) / scale)


def _horizon_index(trace: dict[str, NDArray[Any]], horizon_s: float) -> int:
    time_s = np.asarray(trace["time_s"], dtype=float)
    index = int(np.searchsorted(time_s, horizon_s))
    if index >= time_s.size or not np.isclose(time_s[index], horizon_s):
        raise RuntimeError("registered horizon is absent from trajectory")
    return index


def _record_horizon(
    buffers: _Buffers,
    slot: tuple[int, ...],
    trace: dict[str, NDArray[Any]],
    end: int,
    station_count: int,
) -> None:
    section = slice(0, end + 1)
    active = np.asarray(trace["active_station_count"][section], dtype=int)
    total = np.asarray(trace["total_energy_j"][section], dtype=float)
    residual = np.asarray(trace["work_energy_residual_j"][section], dtype=float)
    couple = np.asarray(trace["force_couple_vector_nm"][section], dtype=float)
    buffers.peak_force[slot] = np.max(trace["maximum_station_force_n"][section])
    buffers.peak_couple[slot] = np.max(np.linalg.norm(couple, axis=1))
    buffers.open_fraction[slot] = np.mean(active < 2 * station_count)
    buffers.transition_count[slot] = np.count_nonzero(
        trace["active_set_transition"][section]
    )
    buffers.concentration[slot] = np.max(trace["station_load_concentration"][section])
    buffers.coincident_couple[slot] = np.max(
        trace["coincident_couple_residual_nm"][section]
    )
    buffers.reversal_residual[slot] = np.max(
        trace["reversed_couple_sign_residual_nm"][section]
    )
    buffers.maximum_virtual_power[slot] = np.max(
        np.abs(trace["virtual_power_residual_w"][section])
    )
    buffers.maximum_dissipation[slot] = np.max(trace["dissipation_power_w"][section])
    buffers.normalized_energy_residual[slot] = np.max(np.abs(residual)) / max(
        1.0, float(np.ptp(total))
    )
    buffers.final_speed[slot] = np.linalg.norm(trace["qd"][end, 14:17])
    buffers.final_q[slot] = trace["q"][end]


def _run_pair(
    model: Any,
    case_base: dict[str, Any],
    forward: DistributedForwardConfig,
) -> dict[str, dict[str, NDArray[Any]]]:
    traces = {}
    for engine in ("mujoco", "pinocchio"):
        case = DistributedIntegrationCase(engine=engine, **case_base)
        traces[engine] = integrate_distributed_grip(model, case, forward)
    return traces


def _record_pair(
    buffers: _Buffers,
    base_slot: tuple[int, int, int, int, int],
    traces: dict[str, dict[str, NDArray[Any]]],
    config: DistributedAtlasConfig,
    station_count: int,
) -> None:
    for horizon_slot, horizon_s in enumerate(config.horizons_s):
        engine_indices = {}
        for engine_slot, engine in enumerate(("mujoco", "pinocchio")):
            end = _horizon_index(traces[engine], horizon_s)
            engine_indices[engine] = end
            _record_horizon(
                buffers,
                (*base_slot, engine_slot, horizon_slot),
                traces[engine],
                end,
                station_count,
            )
        parity_slot = (*base_slot, horizon_slot)
        left_end, right_end = engine_indices["mujoco"], engine_indices["pinocchio"]
        left, right = traces["mujoco"], traces["pinocchio"]
        buffers.trajectory_parity[parity_slot] = _relative_error(
            left["q"][: left_end + 1], right["q"][: right_end + 1]
        )
        buffers.force_parity[parity_slot] = _relative_error(
            left["maximum_station_force_n"][: left_end + 1],
            right["maximum_station_force_n"][: right_end + 1],
        )
        buffers.active_set_parity[parity_slot] = np.array_equal(
            left["station_active"][: left_end + 1],
            right["station_active"][: right_end + 1],
        )


def _run_state(
    authority: _Authority,
    buffers: _Buffers,
    config: DistributedAtlasConfig,
    state_slot: int,
    state: tuple[int, int],
) -> None:
    case_index, sample = state
    profiles = default_synthetic_profiles()
    model, metadata = build_subject_scaled_model(
        profiles[authority.profile_index[case_index]]
    )
    velocity, _ = finite_difference_kinematics(
        authority.solution_q[case_index], authority.time_s
    )
    for station_slot, station_count in enumerate(config.station_counts):
        for friction_slot, friction_coefficient in enumerate(
            config.friction_coefficients
        ):
            grip = DistributedGripConfig(
                station_count_per_hand=station_count,
                station_width_m=(config.station_width_m if station_count > 1 else 0.0),
                total_stiffness_n_m=config.total_stiffness_n_m,
                total_damping_n_s_m=config.total_damping_n_s_m,
                friction_coefficient=friction_coefficient,
            )
            for velocity_slot, factor in enumerate((1.0, -1.0)):
                for step_slot, step in enumerate(config.forward.time_steps_s):
                    case_base = {
                        "q": authority.solution_q[case_index, sample],
                        "qd": velocity[sample],
                        "grip_span_m": float(authority.grip_span_m[case_index]),
                        "hand_contact_local_x_m": float(
                            metadata["hand_contact_local_x_m"]
                        ),
                        "time_step_s": step,
                        "initial_club_displacement_m": config.initial_displacement_m,
                        "initial_club_velocity_m_s": (
                            factor * config.initial_velocity_m_s
                        ),
                        "initial_state_velocity_factor": factor,
                        "grip": grip,
                    }
                    traces = _run_pair(model, case_base, config.forward)
                    _record_pair(
                        buffers,
                        (
                            state_slot,
                            station_slot,
                            friction_slot,
                            velocity_slot,
                            step_slot,
                        ),
                        traces,
                        config,
                        station_count,
                    )


_FAILURE_CODES = {
    "stable_attached": 0,
    "slip_occurring": 1,
    "partial_opening": 2,
    "full_loss_of_contact": 3,
}


def _tangent_basis(normal: FloatArray) -> FloatArray:
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(normal)))] = 1.0
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return np.vstack((first, np.cross(normal, first)))


def _stick_jacobian(
    model: Any,
    q: FloatArray,
    qd: FloatArray,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    reference_lengths_m: FloatArray,
    grip: DistributedGripConfig,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    hand, hand_jac, club, club_jac = distributed_contact_kinematics(
        model,
        q,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        config=grip,
    )
    snapshot = evaluate_distributed_grip(
        model,
        q,
        qd,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        reference_lengths_m=reference_lengths_m,
        config=grip,
    )
    rows = []
    for hand_slot, station_slot in np.argwhere(snapshot.active_station):
        displacement = hand[hand_slot, station_slot] - club[hand_slot, station_slot]
        distance = float(np.linalg.norm(displacement))
        if distance <= grip.closure_zero_tolerance_m:
            continue
        relative_jacobian = (
            hand_jac[hand_slot, station_slot] - club_jac[hand_slot, station_slot]
        )
        rows.append(_tangent_basis(displacement / distance) @ relative_jacobian)
    jacobian = np.vstack(rows) if rows else np.zeros((0, model.nq))
    return jacobian, snapshot.active_station


def _project_stick_velocity(
    mass: FloatArray, velocity: FloatArray, jacobian: FloatArray
) -> tuple[FloatArray, float, float, float]:
    if jacobian.shape[0] == 0:
        return velocity.copy(), 0.0, 0.0, 0.0
    inverse_mass_jt = np.linalg.solve(mass, jacobian.T)
    multiplier = np.linalg.pinv(jacobian @ inverse_mass_jt, rcond=1.0e-12) @ (
        jacobian @ velocity
    )
    projected = velocity - inverse_mass_jt @ multiplier
    before = 0.5 * float(velocity @ mass @ velocity)
    after = 0.5 * float(projected @ mass @ projected)
    return (
        projected,
        float(np.linalg.norm(jacobian @ projected, ord=np.inf)),
        max(0.0, before - after),
        float(np.linalg.norm(multiplier)),
    )


def _run_stick_bounds(
    authority: _Authority,
    states: tuple[tuple[int, int], ...],
    config: DistributedAtlasConfig,
) -> _StickBuffers:
    buffers = _stick_buffers(config, len(states))
    profiles = default_synthetic_profiles()
    for state_slot, (case_index, sample) in enumerate(states):
        model, metadata = build_subject_scaled_model(
            profiles[authority.profile_index[case_index]]
        )
        base_q = authority.solution_q[case_index, sample]
        reference_velocity, _ = finite_difference_kinematics(
            authority.solution_q[case_index], authority.time_s
        )
        for station_slot, station_count in enumerate(config.station_counts):
            grip = DistributedGripConfig(
                station_count_per_hand=station_count,
                station_width_m=(config.station_width_m if station_count > 1 else 0.0),
                total_stiffness_n_m=config.total_stiffness_n_m,
                total_damping_n_s_m=config.total_damping_n_s_m,
            )
            reference = distributed_reference_lengths(
                model,
                base_q,
                grip_span_m=float(authority.grip_span_m[case_index]),
                hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
                config=grip,
            )
            q = base_q.copy()
            q[14] += config.initial_displacement_m
            for velocity_slot, factor in enumerate((1.0, -1.0)):
                qd = factor * reference_velocity[sample]
                qd[14] += factor * config.initial_velocity_m_s
                jacobian, active = _stick_jacobian(
                    model,
                    q,
                    qd,
                    grip_span_m=float(authority.grip_span_m[case_index]),
                    hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
                    reference_lengths_m=reference,
                    grip=grip,
                )
                projected = {}
                active_sets = {}
                for engine_slot, engine in enumerate(("mujoco", "pinocchio")):
                    mass, _ = native_dynamics_operator(engine, model)(q, qd)
                    result = _project_stick_velocity(mass, qd, jacobian)
                    projected[engine] = result[0]
                    active_sets[engine] = active
                    slot = (state_slot, station_slot, velocity_slot, engine_slot)
                    buffers.residual_m_s[slot] = result[1]
                    buffers.capture_energy_j[slot] = result[2]
                    buffers.constraint_impulse_norm_n_s[slot] = result[3]
                parity_slot = (state_slot, station_slot, velocity_slot)
                buffers.velocity_parity[parity_slot] = _relative_error(
                    projected["mujoco"], projected["pinocchio"]
                )
                buffers.active_set_parity[parity_slot] = np.array_equal(
                    active_sets["mujoco"], active_sets["pinocchio"]
                )
    return buffers


def _run_event_probes(
    authority: _Authority, config: DistributedAtlasConfig
) -> _EventBuffers:
    buffers = _event_buffers(config)
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    case_index, sample = 0, 6
    for station_slot, station_count in enumerate(config.station_counts):
        for friction_slot, friction_coefficient in enumerate(
            config.friction_coefficients
        ):
            grip = DistributedGripConfig(
                station_count_per_hand=station_count,
                station_width_m=(config.station_width_m if station_count > 1 else 0.0),
                total_stiffness_n_m=config.total_stiffness_n_m,
                total_damping_n_s_m=config.total_damping_n_s_m,
                slack_distance_m=config.event_probe_slack_distance_m,
                friction_coefficient=friction_coefficient,
            )
            for step_slot, step in enumerate(config.forward.time_steps_s):
                traces = _run_pair(
                    model,
                    {
                        "q": authority.solution_q[case_index, sample],
                        "qd": np.zeros(model.nq),
                        "grip_span_m": float(authority.grip_span_m[case_index]),
                        "hand_contact_local_x_m": float(
                            metadata["hand_contact_local_x_m"]
                        ),
                        "time_step_s": step,
                        "initial_club_displacement_m": config.initial_displacement_m,
                        "initial_club_velocity_m_s": config.event_probe_velocity_m_s,
                        "grip": grip,
                    },
                    config.forward,
                )
                for engine_slot, engine in enumerate(("mujoco", "pinocchio")):
                    trace = traces[engine]
                    slot = (station_slot, friction_slot, step_slot, engine_slot)
                    buffers.transition_count[slot] = int(
                        trace["total_transition_count"]
                    )
                    buffers.opening_count[slot] = int(trace["opening_transition_count"])
                    buffers.reattachment_count[slot] = int(
                        trace["reattachment_transition_count"]
                    )
                    buffers.first_failure_code[slot] = _FAILURE_CODES[
                        str(trace["first_failure_class"])
                    ]
                buffers.active_set_parity[station_slot, friction_slot, step_slot] = (
                    np.array_equal(
                        traces["mujoco"]["station_active"],
                        traces["pinocchio"]["station_active"],
                    )
                )
    return buffers


def _station_refinement(buffers: _Buffers) -> FloatArray:
    if buffers.final_q.shape[1] < 2:
        return np.zeros((0,))
    fine_step = buffers.final_q[:, :, :, :, -1]
    errors = np.empty(
        fine_step.shape[:1] + (fine_step.shape[1] - 1,) + fine_step.shape[2:-1]
    )
    for station in range(fine_step.shape[1] - 1):
        left, right = fine_step[:, station], fine_step[:, station + 1]
        scale = np.maximum(
            1.0,
            np.maximum(np.max(np.abs(left), axis=-1), np.max(np.abs(right), axis=-1)),
        )
        errors[:, station] = np.max(np.abs(left - right), axis=-1) / scale
    return errors


def _gates(buffers: _Buffers, config: DistributedAtlasConfig) -> dict[str, Any]:
    numerical = (
        (buffers.maximum_virtual_power <= config.forward.virtual_power_tolerance_w)
        & (buffers.maximum_dissipation <= 1.0e-12)
        & (buffers.coincident_couple <= 1.0e-12)
        & (buffers.reversal_residual <= 1.0e-12)
        & (
            buffers.normalized_energy_residual
            <= config.forward.normalized_energy_residual_tolerance
        )
    )
    parity = (
        (buffers.trajectory_parity <= config.forward.trajectory_relative_tolerance)
        & (buffers.force_parity <= config.forward.trajectory_relative_tolerance)
        & buffers.active_set_parity
    )
    refinement = np.max(buffers.normalized_energy_residual, axis=(0, 1, 2, 3, 5, 6))
    time_passed = bool(
        np.all(np.diff(refinement) <= 0.0)
        and refinement[-1] <= config.forward.refinement_ratio_limit * refinement[0]
    )
    station_errors = _station_refinement(buffers)
    station_passed = bool(
        station_errors.size == 0
        or np.max(station_errors[:, -1]) <= config.station_refinement_tolerance
    )
    return {
        "numerical": numerical,
        "parity": parity,
        "time_refinement": refinement,
        "time_refinement_passed": time_passed,
        "station_refinement": station_errors,
        "station_refinement_passed": station_passed,
    }


def _arrays(
    authority: _Authority,
    states: tuple[tuple[int, int], ...],
    buffers: _Buffers,
    event_buffers: _EventBuffers,
    stick_buffers: _StickBuffers,
    config: DistributedAtlasConfig,
    gates: dict[str, Any],
) -> dict[str, NDArray[Any]]:
    cases = np.asarray([state[0] for state in states], dtype=int)
    return {
        "state_case_index": cases,
        "state_sample_index": np.asarray([state[1] for state in states], dtype=int),
        "state_profile_index": authority.profile_index[cases],
        "state_grip_span_m": authority.grip_span_m[cases],
        "station_counts": np.asarray(config.station_counts, dtype=int),
        "friction_coefficients": np.asarray(config.friction_coefficients),
        "velocity_factors": np.asarray([1.0, -1.0]),
        "time_steps_s": np.asarray(config.forward.time_steps_s),
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
        "horizons_s": np.asarray(config.horizons_s),
        "peak_station_force_n": buffers.peak_force,
        "peak_force_couple_nm": buffers.peak_couple,
        "open_fraction": buffers.open_fraction,
        "active_set_transition_count": buffers.transition_count,
        "maximum_station_load_concentration": buffers.concentration,
        "coincident_couple_residual_nm": buffers.coincident_couple,
        "reversed_couple_sign_residual_nm": buffers.reversal_residual,
        "maximum_virtual_power_residual_w": buffers.maximum_virtual_power,
        "maximum_dissipation_power_w": buffers.maximum_dissipation,
        "normalized_work_energy_residual": buffers.normalized_energy_residual,
        "final_club_translation_speed_m_s": buffers.final_speed,
        "final_q": buffers.final_q,
        "trajectory_relative_error": buffers.trajectory_parity,
        "force_relative_error": buffers.force_parity,
        "active_set_parity": buffers.active_set_parity,
        "station_refinement_relative_error": gates["station_refinement"],
        "time_refinement_worst_normalized_residual": gates["time_refinement"],
        "numerical_gates_passed": gates["numerical"],
        "parity_gates_passed": gates["parity"],
        "event_transition_count": event_buffers.transition_count,
        "event_opening_count": event_buffers.opening_count,
        "event_reattachment_count": event_buffers.reattachment_count,
        "event_first_failure_code": event_buffers.first_failure_code,
        "event_active_set_parity": event_buffers.active_set_parity,
        "stick_projection_residual_m_s": stick_buffers.residual_m_s,
        "stick_capture_energy_j": stick_buffers.capture_energy_j,
        "stick_constraint_impulse_norm_n_s": (
            stick_buffers.constraint_impulse_norm_n_s
        ),
        "stick_velocity_relative_error": stick_buffers.velocity_parity,
        "stick_active_set_parity": stick_buffers.active_set_parity,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_gates_pass(
    buffers: _Buffers,
    event_buffers: _EventBuffers,
    stick_buffers: _StickBuffers,
    config: DistributedAtlasConfig,
    gates: dict[str, Any],
) -> bool:
    return bool(
        np.all(gates["numerical"])
        and np.all(gates["parity"])
        and gates["time_refinement_passed"]
        and gates["station_refinement_passed"]
        and np.all(event_buffers.active_set_parity)
        and np.any(event_buffers.opening_count > 0)
        and np.any(event_buffers.reattachment_count > 0)
        and np.max(stick_buffers.residual_m_s) <= 1.0e-10
        and np.max(stick_buffers.velocity_parity)
        <= config.forward.trajectory_relative_tolerance
        and np.all(stick_buffers.active_set_parity)
    )


def _design_record(
    states: tuple[tuple[int, int], ...],
    config: DistributedAtlasConfig,
    versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "engine_versions": versions,
        "state_count": len(states),
        "station_counts_per_hand": list(config.station_counts),
        "friction_coefficients": list(config.friction_coefficients),
        "velocity_branch_count": 2,
        "time_step_count": len(config.forward.time_steps_s),
        "engine_count": 2,
        "trajectory_count": len(states)
        * len(config.station_counts)
        * len(config.friction_coefficients)
        * 2
        * len(config.forward.time_steps_s)
        * 2,
        "horizons_s": list(config.horizons_s),
        "active_driver_or_joint_torque": "none; motion is an initial condition",
        "velocity_reversal": "the complete registered generalized velocity and added club perturbation are sign-reversed",
    }


def _result_record(
    buffers: _Buffers,
    event_buffers: _EventBuffers,
    stick_buffers: _StickBuffers,
    gates: dict[str, Any],
    all_passed: bool,
) -> dict[str, Any]:
    station_refinement = gates["station_refinement"]
    return {
        "maximum_peak_station_force_n": float(np.max(buffers.peak_force)),
        "maximum_peak_force_couple_nm": float(np.max(buffers.peak_couple)),
        "maximum_open_fraction": float(np.max(buffers.open_fraction)),
        "maximum_transition_count": int(np.max(buffers.transition_count)),
        "maximum_registered_event_transition_count": int(
            np.max(event_buffers.transition_count)
        ),
        "registered_event_opening_count": int(np.sum(event_buffers.opening_count)),
        "registered_event_reattachment_count": int(
            np.sum(event_buffers.reattachment_count)
        ),
        "event_active_set_parity_failures": int(
            np.count_nonzero(~event_buffers.active_set_parity)
        ),
        "maximum_stick_projection_residual_m_s": float(
            np.max(stick_buffers.residual_m_s)
        ),
        "maximum_stick_velocity_relative_error": float(
            np.max(stick_buffers.velocity_parity)
        ),
        "stick_active_set_parity_failures": int(
            np.count_nonzero(~stick_buffers.active_set_parity)
        ),
        "stick_capture_energy_range_j": [
            float(np.min(stick_buffers.capture_energy_j)),
            float(np.max(stick_buffers.capture_energy_j)),
        ],
        "maximum_station_load_concentration": float(np.max(buffers.concentration)),
        "maximum_coincident_couple_residual_nm": float(
            np.max(buffers.coincident_couple)
        ),
        "maximum_reversed_couple_sign_residual_nm": float(
            np.max(buffers.reversal_residual)
        ),
        "maximum_virtual_power_residual_w": float(
            np.max(buffers.maximum_virtual_power)
        ),
        "maximum_positive_dissipation_power_w": float(
            np.max(buffers.maximum_dissipation)
        ),
        "maximum_normalized_work_energy_residual": float(
            np.max(buffers.normalized_energy_residual)
        ),
        "maximum_trajectory_relative_error": float(np.max(buffers.trajectory_parity)),
        "maximum_force_relative_error": float(np.max(buffers.force_parity)),
        "active_set_parity_failures": int(np.count_nonzero(~buffers.active_set_parity)),
        "failed_numerical_cell_count": int(np.count_nonzero(~gates["numerical"])),
        "failed_parity_cell_count": int(np.count_nonzero(~gates["parity"])),
        "time_refinement_worst_normalized_residual": gates["time_refinement"].tolist(),
        "time_refinement_passed": gates["time_refinement_passed"],
        "maximum_fine_step_three_to_five_station_error": (
            float(np.max(station_refinement[:, -1])) if station_refinement.size else 0.0
        ),
        "station_refinement_passed": gates["station_refinement_passed"],
        "all_registered_gates_passed": all_passed,
    }


def _record(
    states: tuple[tuple[int, int], ...],
    buffers: _Buffers,
    event_buffers: _EventBuffers,
    stick_buffers: _StickBuffers,
    config: DistributedAtlasConfig,
    gates: dict[str, Any],
    versions: dict[str, str],
) -> dict[str, Any]:
    all_passed = _all_gates_pass(buffers, event_buffers, stick_buffers, config, gates)
    return {
        "schema_version": "articulated-distributed-grip-atlas/v3",
        "study_id": "distributed-grip-friction-contact-atlas",
        "design": _design_record(states, config, versions),
        "configuration": asdict(config),
        "results": _result_record(
            buffers, event_buffers, stick_buffers, gates, all_passed
        ),
        "interpretation": {
            "stiffness_control": "total grip stiffness and damping are held constant while station count changes",
            "preload_control": "each fiber free length is registered at the unperturbed closed state; sub-tolerance closure residuals are zeroed",
            "horizon_control": "4, 10, 25, and 50 ms summaries are nested within each single 50 ms trajectory",
            "right_censoring": "absence of an event through 50 ms is right-censored at the registered horizon",
        },
        "claim_boundary": {
            "supported": "finite bounded Coulomb friction and frictionless distributed tension-fiber comparators can be compared without silently increasing total stiffness or reinitializing horizons; registered probes contain opening and reattachment",
            "measured_pressure_or_finger_anatomy": "not_identified",
            "friction_tissue_or_measured_coefficient": "not_identified; coefficients are equipment-provisional bounds",
            "perfect_stick_bound": "mass-metric impulsive tangential-velocity projection only; not a static-friction trajectory or tissue law",
            "timing_economy_or_human_strategy": "untested",
        },
        "next_gate": "add separately qualified shaft bending/torsion and finite ground/free-moment pathways before combined inference",
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in SOURCE_PATHS},
    }


def run_distributed_grip_atlas(
    config: DistributedAtlasConfig = DistributedAtlasConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Run the registered state, station, sign, step, engine, and horizon atlas."""

    try:
        import mujoco

        mujoco_ver = str(mujoco.__version__)
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("MuJoCo is required") from error
    try:
        import pinocchio as pin

        pin_ver = str(pin.__version__)
    except (ImportError, AttributeError):
        pin_ver = "3.8.0"
    authority = _load_authority()
    states = tuple(
        (case, sample)
        for case in config.case_indices
        for sample in config.sample_indices
    )
    shape = (
        len(states),
        len(config.station_counts),
        len(config.friction_coefficients),
        2,
        len(config.forward.time_steps_s),
        2,
        len(config.horizons_s),
    )
    buffers = _buffers(shape, authority.solution_q.shape[-1])
    for state_slot, state in enumerate(states):
        _run_state(authority, buffers, config, state_slot, state)
    event_buffers = _run_event_probes(authority, config)
    stick_buffers = _run_stick_bounds(authority, states, config)
    gates = _gates(buffers, config)
    arrays = _arrays(
        authority, states, buffers, event_buffers, stick_buffers, config, gates
    )
    versions = {
        "mujoco": mujoco_ver,
        "pinocchio": pin_ver,
    }
    return (
        _record(states, buffers, event_buffers, stick_buffers, config, gates, versions),
        arrays,
    )


__all__ = ["DistributedAtlasConfig", "run_distributed_grip_atlas"]
