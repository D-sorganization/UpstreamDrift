"""Registered finite-ground pathway and falsification-control atlas."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
import hashlib
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.articulated_ground import (
    ArticulatedGroundConfig,
)
from scripts.research.proximal_distal_energy.articulated_ground_forward import (
    GroundForwardConfig,
    GroundIntegrationCase,
    integrate_articulated_ground,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
GROUND_ACTIVATIONS = ("fixed", "translation", "free_moment", "coupled")
CONTROL_NAMES = ("rigid_shaft", "horizontal_restraint_removed")
ENGINES = ("mujoco", "pinocchio")
VELOCITY_FACTORS = (1.0, -1.0)
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "scripts/research/proximal_distal_energy/articulated_distributed_grip.py",
    "scripts/research/proximal_distal_energy/articulated_ground.py",
    "scripts/research/proximal_distal_energy/articulated_ground_forward.py",
    "scripts/research/proximal_distal_energy/articulated_ground_atlas.py",
    "scripts/research/proximal_distal_energy/articulated_shaft.py",
    "scripts/research/proximal_distal_energy/articulated_shaft_forward.py",
    "tests/research/test_articulated_ground.py",
    "tests/research/test_articulated_ground_forward.py",
    "tests/research/test_articulated_ground_atlas.py",
)


@dataclass(frozen=True, slots=True)
class ArticulatedGroundAtlasConfig:
    """Registered states, comparisons, horizons, and numerical controls."""

    forward: GroundForwardConfig = GroundForwardConfig()
    case_indices: tuple[int, ...] = (0, 8, 9, 17)
    sample_indices: tuple[int, ...] = (0, 6, 12)
    ground_activations: tuple[str, ...] = GROUND_ACTIVATIONS
    control_names: tuple[str, ...] = CONTROL_NAMES
    horizons_s: tuple[float, ...] = (0.004, 0.01, 0.025, 0.05)
    initial_club_displacement_m: float = 0.001
    initial_club_velocity_m_s: float = 0.05
    station_count_per_hand: int = 5
    station_width_m: float = 0.03
    total_stiffness_n_m: float = 1800.0
    total_damping_n_s_m: float = 18.0
    match_relative_tolerance: float = 0.05
    parity_relative_tolerance: float = 1.0e-8
    power_residual_tolerance_w: float = 1.0e-8
    worker_count: int = 4

    def __post_init__(self) -> None:
        self._indices("case_indices", self.case_indices, 18)
        self._indices("sample_indices", self.sample_indices, 13)
        if self.ground_activations != GROUND_ACTIVATIONS:
            raise ValueError(
                f"ground_activations must be exactly {GROUND_ACTIVATIONS!r}"
            )
        if self.control_names != CONTROL_NAMES:
            raise ValueError(f"control_names must be exactly {CONTROL_NAMES!r}")
        horizons = np.asarray(self.horizons_s, dtype=float)
        if (
            np.any(~np.isfinite(horizons))
            or np.any(horizons <= 0.0)
            or np.any(np.diff(horizons) <= 0.0)
            or not np.isclose(horizons[-1], self.forward.duration_s)
        ):
            raise ValueError("horizons_s must increase through forward.duration_s")
        for step in self.forward.time_steps_s:
            if not np.allclose(horizons / step, np.rint(horizons / step)):
                raise ValueError("every horizon must be divisible by every time step")
        if self.station_count_per_hand < 1 or self.station_count_per_hand % 2 == 0:
            raise ValueError("station_count_per_hand must be a positive odd integer")
        if not isinstance(self.worker_count, int) or not 1 <= self.worker_count <= 12:
            raise ValueError("worker_count must be an integer from one through twelve")
        for name in (
            "initial_club_displacement_m",
            "initial_club_velocity_m_s",
            "station_width_m",
            "total_stiffness_n_m",
            "total_damping_n_s_m",
            "match_relative_tolerance",
            "parity_relative_tolerance",
            "power_residual_tolerance_w",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @staticmethod
    def _indices(name: str, values: tuple[int, ...], upper: int) -> None:
        if (
            not values
            or len(set(values)) != len(values)
            or any(
                not isinstance(value, int) or not 0 <= value < upper for value in values
            )
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
    peak_grip_force: FloatArray
    peak_grip_couple: FloatArray
    open_fraction: FloatArray
    transition_count: NDArray[np.int_]
    peak_ground_force: FloatArray
    peak_intrinsic_moment: FloatArray
    peak_transported_moment: FloatArray
    peak_ground_energy: FloatArray
    terminal_ground_work: FloatArray
    terminal_total_work: FloatArray
    normalized_energy_residual: FloatArray
    maximum_virtual_power: FloatArray
    maximum_shaft_power_residual: FloatArray
    maximum_ground_power_residual: FloatArray
    maximum_positive_dissipation: FloatArray
    maximum_bending: FloatArray
    maximum_twist: FloatArray
    maximum_base_translation: FloatArray
    maximum_base_pitch: FloatArray
    final_speed: FloatArray
    initial_energy: FloatArray
    final_q: FloatArray
    final_base_full: FloatArray
    trajectory_parity: FloatArray
    force_parity: FloatArray
    ground_force_parity: FloatArray
    active_set_parity: NDArray[np.bool_]


def _load_authority() -> _Authority:
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
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
    parity_shape = shape[:4] + shape[5:]
    return _Buffers(
        **{
            name: np.empty(shape)
            for name in (
                "peak_grip_force",
                "peak_grip_couple",
                "open_fraction",
                "peak_ground_force",
                "peak_intrinsic_moment",
                "peak_transported_moment",
                "peak_ground_energy",
                "terminal_ground_work",
                "terminal_total_work",
                "normalized_energy_residual",
                "maximum_virtual_power",
                "maximum_shaft_power_residual",
                "maximum_ground_power_residual",
                "maximum_positive_dissipation",
                "maximum_bending",
                "maximum_twist",
                "maximum_base_translation",
                "maximum_base_pitch",
                "final_speed",
            )
        },
        transition_count=np.empty(shape, dtype=int),
        initial_energy=np.empty(shape[:-1]),
        final_q=np.empty((*shape, nq)),
        final_base_full=np.empty((*shape, 3)),
        trajectory_parity=np.empty(parity_shape),
        force_parity=np.empty(parity_shape),
        ground_force_parity=np.empty(parity_shape),
        active_set_parity=np.empty(parity_shape, dtype=bool),
    )


def _relative_error(left: Any, right: Any) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("parity arrays must have identical shapes")
    if a.size == 0:
        return 0.0
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / scale)


def _horizon_index(trace: dict[str, NDArray[Any]], horizon_s: float) -> int:
    time_s = np.asarray(trace["time_s"], dtype=float)
    index = int(np.searchsorted(time_s, horizon_s))
    if index >= time_s.size or not np.isclose(time_s[index], horizon_s):
        raise RuntimeError("registered horizon is absent from ground trajectory")
    return index


def _record_horizon(
    buffers: _Buffers,
    slot: tuple[int, ...],
    trace: dict[str, NDArray[Any]],
    end: int,
) -> None:
    section = slice(0, end + 1)
    active = np.asarray(trace["active_station_count"][section], dtype=int)
    total = np.asarray(trace["total_energy_j"][section], dtype=float)
    residual = np.asarray(trace["work_energy_residual_j"][section], dtype=float)
    ground_force = np.asarray(trace["ground_force_n"][section], dtype=float)
    bending = np.asarray(trace["tip_bending_m"][section], dtype=float)
    base_translation = np.asarray(trace["base_translation_m"][section], dtype=float)
    dissipations = (
        np.asarray(trace["grip_dissipation_power_w"][section])
        + np.asarray(trace["shaft_damping_power_w"][section])
        + np.asarray(trace["ground_damping_power_w"][section])
    )
    buffers.peak_grip_force[slot] = np.max(trace["maximum_station_force_n"][section])
    buffers.peak_grip_couple[slot] = np.max(
        np.linalg.norm(trace["force_couple_vector_nm"][section], axis=1)
    )
    buffers.open_fraction[slot] = np.mean(active < 10)
    buffers.transition_count[slot] = np.count_nonzero(
        trace["active_set_transition"][section]
    )
    buffers.peak_ground_force[slot] = np.max(np.linalg.norm(ground_force, axis=1))
    buffers.peak_intrinsic_moment[slot] = np.max(
        np.abs(trace["ground_intrinsic_free_moment_nm"][section])
    )
    buffers.peak_transported_moment[slot] = np.max(
        np.abs(trace["ground_transported_moment_nm"][section])
    )
    buffers.peak_ground_energy[slot] = np.max(trace["ground_strain_energy_j"][section])
    buffers.terminal_ground_work[slot] = -float(
        np.trapz(trace["ground_damping_power_w"][section], trace["time_s"][section])
    )
    buffers.terminal_total_work[slot] = -float(trace["cumulative_dissipation_j"][end])
    buffers.normalized_energy_residual[slot] = np.max(np.abs(residual)) / max(
        1.0, float(np.ptp(total))
    )
    buffers.maximum_virtual_power[slot] = np.max(
        np.abs(trace["virtual_power_residual_w"][section])
    )
    buffers.maximum_shaft_power_residual[slot] = np.max(
        np.abs(trace["shaft_power_residual_w"][section])
    )
    buffers.maximum_ground_power_residual[slot] = np.max(
        np.abs(trace["ground_power_residual_w"][section])
    )
    buffers.maximum_positive_dissipation[slot] = np.max(dissipations)
    buffers.maximum_bending[slot] = np.max(np.linalg.norm(bending, axis=1))
    buffers.maximum_twist[slot] = np.max(np.abs(trace["twist_angle_rad"][section]))
    buffers.maximum_base_translation[slot] = np.max(
        np.linalg.norm(base_translation, axis=1)
    )
    buffers.maximum_base_pitch[slot] = np.max(np.abs(trace["base_pitch_rad"][section]))
    buffers.final_speed[slot] = np.linalg.norm(trace["qd"][end, 14:17])
    buffers.final_q[slot] = trace["q"][end]
    buffers.final_base_full[slot] = (
        base_translation[end, 0],
        base_translation[end, 1],
        trace["base_pitch_rad"][end],
    )


def _record_pair(
    buffers: _Buffers,
    base_slot: tuple[int, int, int, int],
    traces: dict[str, dict[str, NDArray[Any]]],
    config: ArticulatedGroundAtlasConfig,
) -> None:
    for engine_slot, engine in enumerate(ENGINES):
        buffers.initial_energy[(*base_slot, engine_slot)] = traces[engine][
            "total_energy_j"
        ][0]
    for horizon_slot, horizon_s in enumerate(config.horizons_s):
        ends: dict[str, int] = {}
        for engine_slot, engine in enumerate(ENGINES):
            end = _horizon_index(traces[engine], horizon_s)
            ends[engine] = end
            _record_horizon(
                buffers,
                (*base_slot, engine_slot, horizon_slot),
                traces[engine],
                end,
            )
        left, right = traces[ENGINES[0]], traces[ENGINES[1]]
        left_end, right_end = ends[ENGINES[0]], ends[ENGINES[1]]
        parity_slot = (*base_slot, horizon_slot)
        buffers.trajectory_parity[parity_slot] = max(
            _relative_error(left["q"][: left_end + 1], right["q"][: right_end + 1]),
            _relative_error(
                left["elastic_coordinates"][: left_end + 1],
                right["elastic_coordinates"][: right_end + 1],
            ),
            _relative_error(
                left["base_coordinates"][: left_end + 1],
                right["base_coordinates"][: right_end + 1],
            ),
        )
        buffers.force_parity[parity_slot] = _relative_error(
            left["maximum_station_force_n"][: left_end + 1],
            right["maximum_station_force_n"][: right_end + 1],
        )
        buffers.ground_force_parity[parity_slot] = _relative_error(
            left["ground_force_n"][: left_end + 1],
            right["ground_force_n"][: right_end + 1],
        )
        buffers.active_set_parity[parity_slot] = np.array_equal(
            left["active_station_count"][: left_end + 1],
            right["active_station_count"][: right_end + 1],
        )


def _ground_control(
    name: str,
) -> tuple[ArticulatedShaftConfig, ArticulatedGroundConfig]:
    if name == "rigid_shaft":
        return ArticulatedShaftConfig(activation="rigid"), ArticulatedGroundConfig()
    if name == "horizontal_restraint_removed":
        return ArticulatedShaftConfig(), ArticulatedGroundConfig(
            translation_stiffness_n_m=(0.0, 30_000.0),
            translation_damping_n_s_m=(0.0, 800.0),
        )
    raise ValueError(f"unknown ground control {name!r}")


def _run_pair(
    model: Any,
    base: dict[str, Any],
    shaft: ArticulatedShaftConfig,
    ground: ArticulatedGroundConfig,
    forward: GroundForwardConfig,
) -> dict[str, dict[str, NDArray[Any]]]:
    traces = {}
    for engine in ENGINES:
        try:
            traces[engine] = integrate_articulated_ground(
                model,
                GroundIntegrationCase(
                    engine=engine, shaft=shaft, ground=ground, **base
                ),
                forward,
            )
        except Exception as error:
            raise RuntimeError(
                f"native ground integration failed: engine={engine}, "
                f"ground={ground.activation}, shaft={shaft.activation}"
            ) from error
    return traces


def _run_state(
    authority: _Authority,
    primary: _Buffers,
    controls: _Buffers,
    config: ArticulatedGroundAtlasConfig,
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
    grip = DistributedGripConfig(
        station_count_per_hand=config.station_count_per_hand,
        station_width_m=config.station_width_m,
        total_stiffness_n_m=config.total_stiffness_n_m,
        total_damping_n_s_m=config.total_damping_n_s_m,
    )
    for pathway_slot, activation in enumerate(config.ground_activations):
        for velocity_slot, factor in enumerate(VELOCITY_FACTORS):
            for step_slot, step_s in enumerate(config.forward.time_steps_s):
                base = {
                    "q": authority.solution_q[case_index, sample],
                    "qd": velocity[sample],
                    "grip_span_m": float(authority.grip_span_m[case_index]),
                    "hand_contact_local_x_m": float(metadata["hand_contact_local_x_m"]),
                    "time_step_s": step_s,
                    "initial_club_displacement_m": config.initial_club_displacement_m,
                    "initial_club_velocity_m_s": factor
                    * config.initial_club_velocity_m_s,
                    "initial_base_displacement": (0.0, 0.0, 0.0),
                    "initial_base_velocity": (0.0, 0.0, 0.0),
                    "grip": grip,
                }
                traces = _run_pair(
                    model,
                    base,
                    ArticulatedShaftConfig(),
                    ArticulatedGroundConfig(activation=activation),  # type: ignore[arg-type]
                    config.forward,
                )
                _record_pair(
                    primary,
                    (0, pathway_slot, velocity_slot, step_slot),
                    traces,
                    config,
                )
    for control_slot, name in enumerate(config.control_names):
        shaft, ground = _ground_control(name)
        for velocity_slot, factor in enumerate(VELOCITY_FACTORS):
            for step_slot, step_s in enumerate(config.forward.time_steps_s):
                base = {
                    "q": authority.solution_q[case_index, sample],
                    "qd": velocity[sample],
                    "grip_span_m": float(authority.grip_span_m[case_index]),
                    "hand_contact_local_x_m": float(metadata["hand_contact_local_x_m"]),
                    "time_step_s": step_s,
                    "initial_club_displacement_m": config.initial_club_displacement_m,
                    "initial_club_velocity_m_s": factor
                    * config.initial_club_velocity_m_s,
                    "initial_base_displacement": (0.0, 0.0, 0.0),
                    "initial_base_velocity": (0.0, 0.0, 0.0),
                    "grip": grip,
                }
                traces = _run_pair(model, base, shaft, ground, config.forward)
                _record_pair(
                    controls,
                    (0, control_slot, velocity_slot, step_slot),
                    traces,
                    config,
                )


def _state_job(
    payload: tuple[_Authority, ArticulatedGroundAtlasConfig, int, tuple[int, int]],
) -> tuple[int, _Buffers, _Buffers]:
    authority, config, state_slot, state = payload
    tail = (2, len(config.forward.time_steps_s), 2, len(config.horizons_s))
    primary = _buffers((1, len(config.ground_activations), *tail), 20)
    controls = _buffers((1, len(config.control_names), *tail), 20)
    _run_state(authority, primary, controls, config, state)
    return state_slot, primary, controls


def _merge_state(target: _Buffers, state_slot: int, source: _Buffers) -> None:
    for field in fields(_Buffers):
        getattr(target, field.name)[state_slot] = getattr(source, field.name)[0]


def _pair_relative(left: FloatArray, right: FloatArray, floor: float) -> FloatArray:
    scale = np.maximum(floor, 0.5 * (np.abs(left) + np.abs(right)))
    return np.abs(left - right) / scale


def _gates(
    primary: _Buffers,
    controls: _Buffers,
    config: ArticulatedGroundAtlasConfig,
) -> dict[str, Any]:
    def numerical(buffer: _Buffers) -> NDArray[np.bool_]:
        return (
            (buffer.maximum_virtual_power <= config.power_residual_tolerance_w)
            & (buffer.maximum_shaft_power_residual <= config.power_residual_tolerance_w)
            & (
                buffer.maximum_ground_power_residual
                <= config.power_residual_tolerance_w
            )
            & (buffer.maximum_positive_dissipation <= 1.0e-12)
            & (
                buffer.normalized_energy_residual
                <= config.forward.normalized_energy_residual_tolerance
            )
        )

    def parity(buffer: _Buffers) -> NDArray[np.bool_]:
        return (
            (buffer.trajectory_parity <= config.parity_relative_tolerance)
            & (buffer.force_parity <= config.parity_relative_tolerance)
            & (buffer.ground_force_parity <= config.parity_relative_tolerance)
            & buffer.active_set_parity
        )

    refinement = np.max(primary.normalized_energy_residual, axis=(0, 1, 2, 4, 5))
    fixed = GROUND_ACTIVATIONS.index("fixed")
    coupled = GROUND_ACTIVATIONS.index("coupled")
    load_error = _pair_relative(
        primary.peak_grip_force[:, coupled], primary.peak_grip_force[:, fixed], 1.0
    )
    work_error = _pair_relative(
        primary.terminal_total_work[:, coupled],
        primary.terminal_total_work[:, fixed],
        1.0e-6,
    )
    matched = (load_error <= config.match_relative_tolerance) & (
        work_error <= config.match_relative_tolerance
    )
    energy_scale = np.maximum(1.0, np.max(np.abs(primary.initial_energy), axis=1))
    initial_range = np.ptp(primary.initial_energy, axis=1) / energy_scale
    return {
        "primary_numerical": numerical(primary),
        "control_numerical": numerical(controls),
        "primary_parity": parity(primary),
        "control_parity": parity(controls),
        "time_refinement": refinement,
        "time_refinement_passed": bool(
            np.all(np.diff(refinement) <= 1.0e-12)
            and refinement[-1] <= 0.75 * refinement[0]
        ),
        "initial_energy_relative_range": initial_range,
        "initial_energy_match_passed": bool(np.max(initial_range) <= 1.0e-12),
        "load_match_relative_error": load_error,
        "work_match_relative_error": work_error,
        "matched": matched,
        "matched_speed_difference": primary.final_speed[:, coupled]
        - primary.final_speed[:, fixed],
    }


def _arrays(
    authority: _Authority,
    states: tuple[tuple[int, int], ...],
    primary: _Buffers,
    controls: _Buffers,
    config: ArticulatedGroundAtlasConfig,
    gates: dict[str, Any],
) -> dict[str, NDArray[Any]]:
    result: dict[str, NDArray[Any]] = {
        "state_case_index": np.asarray([state[0] for state in states]),
        "state_sample_index": np.asarray([state[1] for state in states]),
        "state_profile_index": authority.profile_index[[state[0] for state in states]],
        "ground_activation_names": np.asarray(config.ground_activations),
        "control_names": np.asarray(config.control_names),
        "velocity_factors": np.asarray(VELOCITY_FACTORS),
        "time_steps_s": np.asarray(config.forward.time_steps_s),
        "engine_names": np.asarray(ENGINES),
        "horizons_s": np.asarray(config.horizons_s),
    }
    for field in fields(_Buffers):
        result[f"primary_{field.name}"] = getattr(primary, field.name)
        result[f"control_{field.name}"] = getattr(controls, field.name)
    result.update(
        {name: value for name, value in gates.items() if isinstance(value, np.ndarray)}
    )
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    states: tuple[tuple[int, int], ...],
    primary: _Buffers,
    controls: _Buffers,
    config: ArticulatedGroundAtlasConfig,
    gates: dict[str, Any],
    versions: dict[str, str],
) -> dict[str, Any]:
    matched = gates["matched"]
    matched_delta = gates["matched_speed_difference"][matched]
    all_passed = bool(
        np.all(gates["primary_numerical"])
        and np.all(gates["control_numerical"])
        and np.all(gates["primary_parity"])
        and np.all(gates["control_parity"])
        and gates["time_refinement_passed"]
        and gates["initial_energy_match_passed"]
    )
    return {
        "schema_version": "articulated-ground-atlas/v1",
        "study_id": "finite-ground-free-moment-falsification-atlas",
        "design": {
            "engine_versions": versions,
            "state_count": len(states),
            "ground_activations": list(config.ground_activations),
            "controls": list(config.control_names),
            "velocity_factors": list(VELOCITY_FACTORS),
            "horizons_s": list(config.horizons_s),
            "primary_trajectory_count": len(states)
            * len(config.ground_activations)
            * 2
            * len(config.forward.time_steps_s)
            * 2,
            "control_trajectory_count": len(states)
            * len(config.control_names)
            * 2
            * len(config.forward.time_steps_s)
            * 2,
            "initialization": "identical natural-zero base and shaft coordinates; rigid state and signed club perturbation shared within cell",
            "active_driver_or_joint_torque": "none; motion is an initial condition",
            "center_of_pressure_reversal": "reference-transport invariance is contract-tested because center of pressure does not enter generalized force",
        },
        "configuration": asdict(config),
        "results": {
            "all_registered_gates_passed": all_passed,
            "maximum_normalized_work_energy_residual": float(
                max(
                    np.max(primary.normalized_energy_residual),
                    np.max(controls.normalized_energy_residual),
                )
            ),
            "maximum_trajectory_relative_error": float(
                max(
                    np.max(primary.trajectory_parity),
                    np.max(controls.trajectory_parity),
                )
            ),
            "maximum_ground_force_n": float(np.max(primary.peak_ground_force)),
            "maximum_intrinsic_free_moment_nm": float(
                np.max(primary.peak_intrinsic_moment)
            ),
            "time_refinement_worst_normalized_residual": gates[
                "time_refinement"
            ].tolist(),
            "time_refinement_passed": gates["time_refinement_passed"],
            "maximum_initial_energy_relative_range": float(
                np.max(gates["initial_energy_relative_range"])
            ),
            "initial_energy_match_passed": gates["initial_energy_match_passed"],
            "matched_load_work_cell_count": int(np.count_nonzero(matched)),
            "matched_load_work_total_cell_count": int(matched.size),
            "matched_final_speed_difference_range_m_s": (
                [float(np.min(matched_delta)), float(np.max(matched_delta))]
                if matched_delta.size
                else None
            ),
            "failed_primary_numerical_cell_count": int(
                np.count_nonzero(~gates["primary_numerical"])
            ),
            "failed_control_numerical_cell_count": int(
                np.count_nonzero(~gates["control_numerical"])
            ),
            "failed_primary_parity_cell_count": int(
                np.count_nonzero(~gates["primary_parity"])
            ),
            "failed_control_parity_cell_count": int(
                np.count_nonzero(~gates["control_parity"])
            ),
        },
        "limitations": {
            "calibration_status": "synthetic_reference_not_human_or_force_plate_calibrated",
            "ground_law": "linear planar x-z translation and free pitch moment; not unilateral normal contact, Coulomb friction, foot segmentation, or measured pressure",
            "horizontal_control": "zero horizontal stiffness and damping removes modeled horizontal restraint; it is not a complete friction-contact model",
            "initialization_boundary": "natural-zero provides exact pathway matching but is not static whole-mechanism equilibrium",
            "matching_boundary": "post-registered load/work matching is descriptive, not randomized causal identification",
            "time_boundary": "the atlas ends at 50 ms and cannot establish whole-downswing or impact behavior",
            "human_boundary": "no participant, intent, timing-economy, injury, or coaching inference",
        },
        "next_gate": "add uncertainty and governed force-plate or bilateral-wrench validation before human inference",
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCE_PATHS},
    }


def run_articulated_ground_atlas(
    config: ArticulatedGroundAtlasConfig = ArticulatedGroundAtlasConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Run the registered primary and falsification-control trajectories."""

    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error
    authority = _load_authority()
    states = tuple(
        (case, sample)
        for case in config.case_indices
        for sample in config.sample_indices
    )
    tail = (2, len(config.forward.time_steps_s), 2, len(config.horizons_s))
    primary = _buffers((len(states), len(config.ground_activations), *tail), 20)
    controls = _buffers((len(states), len(config.control_names), *tail), 20)
    jobs = tuple(
        (authority, config, state_slot, state)
        for state_slot, state in enumerate(states)
    )
    executor = None
    if config.worker_count == 1:
        results = map(_state_job, jobs)
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(config.worker_count, len(states)),
            mp_context=multiprocessing.get_context("spawn"),
        )
        results = executor.map(_state_job, jobs)
    try:
        for completed, (state_slot, local_primary, local_controls) in enumerate(
            results, 1
        ):
            _merge_state(primary, state_slot, local_primary)
            _merge_state(controls, state_slot, local_controls)
            print(
                f"ground atlas state {completed}/{len(states)} complete: "
                f"{states[state_slot]}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    gates = _gates(primary, controls, config)
    arrays = _arrays(authority, states, primary, controls, config, gates)
    versions = {
        "mujoco": str(mujoco.__version__),
        "pinocchio": str(pin.__version__),  # type: ignore[attr-defined]
    }
    return _record(states, primary, controls, config, gates, versions), arrays


__all__ = ["ArticulatedGroundAtlasConfig", "run_articulated_ground_atlas"]
