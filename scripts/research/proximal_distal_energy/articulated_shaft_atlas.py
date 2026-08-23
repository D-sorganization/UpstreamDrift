"""Registered articulated distributed-grip and passive-shaft atlas."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
)
from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
    load_default_atlas_authority,
)
from scripts.research.proximal_distal_energy.articulated_atlas_runtime_authority import (
    AtlasStateSelection,
    resolve_atlas_states,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftConfig,
    ArticulatedShaftProperties,
    build_articulated_shaft,
)
from scripts.research.proximal_distal_energy.articulated_shaft_forward import (
    ShaftForwardConfig,
    ShaftIntegrationCase,
    integrate_articulated_shaft,
)

FloatArray = NDArray[np.float64]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
ACTIVATIONS = ("rigid", "bending", "torsion", "coupled")
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "docs/research/proximal_distal_energy_transfer/data/articulated_structural_authority_nominal.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_structural_authority_nominal.npz",
    "docs/research/proximal_distal_energy_transfer/data/shaft_beam_reference.json",
    "scripts/research/proximal_distal_energy/articulated_atlas_authority.py",
    "scripts/research/proximal_distal_energy/articulated_structural_atlas_execution.py",
    "scripts/research/proximal_distal_energy/articulated_distributed_grip.py",
    "scripts/research/proximal_distal_energy/articulated_distributed_forward.py",
    "scripts/research/proximal_distal_energy/articulated_shaft.py",
    "scripts/research/proximal_distal_energy/articulated_shaft_forward.py",
    "scripts/research/proximal_distal_energy/articulated_shaft_atlas.py",
    "scripts/research/proximal_distal_energy/shaft_beam_reference.py",
    "scripts/research/proximal_distal_energy/moving_base_modal_shaft.py",
    "tests/research/test_articulated_shaft.py",
    "tests/research/test_articulated_shaft_forward.py",
    "tests/research/test_articulated_shaft_atlas.py",
)


@dataclass(frozen=True, slots=True)
class ArticulatedShaftAtlasConfig:
    """Registered states, branches, horizons, matching, and numerical gates."""

    forward: ShaftForwardConfig = ShaftForwardConfig()
    case_indices: tuple[int, ...] = (0, 8, 9, 17)
    sample_indices: tuple[int, ...] = (0, 6, 12)
    activations: tuple[str, ...] = ACTIVATIONS
    horizons_s: tuple[float, ...] = (0.004, 0.01, 0.025, 0.05)
    initial_displacement_m: float = 0.001
    initial_velocity_m_s: float = 0.05
    station_count_per_hand: int = 5
    station_width_m: float = 0.03
    total_stiffness_n_m: float = 1800.0
    total_damping_n_s_m: float = 18.0
    shaft_damping_ratio: float = 0.018
    bending_frequency_scale: float = 1.0
    torsional_stiffness_scale: float = 1.0
    match_relative_tolerance: float = 0.05
    worker_count: int = 4

    def __post_init__(self) -> None:
        self._indices("case_indices", self.case_indices, 18)
        self._indices("sample_indices", self.sample_indices, 13)
        if self.activations != ACTIVATIONS:
            raise ValueError(f"activations must be exactly {ACTIVATIONS!r}")
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
                raise ValueError("each horizon must be divisible by each time step")
        if self.station_count_per_hand < 1 or self.station_count_per_hand % 2 == 0:
            raise ValueError("station_count_per_hand must be a positive odd integer")
        if not isinstance(self.worker_count, int) or not 1 <= self.worker_count <= 12:
            raise ValueError("worker_count must be an integer from one through twelve")
        for name in (
            "initial_displacement_m",
            "initial_velocity_m_s",
            "station_width_m",
            "total_stiffness_n_m",
            "total_damping_n_s_m",
            "bending_frequency_scale",
            "torsional_stiffness_scale",
            "match_relative_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.shaft_damping_ratio < 1.0:
            raise ValueError("shaft_damping_ratio must lie in [0, 1)")

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


@dataclass(slots=True)
class _Buffers:
    peak_force: FloatArray
    peak_couple: FloatArray
    open_fraction: FloatArray
    transition_count: NDArray[np.int_]
    maximum_virtual_power: FloatArray
    maximum_positive_dissipation: FloatArray
    maximum_shaft_power_residual: FloatArray
    normalized_energy_residual: FloatArray
    maximum_bending: FloatArray
    maximum_twist: FloatArray
    peak_shaft_energy: FloatArray
    terminal_dissipated_work: FloatArray
    final_speed: FloatArray
    initial_energy: FloatArray
    final_state: FloatArray
    final_elastic: FloatArray
    trajectory_parity: FloatArray
    force_parity: FloatArray
    active_set_parity: NDArray[np.bool_]


def _load_authority() -> ArticulatedAtlasAuthority:
    return load_default_atlas_authority()


def _resolve_states(
    authority: ArticulatedAtlasAuthority,
    config: ArticulatedShaftAtlasConfig,
) -> AtlasStateSelection:
    selection = resolve_atlas_states(
        authority,
        config.case_indices,
        config.sample_indices,
    )
    if not selection.feasible_states:
        raise RuntimeError("registered shaft atlas contains no feasible states")
    return selection


def _buffers(shape: tuple[int, ...], nq: int) -> _Buffers:
    trace_shape = shape[:-1]
    parity_shape = shape[:4] + shape[5:]
    return _Buffers(
        peak_force=np.empty(shape),
        peak_couple=np.empty(shape),
        open_fraction=np.empty(shape),
        transition_count=np.empty(shape, dtype=int),
        maximum_virtual_power=np.empty(shape),
        maximum_positive_dissipation=np.empty(shape),
        maximum_shaft_power_residual=np.empty(shape),
        normalized_energy_residual=np.empty(shape),
        maximum_bending=np.empty(shape),
        maximum_twist=np.empty(shape),
        peak_shaft_energy=np.empty(shape),
        terminal_dissipated_work=np.empty(shape),
        final_speed=np.empty(shape),
        initial_energy=np.empty(trace_shape),
        final_state=np.empty((*shape, nq)),
        final_elastic=np.full((*shape, 3), np.nan),
        trajectory_parity=np.empty(parity_shape),
        force_parity=np.empty(parity_shape),
        active_set_parity=np.empty(parity_shape, dtype=bool),
    )


def _relative_error(left: FloatArray, right: FloatArray) -> float:
    if left.shape != right.shape:
        raise ValueError("parity arrays must have identical shapes")
    if left.size == 0:
        return 0.0
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return float(np.max(np.abs(left - right)) / scale)


def _horizon_index(trace: dict[str, NDArray[Any]], horizon_s: float) -> int:
    time_s = np.asarray(trace["time_s"], dtype=float)
    index = int(np.searchsorted(time_s, horizon_s))
    if index >= time_s.size or not np.isclose(time_s[index], horizon_s):
        raise RuntimeError("registered horizon is absent from shaft trajectory")
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
    couple = np.asarray(trace["force_couple_vector_nm"][section], dtype=float)
    bending = np.asarray(trace["tip_bending_m"][section], dtype=float)
    elastic = np.asarray(trace["elastic_coordinates"], dtype=float)
    buffers.peak_force[slot] = np.max(trace["maximum_station_force_n"][section])
    buffers.peak_couple[slot] = np.max(np.linalg.norm(couple, axis=1))
    buffers.open_fraction[slot] = np.mean(active < 10)
    buffers.transition_count[slot] = np.count_nonzero(
        trace["active_set_transition"][section]
    )
    buffers.maximum_virtual_power[slot] = np.max(
        np.abs(trace["virtual_power_residual_w"][section])
    )
    combined_dissipation = np.asarray(
        trace["grip_dissipation_power_w"][section]
    ) + np.asarray(trace["shaft_damping_power_w"][section])
    buffers.maximum_positive_dissipation[slot] = np.max(combined_dissipation)
    buffers.maximum_shaft_power_residual[slot] = np.max(
        trace["shaft_power_residual_w"][section]
    )
    buffers.normalized_energy_residual[slot] = np.max(np.abs(residual)) / max(
        1.0, float(np.ptp(total))
    )
    buffers.maximum_bending[slot] = np.max(np.linalg.norm(bending, axis=1))
    buffers.maximum_twist[slot] = np.max(np.abs(trace["twist_angle_rad"][section]))
    buffers.peak_shaft_energy[slot] = np.max(trace["shaft_strain_energy_j"][section])
    buffers.terminal_dissipated_work[slot] = -float(
        trace["cumulative_dissipation_j"][end]
    )
    buffers.final_speed[slot] = np.linalg.norm(trace["qd"][end, 14:17])
    buffers.final_state[slot] = trace["q"][end]
    if elastic.shape[1]:
        labels = tuple(str(value) for value in trace["active_labels"])
        for source_index, label in enumerate(labels):
            target = {"bend_x": 0, "bend_y": 1, "torsion": 2}[label]
            buffers.final_elastic[(*slot, target)] = elastic[end, source_index]


def _run_pair(
    model: Any,
    case_base: dict[str, Any],
    activation: str,
    forward: ShaftForwardConfig,
    atlas_config: ArticulatedShaftAtlasConfig,
) -> dict[str, dict[str, NDArray[Any]]]:
    traces = {}
    for engine in ("mujoco", "pinocchio"):
        try:
            traces[engine] = integrate_articulated_shaft(
                model,
                ShaftIntegrationCase(
                    engine=engine,
                    shaft=ArticulatedShaftConfig(
                        activation=activation,  # type: ignore[arg-type]
                        damping_ratio=atlas_config.shaft_damping_ratio,
                        bending_frequency_scale=atlas_config.bending_frequency_scale,
                        torsional_stiffness_scale=atlas_config.torsional_stiffness_scale,
                    ),
                    **case_base,
                ),
                forward,
            )
        except Exception as error:
            raise RuntimeError(
                f"native shaft integration failed: engine={engine}"
            ) from error
    return traces


def _record_pair(
    buffers: _Buffers,
    base_slot: tuple[int, int, int, int],
    traces: dict[str, dict[str, NDArray[Any]]],
    config: ArticulatedShaftAtlasConfig,
) -> None:
    for engine_slot, engine in enumerate(("mujoco", "pinocchio")):
        buffers.initial_energy[(*base_slot, engine_slot)] = traces[engine][
            "total_energy_j"
        ][0]
    for horizon_slot, horizon_s in enumerate(config.horizons_s):
        ends = {}
        for engine_slot, engine in enumerate(("mujoco", "pinocchio")):
            end = _horizon_index(traces[engine], horizon_s)
            ends[engine] = end
            _record_horizon(
                buffers,
                (*base_slot, engine_slot, horizon_slot),
                traces[engine],
                end,
            )
        left, right = traces["mujoco"], traces["pinocchio"]
        left_end, right_end = ends["mujoco"], ends["pinocchio"]
        parity_slot = (*base_slot, horizon_slot)
        rigid_error = _relative_error(
            left["q"][: left_end + 1], right["q"][: right_end + 1]
        )
        elastic_error = _relative_error(
            left["elastic_coordinates"][: left_end + 1],
            right["elastic_coordinates"][: right_end + 1],
        )
        buffers.trajectory_parity[parity_slot] = max(rigid_error, elastic_error)
        buffers.force_parity[parity_slot] = _relative_error(
            left["maximum_station_force_n"][: left_end + 1],
            right["maximum_station_force_n"][: right_end + 1],
        )
        buffers.active_set_parity[parity_slot] = np.array_equal(
            left["active_station_count"][: left_end + 1],
            right["active_station_count"][: right_end + 1],
        )


def _run_activation(
    authority: ArticulatedAtlasAuthority,
    buffers: _Buffers,
    config: ArticulatedShaftAtlasConfig,
    state_slot: int,
    state: tuple[int, int],
    activation_slot: int,
    *,
    buffer_activation_slot: int | None = None,
) -> ArticulatedShaftProperties:
    if not 0 <= activation_slot < len(config.activations):
        raise ValueError("activation_slot is outside the registered design")
    output_slot = (
        activation_slot if buffer_activation_slot is None else buffer_activation_slot
    )
    if not 0 <= output_slot < buffers.peak_force.shape[1]:
        raise ValueError("buffer_activation_slot is outside the output buffer")
    case_index, sample = state
    resolved = authority.resolve_state(case_index, sample)
    model, metadata = resolved.model, resolved.model_metadata
    grip = DistributedGripConfig(
        station_count_per_hand=config.station_count_per_hand,
        station_width_m=config.station_width_m,
        total_stiffness_n_m=config.total_stiffness_n_m,
        total_damping_n_s_m=config.total_damping_n_s_m,
    )
    activation = config.activations[activation_slot]
    for velocity_slot, factor in enumerate((1.0, -1.0)):
        for step_slot, step in enumerate(config.forward.time_steps_s):
            base = {
                "q": resolved.q,
                "qd": resolved.qd,
                "grip_span_m": resolved.grip_span_m,
                "hand_contact_local_x_m": float(metadata["hand_contact_local_x_m"]),
                "time_step_s": step,
                "initial_club_displacement_m": config.initial_displacement_m,
                "initial_club_velocity_m_s": factor * config.initial_velocity_m_s,
                "grip": grip,
            }
            try:
                traces = _run_pair(model, base, activation, config.forward, config)
            except Exception as error:
                raise RuntimeError(
                    "shaft atlas cell failed: "
                    f"state={state}, activation={activation}, "
                    f"velocity_factor={factor}, step_s={step}"
                ) from error
            _record_pair(
                buffers,
                (state_slot, output_slot, velocity_slot, step_slot),
                traces,
                config,
            )
    return build_articulated_shaft(
        model,
        ArticulatedShaftConfig(
            damping_ratio=config.shaft_damping_ratio,
            bending_frequency_scale=config.bending_frequency_scale,
            torsional_stiffness_scale=config.torsional_stiffness_scale,
        ),
    )


def _run_state(
    authority: ArticulatedAtlasAuthority,
    buffers: _Buffers,
    config: ArticulatedShaftAtlasConfig,
    state_slot: int,
    state: tuple[int, int],
) -> ArticulatedShaftProperties:
    properties = None
    for activation_slot in range(len(config.activations)):
        properties = _run_activation(
            authority,
            buffers,
            config,
            state_slot,
            state,
            activation_slot,
        )
    if properties is None:
        raise RuntimeError("registered shaft atlas contains no activations")
    return properties


def _run_state_job(
    payload: tuple[
        ArticulatedAtlasAuthority,
        ArticulatedShaftAtlasConfig,
        int,
        tuple[int, int],
    ],
) -> tuple[int, _Buffers, ArticulatedShaftProperties]:
    authority, config, state_slot, state = payload
    shape = (
        1,
        len(config.activations),
        2,
        len(config.forward.time_steps_s),
        2,
        len(config.horizons_s),
    )
    local = _buffers(shape, authority.solution_q.shape[-1])
    properties = _run_state(authority, local, config, 0, state)
    return state_slot, local, properties


def _run_activation_job(
    payload: tuple[
        ArticulatedAtlasAuthority,
        ArticulatedShaftAtlasConfig,
        int,
        tuple[int, int],
        int,
    ],
) -> tuple[int, tuple[int, int], int, _Buffers, ArticulatedShaftProperties]:
    authority, config, state_slot, state, activation_slot = payload
    shape = (
        1,
        1,
        2,
        len(config.forward.time_steps_s),
        2,
        len(config.horizons_s),
    )
    local = _buffers(shape, authority.solution_q.shape[-1])
    properties = _run_activation(
        authority,
        local,
        config,
        0,
        state,
        activation_slot,
        buffer_activation_slot=0,
    )
    return state_slot, state, activation_slot, local, properties


def _merge_state(target: _Buffers, state_slot: int, source: _Buffers) -> None:
    for field in fields(_Buffers):
        target_array = getattr(target, field.name)
        source_array = getattr(source, field.name)
        target_array[state_slot] = source_array[0]


def _excluded_step_probe(
    authority: ArticulatedAtlasAuthority,
    config: ArticulatedShaftAtlasConfig,
    *,
    state: tuple[int, int],
    velocity_factor: float,
    step_s: float,
) -> dict[str, Any]:
    """Confirm that one excluded coarse torsion branch fails closed."""

    case_index, sample = state
    resolved = authority.resolve_state(case_index, sample)
    model, metadata = resolved.model, resolved.model_metadata
    grip = DistributedGripConfig(
        station_count_per_hand=config.station_count_per_hand,
        station_width_m=config.station_width_m,
        total_stiffness_n_m=config.total_stiffness_n_m,
        total_damping_n_s_m=config.total_damping_n_s_m,
    )
    case = ShaftIntegrationCase(
        q=resolved.q,
        qd=resolved.qd,
        grip_span_m=resolved.grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
        time_step_s=step_s,
        initial_club_displacement_m=config.initial_displacement_m,
        initial_club_velocity_m_s=velocity_factor * config.initial_velocity_m_s,
        engine="mujoco",
        grip=grip,
        shaft=ArticulatedShaftConfig(
            activation="torsion",
            damping_ratio=config.shaft_damping_ratio,
            bending_frequency_scale=config.bending_frequency_scale,
            torsional_stiffness_scale=config.torsional_stiffness_scale,
        ),
    )
    try:
        integrate_articulated_shaft(
            model,
            case,
            ShaftForwardConfig(
                duration_s=config.forward.duration_s,
                time_steps_s=(step_s, step_s / 2.0),
            ),
        )
    except RuntimeError as error:
        message = str(error)
        if "linear shaft domain exceeded" not in message:
            raise
        return {
            "step_s": step_s,
            "state": [case_index, sample],
            "activation": "torsion",
            "velocity_factor": velocity_factor,
            "engine": "mujoco",
            "excluded": True,
            "reason": message,
        }
    raise RuntimeError(
        f"excluded {step_s:g} s shaft probe unexpectedly remained in-domain"
    )


def _excluded_step_probes(
    authority: ArticulatedAtlasAuthority, config: ArticulatedShaftAtlasConfig
) -> list[dict[str, Any]]:
    return [
        _excluded_step_probe(
            authority,
            config,
            state=(0, 0),
            velocity_factor=-1.0,
            step_s=0.001,
        ),
        _excluded_step_probe(
            authority,
            config,
            state=(8, 0),
            velocity_factor=1.0,
            step_s=0.0005,
        ),
    ]


def _pair_relative(left: FloatArray, right: FloatArray, floor: float) -> FloatArray:
    scale = np.maximum(floor, 0.5 * (np.abs(left) + np.abs(right)))
    return np.abs(left - right) / scale


def _excitation_controls(buffers: _Buffers) -> dict[str, bool]:
    bending = ACTIVATIONS.index("bending")
    torsion = ACTIVATIONS.index("torsion")
    coupled = ACTIVATIONS.index("coupled")
    return {
        "bending": bool(np.max(buffers.maximum_bending[:, bending]) > 1.0e-10),
        "torsion": bool(np.max(buffers.maximum_twist[:, torsion]) > 1.0e-12),
        "coupled_bending": bool(np.max(buffers.maximum_bending[:, coupled]) > 1.0e-10),
        "coupled_torsion": bool(np.max(buffers.maximum_twist[:, coupled]) > 1.0e-12),
    }


def _gates(
    buffers: _Buffers,
    config: ArticulatedShaftAtlasConfig,
    properties: ArticulatedShaftProperties,
    beam_record: dict[str, Any],
) -> dict[str, Any]:
    numerical = (
        (buffers.maximum_virtual_power <= config.forward.virtual_power_tolerance_w)
        & (buffers.maximum_positive_dissipation <= 1.0e-12)
        & (buffers.maximum_shaft_power_residual <= 1.0e-10)
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
    small_deflection = (
        buffers.maximum_bending / properties.config.shaft_length_m
        <= properties.config.small_deflection_limit
    )
    twist = buffers.maximum_twist <= properties.config.twist_limit_rad
    refinement = np.max(
        buffers.normalized_energy_residual,
        axis=(0, 1, 2, 4, 5),
    )
    refinement_passed = bool(
        np.all(np.diff(refinement) <= 1.0e-12)
        and refinement[-1] <= config.forward.refinement_ratio_limit * refinement[0]
    )
    energy_scale = np.maximum(
        1.0,
        np.max(np.abs(buffers.initial_energy), axis=1),
    )
    initial_energy_relative_range = (
        np.ptp(buffers.initial_energy, axis=1) / energy_scale
    )
    rigid, coupled = ACTIVATIONS.index("rigid"), ACTIVATIONS.index("coupled")
    load_error = _pair_relative(
        buffers.peak_force[:, coupled], buffers.peak_force[:, rigid], 1.0
    )
    work_error = _pair_relative(
        buffers.terminal_dissipated_work[:, coupled],
        buffers.terminal_dissipated_work[:, rigid],
        1.0e-6,
    )
    matched = (load_error <= config.match_relative_tolerance) & (
        work_error <= config.match_relative_tolerance
    )
    matched_speed_difference = (
        buffers.final_speed[:, coupled] - buffers.final_speed[:, rigid]
    )
    expected_nan = np.ones((4, 3), dtype=bool)
    expected_nan[ACTIVATIONS.index("bending"), :2] = False
    expected_nan[ACTIVATIONS.index("torsion"), 2] = False
    expected_nan[ACTIVATIONS.index("coupled"), :] = False
    observed_nan = np.all(np.isnan(buffers.final_elastic), axis=(0, 2, 3, 4, 5))
    excitation = _excitation_controls(buffers)
    structural_reference_passed = bool(
        properties.fe_bending_frequency_relative_error <= 1.0e-12
        and beam_record["convergence"]["maximum_relative_change_24_to_48"] <= 0.01
        and beam_record["comparison"]["reduced_mode_count"] == 1
        and beam_record["comparison"]["reference_mode_count"] == 6
    )
    return {
        "numerical": numerical,
        "parity": parity,
        "small_deflection": small_deflection,
        "twist": twist,
        "time_refinement": refinement,
        "time_refinement_passed": refinement_passed,
        "initial_energy_relative_range": initial_energy_relative_range,
        "initial_energy_match_passed": bool(
            np.max(initial_energy_relative_range) <= 1.0e-12
        ),
        "load_match_relative_error": load_error,
        "work_match_relative_error": work_error,
        "matched": matched,
        "matched_speed_difference": matched_speed_difference,
        "activation_nan_pattern_passed": bool(
            np.array_equal(observed_nan, expected_nan)
        ),
        "excitation": excitation,
        "structural_reference_passed": structural_reference_passed,
    }


def _arrays(
    authority: ArticulatedAtlasAuthority,
    states: tuple[tuple[int, int], ...],
    buffers: _Buffers,
    config: ArticulatedShaftAtlasConfig,
    gates: dict[str, Any],
) -> dict[str, NDArray[Any]]:
    cases = np.asarray([state[0] for state in states], dtype=int)
    return {
        "state_case_index": cases,
        "state_sample_index": np.asarray([state[1] for state in states], dtype=int),
        "state_profile_index": authority.profile_index[cases],
        "state_grip_span_m": authority.grip_span_m[cases],
        "activation_names": np.asarray(config.activations),
        "velocity_factors": np.asarray([1.0, -1.0]),
        "time_steps_s": np.asarray(config.forward.time_steps_s),
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
        "horizons_s": np.asarray(config.horizons_s),
        "peak_station_force_n": buffers.peak_force,
        "peak_force_couple_nm": buffers.peak_couple,
        "open_fraction": buffers.open_fraction,
        "active_set_transition_count": buffers.transition_count,
        "maximum_virtual_power_residual_w": buffers.maximum_virtual_power,
        "maximum_positive_dissipation_power_w": buffers.maximum_positive_dissipation,
        "maximum_shaft_power_residual_w": buffers.maximum_shaft_power_residual,
        "normalized_work_energy_residual": buffers.normalized_energy_residual,
        "maximum_tip_bending_m": buffers.maximum_bending,
        "maximum_twist_angle_rad": buffers.maximum_twist,
        "peak_shaft_strain_energy_j": buffers.peak_shaft_energy,
        "terminal_dissipated_work_j": buffers.terminal_dissipated_work,
        "final_club_translation_speed_m_s": buffers.final_speed,
        "initial_total_energy_j": buffers.initial_energy,
        "final_q": buffers.final_state,
        "final_elastic_coordinates": buffers.final_elastic,
        "trajectory_relative_error": buffers.trajectory_parity,
        "force_relative_error": buffers.force_parity,
        "active_set_parity": buffers.active_set_parity,
        "time_refinement_worst_normalized_residual": gates["time_refinement"],
        "initial_energy_relative_range": gates["initial_energy_relative_range"],
        "load_match_relative_error": gates["load_match_relative_error"],
        "work_match_relative_error": gates["work_match_relative_error"],
        "matched_load_work": gates["matched"],
        "matched_final_speed_difference_m_s": gates["matched_speed_difference"],
        "numerical_gates_passed": gates["numerical"],
        "parity_gates_passed": gates["parity"],
        "small_deflection_gate_passed": gates["small_deflection"],
        "twist_gate_passed": gates["twist"],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structural_record(
    properties: ArticulatedShaftProperties,
    beam_record: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    comparison = beam_record["comparison"]
    return {
        "bending_frequency_hz": properties.bending_frequency_hz,
        "torsion_frequency_hz": properties.torsion_frequency_hz,
        "fe_bending_frequency_relative_error": properties.fe_bending_frequency_relative_error,
        "one_mode_low_frequency_tip_rms_discrepancy_m": comparison[
            "low_frequency_tip_rms_discrepancy_m"
        ],
        "one_mode_high_frequency_tip_rms_discrepancy_m": comparison[
            "high_frequency_tip_rms_discrepancy_m"
        ],
        "reference_mode_count": comparison["reference_mode_count"],
        "structural_reference_passed": gates["structural_reference_passed"],
    }


def _result_record(
    buffers: _Buffers, gates: dict[str, Any], all_passed: bool
) -> dict[str, Any]:
    matched = gates["matched"]
    matched_delta = gates["matched_speed_difference"][matched]
    return {
        "all_registered_gates_passed": all_passed,
        "maximum_peak_station_force_n": float(np.max(buffers.peak_force)),
        "maximum_peak_force_couple_nm": float(np.max(buffers.peak_couple)),
        "maximum_tip_bending_m": float(np.max(buffers.maximum_bending)),
        "maximum_twist_angle_rad": float(np.max(buffers.maximum_twist)),
        "maximum_peak_shaft_strain_energy_j": float(np.max(buffers.peak_shaft_energy)),
        "maximum_normalized_work_energy_residual": float(
            np.max(buffers.normalized_energy_residual)
        ),
        "maximum_trajectory_relative_error": float(np.max(buffers.trajectory_parity)),
        "maximum_force_relative_error": float(np.max(buffers.force_parity)),
        "active_set_parity_failures": int(np.count_nonzero(~buffers.active_set_parity)),
        "time_refinement_worst_normalized_residual": gates["time_refinement"].tolist(),
        "time_refinement_passed": gates["time_refinement_passed"],
        "maximum_initial_energy_relative_range": float(
            np.max(gates["initial_energy_relative_range"])
        ),
        "initial_energy_match_passed": gates["initial_energy_match_passed"],
        "activation_coordinate_pattern_passed": gates["activation_nan_pattern_passed"],
        "excitation_controls": gates["excitation"],
        "matched_load_work_cell_count": int(np.count_nonzero(matched)),
        "matched_load_work_total_cell_count": int(matched.size),
        "matched_final_speed_difference_range_m_s": (
            [float(np.min(matched_delta)), float(np.max(matched_delta))]
            if matched_delta.size
            else None
        ),
        "failed_numerical_cell_count": int(np.count_nonzero(~gates["numerical"])),
        "failed_parity_cell_count": int(np.count_nonzero(~gates["parity"])),
        "failed_small_deflection_cell_count": int(
            np.count_nonzero(~gates["small_deflection"])
        ),
        "failed_twist_cell_count": int(np.count_nonzero(~gates["twist"])),
    }


@dataclass(frozen=True, slots=True)
class _RecordContext:
    authority: ArticulatedAtlasAuthority
    selection: AtlasStateSelection
    buffers: _Buffers
    config: ArticulatedShaftAtlasConfig
    properties: ArticulatedShaftProperties
    beam_record: dict[str, Any]
    gates: dict[str, Any]
    versions: dict[str, str]
    coarse_probes: list[dict[str, Any]]


def _record(
    context: _RecordContext,
) -> dict[str, Any]:
    authority = context.authority
    selection = context.selection
    buffers = context.buffers
    config = context.config
    properties = context.properties
    beam_record = context.beam_record
    gates = context.gates
    versions = context.versions
    coarse_probes = context.coarse_probes
    states = selection.feasible_states
    all_passed = bool(
        np.all(gates["numerical"])
        and np.all(gates["parity"])
        and np.all(gates["small_deflection"])
        and np.all(gates["twist"])
        and gates["time_refinement_passed"]
        and gates["initial_energy_match_passed"]
        and gates["activation_nan_pattern_passed"]
        and all(gates["excitation"].values())
        and gates["structural_reference_passed"]
        and all(probe["excluded"] for probe in coarse_probes)
    )
    return {
        "schema_version": "articulated-shaft-atlas/v1",
        "study_id": "distributed-grip-articulated-shaft-bending-torsion",
        "design": {
            "engine_versions": versions,
            "planned_state_count": len(selection.planned_states),
            "feasible_state_count": len(states),
            "retained_failures": [dict(row) for row in selection.retained_failures],
            "state_count": len(states),
            "activations": list(config.activations),
            "velocity_branch_count": 2,
            "time_step_count": len(config.forward.time_steps_s),
            "engine_count": 2,
            "trajectory_count": len(states)
            * len(config.activations)
            * 2
            * len(config.forward.time_steps_s)
            * 2,
            "horizons_s": list(config.horizons_s),
            "grip_station_count_per_hand": config.station_count_per_hand,
            "active_driver_or_joint_torque": "none; motion is an initial condition",
            "initial_energy_control": "zero elastic state gives common initial total energy across activations",
            "load_work_match": "post-registered coupled-versus-rigid cells within the declared symmetric relative tolerance",
        },
        "configuration": asdict(config),
        "state_authority": authority.provenance_record(),
        "excluded_coarse_step_probes": coarse_probes,
        "structural_authority": _structural_record(properties, beam_record, gates),
        "results": _result_record(buffers, gates, all_passed),
        "limitations": {
            "calibration_status": properties.calibration_status,
            "shaft_model": "linear first-mode bending in two planes plus linear torsion using rigid lumped shaft/head inertia",
            "higher_mode_boundary": "the committed six-mode beam shows materially larger discrepancy under short high-frequency loading than under slow loading",
            "contact_boundary": "frictionless tension fibers with state-registered free lengths; not measured pressure, fingers, or tissue",
            "support_boundary": "ground reaction and free moment are absent",
            "matching_boundary": "post-registration matching is descriptive and does not randomize or identify a causal shaft effect",
            "time_step_boundary": "registered 1.0 and 0.50 ms torsion probes leave the declared linear domain in different articulated states and are excluded; the atlas uses the bounded 0.25/0.125 ms refinement pair",
            "human_boundary": "no participant, intent, timing-economy, injury, or coaching inference",
        },
        "next_gate": "add finite ground/free-moment and uncertainty pathways before full-delivery or human inference",
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in SOURCE_PATHS},
    }


def run_articulated_shaft_atlas(
    config: ArticulatedShaftAtlasConfig = ArticulatedShaftAtlasConfig(),
    *,
    authority: ArticulatedAtlasAuthority | None = None,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Run the registered state, shaft, sign, step, engine, and horizon atlas."""

    authority = authority if authority is not None else _load_authority()
    selection = _resolve_states(authority, config)
    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error
    coarse_probes = _excluded_step_probes(authority, config)
    states = selection.feasible_states
    shape = (
        len(states),
        len(config.activations),
        2,
        len(config.forward.time_steps_s),
        2,
        len(config.horizons_s),
    )
    buffers = _buffers(shape, authority.solution_q.shape[-1])
    properties = None
    jobs = tuple(
        (authority, config, state_slot, state)
        for state_slot, state in enumerate(states)
    )
    results: Iterator[tuple[int, _Buffers, ArticulatedShaftProperties]]
    if config.worker_count == 1:
        results = map(_run_state_job, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(config.worker_count, len(states)),
            mp_context=multiprocessing.get_context("spawn"),
        )
        results = executor.map(_run_state_job, jobs)
    try:
        for completed, (state_slot, local, state_properties) in enumerate(results, 1):
            _merge_state(buffers, state_slot, local)
            properties = state_properties
            print(
                f"shaft atlas state {completed}/{len(states)} complete: "
                f"{states[state_slot]}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if properties is None:
        raise RuntimeError("registered shaft atlas contains no states")
    beam_record = json.loads(
        (DATA_DIR / "shaft_beam_reference.json").read_text(encoding="utf-8")
    )
    gates = _gates(buffers, config, properties, beam_record)
    arrays = _arrays(authority, states, buffers, config, gates)
    versions = {
        "mujoco": str(mujoco.__version__),
        "pinocchio": str(pin.__version__),  # type: ignore[attr-defined]
    }
    return (
        _record(
            _RecordContext(
                authority=authority,
                selection=selection,
                buffers=buffers,
                config=config,
                properties=properties,
                beam_record=beam_record,
                gates=gates,
                versions=versions,
                coarse_probes=coarse_probes,
            )
        ),
        arrays,
    )


__all__ = ["ArticulatedShaftAtlasConfig", "run_articulated_shaft_atlas"]
