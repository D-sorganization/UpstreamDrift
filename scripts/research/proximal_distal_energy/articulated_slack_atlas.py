"""Registered atlas for typed unilateral slack in articulated contact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    finite_difference_kinematics,
)
from scripts.research.proximal_distal_energy.articulated_slack_contact import (
    AttachmentLawConfig,
    AttachmentLawKind,
)
from scripts.research.proximal_distal_energy.articulated_slack_forward import (
    ArticulatedSlackForwardConfig,
    SlackIntegrationCase,
    integrate_articulated_slack,
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
    "scripts/research/proximal_distal_energy/articulated_slack_contact.py",
    "scripts/research/proximal_distal_energy/articulated_slack_forward.py",
    "scripts/research/proximal_distal_energy/articulated_slack_atlas.py",
    "scripts/research/proximal_distal_energy/run_articulated_slack_atlas.py",
    "tests/research/test_articulated_slack_contact.py",
    "tests/research/test_articulated_slack_forward.py",
)


@dataclass(frozen=True, slots=True)
class ArticulatedSlackAtlasConfig:
    """Registered cohort, laws, matched controls, and solver gates."""

    forward: ArticulatedSlackForwardConfig = ArticulatedSlackForwardConfig()
    case_indices: tuple[int, ...] = (0, 4, 8, 9, 13, 17)
    sample_indices: tuple[int, ...] = (0, 6, 12)
    stiffness_n_m: float = 1800.0
    damping_n_s_m: float = 18.0
    base_extension_m: float = 1.0e-3
    initial_velocity_m_s: float = 5.0e-2
    slack_distances_m: tuple[float, ...] = (5.0e-4, 1.5e-3)

    def __post_init__(self) -> None:
        if not isinstance(self.forward, ArticulatedSlackForwardConfig):
            raise TypeError("forward must be an ArticulatedSlackForwardConfig")
        for name in (
            "stiffness_n_m",
            "base_extension_m",
            "initial_velocity_m_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.damping_n_s_m) or self.damping_n_s_m < 0.0:
            raise ValueError("damping_n_s_m must be finite and nonnegative")
        if any(
            not np.isfinite(value) or value <= 0.0 for value in self.slack_distances_m
        ):
            raise ValueError("slack_distances_m must be finite and positive")
        self._validate_indices("case_indices", self.case_indices, 18)
        self._validate_indices("sample_indices", self.sample_indices, 13)

    @staticmethod
    def _validate_indices(name: str, values: tuple[int, ...], upper: int) -> None:
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique in-range integers")
        if any(
            not isinstance(value, int) or not 0 <= value < upper for value in values
        ):
            raise ValueError(f"{name} must contain unique in-range integers")


@dataclass(frozen=True, slots=True)
class SlackCondition:
    """One law, preload-control, and initial-velocity branch."""

    name: str
    law: AttachmentLawConfig
    preload_mode: str
    state_velocity_factor: float
    club_velocity_m_s: float
    initial_displacement_m: float


@dataclass(frozen=True, slots=True)
class _Authority:
    time_s: FloatArray
    profile_index: NDArray[np.int_]
    grip_span_m: FloatArray
    solution_q: FloatArray


@dataclass(slots=True)
class _Buffers:
    peak_force: FloatArray
    open_fraction: FloatArray
    transition_count: NDArray[np.int_]
    first_opening_s: FloatArray
    first_reattachment_s: FloatArray
    opening_observed: NDArray[np.bool_]
    reattachment_observed: NDArray[np.bool_]
    initial_strain_energy: FloatArray
    final_club_speed: FloatArray
    maximum_virtual_power: FloatArray
    maximum_dissipation: FloatArray
    normalized_energy_residual: FloatArray
    trajectory_relative_error: FloatArray
    force_relative_error: FloatArray
    active_set_parity: NDArray[np.bool_]


def registered_slack_conditions(
    config: ArticulatedSlackAtlasConfig,
) -> tuple[SlackCondition, ...]:
    """Return the full law-by-preload-by-velocity comparison matrix."""

    laws = [
        ("bilateral", AttachmentLawKind.BILATERAL, 0.0),
        ("tension_only", AttachmentLawKind.TENSION_ONLY, 0.0),
        *[
            (
                f"dead_zone_{distance * 1000:g}mm",
                AttachmentLawKind.DEAD_ZONE_TENSION,
                distance,
            )
            for distance in config.slack_distances_m
        ],
    ]
    conditions = []
    for law_name, kind, slack in laws:
        law = AttachmentLawConfig(
            kind=kind,
            stiffness=config.stiffness_n_m,
            damping=config.damping_n_s_m,
            slack_distance_m=slack,
        )
        for preload_mode in ("common_displacement", "matched_extension"):
            displacement = config.base_extension_m
            if preload_mode == "matched_extension":
                displacement += slack
            for velocity_factor in (1.0, -1.0):
                sign = "forward" if velocity_factor > 0.0 else "reversed"
                conditions.append(
                    SlackCondition(
                        name=f"{law_name}__{preload_mode}__{sign}",
                        law=law,
                        preload_mode=preload_mode,
                        state_velocity_factor=1.0,
                        club_velocity_m_s=(
                            config.initial_velocity_m_s * velocity_factor
                        ),
                        initial_displacement_m=displacement,
                    )
                )
    event_law = AttachmentLawConfig(
        kind=AttachmentLawKind.DEAD_ZONE_TENSION,
        stiffness=config.stiffness_n_m,
        damping=config.damping_n_s_m,
        slack_distance_m=max(config.slack_distances_m),
    )
    conditions.extend(
        (
            SlackCondition(
                name="dead_zone_event_probe__open_to_taut",
                law=event_law,
                preload_mode="event_probe_open",
                state_velocity_factor=0.0,
                club_velocity_m_s=1.0,
                initial_displacement_m=config.base_extension_m,
            ),
            SlackCondition(
                name="dead_zone_event_probe__taut_to_open",
                law=event_law,
                preload_mode="event_probe_taut",
                state_velocity_factor=0.0,
                club_velocity_m_s=-1.0,
                initial_displacement_m=(
                    config.base_extension_m + max(config.slack_distances_m)
                ),
            ),
        )
    )
    return tuple(conditions)


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


def _buffers(shape: tuple[int, ...]) -> _Buffers:
    parity_shape = shape[:-1]
    return _Buffers(
        peak_force=np.empty(shape),
        open_fraction=np.empty(shape),
        transition_count=np.empty(shape, dtype=int),
        first_opening_s=np.empty(shape),
        first_reattachment_s=np.empty(shape),
        opening_observed=np.empty(shape, dtype=bool),
        reattachment_observed=np.empty(shape, dtype=bool),
        initial_strain_energy=np.empty(shape),
        final_club_speed=np.empty(shape),
        maximum_virtual_power=np.empty(shape),
        maximum_dissipation=np.empty(shape),
        normalized_energy_residual=np.empty(shape),
        trajectory_relative_error=np.empty(parity_shape),
        force_relative_error=np.empty(parity_shape),
        active_set_parity=np.empty(parity_shape, dtype=bool),
    )


def _event_time(time_s: FloatArray, event: NDArray[np.bool_]) -> tuple[float, bool]:
    indices = np.flatnonzero(event)
    if indices.size:
        return float(time_s[indices[0]]), True
    return float(time_s[-1]), False


def _record_trace(
    buffers: _Buffers,
    slots: tuple[int, int, int, int],
    trace: dict[str, NDArray[Any]],
) -> None:
    active = np.asarray(trace["active_interface_count"], dtype=int)
    transitions = np.asarray(trace["active_set_transition"], dtype=bool)
    time_s = np.asarray(trace["time_s"])
    opening = np.zeros(active.size, dtype=bool)
    reattachment = np.zeros(active.size, dtype=bool)
    opening[1:] = active[1:] < active[:-1]
    reattachment[1:] = active[1:] > active[:-1]
    buffers.peak_force[slots] = np.max(trace["maximum_contact_force_n"])
    buffers.open_fraction[slots] = np.mean(active < 2)
    buffers.transition_count[slots] = np.count_nonzero(transitions)
    buffers.first_opening_s[slots], buffers.opening_observed[slots] = _event_time(
        time_s, opening
    )
    buffers.first_reattachment_s[slots], buffers.reattachment_observed[slots] = (
        _event_time(time_s, reattachment)
    )
    buffers.initial_strain_energy[slots] = trace["strain_energy_j"][0]
    buffers.final_club_speed[slots] = np.linalg.norm(trace["qd"][-1, 14:17])
    buffers.maximum_virtual_power[slots] = np.max(
        np.abs(trace["virtual_power_residual_w"])
    )
    buffers.maximum_dissipation[slots] = np.max(trace["dissipation_power_w"])
    total = np.asarray(trace["total_energy_j"])
    residual = np.asarray(trace["work_energy_residual_j"])
    buffers.normalized_energy_residual[slots] = np.max(np.abs(residual)) / max(
        1.0, float(np.ptp(total))
    )


def _relative_error(left: FloatArray, right: FloatArray) -> float:
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return float(np.max(np.abs(left - right)) / scale)


def _run_state(
    authority: _Authority,
    buffers: _Buffers,
    config: ArticulatedSlackAtlasConfig,
    state_slot: int,
    state: tuple[int, int],
    conditions: tuple[SlackCondition, ...],
) -> None:
    case_index, sample = state
    profiles = default_synthetic_profiles()
    model, metadata = build_subject_scaled_model(
        profiles[authority.profile_index[case_index]]
    )
    velocity, _ = finite_difference_kinematics(
        authority.solution_q[case_index], authority.time_s
    )
    for condition_slot, condition in enumerate(conditions):
        for step_slot, time_step_s in enumerate(config.forward.time_steps_s):
            traces = {}
            for engine_slot, engine in enumerate(("mujoco", "pinocchio")):
                case = SlackIntegrationCase(
                    q=authority.solution_q[case_index, sample],
                    qd=velocity[sample] * condition.state_velocity_factor,
                    grip_span_m=float(authority.grip_span_m[case_index]),
                    hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
                    time_step_s=time_step_s,
                    initial_club_displacement_m=condition.initial_displacement_m,
                    initial_club_velocity_m_s=condition.club_velocity_m_s,
                    engine=engine,
                    law=condition.law,
                )
                trace = integrate_articulated_slack(model, case, config.forward)
                traces[engine] = trace
                _record_trace(
                    buffers,
                    (state_slot, condition_slot, step_slot, engine_slot),
                    trace,
                )
            parity_slot = (state_slot, condition_slot, step_slot)
            buffers.trajectory_relative_error[parity_slot] = _relative_error(
                traces["mujoco"]["q"], traces["pinocchio"]["q"]
            )
            buffers.force_relative_error[parity_slot] = _relative_error(
                traces["mujoco"]["maximum_contact_force_n"],
                traces["pinocchio"]["maximum_contact_force_n"],
            )
            buffers.active_set_parity[parity_slot] = np.array_equal(
                traces["mujoco"]["active_interface_count"],
                traces["pinocchio"]["active_interface_count"],
            )


def _gates(
    buffers: _Buffers, config: ArticulatedSlackAtlasConfig
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], FloatArray, bool]:
    numerical = (
        (buffers.maximum_virtual_power <= config.forward.virtual_power_tolerance_w)
        & (buffers.maximum_dissipation <= 1.0e-12)
        & (
            buffers.normalized_energy_residual
            <= config.forward.normalized_energy_residual_tolerance
        )
    )
    parity = (
        (
            buffers.trajectory_relative_error
            <= config.forward.trajectory_relative_tolerance
        )
        & (buffers.force_relative_error <= config.forward.trajectory_relative_tolerance)
        & buffers.active_set_parity
    )
    refinement = np.max(buffers.normalized_energy_residual, axis=(0, 1, 3))
    refinement_passed = bool(
        np.all(np.diff(refinement) <= 0.0)
        and refinement[-1] <= config.forward.refinement_ratio_limit * refinement[0]
    )
    return numerical, parity, refinement, refinement_passed


def _arrays(
    authority: _Authority,
    buffers: _Buffers,
    states: tuple[tuple[int, int], ...],
    conditions: tuple[SlackCondition, ...],
    config: ArticulatedSlackAtlasConfig,
    gates: tuple[NDArray[np.bool_], NDArray[np.bool_], FloatArray, bool],
) -> dict[str, NDArray[Any]]:
    state_cases = np.asarray([state[0] for state in states], dtype=int)
    state_samples = np.asarray([state[1] for state in states], dtype=int)
    return {
        "state_case_index": state_cases,
        "state_sample_index": state_samples,
        "state_profile_index": authority.profile_index[state_cases],
        "state_grip_span_m": authority.grip_span_m[state_cases],
        "condition_names": np.asarray([condition.name for condition in conditions]),
        "law_kinds": np.asarray([condition.law.kind.value for condition in conditions]),
        "slack_distance_m": np.asarray(
            [condition.law.slack_distance_m for condition in conditions]
        ),
        "preload_modes": np.asarray(
            [condition.preload_mode for condition in conditions]
        ),
        "state_velocity_factors": np.asarray(
            [condition.state_velocity_factor for condition in conditions]
        ),
        "club_velocity_m_s": np.asarray(
            [condition.club_velocity_m_s for condition in conditions]
        ),
        "initial_displacement_m": np.asarray(
            [condition.initial_displacement_m for condition in conditions]
        ),
        "time_steps_s": np.asarray(config.forward.time_steps_s),
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
        "peak_contact_force_n": buffers.peak_force,
        "open_fraction": buffers.open_fraction,
        "active_set_transition_count": buffers.transition_count,
        "first_opening_s": buffers.first_opening_s,
        "first_reattachment_s": buffers.first_reattachment_s,
        "opening_observed": buffers.opening_observed,
        "reattachment_observed": buffers.reattachment_observed,
        "initial_strain_energy_j": buffers.initial_strain_energy,
        "final_club_translation_speed_m_s": buffers.final_club_speed,
        "maximum_virtual_power_residual_w": buffers.maximum_virtual_power,
        "maximum_dissipation_power_w": buffers.maximum_dissipation,
        "normalized_work_energy_residual": buffers.normalized_energy_residual,
        "trajectory_relative_error": buffers.trajectory_relative_error,
        "force_relative_error": buffers.force_relative_error,
        "active_set_parity": buffers.active_set_parity,
        "numerical_gates_passed": gates[0],
        "parity_gates_passed": gates[1],
        "refinement_worst_normalized_residual": gates[2],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    buffers: _Buffers,
    states: tuple[tuple[int, int], ...],
    conditions: tuple[SlackCondition, ...],
    config: ArticulatedSlackAtlasConfig,
    gates: tuple[NDArray[np.bool_], NDArray[np.bool_], FloatArray, bool],
    engine_versions: dict[str, str],
) -> dict[str, Any]:
    all_passed = bool(np.all(gates[0]) and np.all(gates[1]) and gates[3])
    return {
        "schema_version": "articulated-slack-atlas/v1",
        "study_id": "typed-unilateral-slack-articulated-contact",
        "design": {
            "engine_versions": engine_versions,
            "state_count": len(states),
            "condition_count": len(conditions),
            "time_step_count": len(config.forward.time_steps_s),
            "trajectory_count": int(np.prod(buffers.peak_force.shape)),
            "horizon_s": config.forward.duration_s,
            "active_driver_or_joint_torque": "none; motion is an initial condition",
        },
        "configuration": asdict(config),
        "conditions": [
            {
                "name": condition.name,
                "law": asdict(condition.law),
                "preload_mode": condition.preload_mode,
                "state_velocity_factor": condition.state_velocity_factor,
                "club_velocity_m_s": condition.club_velocity_m_s,
                "initial_displacement_m": condition.initial_displacement_m,
            }
            for condition in conditions
        ],
        "results": {
            "opening_cell_count": int(np.count_nonzero(buffers.opening_observed)),
            "reattachment_cell_count": int(
                np.count_nonzero(buffers.reattachment_observed)
            ),
            "maximum_open_fraction": float(np.max(buffers.open_fraction)),
            "maximum_transition_count": int(np.max(buffers.transition_count)),
            "maximum_peak_contact_force_n": float(np.max(buffers.peak_force)),
            "maximum_virtual_power_residual_w": float(
                np.max(buffers.maximum_virtual_power)
            ),
            "maximum_positive_dissipation_power_w": float(
                np.max(buffers.maximum_dissipation)
            ),
            "maximum_normalized_work_energy_residual": float(
                np.max(buffers.normalized_energy_residual)
            ),
            "maximum_trajectory_relative_error": float(
                np.max(buffers.trajectory_relative_error)
            ),
            "maximum_force_relative_error": float(np.max(buffers.force_relative_error)),
            "active_set_parity_failures": int(
                np.count_nonzero(~buffers.active_set_parity)
            ),
            "failed_numerical_cell_count": int(np.count_nonzero(~gates[0])),
            "failed_parity_cell_count": int(np.count_nonzero(~gates[1])),
            "refinement_worst_normalized_residual": gates[2].tolist(),
            "refinement_direction_passed": gates[3],
            "all_registered_gates_passed": all_passed,
        },
        "interpretation": {
            "matched_controls": "common-displacement and approximately matched radial-extension branches separate dead-zone opening from preload-energy changes",
            "events": "opening and reattachment are active-set events in a synthetic tension law, not measured biological slack",
            "right_censoring": "absence of an event before five milliseconds is right-censored at the registered horizon",
        },
        "claim_boundary": {
            "supported": "typed bilateral, tension-only, and dead-zone laws can be compared under matched articulated initial states and registered numerical controls",
            "timing_economy_or_self_correction": "not_identified",
            "biological_slack_or_intent": "not_identified",
            "delivery_or_clubhead_speed_strategy": "not_established",
            "human_transfer": "untested",
        },
        "next_gate": "extend event-qualified laws to distributed grip and shaft compliance, longer horizons, ground coupling, and governed bilateral human wrenches",
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in SOURCE_PATHS},
    }


def run_articulated_slack_atlas(
    config: ArticulatedSlackAtlasConfig = ArticulatedSlackAtlasConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Run the registered typed-law, preload, velocity, state, and engine atlas."""

    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error
    authority = _load_authority()
    states = tuple(
        (case, sample)
        for case in config.case_indices
        for sample in config.sample_indices
    )
    conditions = registered_slack_conditions(config)
    shape = (len(states), len(conditions), len(config.forward.time_steps_s), 2)
    buffers = _buffers(shape)
    for state_slot, state in enumerate(states):
        _run_state(authority, buffers, config, state_slot, state, conditions)
    gates = _gates(buffers, config)
    arrays = _arrays(authority, buffers, states, conditions, config, gates)
    versions = {
        "mujoco": str(mujoco.__version__),
        "pinocchio": str(pin.__version__),  # type: ignore[attr-defined]
    }
    return _record(buffers, states, conditions, config, gates, versions), arrays


__all__ = [
    "ArticulatedSlackAtlasConfig",
    "SlackCondition",
    "registered_slack_conditions",
    "run_articulated_slack_atlas",
]
