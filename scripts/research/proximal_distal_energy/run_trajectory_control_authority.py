"""Generate or validate trajectory-varying control-authority evidence."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.research.proximal_distal_energy.numeric_evidence import (
    canonicalize_published_numbers,
)
from scripts.research.proximal_distal_energy.phase_event_stability import (
    StateScales,
    first_positive_crossing,
    rollout_state_history,
    state_derivative,
)
from scripts.research.proximal_distal_energy.run_experiments import INITIAL_Q
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    ControlScales,
    EventConditionedGramian,
    GuardCrossingConfig,
    PulseSensitivityRequest,
    RefinedCrossing,
    StepLinearization,
    StepLinearizationConfig,
    TrajectoryLinearization,
    direct_variable_terminal_pulse_sensitivity,
    event_conditioned_gramian,
    frozen_local_gramian,
    linearize_trajectory,
    propagated_terminal_input_sensitivity,
    reachability_history,
    refine_guard_crossing,
    scale_step_matrices,
    step_linearization,
)
from src.shared.python.simulation_backends import GolfModelParams

DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"
REPORT_PATH = DATA_DIR / "trajectory_control_authority.json"
ARRAY_PATH = DATA_DIR / "trajectory_control_authority.npz"
SOURCE_REPORT_PATH = DATA_DIR / "phase_event_stability.json"
SOURCE_ARRAY_PATH = DATA_DIR / "phase_event_stability.npz"

SCHEMA_VERSION = "proximal-distal-trajectory-control-authority/v1"
BASE_DT_S = 0.000125
SOURCE_HORIZON_S = 0.9
STATE_STEPS = np.array([1e-6, 1e-6, 1e-5, 1e-5], dtype=float)
CONTROL_STEPS = np.array([1e-4, 1e-4], dtype=float)
CONTROL_STEP_MULTIPLIERS = (0.5, 1.0, 2.0)
INTEGRATION_STEP_MULTIPLIERS = (0.5, 1.0, 2.0)
CONTROL_SCALES = ControlScales((100.0, 100.0))
GUARD_GRADIENT = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
TRANSVERSALITY_THRESHOLD_PER_S = 1e-6
GUARD_RESIDUAL_GATE = 1e-10
ADDITIVITY_RESIDUAL_GATE = 1e-10
DIRECT_PULSE_RESIDUAL_GATE = 1e-4
INPUT_REFINEMENT_GATE = 1e-3
INTEGRATION_REFINEMENT_GATE = 5e-2
EQUIVALENT_UNIT_RESIDUAL_GATE = 1e-12
RANK_ABSOLUTE_TOLERANCE = 1e-12
RANK_RELATIVE_TOLERANCE = 1e-9
PULSE_PERTURBATION_SCALE = 1e-6
PHASE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
PULSE_FRACTIONS = (0.2, 0.5, 0.8)
INFERENCE_BOUNDARY = (
    "This is local first-order finite-window authority on one synthetic "
    "analytical-double-pendulum trajectory. The frozen-local result is a "
    "countermodel, not a global nonlinear reachability result. A Gramian does "
    "not establish bounded-control feasibility, controller superiority, human "
    "strength or neural strategy, passive torque, robustness, or coaching "
    "benefit."
)


@dataclass(frozen=True, slots=True)
class _EventTrajectory:
    trajectory: TrajectoryLinearization
    step_durations_s: np.ndarray
    crossing_count: int
    event_time_s: float
    guard_residual: float
    event_flow: np.ndarray
    source_interpolated_event_time_s: float


@dataclass(frozen=True, slots=True)
class _Authority:
    trajectory: _EventTrajectory
    histories: dict[str, np.ndarray]
    event_cases: dict[str, EventConditionedGramian]


@dataclass(frozen=True, slots=True)
class _EvidenceParts:
    """Intermediate results shared by the portable report and raw arrays."""

    base: _Authority
    additivity: float
    zero_maximum: float
    input_trials: list[dict[str, Any]]
    input_arrays: list[np.ndarray]
    integration_trials: list[dict[str, Any]]
    integration_arrays: list[np.ndarray]
    pulse_records: list[dict[str, Any]]
    pulse_predicted: np.ndarray
    pulse_observed: np.ndarray
    frozen_records: list[dict[str, Any]]
    varying_windows: np.ndarray
    frozen_windows: np.ndarray


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_report() -> dict[str, Any]:
    report = json.loads(SOURCE_REPORT_PATH.read_text(encoding="utf-8"))
    if report.get("schema_version") != "proximal-distal-phase-event-stability/v1":
        raise ValueError("source phase/event report schema is not qualified")
    if report["reference_event"].get("crossing_count") != 1:
        raise ValueError("source phase/event trajectory must have one crossing")
    return report


def _state_scales() -> StateScales:
    values = tuple(
        float(value) for value in _source_report()["registration"]["state_scales"]
    )
    return StateScales(values)


def _commands(dt_s: float) -> np.ndarray:
    horizon = int(round(SOURCE_HORIZON_S / dt_s))
    if not math.isclose(horizon * dt_s, SOURCE_HORIZON_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("integration step must divide the source horizon")
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    return program.controls(horizon, dt_s)


def _linearization_config(
    dt_s: float, control_step_multiplier: float = 1.0
) -> StepLinearizationConfig:
    return StepLinearizationConfig(
        dt_s=dt_s,
        state_steps=tuple(STATE_STEPS),
        control_steps=tuple(CONTROL_STEPS * control_step_multiplier),
        state_scales=_state_scales(),
        control_scales=CONTROL_SCALES,
    )


def _append_refined_event_step(
    prefix: TrajectoryLinearization,
    final_step: StepLinearization,
    refined: RefinedCrossing,
    event_control: np.ndarray,
    dt_s: float,
) -> tuple[TrajectoryLinearization, np.ndarray]:
    """Append the exact partial event step to a full-step prefix."""

    durations = np.concatenate(
        (np.full(prefix.controls.shape[0], dt_s), [refined.partial_dt_s])
    )
    trajectory = TrajectoryLinearization(
        time_s=np.concatenate((prefix.time_s, [refined.time_s])),
        state=np.vstack((prefix.state, refined.state)),
        controls=np.vstack((prefix.controls, event_control)),
        state_matrices=np.concatenate(
            (prefix.state_matrices, final_step.state_matrix[np.newaxis]), axis=0
        ),
        input_matrices=np.concatenate(
            (prefix.input_matrices, final_step.input_matrix[np.newaxis]), axis=0
        ),
        scaled_state_matrices=np.concatenate(
            (
                prefix.scaled_state_matrices,
                final_step.scaled_state_matrix[np.newaxis],
            ),
            axis=0,
        ),
        scaled_sample_input_matrices=np.concatenate(
            (
                prefix.scaled_sample_input_matrices,
                final_step.scaled_sample_input_matrix[np.newaxis],
            ),
            axis=0,
        ),
        scaled_energy_input_matrices=np.concatenate(
            (
                prefix.scaled_energy_input_matrices,
                final_step.scaled_energy_input_matrix[np.newaxis],
            ),
            axis=0,
        ),
    )
    return trajectory, durations


def _event_trajectory(
    *,
    dt_s: float,
    control_step_multiplier: float,
) -> _EventTrajectory:
    params = GolfModelParams.default()
    initial = np.array([*INITIAL_Q, 0.0, 0.0], dtype=float)
    controls = _commands(dt_s)
    time_s, state = rollout_state_history(
        params,
        initial_state=initial,
        controls=controls,
        dt_s=dt_s,
    )
    crossing = first_positive_crossing(time_s, state @ GUARD_GRADIENT)
    if crossing.crossing_count != 1 or crossing.sample_index is None:
        raise ValueError("trajectory must retain exactly one registered crossing")
    if crossing.time_s is None:
        raise ValueError("registered crossing time must be available")
    index = crossing.sample_index
    scales = _state_scales()
    prefix = linearize_trajectory(
        params=params,
        initial_state=initial,
        controls=controls[:index],
        dt_s=dt_s,
        state_steps=STATE_STEPS,
        control_steps=CONTROL_STEPS * control_step_multiplier,
        state_scales=scales,
        control_scales=CONTROL_SCALES,
    )
    refined = refine_guard_crossing(
        params=params,
        state_before=prefix.state[-1],
        control=controls[index],
        time_before_s=float(prefix.time_s[-1]),
        bracket_dt_s=dt_s,
        config=GuardCrossingConfig(
            guard_gradient=tuple(GUARD_GRADIENT),
            guard_tolerance=GUARD_RESIDUAL_GATE / 10.0,
            time_tolerance_s=1e-13,
            transversality_threshold=TRANSVERSALITY_THRESHOLD_PER_S,
        ),
    )
    if refined.status != "transverse_candidate":
        raise ValueError("registered event must remain transverse")
    final_step = step_linearization(
        params=params,
        state=prefix.state[-1],
        control=controls[index],
        time_s=float(prefix.time_s[-1]),
        config=_linearization_config(refined.partial_dt_s, control_step_multiplier),
    )
    trajectory, durations = _append_refined_event_step(
        prefix,
        final_step,
        refined,
        controls[index],
        dt_s,
    )
    flow = state_derivative(params, refined.state, controls[index])
    return _EventTrajectory(
        trajectory=trajectory,
        step_durations_s=durations,
        crossing_count=crossing.crossing_count,
        event_time_s=refined.time_s,
        guard_residual=abs(refined.guard_residual),
        event_flow=flow,
        source_interpolated_event_time_s=crossing.time_s,
    )


def _condition_event(
    gramian: np.ndarray, event: _EventTrajectory
) -> EventConditionedGramian:
    scales = _state_scales().array
    scaled_flow = event.event_flow / scales
    scaled_gradient = GUARD_GRADIENT * scales
    return event_conditioned_gramian(
        gramian,
        event_flow=scaled_flow,
        guard_gradient=scaled_gradient,
        transversality_threshold=TRANSVERSALITY_THRESHOLD_PER_S,
    )


def _authority(event: _EventTrajectory) -> _Authority:
    state = event.trajectory.scaled_state_matrices
    control = event.trajectory.scaled_energy_input_matrices
    masks = {
        "full": None,
        "shoulder_only": np.array([1.0, 0.0]),
        "wrist_only": np.array([0.0, 1.0]),
        "zero_input": np.array([0.0, 0.0]),
    }
    histories = {
        name: reachability_history(state, control, channel_mask=mask)
        for name, mask in masks.items()
    }
    events = {
        name: _condition_event(history[-1], event)
        for name, history in histories.items()
    }
    return _Authority(trajectory=event, histories=histories, event_cases=events)


def _relative_residual(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _spectrum_summary(matrix: np.ndarray) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[::-1]
    leading = max(float(eigenvalues[0]), 0.0)
    negative_floor = -max(RANK_ABSOLUTE_TOLERANCE, leading * 1e-12)
    if float(eigenvalues[-1]) < negative_floor:
        raise ValueError("Gramian lost positive-semidefinite closure")
    clipped = np.maximum(eigenvalues, 0.0)
    threshold = max(RANK_ABSOLUTE_TOLERANCE, RANK_RELATIVE_TOLERANCE * leading)
    rank = int(np.count_nonzero(clipped > threshold))
    retained = float(clipped[rank - 1]) if rank else None
    condition = leading / retained if retained is not None else None
    return {
        "dimension": int(matrix.shape[0]),
        "eigenvalues": clipped.tolist(),
        "rank": rank,
        "rank_threshold": threshold,
        "retained_condition_number": condition,
        "trace": float(np.trace(matrix)),
    }


def _channel_payload(authority: _Authority) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, history in authority.histories.items():
        event = authority.event_cases[name]
        if event.tangent_gramian is None:
            raise ValueError("registered event tangent authority must be available")
        payload[name] = {
            "full_state": _spectrum_summary(history[-1]),
            "event_tangent": _spectrum_summary(event.tangent_gramian),
        }
    return payload


def _window_indices(step_count: int) -> list[int]:
    indices = [int(round(fraction * step_count)) for fraction in PHASE_FRACTIONS]
    indices[0] = 0
    indices[-1] = step_count
    if any(
        right <= left for left, right in zip(indices[:-1], indices[1:], strict=True)
    ):
        raise ValueError("phase windows must contain at least one step")
    return indices


def _frozen_countermodels(
    event: _EventTrajectory,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    trajectory = event.trajectory
    indices = _window_indices(trajectory.controls.shape[0])
    varying_arrays: list[np.ndarray] = []
    frozen_arrays: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    params = GolfModelParams.default()
    for window_index, (start, end) in enumerate(
        zip(indices[:-1], indices[1:], strict=True)
    ):
        varying = reachability_history(
            trajectory.scaled_state_matrices[start:end],
            trajectory.scaled_energy_input_matrices[start:end],
        )[-1]
        duration_count = end - start
        frozen_state = np.repeat(
            trajectory.scaled_state_matrices[start][np.newaxis],
            duration_count,
            axis=0,
        )
        frozen_input = np.repeat(
            trajectory.scaled_energy_input_matrices[start][np.newaxis],
            duration_count,
            axis=0,
        )
        if not np.allclose(
            event.step_durations_s[start:end], event.step_durations_s[start]
        ):
            last = step_linearization(
                params=params,
                state=trajectory.state[start],
                control=trajectory.controls[start],
                time_s=float(trajectory.time_s[start]),
                config=_linearization_config(float(event.step_durations_s[end - 1])),
            )
            frozen_state[-1] = last.scaled_state_matrix
            frozen_input[-1] = last.scaled_energy_input_matrix
        frozen = reachability_history(frozen_state, frozen_input)[-1]
        varying_arrays.append(varying)
        frozen_arrays.append(frozen)
        records.append(
            {
                "window_index": window_index,
                "start_phase_fraction": PHASE_FRACTIONS[window_index],
                "end_phase_fraction": PHASE_FRACTIONS[window_index + 1],
                "elapsed_time_s": float(np.sum(event.step_durations_s[start:end])),
                "same_phase_and_horizon": True,
                "trajectory_varying": _spectrum_summary(varying),
                "frozen_local": _spectrum_summary(frozen),
                "relative_gramian_difference": _relative_residual(frozen, varying),
                "interpretation": "matched-window structural countermodel only",
            }
        )
    return records, np.stack(varying_arrays), np.stack(frozen_arrays)


def _direct_pulses(
    authority: _Authority,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    trajectory = authority.trajectory.trajectory
    durations = authority.trajectory.step_durations_s
    step_count = trajectory.controls.shape[0]
    predicted_rows: list[np.ndarray] = []
    observed_rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for phase_fraction in PULSE_FRACTIONS:
        pulse_step = min(int(round(phase_fraction * step_count)), step_count - 1)
        for channel_index, channel_name in enumerate(("shoulder", "wrist")):
            predicted = propagated_terminal_input_sensitivity(
                trajectory.scaled_state_matrices,
                trajectory.scaled_energy_input_matrices,
                pulse_step=pulse_step,
                channel_index=channel_index,
            )
            observed = direct_variable_terminal_pulse_sensitivity(
                params=GolfModelParams.default(),
                initial_state=trajectory.state[0],
                controls=trajectory.controls,
                step_durations_s=durations,
                state_scales=_state_scales(),
                control_scales=CONTROL_SCALES,
                request=PulseSensitivityRequest(
                    pulse_step=pulse_step,
                    channel_index=channel_index,
                    perturbation_scale=PULSE_PERTURBATION_SCALE,
                ),
            )
            residual = float(np.max(np.abs(observed - predicted)))
            if residual >= DIRECT_PULSE_RESIDUAL_GATE:
                raise ValueError("direct pulse residual exceeds its raw gate")
            predicted_rows.append(predicted)
            observed_rows.append(observed)
            records.append(
                {
                    "phase_fraction": phase_fraction,
                    "pulse_step": pulse_step,
                    "channel": channel_name,
                    "maximum_abs_residual": residual,
                }
            )
    return records, np.stack(predicted_rows), np.stack(observed_rows)


def _equivalent_unit_residual(event: _EventTrajectory) -> float:
    trajectory = event.trajectory
    state_units = np.diag(np.full(4, 180.0 / np.pi))
    control_units = np.diag([1000.0, 1000.0])
    converted_state_scales = StateScales(
        tuple(state_units.diagonal() * _state_scales().array)
    )
    converted_control_scales = ControlScales(
        tuple(control_units.diagonal() * CONTROL_SCALES.array)
    )
    maximum = 0.0
    for index, duration in enumerate(event.step_durations_s):
        scaled = scale_step_matrices(
            state_units @ trajectory.state_matrices[index] @ np.linalg.inv(state_units),
            state_units
            @ trajectory.input_matrices[index]
            @ np.linalg.inv(control_units),
            dt_s=float(duration),
            state_scales=converted_state_scales,
            control_scales=converted_control_scales,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(scaled[0] - trajectory.scaled_state_matrices[index]))),
            float(
                np.max(
                    np.abs(scaled[2] - trajectory.scaled_energy_input_matrices[index])
                )
            ),
        )
    if maximum >= EQUIVALENT_UNIT_RESIDUAL_GATE:
        raise ValueError("equivalent-unit residual exceeds its raw gate")
    return maximum


def _source_identity() -> dict[str, Any]:
    return {
        "phase_event_array_sha256": _sha256(SOURCE_ARRAY_PATH),
        "phase_event_report_schema": _source_report()["schema_version"],
        "phase_event_report_sha256": _sha256(SOURCE_REPORT_PATH),
        "required_parent_pr": 9117,
        "required_parent_issue": 9116,
    }


def _report_payload(parts: _EvidenceParts) -> dict[str, Any]:
    base = parts.base
    event = base.trajectory
    full_event = base.event_cases["full"]
    if full_event.tangent_basis is None or full_event.tangent_gramian is None:
        raise ValueError("event-conditioned authority must remain available")
    source_event_time = float(_source_report()["reference_event"]["time_s"])
    return {
        "schema_version": SCHEMA_VERSION,
        "model_tier": "analytical_double_pendulum",
        "source_identity": _source_identity(),
        "registration": {
            "state_coordinates": [
                "shoulder_angle_rad",
                "wrist_relative_angle_rad",
                "shoulder_rate_rad_s",
                "wrist_relative_rate_rad_s",
            ],
            "control_coordinates": ["shoulder_torque_nm", "wrist_torque_nm"],
            "state_scales": list(_state_scales().values),
            "control_scales_nm": list(CONTROL_SCALES.values),
            "state_steps": STATE_STEPS.tolist(),
            "control_steps_nm": CONTROL_STEPS.tolist(),
            "base_dt_s": BASE_DT_S,
            "input_normalization": "continuous_energy_equivalent_Bd_divided_by_sqrt_dt",
            "rank_absolute_tolerance": RANK_ABSOLUTE_TOLERANCE,
            "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
            "guard": "theta_shoulder + theta_wrist_relative = 0, positive crossing",
            "guard_residual_gate": GUARD_RESIDUAL_GATE,
            "transversality_threshold_per_s": TRANSVERSALITY_THRESHOLD_PER_S,
        },
        "event_conditioned_authority": {
            "status": full_event.status,
            "unique_crossing": event.crossing_count == 1,
            "full_state_dimension": 4,
            "tangent_dimension": int(full_event.tangent_basis.shape[1]),
            "guard_normal_null_direction_is_actuator_loss": False,
            "event_time_s": event.event_time_s,
            "source_interpolated_event_time_s": source_event_time,
            "exact_step_minus_source_event_time_s": event.event_time_s
            - source_event_time,
            "guard_residual": event.guard_residual,
            "transversality_per_s": full_event.transversality_per_s,
            "full_state": _spectrum_summary(base.histories["full"][-1]),
            "event_tangent": _spectrum_summary(full_event.tangent_gramian),
        },
        "channel_cases": _channel_payload(base),
        "frozen_local_countermodel": parts.frozen_records,
        "falsification_controls": {
            "zero_input": {"maximum_abs_gramian_entry": parts.zero_maximum},
            "channel_additivity": {"maximum_abs_residual": parts.additivity},
            "direct_pulses": parts.pulse_records,
            "input_step_refinement": parts.input_trials,
            "integration_step_refinement": parts.integration_trials,
            "equivalent_units": {
                "maximum_abs_residual": _equivalent_unit_residual(event)
            },
        },
        "availability": {
            "bounded_control_feasibility": "unavailable",
            "controller_ranking": "unavailable",
            "human_actuator_interpretation": "unavailable",
            "coaching_recommendation": "unavailable",
        },
        "inference_boundary": INFERENCE_BOUNDARY,
        "limitations": [
            "The Jacobians differentiate the same analytical RK4 operator as the parent phase/event study; this is not independent model validation.",
            "The trajectory is synthetic and open loop, with no participant, muscle, fatigue, delay, noise, or bilateral-wrench observations.",
            "Gramian eigenvalues depend on the declared state, torque, energy, and rank scales; raw dimensional rankings are not reported.",
            "The event is a geometric club-vertical guard, not measured club-ball impact.",
        ],
    }


def _array_payload(parts: _EvidenceParts) -> dict[str, np.ndarray]:
    base = parts.base
    event = base.trajectory
    trajectory = event.trajectory
    full_event = base.event_cases["full"]
    if (
        full_event.projection is None
        or full_event.tangent_basis is None
        or full_event.tangent_gramian is None
    ):
        raise ValueError("full event case must retain its transverse projection")
    return {
        "time_s": trajectory.time_s,
        "step_durations_s": event.step_durations_s,
        "state": trajectory.state,
        "controls": trajectory.controls,
        "physical_state_matrices": trajectory.state_matrices,
        "physical_input_matrices": trajectory.input_matrices,
        "scaled_state_matrices": trajectory.scaled_state_matrices,
        "scaled_sample_input_matrices": trajectory.scaled_sample_input_matrices,
        "scaled_energy_input_matrices": trajectory.scaled_energy_input_matrices,
        "full_gramian_history": base.histories["full"],
        "shoulder_gramian_history": base.histories["shoulder_only"],
        "wrist_gramian_history": base.histories["wrist_only"],
        "zero_gramian_history": base.histories["zero_input"],
        "event_projection": full_event.projection,
        "event_tangent_basis": full_event.tangent_basis,
        "event_tangent_gramian": full_event.tangent_gramian,
        "direct_pulse_predicted": parts.pulse_predicted,
        "direct_pulse_observed": parts.pulse_observed,
        "input_refinement_event_gramians": np.stack(parts.input_arrays),
        "integration_refinement_event_gramians": np.stack(parts.integration_arrays),
        "trajectory_varying_window_gramians": parts.varying_windows,
        "frozen_local_window_gramians": parts.frozen_windows,
    }


def _build_parts() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache: dict[tuple[float, float], _Authority] = {}

    def authority(dt_s: float, input_multiplier: float) -> _Authority:
        key = (dt_s, input_multiplier)
        if key not in cache:
            cache[key] = _authority(
                _event_trajectory(
                    dt_s=dt_s,
                    control_step_multiplier=input_multiplier,
                )
            )
        return cache[key]

    base = authority(BASE_DT_S, 1.0)
    full_history = base.histories["full"]
    shoulder_history = base.histories["shoulder_only"]
    wrist_history = base.histories["wrist_only"]
    zero_history = base.histories["zero_input"]
    additivity = float(np.max(np.abs(full_history - shoulder_history - wrist_history)))
    if additivity >= ADDITIVITY_RESIDUAL_GATE:
        raise ValueError("channel additivity residual exceeds its raw gate")
    zero_maximum = float(np.max(np.abs(zero_history)))
    if zero_maximum != 0.0:
        raise ValueError("zero-input authority must remain exactly zero")

    base_event_gramian = full_history[-1]
    input_trials: list[dict[str, Any]] = []
    input_arrays: list[np.ndarray] = []
    for multiplier in CONTROL_STEP_MULTIPLIERS:
        candidate = authority(BASE_DT_S, multiplier).histories["full"][-1]
        residual = _relative_residual(candidate, base_event_gramian)
        if residual >= INPUT_REFINEMENT_GATE:
            raise ValueError("input-step refinement residual exceeds its raw gate")
        input_trials.append(
            {
                "control_step_multiplier": multiplier,
                "relative_event_gramian_residual": residual,
            }
        )
        input_arrays.append(candidate)

    integration_trials: list[dict[str, Any]] = []
    integration_arrays: list[np.ndarray] = []
    for multiplier in INTEGRATION_STEP_MULTIPLIERS:
        candidate_authority = authority(BASE_DT_S * multiplier, 1.0)
        candidate = candidate_authority.histories["full"][-1]
        residual = _relative_residual(candidate, base_event_gramian)
        if residual >= INTEGRATION_REFINEMENT_GATE:
            raise ValueError(
                "integration-step refinement residual exceeds its raw gate"
            )
        integration_trials.append(
            {
                "integration_step_multiplier": multiplier,
                "dt_s": BASE_DT_S * multiplier,
                "event_time_s": candidate_authority.trajectory.event_time_s,
                "relative_event_gramian_residual": residual,
            }
        )
        integration_arrays.append(candidate)

    pulse_records, pulse_predicted, pulse_observed = _direct_pulses(base)
    frozen_records, varying_windows, frozen_windows = _frozen_countermodels(
        base.trajectory
    )
    full_event = base.event_cases["full"]
    if (
        full_event.projection is None
        or full_event.tangent_basis is None
        or full_event.tangent_gramian is None
    ):
        raise ValueError("event-conditioned authority must remain available")

    parts = _EvidenceParts(
        base=base,
        additivity=additivity,
        zero_maximum=zero_maximum,
        input_trials=input_trials,
        input_arrays=input_arrays,
        integration_trials=integration_trials,
        integration_arrays=integration_arrays,
        pulse_records=pulse_records,
        pulse_predicted=pulse_predicted,
        pulse_observed=pulse_observed,
        frozen_records=frozen_records,
        varying_windows=varying_windows,
        frozen_windows=frozen_windows,
    )
    return _report_payload(parts), _array_payload(parts)


def build_evidence() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build portable summaries and full-precision reviewer arrays."""

    report, arrays = _build_parts()
    validate_report(report)
    return canonicalize_published_numbers(report), arrays


def build_report() -> dict[str, Any]:
    """Build the deterministic portable report."""

    report, _ = build_evidence()
    return report


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Fail closed when provenance, controls, or inference boundaries drift."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected trajectory control-authority schema")
    if report.get("source_identity") != _source_identity():
        raise ValueError("parent phase/event source identity is stale")
    event = report.get("event_conditioned_authority", {})
    if event.get("status") != "transverse" or not event.get("unique_crossing"):
        raise ValueError("event must remain uniquely transverse")
    if event.get("full_state_dimension") != 4 or event.get("tangent_dimension") != 3:
        raise ValueError("event tangent dimension must remain explicitly three")
    if event.get("guard_normal_null_direction_is_actuator_loss") is not False:
        raise ValueError("guard-normal null direction cannot be actuator loss")
    if event.get("guard_residual", math.inf) >= GUARD_RESIDUAL_GATE:
        raise ValueError("event guard residual exceeds its gate")
    controls = report.get("falsification_controls", {})
    if controls.get("zero_input", {}).get("maximum_abs_gramian_entry") != 0.0:
        raise ValueError("zero-input authority must remain zero")
    if (
        controls.get("channel_additivity", {}).get("maximum_abs_residual", math.inf)
        >= ADDITIVITY_RESIDUAL_GATE
    ):
        raise ValueError("channel additivity control failed")
    pulses = controls.get("direct_pulses", [])
    if len(pulses) != 6 or any(
        trial.get("maximum_abs_residual", math.inf) >= DIRECT_PULSE_RESIDUAL_GATE
        for trial in pulses
    ):
        raise ValueError("direct pulse control failed")
    input_trials = controls.get("input_step_refinement", [])
    if len(input_trials) != 3 or any(
        trial.get("relative_event_gramian_residual", math.inf) >= INPUT_REFINEMENT_GATE
        for trial in input_trials
    ):
        raise ValueError("input-step refinement control failed")
    integration_trials = controls.get("integration_step_refinement", [])
    if len(integration_trials) != 3 or any(
        trial.get("relative_event_gramian_residual", math.inf)
        >= INTEGRATION_REFINEMENT_GATE
        for trial in integration_trials
    ):
        raise ValueError("integration-step refinement control failed")
    if (
        controls.get("equivalent_units", {}).get("maximum_abs_residual", math.inf)
        >= EQUIVALENT_UNIT_RESIDUAL_GATE
    ):
        raise ValueError("equivalent-unit control failed")
    channels = report.get("channel_cases", {})
    if set(channels) != {"full", "shoulder_only", "wrist_only", "zero_input"}:
        raise ValueError("all four channel cases must remain registered")
    frozen = report.get("frozen_local_countermodel", [])
    if len(frozen) != 4 or not all(
        case.get("same_phase_and_horizon") for case in frozen
    ):
        raise ValueError("frozen-local matched windows are incomplete")
    availability = report.get("availability", {})
    if any(value != "unavailable" for value in availability.values()):
        raise ValueError("unqualified conclusions must remain unavailable")
    boundary = str(report.get("inference_boundary", "")).lower()
    for phrase in (
        "local first-order",
        "countermodel",
        "not a global",
        "does not establish",
        "human",
        "coaching",
    ):
        if phrase not in boundary:
            raise ValueError("inference boundary is incomplete")
    return {
        "channel_case_count": len(channels),
        "direct_pulse_count": len(pulses),
        "frozen_window_count": len(frozen),
        "integration_refinement_count": len(integration_trials),
        "input_refinement_count": len(input_trials),
    }


def write_evidence() -> None:
    """Write deterministic JSON and compressed full-precision arrays."""

    report, arrays = build_evidence()
    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    np.savez_compressed(ARRAY_PATH, **arrays)  # type: ignore[arg-type]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("write", "validate"), default="write"
    )
    args = parser.parse_args()
    if args.mode == "write":
        write_evidence()
        return
    registered = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    expected = build_report()
    if registered != expected:
        raise ValueError("registered trajectory control-authority report is stale")
    print(json.dumps(validate_report(registered), sort_keys=True))


if __name__ == "__main__":
    main()
