"""Generate or validate finite-time and event-sensitivity evidence."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import json
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
    Crossing,
    DirectEventSensitivity,
    EventSensitivity,
    FiniteTimeSpectra,
    PeriodicityGate,
    StateScales,
    TransitionRollout,
    direct_event_time_control,
    direct_transition_control,
    event_time_sensitivity,
    finite_time_spectra,
    first_positive_crossing,
    normalized_transition,
    periodicity_gate,
    propagate_state_transition,
    rollout_state_history,
    saltation_matrix,
    state_derivative,
)
from scripts.research.proximal_distal_energy.run_experiments import (
    DT,
    HORIZON,
    INITIAL_Q,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams

DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"
REPORT_PATH = DATA_DIR / "phase_event_stability.json"
ARRAY_PATH = DATA_DIR / "phase_event_stability.npz"
SCHEMA_VERSION = "proximal-distal-phase-event-stability/v1"
ANALYSIS_DT_S = DT / 8.0
ANALYSIS_HORIZON = HORIZON * 8
STATE_STEPS = np.array([1e-6, 1e-6, 1e-5, 1e-5], dtype=float)
STEP_MULTIPLIERS = (0.1, 1.0, 10.0)
PERTURBATION_SCALES = (5e-7, 1e-6, 2e-6)
TRANSVERSALITY_THRESHOLD_PER_S = 1e-6
PERIODICITY_TOLERANCE = 1e-6
GUARD_GRADIENT = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
INFERENCE_BOUNDARY = (
    "This is a local finite-window diagnostic on one synthetic, open-loop, "
    "nonperiodic analytical trajectory. Finite-time amplification is not "
    "asymptotic or global stability; event sensitivity is not neural timing "
    "demand; and neither result establishes anatomy, passive negative torque, "
    "participant robustness, human strategy, or coaching guidance."
)


@dataclass(frozen=True, slots=True)
class _AnalysisState:
    params: GolfModelParams
    initial: np.ndarray
    controls: np.ndarray
    event_controls: np.ndarray
    crossing: Crossing
    event_time_s: float
    scales: StateScales
    nominal: TransitionRollout
    nominal_event_transition: np.ndarray
    nominal_event_state: np.ndarray
    normalized_maps: np.ndarray
    spectra: FiniteTimeSpectra
    event_flow: np.ndarray
    step_trials: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _AdverseControls:
    grazing: EventSensitivity
    identity: np.ndarray
    time_guard_saltation: np.ndarray
    corrupted_saltation: np.ndarray
    periodicity: PeriodicityGate
    unit_residual: float
    time_unit_residual: float


@dataclass(frozen=True, slots=True)
class _EvidenceParts:
    state: _AnalysisState
    implicit: EventSensitivity
    direct_event_trials: list[dict[str, Any]]
    direct_event_derivatives: list[np.ndarray]
    direct_transition_trials: list[dict[str, Any]]
    direct_transition_matrices: list[np.ndarray]
    adverse: _AdverseControls


def _interpolate(time_s: np.ndarray, values: np.ndarray, target_s: float) -> np.ndarray:
    """Linearly interpolate a vector or matrix history on a strict grid."""

    if values.shape[0] != time_s.size:
        raise ValueError("history and time grid must have equal lengths")
    if not time_s[0] <= target_s <= time_s[-1]:
        raise ValueError("interpolation target must lie on the time grid")
    flat = values.reshape(values.shape[0], -1)
    result = np.asarray(
        [
            np.interp(target_s, time_s, flat[:, column])
            for column in range(flat.shape[1])
        ],
        dtype=float,
    )
    return result.reshape(values.shape[1:])


def _state_scales(state: np.ndarray) -> StateScales:
    rate_scales = np.maximum(np.max(np.abs(state[:, 2:]), axis=0), 1.0)
    return StateScales((np.pi, np.pi, float(rate_scales[0]), float(rate_scales[1])))


def _direct_event_payload(result: DirectEventSensitivity) -> dict[str, Any]:
    return {
        "crossing_statuses": list(result.crossing_statuses),
        "derivative_s_per_scaled_state": (
            None
            if result.derivative_s_per_scaled_state is None
            else result.derivative_s_per_scaled_state.tolist()
        ),
        "status": result.status,
    }


def _checkpoint_payload(
    *,
    time_s: np.ndarray,
    transitions: np.ndarray,
    event_time_s: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        target = fraction * event_time_s
        matrix = _interpolate(time_s, transitions, target)
        singular = np.linalg.svd(matrix, compute_uv=False)
        exponent = (
            np.full(singular.shape, np.nan)
            if target == 0.0
            else np.log(singular) / target
        )
        records.append(
            {
                "finite_time_exponents_per_s": [
                    None if not np.isfinite(value) else float(value)
                    for value in exponent
                ],
                "maximum_amplification": float(singular[0]),
                "minimum_amplification": float(singular[-1]),
                "phase_fraction": fraction,
                "singular_values": singular.tolist(),
                "time_s": target,
            }
        )
    return records


def _transition_refinement(
    params: GolfModelParams,
    initial: np.ndarray,
    event_controls: np.ndarray,
    event_time_s: float,
) -> tuple[TransitionRollout, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rollouts = {
        multiplier: propagate_state_transition(
            params,
            initial_state=initial,
            controls=event_controls,
            dt_s=ANALYSIS_DT_S,
            state_steps=STATE_STEPS * multiplier,
        )
        for multiplier in STEP_MULTIPLIERS
    }
    nominal = rollouts[1.0]
    event_transition = _interpolate(nominal.time_s, nominal.transition, event_time_s)
    event_state = _interpolate(nominal.time_s, nominal.state, event_time_s)
    trials = [
        {
            "event_transition_max_abs_residual_from_nominal": float(
                np.max(
                    np.abs(
                        _interpolate(
                            rollouts[multiplier].time_s,
                            rollouts[multiplier].transition,
                            event_time_s,
                        )
                        - event_transition
                    )
                )
            ),
            "state_step_multiplier": multiplier,
        }
        for multiplier in STEP_MULTIPLIERS
    ]
    return nominal, event_transition, event_state, trials


def _build_analysis_state() -> _AnalysisState:
    params = GolfModelParams.default()
    initial = np.array([*INITIAL_Q, 0.0, 0.0], dtype=float)
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    controls = program.controls(ANALYSIS_HORIZON, ANALYSIS_DT_S)
    full_time, full_state = rollout_state_history(
        params, initial_state=initial, controls=controls, dt_s=ANALYSIS_DT_S
    )
    crossing = first_positive_crossing(full_time, full_state[:, 0] + full_state[:, 1])
    if crossing.crossing_count != 1 or crossing.time_s is None:
        raise RuntimeError("registered club-vertical guard must cross exactly once")
    if crossing.sample_index is None or crossing.fraction is None:
        raise RuntimeError("registered crossing must include interpolation metadata")
    event_controls = controls[: crossing.sample_index + 1]
    scales = _state_scales(full_state[: crossing.sample_index + 2])
    nominal, event_transition, event_state, step_trials = _transition_refinement(
        params, initial, event_controls, crossing.time_s
    )
    normalized_maps = np.asarray(
        [normalized_transition(matrix, scales) for matrix in nominal.transition]
    )
    spectra = finite_time_spectra(normalized_maps, nominal.time_s)
    event_flow = state_derivative(
        params, event_state, event_controls[crossing.sample_index]
    )
    return _AnalysisState(
        params=params,
        initial=initial,
        controls=controls,
        event_controls=event_controls,
        crossing=crossing,
        event_time_s=crossing.time_s,
        scales=scales,
        nominal=nominal,
        nominal_event_transition=event_transition,
        nominal_event_state=event_state,
        normalized_maps=normalized_maps,
        spectra=spectra,
        event_flow=event_flow,
        step_trials=step_trials,
    )


def _implicit_sensitivity(state: _AnalysisState) -> EventSensitivity:
    implicit = event_time_sensitivity(
        state.nominal_event_transition,
        event_flow=state.event_flow,
        guard_gradient=GUARD_GRADIENT,
        state_scales=state.scales,
        transversality_threshold=TRANSVERSALITY_THRESHOLD_PER_S,
    )
    if implicit.derivative_s_per_scaled_state is None:
        raise RuntimeError("registered club-vertical guard must be transverse")
    return implicit


def _direct_event_controls(
    state: _AnalysisState, implicit: EventSensitivity
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    trials: list[dict[str, Any]] = []
    derivatives: list[np.ndarray] = []
    if implicit.derivative_s_per_scaled_state is None:
        raise RuntimeError("implicit event derivative must be available")
    for perturbation_scale in PERTURBATION_SCALES:
        result = direct_event_time_control(
            state.params,
            initial_state=state.initial,
            controls=state.controls,
            dt_s=ANALYSIS_DT_S,
            state_scales=state.scales,
            perturbation_scale=perturbation_scale,
            guard_gradient=GUARD_GRADIENT,
        )
        payload = _direct_event_payload(result)
        payload["perturbation_scale"] = perturbation_scale
        if result.derivative_s_per_scaled_state is not None:
            derivatives.append(result.derivative_s_per_scaled_state)
            payload["maximum_abs_residual_from_implicit_s"] = float(
                np.max(
                    np.abs(
                        result.derivative_s_per_scaled_state
                        - implicit.derivative_s_per_scaled_state
                    )
                )
            )
        trials.append(payload)
    return trials, derivatives


def _direct_transition_controls(
    state: _AnalysisState,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    trials: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    predicted = normalized_transition(state.nominal.transition[-1], state.scales)
    for perturbation_scale in PERTURBATION_SCALES:
        direct = direct_transition_control(
            state.params,
            initial_state=state.initial,
            controls=state.event_controls,
            dt_s=ANALYSIS_DT_S,
            state_scales=state.scales,
            perturbation_scale=perturbation_scale,
        )
        matrices.append(direct)
        trials.append(
            {
                "maximum_abs_residual": float(np.max(np.abs(direct - predicted))),
                "perturbation_scale": perturbation_scale,
            }
        )
    return trials, matrices


def _adverse_controls(state: _AnalysisState) -> _AdverseControls:
    grazing_gradient = np.array(
        [state.event_flow[1], -state.event_flow[0], 0.0, 0.0], dtype=float
    )
    grazing = event_time_sensitivity(
        state.nominal_event_transition,
        event_flow=state.event_flow,
        guard_gradient=grazing_gradient,
        state_scales=state.scales,
        transversality_threshold=TRANSVERSALITY_THRESHOLD_PER_S,
    )
    identity = np.eye(4)
    time_guard = saltation_matrix(
        reset_jacobian=identity,
        flow_before=state.event_flow,
        flow_after=state.event_flow,
        guard_gradient=np.zeros(4),
        guard_time_derivative=1.0,
        reset_time_derivative=np.zeros(4),
    )
    corrupted_reset = identity.copy()
    corrupted_reset[0, 0] = 0.95
    corrupted = saltation_matrix(
        reset_jacobian=corrupted_reset,
        flow_before=state.event_flow,
        flow_after=state.event_flow,
        guard_gradient=np.zeros(4),
        guard_time_derivative=1.0,
        reset_time_derivative=np.zeros(4),
    )
    periodicity = periodicity_gate(
        initial_state=state.initial,
        final_state=state.nominal_event_state,
        state_scales=state.scales,
        tolerance=PERIODICITY_TOLERANCE,
    )
    degree_change = np.diag(np.full(4, 180.0 / np.pi))
    converted = (
        degree_change @ state.nominal_event_transition @ np.linalg.inv(degree_change)
    )
    converted_scales = StateScales(tuple(degree_change.diagonal() * state.scales.array))
    unit_residual = float(
        np.max(
            np.abs(
                normalized_transition(converted, converted_scales)
                - normalized_transition(state.nominal_event_transition, state.scales)
            )
        )
    )
    exponents = state.spectra.exponents_per_s[1:]
    time_unit_residual = float(
        np.nanmax(np.abs(exponents - (exponents / 1000.0) * 1000.0))
    )
    return _AdverseControls(
        grazing=grazing,
        identity=identity,
        time_guard_saltation=time_guard,
        corrupted_saltation=corrupted,
        periodicity=periodicity,
        unit_residual=unit_residual,
        time_unit_residual=time_unit_residual,
    )


def _registration_payload(state: _AnalysisState) -> dict[str, Any]:
    return {
        "control_program": {
            "onset_s": 0.10,
            "shoulder_torque_nm": 60.0,
            "wrist_drive_nm": 15.0,
            "wrist_restrain_nm": 10.0,
        },
        "analysis_dt_s": ANALYSIS_DT_S,
        "analysis_horizon_steps": ANALYSIS_HORIZON,
        "source_reference_dt_s": DT,
        "source_reference_horizon_steps": HORIZON,
        "initial_state": state.initial.tolist(),
        "state_coordinates": [
            "shoulder_angle_rad",
            "wrist_relative_angle_rad",
            "shoulder_rate_rad_s",
            "wrist_relative_rate_rad_s",
        ],
        "state_scales": state.scales.values,
        "state_steps": STATE_STEPS.tolist(),
        "state_step_multipliers": list(STEP_MULTIPLIERS),
        "direct_perturbation_scales": list(PERTURBATION_SCALES),
        "guard": "theta_shoulder + theta_wrist_relative = 0, positive crossing",
        "transversality_threshold_per_s": TRANSVERSALITY_THRESHOLD_PER_S,
        "periodicity_tolerance": PERIODICITY_TOLERANCE,
    }


def _finite_time_payload(state: _AnalysisState) -> dict[str, Any]:
    return {
        "interpretation": "local finite-window amplification on a nonperiodic trajectory",
        "event_transition_matrix_scaled": normalized_transition(
            state.nominal_event_transition, state.scales
        ).tolist(),
        "phase_checkpoints": _checkpoint_payload(
            time_s=state.nominal.time_s,
            transitions=state.normalized_maps,
            event_time_s=state.event_time_s,
        ),
        "maximum_observed_amplification": float(
            np.max(state.spectra.singular_values[:, 0])
        ),
        "minimum_observed_amplification": float(
            np.min(state.spectra.singular_values[:, -1])
        ),
    }


def _event_sensitivity_payload(parts: _EvidenceParts) -> dict[str, Any]:
    derivative = parts.implicit.derivative_s_per_scaled_state
    if derivative is None:
        raise RuntimeError("implicit event derivative must be available")
    grazing = parts.adverse.grazing
    return {
        "implicit": {
            "status": parts.implicit.status,
            "transversality_per_s": parts.implicit.transversality_per_s,
            "derivative_s_per_scaled_state": derivative.tolist(),
        },
        "direct_trials": parts.direct_event_trials,
        "constructed_near_grazing_control": {
            "status": grazing.status,
            "transversality_per_s": grazing.transversality_per_s,
            "derivative_s_per_scaled_state": None,
            "construction": "configuration guard normal orthogonal to event configuration velocity",
        },
    }


def _control_payloads(parts: _EvidenceParts) -> dict[str, dict[str, Any]]:
    adverse = parts.adverse
    saltation = {
        "time_guard_identity_reset_max_abs_residual": float(
            np.max(np.abs(adverse.time_guard_saltation - adverse.identity))
        ),
        "corrupted_reset_diagonal": 0.95,
        "corrupted_reset_max_abs_deviation_from_identity": float(
            np.max(np.abs(adverse.corrupted_saltation - adverse.identity))
        ),
    }
    equivalent_units = {
        "radian_to_degree_scaled_transition_max_abs_residual": adverse.unit_residual,
        "second_to_millisecond_exponent_roundtrip_max_abs_residual_per_s": (
            adverse.time_unit_residual
        ),
    }
    periodicity = {
        **asdict(adverse.periodicity),
        "floquet_multipliers": None,
        "reason": "suppressed because the registered trajectory does not close in scaled state",
    }
    return {
        "saltation_controls": saltation,
        "equivalent_unit_controls": equivalent_units,
        "periodicity_gate": periodicity,
    }


def _report_payload(parts: _EvidenceParts) -> dict[str, Any]:
    state = parts.state
    controls = _control_payloads(parts)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_tier": "analytical_double_pendulum",
        "registration": _registration_payload(state),
        "reference_event": {
            "crossing_count": state.crossing.crossing_count,
            "sample_index_before_crossing": state.crossing.sample_index,
            "interpolation_fraction": state.crossing.fraction,
            "time_s": state.event_time_s,
            "state": state.nominal_event_state.tolist(),
            "flow": state.event_flow.tolist(),
        },
        "finite_time_analysis": _finite_time_payload(state),
        "step_refinement": state.step_trials,
        "direct_transition_controls": parts.direct_transition_trials,
        "event_time_sensitivity": _event_sensitivity_payload(parts),
        "saltation_controls": controls["saltation_controls"],
        "equivalent_unit_controls": controls["equivalent_unit_controls"],
        "periodicity_gate": controls["periodicity_gate"],
        "inference_boundary": INFERENCE_BOUNDARY,
        "limitations": [
            "The state-transition map is a local derivative of the registered RK4 trajectory, not an independent physics engine.",
            "Direct perturbation controls share the same equations and integration family; they verify implementation, not model validity.",
            "The club-vertical guard is a geometric delivery countermodel, not measured club-ball impact.",
            "No periodic orbit is registered, so Floquet multipliers, orbital stability, and basins are unavailable.",
            "No participant data, actuator delay, state-estimation error, fatigue, muscle dynamics, or bilateral wrench data enter this slice.",
        ],
    }


def _array_payload(parts: _EvidenceParts) -> dict[str, np.ndarray]:
    state = parts.state
    return {
        "time_s": state.nominal.time_s,
        "state": state.nominal.state,
        "physical_transition": state.nominal.transition,
        "scaled_transition": state.normalized_maps,
        "singular_values": state.spectra.singular_values,
        "finite_time_exponents_per_s": state.spectra.exponents_per_s,
        "direct_transition_matrices": np.asarray(parts.direct_transition_matrices),
        "direct_event_derivatives_s_per_scaled_state": np.asarray(
            parts.direct_event_derivatives
        ),
    }


def build_evidence() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build deterministic evidence and full-precision reviewer arrays."""

    state = _build_analysis_state()
    implicit = _implicit_sensitivity(state)
    event_trials, event_derivatives = _direct_event_controls(state, implicit)
    transition_trials, transition_matrices = _direct_transition_controls(state)
    parts = _EvidenceParts(
        state=state,
        implicit=implicit,
        direct_event_trials=event_trials,
        direct_event_derivatives=event_derivatives,
        direct_transition_trials=transition_trials,
        direct_transition_matrices=transition_matrices,
        adverse=_adverse_controls(state),
    )
    return canonicalize_published_numbers(_report_payload(parts)), _array_payload(parts)


def build_report() -> dict[str, Any]:
    """Build the canonical JSON projection."""

    report, _ = build_evidence()
    return report


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Fail closed on missing controls or overpromoted interpretation."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected phase/event stability schema")
    if report.get("model_tier") != "analytical_double_pendulum":
        raise ValueError("model tier must remain analytical double pendulum")
    event = report["event_time_sensitivity"]
    if event["implicit"]["status"] != "transverse":
        raise ValueError("registered event must remain transverse")
    if any(
        trial["status"] != "available_transverse_candidates"
        for trial in event["direct_trials"]
    ):
        raise ValueError("direct event trials must retain unique crossings")
    if event["constructed_near_grazing_control"]["status"] != "near_grazing":
        raise ValueError("near-grazing killswitch must suppress sensitivity")
    if (
        report["saltation_controls"]["time_guard_identity_reset_max_abs_residual"]
        != 0.0
    ):
        raise ValueError("time-guard identity saltation control failed")
    if (
        report["saltation_controls"]["corrupted_reset_max_abs_deviation_from_identity"]
        <= 0.0
    ):
        raise ValueError("corrupted reset killswitch did not change saltation")
    periodicity = report["periodicity_gate"]
    if periodicity["periodic"] or periodicity["floquet_eligible"]:
        raise ValueError("open downswing must not pass periodicity")
    if periodicity["floquet_multipliers"] is not None:
        raise ValueError("Floquet output must remain suppressed")
    boundary = str(report.get("inference_boundary", "")).lower()
    for phrase in (
        "not asymptotic",
        "not neural",
        "neither result establishes",
        "coaching",
    ):
        if phrase not in boundary:
            raise ValueError("inference boundary is incomplete")
    return {
        "direct_event_trial_count": len(event["direct_trials"]),
        "direct_transition_trial_count": len(report["direct_transition_controls"]),
        "phase_checkpoint_count": len(
            report["finite_time_analysis"]["phase_checkpoints"]
        ),
        "step_trial_count": len(report["step_refinement"]),
    }


def write_evidence() -> None:
    """Write deterministic JSON and full-precision NPZ evidence."""

    report, arrays = build_evidence()
    validate_report(report)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    np.savez_compressed(ARRAY_PATH, **arrays)


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
        raise ValueError("registered phase/event stability report is stale")
    print(json.dumps(validate_report(registered), sort_keys=True))


if __name__ == "__main__":
    main()
