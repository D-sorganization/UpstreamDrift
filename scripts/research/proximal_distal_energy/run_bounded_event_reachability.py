"""Run the registered bounded nonlinear event-reaching study for #9124.

The module keeps target, channel, horizon, event, scaling, and solver identities
explicit so continuation and killswitch cases remain matched.  All torque and
slew limits are declared model scenarios.  Results do not establish human
capacity, controller superiority, passive torque, or coaching guidance.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, replace
from functools import cache
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

from scripts.research.proximal_distal_energy.bounded_event_multiple_shooting import (
    MultipleShootingConfig,
    MultipleShootingResult,
    MultipleShootingStatus,
    solve_bounded_event_multiple_shooting,
)
from scripts.research.proximal_distal_energy.bounded_event_reachability import (
    BoundedEventReachabilityProblem,
    ControlPerturbationBounds,
    EventReplayStatus,
    FeasibilityStatus,
    replay_guard_event,
)
from scripts.research.proximal_distal_energy.phase_event_stability import StateScales
from scripts.research.proximal_distal_energy.numeric_evidence import (
    canonicalize_published_numbers,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    ControlScales,
    GuardCrossingConfig,
)
from src.shared.python.simulation_backends import GolfModelParams

DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"
REPORT_PATH = DATA_DIR / "bounded_event_reachability.json"
ARRAY_PATH = DATA_DIR / "bounded_event_reachability.npz"
PARENT_REPORT_PATH = DATA_DIR / "trajectory_control_authority.json"
PARENT_ARRAY_PATH = DATA_DIR / "trajectory_control_authority.npz"
SCHEMA_VERSION = "proximal-distal-bounded-event-reachability/v1"
BASE_DT_S = 2e-3
HORIZON_S = 0.40
REGISTERED_SEGMENT_COUNT = 4
TANGENT_TOLERANCE = 2e-6
INITIAL_STATE = (-2.2, -1.57, 0.0, 0.0)
STATE_SCALES = StateScales((math.pi, math.pi, 10.0, 10.0))
CONTROL_SCALES = ControlScales((100.0, 100.0))
GUARD = GuardCrossingConfig(
    guard_gradient=(1.0, 1.0, 0.0, 0.0),
    guard_tolerance=1e-10,
    time_tolerance_s=1e-12,
    transversality_threshold=1e-8,
)
REGISTERED_CONFIG = MultipleShootingConfig(
    segment_count=REGISTERED_SEGMENT_COUNT,
    max_iterations=60,
    constraint_tolerance=TANGENT_TOLERANCE,
    objective_tolerance=1e-10,
    seed=0,
    initial_control_fraction=0.0,
)
MULTISTART_OBJECTIVE_SPREAD_GATE = 0.05
INFERENCE_BOUNDARY = (
    "These are bounded nonlinear model-scenario results for one synthetic "
    "analytical double-pendulum trajectory and one geometric delivery guard. "
    "They do not establish global reachability, human torque or torque-rate "
    "capacity, controller superiority, passive torque, participant behavior, "
    "or a coaching recommendation."
)


@dataclass(frozen=True, slots=True)
class TargetCase:
    """One signed guard-tangent angle target relative to the reference event."""

    name: str
    direction: int
    amplitude_rad: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("target name must be nonempty")
        if self.direction not in (-1, 0, 1):
            raise ValueError("target direction must be -1, 0, or 1")
        if not math.isfinite(self.amplitude_rad) or self.amplitude_rad < 0.0:
            raise ValueError("target amplitude must be finite and nonnegative")
        if (self.amplitude_rad == 0.0) != (self.direction == 0):
            raise ValueError("zero target amplitude and direction must coincide")

    @property
    def offset(self) -> np.ndarray:
        signed = self.direction * self.amplitude_rad
        return np.array([signed, -signed, 0.0, 0.0], dtype=float)


@dataclass(frozen=True, slots=True)
class ChannelCase:
    """Matched actuator killswitch scenario."""

    name: str
    bounds: ControlPerturbationBounds

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("channel name must be nonempty")


def registered_targets() -> tuple[TargetCase, ...]:
    """Return the symmetric registered angle-tangent continuation targets."""

    targets = [TargetCase("zero", 0, 0.0)]
    for amplitude, label in ((5e-4, "0p0005"), (1e-3, "0p001"), (2e-3, "0p002")):
        targets.extend(
            (
                TargetCase(f"minus_{label}", -1, amplitude),
                TargetCase(f"plus_{label}", 1, amplitude),
            )
        )
    return tuple(targets)


def registered_channels() -> tuple[ChannelCase, ...]:
    """Return both-channel, killswitch, and zero-authority scenarios."""

    active_torque = 20.0
    active_rate = 10000.0
    return (
        ChannelCase(
            "both",
            ControlPerturbationBounds(
                lower_nm=(-active_torque, -active_torque),
                upper_nm=(active_torque, active_torque),
                max_rate_nm_per_s=(active_rate, active_rate),
            ),
        ),
        ChannelCase(
            "shoulder_only",
            ControlPerturbationBounds(
                lower_nm=(-active_torque, 0.0),
                upper_nm=(active_torque, 0.0),
                max_rate_nm_per_s=(active_rate, 0.0),
            ),
        ),
        ChannelCase(
            "wrist_only",
            ControlPerturbationBounds(
                lower_nm=(0.0, -active_torque),
                upper_nm=(0.0, active_torque),
                max_rate_nm_per_s=(0.0, active_rate),
            ),
        ),
        ChannelCase("zero", ControlPerturbationBounds.zero()),
    )


def study_matrix() -> tuple[tuple[TargetCase, ChannelCase], ...]:
    """Return the complete target-by-channel matched continuation matrix."""

    return tuple(
        (target, channel)
        for target in registered_targets()
        for channel in registered_channels()
    )


def _controls(dt_s: float) -> np.ndarray:
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    horizon = int(round(HORIZON_S / dt_s))
    if not math.isclose(horizon * dt_s, HORIZON_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("dt_s must divide the registered horizon")
    return restrain_then_drive_program(60.0, 15.0, 10.0, 0.10).controls(horizon, dt_s)


@cache
def _reference_event_values() -> tuple[float, ...]:
    replay = replay_guard_event(
        params=GolfModelParams.default(),
        initial_state=INITIAL_STATE,
        controls=_controls(BASE_DT_S),
        dt_s=BASE_DT_S,
        guard=GUARD,
    )
    if replay.status is not EventReplayStatus.TRANSVERSE or replay.state is None:
        raise ValueError("registered reference event must remain transverse")
    return tuple(float(value) for value in replay.state)


def _reference_event_state() -> np.ndarray:
    return np.asarray(_reference_event_values(), dtype=float)


def build_problem(
    *,
    dt_s: float,
    target: TargetCase,
    channel: ChannelCase,
    initial_state: tuple[float, ...] = INITIAL_STATE,
) -> BoundedEventReachabilityProblem:
    """Build a matched problem against the fixed reference event target."""

    target_state = _reference_event_state() + target.offset
    return BoundedEventReachabilityProblem(
        params=GolfModelParams.default(),
        initial_state=initial_state,
        nominal_controls=_controls(dt_s),
        dt_s=dt_s,
        state_scales=STATE_SCALES,
        control_scales=CONTROL_SCALES,
        bounds=channel.bounds,
        guard=GUARD,
        target_event_state=tuple(target_state),
        tangent_tolerance=TANGENT_TOLERANCE,
    )


def _optional(value: float | None) -> float | None:
    return None if value is None else float(value)


def _trial_record(
    *,
    problem: BoundedEventReachabilityProblem,
    target: TargetCase,
    channel: ChannelCase,
    config: MultipleShootingConfig,
    result: MultipleShootingResult,
) -> dict[str, Any]:
    replay = result.replay
    event = None if replay is None else replay.event
    return {
        "target_name": target.name,
        "target_direction": target.direction,
        "target_amplitude_rad": target.amplitude_rad,
        "target_event_state": list(problem.target_event_state),
        "initial_state": list(problem.initial_state),
        "channel": channel.name,
        "dt_s": problem.dt_s,
        "segment_count": config.segment_count,
        "seed": config.seed,
        "initial_control_fraction": config.initial_control_fraction,
        "solver_status": result.status.value,
        "solver_success": result.solver_success,
        "solver_message": result.message,
        "solver_iterations": result.iterations,
        "objective": result.objective,
        "maximum_continuity_residual": result.maximum_continuity_residual,
        "maximum_target_residual": result.maximum_target_residual,
        "replay_feasibility_status": (
            None if replay is None else replay.feasibility_status.value
        ),
        "replay_constraint_status": (
            None if replay is None else replay.constraint_status.value
        ),
        "replay_authority_status": (
            None if replay is None else replay.authority_status.value
        ),
        "replay_tangent_residual": (
            None if replay is None else _optional(replay.event_tangent_residual)
        ),
        "replay_full_state_residual": (
            None if replay is None else _optional(replay.full_state_residual)
        ),
        "scaled_control_energy": (
            None if replay is None else replay.scaled_control_energy
        ),
        "peak_scaled_control": None if replay is None else replay.peak_scaled_control,
        "event_status": None if event is None else event.status.value,
        "crossing_count": None if event is None else event.crossing_count,
        "event_time_s": None if event is None else _optional(event.time_s),
        "guard_residual": None if event is None else _optional(event.guard_residual),
        "transversality_per_s": (
            None if event is None else _optional(event.transversality_per_s)
        ),
    }


def run_trial(
    *,
    problem: BoundedEventReachabilityProblem,
    target: TargetCase,
    channel: ChannelCase,
    config: MultipleShootingConfig,
) -> tuple[dict[str, Any], MultipleShootingResult]:
    """Solve and serialize one matched trial without discarding failures."""

    result = solve_bounded_event_multiple_shooting(problem, config)
    return (
        _trial_record(
            problem=problem,
            target=target,
            channel=channel,
            config=config,
            result=result,
        ),
        result,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, Any]:
    parent = json.loads(PARENT_REPORT_PATH.read_text(encoding="utf-8"))
    if (
        parent.get("schema_version")
        != "proximal-distal-trajectory-control-authority/v1"
    ):
        raise ValueError("parent trajectory-control report is not qualified")
    source_paths = (
        Path("scripts/research/proximal_distal_energy/bounded_event_reachability.py"),
        Path(
            "scripts/research/proximal_distal_energy/bounded_event_multiple_shooting.py"
        ),
        Path(
            "scripts/research/proximal_distal_energy/run_bounded_event_reachability.py"
        ),
    )
    return {
        "required_parent_issue": 9123,
        "required_parent_report_schema": parent["schema_version"],
        "required_parent_report_sha256": _sha256(PARENT_REPORT_PATH),
        "required_parent_array_sha256": _sha256(PARENT_ARRAY_PATH),
        "source_sha256": {
            path.as_posix(): _sha256(ROOT / path) for path in source_paths
        },
    }


def _anchor_target() -> TargetCase:
    return next(
        target for target in registered_targets() if target.name == "plus_0p001"
    )


def _channel(name: str) -> ChannelCase:
    return next(channel for channel in registered_channels() if channel.name == name)


def _run_continuation() -> tuple[
    list[dict[str, Any]],
    list[MultipleShootingResult],
    dict[tuple[str, str], MultipleShootingResult],
]:
    records: list[dict[str, Any]] = []
    results: list[MultipleShootingResult] = []
    indexed: dict[tuple[str, str], MultipleShootingResult] = {}
    for target, channel in study_matrix():
        problem = build_problem(dt_s=BASE_DT_S, target=target, channel=channel)
        record, result = run_trial(
            problem=problem,
            target=target,
            channel=channel,
            config=REGISTERED_CONFIG,
        )
        records.append(record)
        results.append(result)
        indexed[(target.name, channel.name)] = result
    return records, results, indexed


def _run_multistart(
    base_result: MultipleShootingResult,
) -> tuple[list[dict[str, Any]], list[MultipleShootingResult]]:
    target = _anchor_target()
    channel = _channel("both")
    problem = build_problem(dt_s=BASE_DT_S, target=target, channel=channel)
    records = [
        {
            **_trial_record(
                problem=problem,
                target=target,
                channel=channel,
                config=REGISTERED_CONFIG,
                result=base_result,
            ),
            "control_id": "seed_0_zero_start",
        }
    ]
    results = [base_result]
    seeded = replace(
        REGISTERED_CONFIG,
        seed=1,
        initial_control_fraction=0.05,
    )
    record, result = run_trial(
        problem=problem,
        target=target,
        channel=channel,
        config=seeded,
    )
    records.append({**record, "control_id": "seed_1_five_percent_start"})
    results.append(result)
    return records, results


def _run_mesh_controls(
    base_result: MultipleShootingResult,
) -> tuple[list[dict[str, Any]], list[MultipleShootingResult]]:
    target = _anchor_target()
    channel = _channel("both")
    problem = build_problem(dt_s=BASE_DT_S, target=target, channel=channel)
    records: list[dict[str, Any]] = []
    results: list[MultipleShootingResult] = []
    for segment_count in (3, 4, 5):
        config = replace(REGISTERED_CONFIG, segment_count=segment_count)
        if segment_count == REGISTERED_SEGMENT_COUNT:
            result = base_result
            record = _trial_record(
                problem=problem,
                target=target,
                channel=channel,
                config=config,
                result=result,
            )
        else:
            record, result = run_trial(
                problem=problem,
                target=target,
                channel=channel,
                config=config,
            )
        records.append({**record, "control_id": f"segments_{segment_count}"})
        results.append(result)
    return records, results


def _run_integration_controls(
    base_result: MultipleShootingResult,
) -> tuple[list[dict[str, Any]], list[MultipleShootingResult]]:
    target = _anchor_target()
    channel = _channel("both")
    records: list[dict[str, Any]] = []
    results: list[MultipleShootingResult] = []
    for dt_s in (1e-3, BASE_DT_S, 4e-3):
        problem = build_problem(dt_s=dt_s, target=target, channel=channel)
        if dt_s == BASE_DT_S:
            result = base_result
            record = _trial_record(
                problem=problem,
                target=target,
                channel=channel,
                config=REGISTERED_CONFIG,
                result=result,
            )
        else:
            record, result = run_trial(
                problem=problem,
                target=target,
                channel=channel,
                config=REGISTERED_CONFIG,
            )
        records.append({**record, "control_id": f"dt_{dt_s:.3f}"})
        results.append(result)
    return records, results


def _run_adverse_initial_states() -> tuple[
    list[dict[str, Any]], list[MultipleShootingResult]
]:
    target = _anchor_target()
    channel = _channel("both")
    cases = {
        "angle_tangent_plus": np.array([0.002, -0.002, 0.0, 0.0]),
        "rate_split_plus": np.array([0.0, 0.0, 0.02, -0.02]),
    }
    records: list[dict[str, Any]] = []
    results: list[MultipleShootingResult] = []
    for name, offset in cases.items():
        initial = np.asarray(INITIAL_STATE, dtype=float) + offset
        problem = build_problem(
            dt_s=BASE_DT_S,
            target=target,
            channel=channel,
            initial_state=tuple(initial),
        )
        record, result = run_trial(
            problem=problem,
            target=target,
            channel=channel,
            config=REGISTERED_CONFIG,
        )
        records.append({**record, "control_id": name})
        results.append(result)
    return records, results


def _event_state(result: MultipleShootingResult) -> tuple[np.ndarray, bool]:
    if result.replay is None or result.replay.event is None:
        return np.zeros(4, dtype=float), False
    if result.replay.event.state is None:
        return np.zeros(4, dtype=float), False
    return result.replay.event.state.copy(), True


def _stack_result_arrays(
    prefix: str,
    results: list[MultipleShootingResult],
) -> dict[str, np.ndarray]:
    event_pairs = [_event_state(result) for result in results]
    return {
        f"{prefix}_segment_perturbations": np.stack(
            [result.segment_perturbations for result in results]
        ),
        f"{prefix}_state_nodes": np.stack([result.state_nodes for result in results]),
        f"{prefix}_event_state": np.stack([state for state, _ in event_pairs]),
        f"{prefix}_event_state_available": np.asarray(
            [available for _, available in event_pairs], dtype=bool
        ),
        f"{prefix}_solver_status": np.asarray(
            [result.status.value for result in results]
        ),
    }


def _relative_spread(values: list[float]) -> float:
    if not values:
        return math.inf
    denominator = max(min(values), np.finfo(float).eps)
    return float((max(values) - min(values)) / denominator)


def _qualification(
    *,
    continuation: list[dict[str, Any]],
    multistart: list[dict[str, Any]],
    mesh: list[dict[str, Any]],
    integration: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_multistart = [
        float(record["objective"])
        for record in multistart
        if record["solver_status"] == "converged"
    ]
    spread = _relative_spread(successful_multistart)
    numerical_failure_count = sum(
        record["solver_status"] == "numerical_failure" for record in continuation
    )
    replay_rejection_count = sum(
        record["solver_status"] == "replay_rejected" for record in continuation
    )
    mesh_all_converged = all(record["solver_status"] == "converged" for record in mesh)
    integration_all_converged = all(
        record["solver_status"] == "converged" for record in integration
    )
    multistart_adequate = (
        len(successful_multistart) == len(multistart)
        and spread <= MULTISTART_OBJECTIVE_SPREAD_GATE
    )
    feasibility_evidence_adequate = (
        numerical_failure_count == 0
        and replay_rejection_count == 0
        and mesh_all_converged
        and integration_all_converged
        and len(successful_multistart) == len(multistart)
    )
    return {
        "numerical_failure_count": numerical_failure_count,
        "replay_rejection_count": replay_rejection_count,
        "mesh_all_converged": mesh_all_converged,
        "integration_step_all_converged": integration_all_converged,
        "multistart_relative_objective_spread": spread,
        "multistart_spread_gate": MULTISTART_OBJECTIVE_SPREAD_GATE,
        "multistart_adequate": multistart_adequate,
        "feasibility_evidence_adequate": feasibility_evidence_adequate,
        "optimality_evidence_adequate": multistart_adequate,
        "channel_ranking_available": False,
        "registered_release_adequate": feasibility_evidence_adequate,
    }


def _registration_payload() -> dict[str, Any]:
    """Return the frozen coordinate, target, bound, and solver declaration."""

    return {
        "state_coordinates": [
            "shoulder_angle_rad",
            "wrist_relative_angle_rad",
            "shoulder_rate_rad_s",
            "wrist_relative_rate_rad_s",
        ],
        "control_coordinates": ["shoulder_torque_nm", "wrist_torque_nm"],
        "state_scales": list(STATE_SCALES.values),
        "control_scales_nm": list(CONTROL_SCALES.values),
        "initial_state": list(INITIAL_STATE),
        "reference_event_state": list(_reference_event_values()),
        "base_dt_s": BASE_DT_S,
        "horizon_s": HORIZON_S,
        "segment_count": REGISTERED_SEGMENT_COUNT,
        "target_tangent_tolerance": TANGENT_TOLERANCE,
        "target_offsets": [
            {
                "name": target.name,
                "direction": target.direction,
                "amplitude_rad": target.amplitude_rad,
                "offset": target.offset.tolist(),
            }
            for target in registered_targets()
        ],
        "channel_bounds": {
            channel.name: {
                "lower_nm": list(channel.bounds.lower_nm),
                "upper_nm": list(channel.bounds.upper_nm),
                "max_rate_nm_per_s": list(channel.bounds.max_rate_nm_per_s),
            }
            for channel in registered_channels()
        },
        "bounds_are_human_capacity_evidence": False,
        "guard": "theta_shoulder + theta_wrist_relative = 0, positive crossing",
        "solver": "scipy_slsqp_multiple_shooting_with_exact_rk4_replay",
    }


def build_evidence() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the serial registered study and return report plus raw arrays."""

    continuation, continuation_results, indexed = _run_continuation()
    anchor = indexed[("plus_0p001", "both")]
    multistart, multistart_results = _run_multistart(anchor)
    mesh, mesh_results = _run_mesh_controls(anchor)
    integration, integration_results = _run_integration_controls(anchor)
    adverse, adverse_results = _run_adverse_initial_states()
    all_records = continuation + multistart + mesh + integration + adverse
    report = {
        "schema_version": SCHEMA_VERSION,
        "model_tier": "analytical_double_pendulum",
        "source_identity": _source_identity(),
        "registration": _registration_payload(),
        "continuation_trials": continuation,
        "falsification_controls": {
            "multistart": multistart,
            "mesh_refinement": mesh,
            "integration_step_refinement": integration,
            "adverse_initial_states": adverse,
        },
        "outcome_counts": {
            "solver_status": dict(
                Counter(record["solver_status"] for record in all_records)
            ),
            "replay_feasibility_status": dict(
                Counter(
                    str(record["replay_feasibility_status"]) for record in all_records
                )
            ),
            "event_status": dict(
                Counter(str(record["event_status"]) for record in all_records)
            ),
        },
        "qualification": _qualification(
            continuation=continuation,
            multistart=multistart,
            mesh=mesh,
            integration=integration,
        ),
        "availability": {
            "registered_model_scenario_feasibility": "available",
            "global_nonlinear_reachability": "unavailable",
            "channel_or_controller_ranking": "suppressed",
            "human_actuator_interpretation": "unavailable",
            "passive_torque_interpretation": "unavailable",
            "coaching_recommendation": "unavailable",
        },
        "inference_boundary": INFERENCE_BOUNDARY,
        "limitations": [
            "The study uses a synthetic open-loop analytical double pendulum and a geometric delivery guard, not measured impact.",
            "The final event duration varies only inside the nominal crossing bracket; bracket changes and global topology remain unmapped.",
            "Torque and slew bounds are declared model scenarios without governed human strength or activation evidence.",
            "The objective is a scaled control-energy proxy, not metabolic energy, fatigue, injury risk, or optimal human effort.",
            "A converged solver result is retained only after independent exact-RK4 replay, but this is not independent model validation.",
        ],
    }
    arrays = {
        **_stack_result_arrays("continuation", continuation_results),
        **_stack_result_arrays("multistart", multistart_results),
        **_stack_result_arrays("integration", integration_results),
        **_stack_result_arrays("adverse", adverse_results),
        "continuation_target_state": np.asarray(
            [record["target_event_state"] for record in continuation], dtype=float
        ),
        "continuation_initial_state": np.asarray(
            [record["initial_state"] for record in continuation], dtype=float
        ),
    }
    for record, result in zip(mesh, mesh_results, strict=True):
        key = str(record["control_id"])
        arrays[f"mesh_{key}_segment_perturbations"] = result.segment_perturbations
        arrays[f"mesh_{key}_state_nodes"] = result.state_nodes
    validate_report(report)
    return canonicalize_published_numbers(report), arrays


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Fail closed on provenance, matrix, killswitch, or inference drift."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected bounded event-reachability schema")
    if report.get("source_identity") != _source_identity():
        raise ValueError("bounded event-reachability source identity is stale")
    continuation = report.get("continuation_trials", [])
    if len(continuation) != 28:
        raise ValueError("registered continuation matrix must contain 28 trials")
    identities = {
        (record.get("target_name"), record.get("channel")) for record in continuation
    }
    if len(identities) != 28:
        raise ValueError("continuation target/channel identities must be unique")
    allowed_solver = {status.value for status in MultipleShootingStatus}
    allowed_feasibility = {status.value for status in FeasibilityStatus}
    for record in continuation:
        if record.get("solver_status") not in allowed_solver:
            raise ValueError("continuation solver outcome is untyped")
        if record.get("replay_feasibility_status") not in allowed_feasibility:
            raise ValueError("continuation replay outcome is untyped")
    zero_nominal = next(
        record
        for record in continuation
        if record["target_name"] == "zero" and record["channel"] == "zero"
    )
    if zero_nominal["replay_feasibility_status"] != "feasible":
        raise ValueError("zero-authority nominal target must replay feasible")
    zero_nonzero = [
        record
        for record in continuation
        if record["target_name"] != "zero" and record["channel"] == "zero"
    ]
    if any(
        record["replay_feasibility_status"] != "infeasible" for record in zero_nonzero
    ):
        raise ValueError("zero-authority displaced targets must remain infeasible")
    controls = report.get("falsification_controls", {})
    expected_counts = {
        "multistart": 2,
        "mesh_refinement": 3,
        "integration_step_refinement": 3,
        "adverse_initial_states": 2,
    }
    for name, count in expected_counts.items():
        if len(controls.get(name, [])) != count:
            raise ValueError(f"{name} control count must remain {count}")
    qualification = report.get("qualification", {})
    if qualification.get("registered_release_adequate") != qualification.get(
        "feasibility_evidence_adequate"
    ):
        raise ValueError("release adequacy must follow feasibility evidence")
    if (
        not qualification.get("optimality_evidence_adequate")
        and qualification.get("channel_ranking_available") is not False
    ):
        raise ValueError("failed optimality adequacy must suppress rankings")
    availability = report.get("availability", {})
    if availability.get("channel_or_controller_ranking") != "suppressed":
        raise ValueError("channel/controller rankings must remain suppressed")
    for name in (
        "global_nonlinear_reachability",
        "human_actuator_interpretation",
        "passive_torque_interpretation",
        "coaching_recommendation",
    ):
        if availability.get(name) != "unavailable":
            raise ValueError(f"{name} must remain unavailable")
    boundary = str(report.get("inference_boundary", "")).lower()
    for phrase in (
        "model-scenario",
        "global reachability",
        "human torque",
        "passive torque",
        "coaching recommendation",
    ):
        if phrase not in boundary:
            raise ValueError(f"inference boundary must retain '{phrase}'")
    return {
        "continuation_trials": len(continuation),
        "control_trials": sum(expected_counts.values()),
        "typed_outcomes": len(continuation),
    }


def write_evidence() -> None:
    """Write deterministic portable JSON and full-precision NPZ evidence."""

    report, arrays = build_evidence()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(ARRAY_PATH, **arrays)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("write", "validate"), nargs="?", default="validate"
    )
    args = parser.parse_args()
    if args.mode == "write":
        write_evidence()
        print(f"wrote {REPORT_PATH}")
        print(f"wrote {ARRAY_PATH}")
        return
    registered = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_report(registered), sort_keys=True))


__all__ = [
    "ARRAY_PATH",
    "BASE_DT_S",
    "ChannelCase",
    "INITIAL_STATE",
    "REPORT_PATH",
    "REGISTERED_SEGMENT_COUNT",
    "SCHEMA_VERSION",
    "TANGENT_TOLERANCE",
    "TargetCase",
    "build_evidence",
    "build_problem",
    "registered_channels",
    "registered_targets",
    "run_trial",
    "study_matrix",
    "validate_report",
    "write_evidence",
]


if __name__ == "__main__":
    main()
