"""Generate or validate local double-pendulum rank diagnostics."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Repository modules include both ``src.shared`` and top-level packages rooted
# at ``src``.  Pytest installs both roots, but a documented ``python -m``
# invocation from a clean checkout does not.  Establish the same deterministic
# import contract before loading the simulation backend.
ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.research.proximal_distal_energy.local_linear_diagnostics import (
    INFERENCE_BOUNDARY,
    LinearizationPoint,
    LocalLinearAudit,
    NondimensionalScales,
    RankDiagnostic,
    RankTolerance,
    audit_double_pendulum_configuration_state,
)
from scripts.research.proximal_distal_energy.run_experiments import (
    DT,
    HORIZON,
    INITIAL_Q,
    rollout_program,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams


REPORT_PATH = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/local_linear_diagnostics.json"
)
SCHEMA_VERSION = "proximal-distal-local-linear-diagnostics/v2"
PHASES = (
    ("initial_state", 0.0),
    ("early_downswing", 0.3),
    ("mid_downswing", 0.7),
    ("delivery_event_nearest_sample", 1.0),
)
STEP_MULTIPLIERS = (0.1, 1.0, 10.0)
BASE_STATE_STEPS = np.array([1e-6, 1e-6, 1e-5, 1e-5])
BASE_CONTROL_STEPS = np.array([1e-4, 1e-4])
RANK_TOLERANCE = RankTolerance(absolute=1e-8, relative=1e-7)


@dataclass(frozen=True, slots=True)
class ReferenceTrajectory:
    """Registered rollout data required to construct local audit points."""

    impact_time_s: float
    time: np.ndarray
    q: np.ndarray
    v: np.ndarray
    controls: np.ndarray
    params: GolfModelParams
    scale_scenarios: dict[str, NondimensionalScales]


def _linearization_point(
    state: np.ndarray,
    control: np.ndarray,
    *,
    state_multiplier: float = 1.0,
    control_steps: np.ndarray | None = None,
) -> LinearizationPoint:
    """Build the immutable point contract used by the local audit API."""
    selected_control_steps = (
        BASE_CONTROL_STEPS * state_multiplier
        if control_steps is None
        else control_steps
    )
    return LinearizationPoint(
        state=tuple(float(value) for value in state),
        control=tuple(float(value) for value in control),
        state_steps=tuple(
            float(value) for value in BASE_STATE_STEPS * state_multiplier
        ),
        control_steps=tuple(float(value) for value in selected_control_steps),
    )


def _scale_payload(scales: NondimensionalScales) -> dict[str, Any]:
    return {
        "characteristic_time_s": scales.characteristic_time_s,
        "control": list(scales.control),
        "output": list(scales.output),
        "state": list(scales.state),
    }


def _registered_scale_scenarios(
    *, impact_time_s: float, v: np.ndarray, controls: np.ndarray
) -> dict[str, NondimensionalScales]:
    angle_scale = float(np.pi)
    rate_scales = np.maximum(np.max(np.abs(v), axis=0), 1.0)
    control_scales = np.maximum(np.max(np.abs(controls), axis=0), 1.0)

    def scenario(rate_factor: float, time_factor: float) -> NondimensionalScales:
        return NondimensionalScales(
            state=(
                angle_scale,
                angle_scale,
                float(rate_scales[0] * rate_factor),
                float(rate_scales[1] * rate_factor),
            ),
            control=(float(control_scales[0]), float(control_scales[1])),
            output=(angle_scale, angle_scale),
            characteristic_time_s=float(impact_time_s * time_factor),
        )

    return {
        "short_time_high_rate": scenario(2.0, 0.5),
        "nominal": scenario(1.0, 1.0),
        "long_time_low_rate": scenario(0.5, 2.0),
    }


def _rank_payload(diagnostic: RankDiagnostic) -> dict[str, Any]:
    return {
        "full_rank": diagnostic.full_rank,
        "matrix_shape": list(diagnostic.matrix_shape),
        "rank": diagnostic.rank,
        "retained_condition_number": diagnostic.retained_condition_number,
        "singular_values": list(diagnostic.singular_values),
        "smallest_retained": diagnostic.smallest_retained,
        "threshold": diagnostic.threshold,
    }


def _audit_payload(audit: LocalLinearAudit, multiplier: float) -> dict[str, Any]:
    return {
        "control_steps": list(audit.control_steps),
        "controllability": _rank_payload(audit.controllability),
        "controllability_matrix": audit.controllability_matrix.tolist(),
        "dimensionless_input_matrix": audit.dimensionless_input_matrix.tolist(),
        "dimensionless_output_matrix": audit.dimensionless_output_matrix.tolist(),
        "dimensionless_state_matrix": audit.dimensionless_state_matrix.tolist(),
        "input_matrix": audit.input_matrix.tolist(),
        "multiplier": multiplier,
        "observability": _rank_payload(audit.observability),
        "observability_matrix": audit.observability_matrix.tolist(),
        "output_matrix": audit.output_matrix.tolist(),
        "rank_tolerance": {
            "absolute": RANK_TOLERANCE.absolute,
            "relative": RANK_TOLERANCE.relative,
        },
        "rank_basis": "nondimensional",
        "raw_dimensional_conditioning_status": "not_interpreted",
        "scales": _scale_payload(audit.scales),
        "state_matrix": audit.state_matrix.tolist(),
        "state_steps": list(audit.state_steps),
    }


def _step_trials(
    trajectory: ReferenceTrajectory, state: np.ndarray, control: np.ndarray
) -> list[dict[str, Any]]:
    """Audit finite-difference sensitivity at the nominal scale."""
    nominal_scales = trajectory.scale_scenarios["nominal"]
    return [
        _audit_payload(
            audit_double_pendulum_configuration_state(
                trajectory.params,
                point=_linearization_point(state, control, state_multiplier=multiplier),
                scales=nominal_scales,
                tolerance=RANK_TOLERANCE,
            ),
            multiplier,
        )
        for multiplier in STEP_MULTIPLIERS
    ]


def _scale_trials(
    trajectory: ReferenceTrajectory, state: np.ndarray, control: np.ndarray
) -> list[dict[str, Any]]:
    """Audit the registered nondimensional scale scenarios."""
    trials: list[dict[str, Any]] = []
    for scenario_name, scales in trajectory.scale_scenarios.items():
        payload = _audit_payload(
            audit_double_pendulum_configuration_state(
                trajectory.params,
                point=_linearization_point(state, control),
                scales=scales,
                tolerance=RANK_TOLERANCE,
            ),
            1.0,
        )
        payload["name"] = scenario_name
        trials.append(payload)
    return trials


def _measurement_countermodels(
    trajectory: ReferenceTrajectory, state: np.ndarray, control: np.ndarray
) -> list[dict[str, Any]]:
    """Evaluate declared sensing maps, including the zero-output killswitch."""
    nominal_scales = trajectory.scale_scenarios["nominal"]
    measurement_definitions = (
        (
            "both_joint_angles",
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            nominal_scales.output,
        ),
        ("shoulder_angle_only", np.array([[1.0, 0.0, 0.0, 0.0]]), (np.pi,)),
        (
            "wrist_relative_angle_only",
            np.array([[0.0, 1.0, 0.0, 0.0]]),
            (np.pi,),
        ),
        ("zero_output_killswitch", np.zeros((1, 4)), (1.0,)),
    )
    countermodels: list[dict[str, Any]] = []
    for scenario_name, output_map, output_scales in measurement_definitions:
        scales = NondimensionalScales(
            state=nominal_scales.state,
            control=nominal_scales.control,
            output=tuple(float(value) for value in output_scales),
            characteristic_time_s=nominal_scales.characteristic_time_s,
        )
        audit = audit_double_pendulum_configuration_state(
            trajectory.params,
            point=_linearization_point(state, control),
            scales=scales,
            tolerance=RANK_TOLERANCE,
            output_map=output_map,
        )
        countermodels.append(
            {
                "name": scenario_name,
                "observability": _rank_payload(audit.observability),
                "output_map": output_map.tolist(),
                "rank_basis": "nondimensional",
                "scales": _scale_payload(scales),
            }
        )
    return countermodels


def _actuator_countermodels(
    trajectory: ReferenceTrajectory, state: np.ndarray, control: np.ndarray
) -> list[dict[str, Any]]:
    """Evaluate declared control maps, including the zero-input killswitch."""
    nominal_scales = trajectory.scale_scenarios["nominal"]
    actuator_definitions = (
        ("both_generalized_torques", np.eye(2), control, nominal_scales.control),
        (
            "shoulder_torque_only",
            np.array([[1.0], [0.0]]),
            control[:1],
            nominal_scales.control[:1],
        ),
        (
            "wrist_torque_only",
            np.array([[0.0], [1.0]]),
            control[1:],
            nominal_scales.control[1:],
        ),
        ("zero_input_killswitch", np.zeros((2, 1)), np.zeros(1), (1.0,)),
    )
    countermodels: list[dict[str, Any]] = []
    for (
        scenario_name,
        control_map,
        scenario_control,
        control_scales,
    ) in actuator_definitions:
        scales = NondimensionalScales(
            state=nominal_scales.state,
            control=tuple(float(value) for value in control_scales),
            output=nominal_scales.output,
            characteristic_time_s=nominal_scales.characteristic_time_s,
        )
        audit = audit_double_pendulum_configuration_state(
            trajectory.params,
            point=_linearization_point(
                state,
                scenario_control,
                control_steps=np.full(len(scenario_control), BASE_CONTROL_STEPS[0]),
            ),
            scales=scales,
            tolerance=RANK_TOLERANCE,
            generalized_control_map=control_map,
        )
        countermodels.append(
            {
                "controllability": _rank_payload(audit.controllability),
                "generalized_control_map": control_map.tolist(),
                "name": scenario_name,
                "rank_basis": "nondimensional",
                "scales": _scale_payload(scales),
            }
        )
    return countermodels


def _operating_point(
    *, name: str, phase_fraction: float, trajectory: ReferenceTrajectory
) -> dict[str, Any]:
    """Build one registered operating-point evidence record."""
    target_time = phase_fraction * trajectory.impact_time_s
    index = int(np.argmin(np.abs(trajectory.time - target_time)))
    state = np.concatenate((trajectory.q[index], trajectory.v[index]))
    control = trajectory.controls[index]
    trials = _step_trials(trajectory, state, control)
    ranks = {
        (trial["observability"]["rank"], trial["controllability"]["rank"])
        for trial in trials
    }
    return {
        "actuator_countermodels": _actuator_countermodels(trajectory, state, control),
        "control_nm": control.tolist(),
        "event_time_offset_s": float(trajectory.time[index] - target_time),
        "name": name,
        "phase_fraction_of_impact_time": phase_fraction,
        "measurement_countermodels": _measurement_countermodels(
            trajectory, state, control
        ),
        "rank_stable_across_steps": len(ranks) == 1,
        "sample_index": index,
        "source": "registered_synthetic_reference_rollout",
        "state": {
            "q_rad": trajectory.q[index].tolist(),
            "v_rad_s": trajectory.v[index].tolist(),
        },
        "step_trials": trials,
        "scale_trials": _scale_trials(trajectory, state, control),
        "time_s": float(trajectory.time[index]),
    }


def _scale_contract(
    scale_scenarios: dict[str, NondimensionalScales],
) -> dict[str, Any]:
    """Describe how every nondimensional coordinate scale is obtained."""
    return {
        "characteristic_time_derivation": (
            "delivery_event_time_s multiplied by the registered scenario time_factor"
        ),
        "control_coordinates": [
            "shoulder_generalized_torque_nm",
            "wrist_relative_generalized_torque_nm",
        ],
        "control_scale_derivation": (
            "max(maximum absolute registered-rollout torque, 1 N*m)"
        ),
        "output_coordinates": [
            "shoulder_angle_rad",
            "wrist_relative_angle_rad",
        ],
        "scenario_order": list(scale_scenarios),
        "scenarios": {
            name: _scale_payload(scales) for name, scales in scale_scenarios.items()
        },
        "scope": (
            "registered synthetic reference scales; sensitivity scenarios "
            "are numerical adequacy checks, not population priors"
        ),
        "state_units": ["rad", "rad", "rad/s", "rad/s"],
        "state_coordinates": [
            "shoulder_angle_rad",
            "wrist_relative_angle_rad",
            "shoulder_rate_rad_s",
            "wrist_relative_rate_rad_s",
        ],
        "state_scale_derivation": {
            "angles": "pi rad",
            "rates": (
                "max(maximum absolute registered-rollout rate, 1 rad/s) "
                "multiplied by the registered scenario rate_factor"
            ),
        },
        "control_units": ["N*m", "N*m"],
        "output_units": ["rad", "rad"],
    }


def _reference_rollout_payload(
    *, impact_time_s: float, impact_speed_m_s: float, impact_arm_angle_rad: float
) -> dict[str, Any]:
    """Retain the deterministic synthetic trajectory provenance."""
    return {
        "delivery_event": {
            "arm_angle_rad": impact_arm_angle_rad,
            "clubhead_speed_m_s": impact_speed_m_s,
            "time_s": impact_time_s,
        },
        "dt_s": DT,
        "horizon_steps": HORIZON,
        "initial_q_rad": list(INITIAL_Q),
        "model_parameters": "GolfModelParams.default()",
        "program": {
            "onset_s": 0.10,
            "shoulder_torque_nm": 60.0,
            "wrist_drive_nm": 15.0,
            "wrist_restrain_nm": 10.0,
        },
        "scope": "synthetic reference trajectory; not a participant or coaching strategy",
    }


def _summary_payload(
    points: list[dict[str, Any]],
    scale_scenarios: dict[str, NondimensionalScales],
) -> dict[str, Any]:
    """Summarize registered evidence counts without broadening inference."""
    return {
        "all_rank_decisions_stable": all(
            point["rank_stable_across_steps"] for point in points
        ),
        "actuator_countermodel_count": sum(
            len(point["actuator_countermodels"]) for point in points
        ),
        "control_dimension": 2,
        "operating_point_count": len(points),
        "output_dimension": 2,
        "measurement_countermodel_count": sum(
            len(point["measurement_countermodels"]) for point in points
        ),
        "scale_scenario_count": len(scale_scenarios),
        "state_dimension": 4,
        "step_multiplier_count": len(STEP_MULTIPLIERS),
    }


def build_report() -> dict[str, object]:
    """Build deterministic local-rank evidence from the reference rollout."""
    params = GolfModelParams.default()
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    time, q, v, controls = rollout_program(params, program)
    impact = find_impact(time, q, v, PlanarInertials.from_params(params))
    if impact is None:
        raise RuntimeError("registered reference rollout has no delivery event")
    impact_time_s, impact_speed_m_s, impact_arm_angle_rad = impact
    scale_scenarios = _registered_scale_scenarios(
        impact_time_s=impact_time_s, v=v, controls=controls
    )
    trajectory = ReferenceTrajectory(
        impact_time_s=impact_time_s,
        time=time,
        q=q,
        v=v,
        controls=controls,
        params=params,
        scale_scenarios=scale_scenarios,
    )
    points = [
        _operating_point(name=name, phase_fraction=fraction, trajectory=trajectory)
        for name, fraction in PHASES
    ]
    return {
        "classification": "local_first_order_numerical_rank",
        "conditioning_contract": {
            "interpreted_basis": "nondimensional_matrices",
            "raw_dimensional_conditioning_status": "not_interpreted",
            "unit_invariance_fixture": "equivalent_length_units",
        },
        "falsifiers": [
            "A manufactured observable or controllable linear fixture fails its exact rank expectation.",
            "The zero-input or zero-output killswitch retains the corresponding full-rank decision.",
            "Any operating-point rank decision changes across the registered 0.1x, 1x, and 10x finite-difference steps.",
        ],
        "global_nonlinear_status": "not_evaluated",
        "inference_boundary": INFERENCE_BOUNDARY,
        "issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9092",
        "measurement_contract": {
            "outputs": ["shoulder_angle_rad", "wrist_relative_angle_rad"],
            "unmeasured_states": ["shoulder_rate_rad_s", "wrist_rate_rad_s"],
        },
        "model_tier": "analytical_double_pendulum",
        "nondimensional_scale_contract": _scale_contract(scale_scenarios),
        "operating_points": points,
        "parent_issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9027",
        "practical_identifiability_status": "not_evaluated",
        "reference_rollout": _reference_rollout_payload(
            impact_time_s=impact_time_s,
            impact_speed_m_s=impact_speed_m_s,
            impact_arm_angle_rad=impact_arm_angle_rad,
        ),
        "schema_version": SCHEMA_VERSION,
        "structural_identifiability_status": "not_evaluated",
        "summary": _summary_payload(points, scale_scenarios),
    }


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Fail closed on promoted inference, unstable rank, or stale evidence."""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("structural_identifiability_status") != "not_evaluated":
        raise ValueError("structural identifiability is not established by local rank")
    for field in ("practical_identifiability_status", "global_nonlinear_status"):
        if report.get(field) != "not_evaluated":
            raise ValueError(f"{field} must remain not_evaluated")
    points = report.get("operating_points")
    if not isinstance(points, list) or len(points) != len(PHASES):
        raise ValueError("operating_points must contain the four registered phases")
    for point in points:
        trials = point.get("step_trials", [])
        ranks = [
            (trial["observability"]["rank"], trial["controllability"]["rank"])
            for trial in trials
        ]
        if len(trials) != len(STEP_MULTIPLIERS) or len(set(ranks)) != 1:
            raise ValueError(
                "rank stability gate failed across finite-difference steps"
            )
        if point.get("rank_stable_across_steps") is not True:
            raise ValueError("rank stability flag must be true")
        if len(point.get("scale_trials", [])) != 3:
            raise ValueError("three nondimensional scale trials are required")
        measurements = point.get("measurement_countermodels", [])
        actuators = point.get("actuator_countermodels", [])
        if len(measurements) != 4 or len(actuators) != 4:
            raise ValueError("four measurement and actuator countermodels are required")
        if measurements[-1]["observability"]["rank"] != 0:
            raise ValueError("zero-output killswitch must have zero observability rank")
        if actuators[-1]["controllability"]["rank"] != 0:
            raise ValueError(
                "zero-input killswitch must have zero controllability rank"
            )
    expected = build_report()
    if report != expected:
        raise ValueError("registered report differs from deterministic recomputation")
    return {
        "actuator_countermodel_count": len(points) * 4,
        "measurement_countermodel_count": len(points) * 4,
        "operating_point_count": len(points),
        "rank_decision_count": len(points) * len(STEP_MULTIPLIERS) * 2,
        "scale_sensitivity_count": len(points) * 3,
        "step_multiplier_count": len(STEP_MULTIPLIERS),
    }


def main(argv: list[str] | None = None) -> int:
    """Write or validate the registered evidence artifact."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate"))
    args = parser.parse_args(argv)
    if args.action == "write":
        payload = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
        REPORT_PATH.write_text(payload, encoding="utf-8")
        print(REPORT_PATH)
        return 0
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_report(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
