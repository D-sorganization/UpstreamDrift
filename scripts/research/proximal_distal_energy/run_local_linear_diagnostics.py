"""Generate or validate local double-pendulum rank diagnostics."""

from __future__ import annotations

from argparse import ArgumentParser
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


def _scale_payload(scales: NondimensionalScales) -> dict[str, object]:
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


def _rank_payload(diagnostic: RankDiagnostic) -> dict[str, object]:
    return {
        "full_rank": diagnostic.full_rank,
        "matrix_shape": list(diagnostic.matrix_shape),
        "rank": diagnostic.rank,
        "retained_condition_number": diagnostic.retained_condition_number,
        "singular_values": list(diagnostic.singular_values),
        "smallest_retained": diagnostic.smallest_retained,
        "threshold": diagnostic.threshold,
    }


def _audit_payload(audit: LocalLinearAudit, multiplier: float) -> dict[str, object]:
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


def _operating_point(
    *,
    name: str,
    phase_fraction: float,
    impact_time_s: float,
    time: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    controls: np.ndarray,
    params: GolfModelParams,
    scale_scenarios: dict[str, NondimensionalScales],
) -> dict[str, object]:
    target_time = phase_fraction * impact_time_s
    index = int(np.argmin(np.abs(time - target_time)))
    state = np.concatenate((q[index], v[index]))
    control = controls[index]
    nominal_scales = scale_scenarios["nominal"]
    trials = [
        _audit_payload(
            audit_double_pendulum_configuration_state(
                params,
                state=state,
                control=control,
                state_steps=BASE_STATE_STEPS * multiplier,
                control_steps=BASE_CONTROL_STEPS * multiplier,
                scales=nominal_scales,
                tolerance=RANK_TOLERANCE,
            ),
            multiplier,
        )
        for multiplier in STEP_MULTIPLIERS
    ]
    ranks = {
        (trial["observability"]["rank"], trial["controllability"]["rank"])
        for trial in trials
    }
    scale_trials = []
    for scenario_name, scales in scale_scenarios.items():
        payload = _audit_payload(
            audit_double_pendulum_configuration_state(
                params,
                state=state,
                control=control,
                state_steps=BASE_STATE_STEPS,
                control_steps=BASE_CONTROL_STEPS,
                scales=scales,
                tolerance=RANK_TOLERANCE,
            ),
            1.0,
        )
        payload["name"] = scenario_name
        scale_trials.append(payload)

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
    measurement_countermodels = []
    for scenario_name, output_map, output_scales in measurement_definitions:
        scales = NondimensionalScales(
            state=nominal_scales.state,
            control=nominal_scales.control,
            output=tuple(float(value) for value in output_scales),
            characteristic_time_s=nominal_scales.characteristic_time_s,
        )
        audit = audit_double_pendulum_configuration_state(
            params,
            state=state,
            control=control,
            state_steps=BASE_STATE_STEPS,
            control_steps=BASE_CONTROL_STEPS,
            scales=scales,
            tolerance=RANK_TOLERANCE,
            output_map=output_map,
        )
        measurement_countermodels.append(
            {
                "name": scenario_name,
                "observability": _rank_payload(audit.observability),
                "output_map": output_map.tolist(),
                "rank_basis": "nondimensional",
                "scales": _scale_payload(scales),
            }
        )

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
    actuator_countermodels = []
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
            params,
            state=state,
            control=scenario_control,
            state_steps=BASE_STATE_STEPS,
            control_steps=np.full(len(scenario_control), BASE_CONTROL_STEPS[0]),
            scales=scales,
            tolerance=RANK_TOLERANCE,
            generalized_control_map=control_map,
        )
        actuator_countermodels.append(
            {
                "controllability": _rank_payload(audit.controllability),
                "generalized_control_map": control_map.tolist(),
                "name": scenario_name,
                "rank_basis": "nondimensional",
                "scales": _scale_payload(scales),
            }
        )
    return {
        "actuator_countermodels": actuator_countermodels,
        "control_nm": control.tolist(),
        "event_time_offset_s": float(time[index] - target_time),
        "name": name,
        "phase_fraction_of_impact_time": phase_fraction,
        "measurement_countermodels": measurement_countermodels,
        "rank_stable_across_steps": len(ranks) == 1,
        "sample_index": index,
        "source": "registered_synthetic_reference_rollout",
        "state": {
            "q_rad": q[index].tolist(),
            "v_rad_s": v[index].tolist(),
        },
        "step_trials": trials,
        "scale_trials": scale_trials,
        "time_s": float(time[index]),
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
    points = [
        _operating_point(
            name=name,
            phase_fraction=fraction,
            impact_time_s=impact_time_s,
            time=time,
            q=q,
            v=v,
            controls=controls,
            params=params,
            scale_scenarios=scale_scenarios,
        )
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
        "issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9027",
        "measurement_contract": {
            "outputs": ["shoulder_angle_rad", "wrist_relative_angle_rad"],
            "unmeasured_states": ["shoulder_rate_rad_s", "wrist_rate_rad_s"],
        },
        "nondimensional_scale_contract": {
            "scenario_order": list(scale_scenarios),
            "scenarios": {
                name: _scale_payload(scales) for name, scales in scale_scenarios.items()
            },
            "scope": (
                "registered synthetic reference scales; sensitivity scenarios "
                "are numerical adequacy checks, not population priors"
            ),
            "state_units": ["rad", "rad", "rad/s", "rad/s"],
            "control_units": ["N*m", "N*m"],
            "output_units": ["rad", "rad"],
        },
        "model_tier": "analytical_double_pendulum",
        "operating_points": points,
        "practical_identifiability_status": "not_evaluated",
        "reference_rollout": {
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
        },
        "schema_version": SCHEMA_VERSION,
        "structural_identifiability_status": "not_evaluated",
        "summary": {
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
        },
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
