"""Qualify a pure controller-facing map against the canonical ODE plant.

This module establishes state, control, integration, and parity semantics only.
It deliberately runs no controller optimization and grants no ranking authority.
"""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path
from typing import Final

import numpy as np

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumState,
)
from src.shared.python.simulation_backends.factory import make_backend
from src.shared.python.simulation_backends.model_params import GolfModelParams
from src.shared.python.simulation_backends.protocol import SimState

from .nonlinear_controller_numerics import FloatArray, finite_vector

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REPORT_PATH = ARTICLE / "data/nonlinear_controller_plant_transport.json"
ENVIRONMENT_LOCK_PATH = ROOT / "requirements.lock"
REGISTRATION_RELATIVE_PATH = Path(
    "docs/research/proximal_distal_energy_transfer/data/"
    "nonlinear_controller_comparison_registration.json"
)
SCHEMA_VERSION = "proximal-distal-nonlinear-controller-plant-transport/v2"
STEP_SIZES: Final = (0.0005, 0.001, 0.002)
PARITY_ERROR_LIMIT: Final = 1.0e-12
STATE_NAMES: Final = (
    "shoulder_angle_rad",
    "wrist_relative_angle_rad",
    "shoulder_rate_rad_s",
    "wrist_relative_rate_rad_s",
)
CONTROL_NAMES: Final = ("shoulder_torque_nm", "wrist_torque_nm")
PARITY_CASES: Final = (
    ((-2.20, -1.57, 0.00, 0.00), (0.00, 0.00)),
    ((-1.40, -1.10, 3.50, -2.00), (60.00, -15.00)),
    ((-0.35, -0.65, 12.00, 8.00), (60.00, 15.00)),
    ((0.10, 0.20, -4.00, 3.00), (-60.00, 15.00)),
)
INVALID_CASES: Final = (
    ((0.0, 0.0, 0.0), (0.0, 0.0), "wrong_state_size"),
    ((0.0, 0.0, 0.0, 0.0), (0.0,), "wrong_control_size"),
    ((0.0, 0.0, math.nan, 0.0), (0.0, 0.0), "nonfinite_state"),
    ((0.0, 0.0, 0.0, 0.0), (0.0, math.inf), "nonfinite_control"),
)


class RegisteredDoublePendulumPlant:
    """Pure four-state, two-torque RK4 map for controller qualification."""

    def __init__(self, parameters: GolfModelParams, *, step_s: float) -> None:
        if not math.isfinite(step_s) or step_s <= 0.0:
            raise ValueError("step_s must be positive and finite")
        self._step_s = float(step_s)
        self._dynamics = DoublePendulumDynamics(
            parameters.to_double_pendulum_parameters()
        )

    def __call__(self, state: FloatArray, control: FloatArray) -> FloatArray:
        """Advance one constant-control step without mutating caller inputs."""
        state_vector = finite_vector("state", state, 4).copy()
        control_vector = finite_vector("control", control, 2).copy()
        step = self._step_s
        k1 = self._derivative(state_vector, control_vector)
        k2 = self._derivative(state_vector + 0.5 * step * k1, control_vector)
        k3 = self._derivative(state_vector + 0.5 * step * k2, control_vector)
        k4 = self._derivative(state_vector + step * k3, control_vector)
        candidate = state_vector + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(candidate)):
            raise ValueError("double-pendulum step produced a non-finite state")
        return candidate

    def _derivative(self, state: FloatArray, control: FloatArray) -> FloatArray:
        model_state = DoublePendulumState(
            theta1=float(state[0]),
            theta2=float(state[1]),
            omega1=float(state[2]),
            omega2=float(state[3]),
        )
        drift, control_map = self._dynamics.control_affine(model_state)
        acceleration_1 = (
            drift[2] + control_map[2][0] * control[0] + control_map[2][1] * control[1]
        )
        acceleration_2 = (
            drift[3] + control_map[3][0] * control[0] + control_map[3][1] * control[1]
        )
        return np.array((drift[0], drift[1], acceleration_1, acceleration_2))


def build_transport_qualification(root: Path = ROOT) -> dict[str, object]:
    """Build deterministic multi-step-size parity evidence without controls."""
    root = root.resolve()
    parameters = GolfModelParams.default()
    records: list[dict[str, object]] = []
    parity_errors: list[float] = []
    deterministic = True
    immutable = True
    for step_s in STEP_SIZES:
        plant = RegisteredDoublePendulumPlant(parameters, step_s=step_s)
        for state_values, control_values in PARITY_CASES:
            state = np.asarray(state_values, dtype=float)
            control = np.asarray(control_values, dtype=float)
            state_before = state.copy()
            control_before = control.copy()
            observed = plant(state, control)
            replay = plant(state, control)
            reference = _reference_step(parameters, state, control, step_s)
            error = float(np.max(np.abs(observed - reference)))
            parity_errors.append(error)
            deterministic = deterministic and np.array_equal(observed, replay)
            immutable = immutable and np.array_equal(state, state_before)
            immutable = immutable and np.array_equal(control, control_before)
            records.append(
                {
                    "step_s": step_s,
                    "state": state.tolist(),
                    "control": control.tolist(),
                    "maximum_state_error": error,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "canonical_double_pendulum_plant_transport",
        "evidence_status": "plant_transport_qualified_no_controller_evaluation",
        "state_contract": list(STATE_NAMES),
        "control_contract": list(CONTROL_NAMES),
        "integration_contract": {
            "method": "fixed_step_rk4_constant_control",
            "step_sizes_s": list(STEP_SIZES),
            "reference_backend": "ode",
        },
        "step_sizes_s": list(STEP_SIZES),
        "registration_authority": _authority(root, REGISTRATION_RELATIVE_PATH),
        "parameter_authority": {
            "canonical_json": parameters.model_dump(mode="json"),
            "sha256": _parameter_sha256(parameters),
        },
        "source_authorities": _source_authorities(root),
        "environment_authority": _authority(
            root, ENVIRONMENT_LOCK_PATH.relative_to(ROOT)
        ),
        "parity_thresholds": {"maximum_state_error": PARITY_ERROR_LIMIT},
        "parity_cases": records,
        "parity_case_count": len(records),
        "maximum_state_parity_error": max(parity_errors),
        "deterministic_replay_passed": bool(deterministic),
        "input_immutability_passed": bool(immutable),
        "invalid_input_cases": _invalid_input_records(parameters),
        "invalid_input_case_count": len(INVALID_CASES),
        "step_size_count": len(STEP_SIZES),
        "invalid_input_policy": "raise_value_error_without_trajectory",
        "controller_evaluation_count": 0,
        "ranking_eligible_method_count": 0,
        "eligible_for_ranking": False,
        "remaining_gates": [
            "prospective_outcome_blind_tuning",
            "typed_plant_outcomes",
            "matched_solver_replay",
            "frozen_held_out_execution",
            "event_retention_and_failure_region_map",
            "optimality_adequacy_or_ranking_suppression",
        ],
        "inference_boundary": (
            "Plant parity establishes numerical transport only. It does not "
            "qualify a controller, compare policies, identify human control, "
            "establish passive torque, or support a coaching recommendation."
        ),
    }


def validate_transport_qualification(
    report: dict[str, object], root: Path = ROOT
) -> dict[str, int]:
    """Fail closed on parity, replay, provenance, failure, or scope drift."""
    if report != build_transport_qualification(root):
        raise ValueError("transport report differs from deterministic authority")
    cases = report.get("parity_cases")
    expected_cases = len(STEP_SIZES) * len(PARITY_CASES)
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError("all registered step-size parity cases are required")
    if _exact_int(report, "parity_case_count") != len(cases):
        raise ValueError("parity case count drifted")
    maximum_error = report.get("maximum_state_parity_error")
    if not isinstance(maximum_error, (int, float)) or isinstance(maximum_error, bool):
        raise ValueError("maximum parity error must be numeric")
    if not math.isfinite(float(maximum_error)):
        raise ValueError("maximum parity error must be finite")
    if float(maximum_error) > PARITY_ERROR_LIMIT:
        raise ValueError("canonical ODE parity gate failed")
    if report.get("deterministic_replay_passed") is not True:
        raise ValueError("deterministic replay gate failed")
    if report.get("input_immutability_passed") is not True:
        raise ValueError("input immutability gate failed")
    invalid_cases = report.get("invalid_input_cases")
    if not isinstance(invalid_cases, list) or not all(
        item.get("typed_failure") == "ValueError"
        for item in invalid_cases
        if isinstance(item, dict)
    ):
        raise ValueError("invalid inputs must produce typed failures")
    if _exact_int(report, "invalid_input_case_count") != len(INVALID_CASES):
        raise ValueError("invalid input case count drifted")
    if _exact_int(report, "step_size_count") != len(STEP_SIZES):
        raise ValueError("step-size count drifted")
    evaluations = _exact_int(report, "controller_evaluation_count")
    eligible = int(report.get("eligible_for_ranking") is True)
    if evaluations != 0 or eligible != 0:
        raise ValueError("transport evidence cannot rank controllers")
    if _exact_int(report, "ranking_eligible_method_count") != eligible:
        raise ValueError("ranking-eligible method count drifted")
    return {
        "parity_case_count": len(cases),
        "controller_evaluation_count": evaluations,
        "ranking_eligible_count": eligible,
    }


def _reference_step(
    parameters: GolfModelParams,
    state: FloatArray,
    control: FloatArray,
    step_s: float,
) -> FloatArray:
    backend = make_backend("ode", parameters)
    backend.reset(SimState(q=state[:2], v=state[2:], time=0.0))
    backend.set_control(control)
    backend.step(step_s)
    result = backend.get_state()
    return np.concatenate((result.q, result.v))


def _invalid_input_records(
    parameters: GolfModelParams,
) -> list[dict[str, object]]:
    plant = RegisteredDoublePendulumPlant(parameters, step_s=0.001)
    records: list[dict[str, object]] = []
    for state_values, control_values, case_name in INVALID_CASES:
        try:
            plant(
                np.asarray(state_values, dtype=float),
                np.asarray(control_values, dtype=float),
            )
        except ValueError:
            records.append(
                {
                    "case": case_name,
                    "typed_failure": "ValueError",
                    "trajectory_emitted": False,
                }
            )
        else:
            raise ValueError(f"{case_name}: invalid plant input was accepted")
    return records


def _source_authorities(root: Path) -> list[dict[str, str]]:
    paths = (
        Path("src/shared/python/simulation_backends/model_params.py"),
        Path("src/shared/python/simulation_backends/ode_backend.py"),
        Path(
            "src/engines/pendulum_models/python/double_pendulum_model/physics/"
            "double_pendulum.py"
        ),
        Path(
            "scripts/research/proximal_distal_energy/nonlinear_controller_numerics.py"
        ),
    )
    return [_authority(root, path) for path in paths]


def _authority(root: Path, relative_path: Path) -> dict[str, str]:
    return {
        "path": relative_path.as_posix(),
        "sha256": _sha256(root / relative_path),
    }


def _parameter_sha256(parameters: GolfModelParams) -> str:
    payload = json.dumps(
        parameters.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_int(report: dict[str, object], field: str) -> int:
    value = report.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("write", "validate"), nargs="?", default="validate"
    )
    args = parser.parse_args()
    if args.command == "write":
        report = build_transport_qualification(ROOT)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_transport_qualification(report, ROOT), indent=2))


if __name__ == "__main__":
    main()
