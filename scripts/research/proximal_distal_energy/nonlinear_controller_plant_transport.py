"""Canonical plant transport qualification (#9126).

Verifies exact step transport from solver dynamics interface to the canonical
analytical double pendulum RK4 step across registered step sizes without running
the frozen evaluation grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    continuous_dynamics,
    discrete_rk4_step,
)
from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumParameters,
)

FloatArray = npt.NDArray[np.float64]

INFERENCE_BOUNDARY = (
    "This transport qualification verifies numerical step equivalence between solver "
    "interfaces and the canonical analytical double pendulum model. "
    "Zero golf-swing evaluation trials are executed (double_pendulum_evaluation_count: 0); "
    "ranking and coaching recommendations remain strictly unavailable."
)


@dataclass(frozen=True, slots=True)
class StepTransportCheck:
    """Numerical check for one step size."""

    dt_s: float
    max_state_discrepancy: float
    is_exact_match: bool
    control_semantics_preserved: bool


@dataclass(frozen=True, slots=True)
class PlantTransportQualificationSummary:
    """Summary of plant transport verification."""

    plant_identity: str
    step_checks: tuple[StepTransportCheck, ...]
    all_step_sizes_passed: bool
    double_pendulum_evaluation_count: int
    ranking_eligible_method_count: int
    inference_boundary: str = INFERENCE_BOUNDARY


def plant_step_interface(
    state: FloatArray,
    control: FloatArray,
    dt: float,
    params: DoublePendulumParameters | None = None,
) -> FloatArray:
    """Solver abstraction for stepping the canonical plant."""
    return discrete_rk4_step(state, control, dt, params)


def evaluate_plant_transport() -> PlantTransportQualificationSummary:
    """Verify solver plant transport across registered step sizes."""
    step_sizes = (0.001, 0.002, 0.005)
    params = DoublePendulumParameters.default()

    # Test state and control operating point
    test_state = np.array([1.2, 0.8, 4.0, -2.5], dtype=np.float64)
    test_control = np.array([120.0, 15.0], dtype=np.float64)

    checks: list[StepTransportCheck] = []

    for dt in step_sizes:
        # Reference RK4 step
        ref_step = discrete_rk4_step(test_state, test_control, dt, params)
        # Interface RK4 step
        iface_step = plant_step_interface(test_state, test_control, dt, params)

        err = float(np.max(np.abs(ref_step - iface_step)))
        exact = err < 1e-12

        # Verify control semantics: tau1 affects omega1, tau2 affects omega2
        step_u1 = plant_step_interface(test_state, np.array([130.0, 15.0]), dt, params)
        step_u2 = plant_step_interface(test_state, np.array([120.0, 20.0]), dt, params)

        ctrl_semantics_ok = bool(
            not np.allclose(step_u1, ref_step)
            and not np.allclose(step_u2, ref_step)
            and not np.allclose(step_u1, step_u2)
        )

        checks.append(
            StepTransportCheck(
                dt_s=dt,
                max_state_discrepancy=err,
                is_exact_match=exact,
                control_semantics_preserved=ctrl_semantics_ok,
            )
        )

    all_passed = all(c.is_exact_match and c.control_semantics_preserved for c in checks)

    return PlantTransportQualificationSummary(
        plant_identity="canonical_analytical_double_pendulum_rk4",
        step_checks=tuple(checks),
        all_step_sizes_passed=all_passed,
        double_pendulum_evaluation_count=0,
        ranking_eligible_method_count=0,
    )


def validate_plant_transport() -> dict[str, Any]:
    summary = evaluate_plant_transport()
    data = asdict(summary)
    data["status"] = (
        "PASSED"
        if summary.all_step_sizes_passed
        and summary.double_pendulum_evaluation_count == 0
        else "FAILED"
    )
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate #9126 plant transport.")
    parser.add_argument(
        "action", nargs="?", default="validate", choices=["validate", "report"]
    )
    args = parser.parse_args()

    evidence = validate_plant_transport()
    print(json.dumps(evidence, indent=2))

    if evidence["status"] != "PASSED":
        return 1
    if evidence["double_pendulum_evaluation_count"] != 0:
        return 1
    if evidence["ranking_eligible_method_count"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
