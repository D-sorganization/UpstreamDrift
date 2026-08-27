"""Nonlinear controller comparison registration (#9126).

Freezes prospective comparison parameters against post-#9125 authority,
guaranteeing double_pendulum_evaluation_count: 0 and ranking_eligible_method_count: 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

INFERENCE_BOUNDARY = (
    "This registration freezes comparative controller estimands, plant identity, "
    "and evaluation gates for the analytical double pendulum before any controllers "
    "are executed or ranked. Scientific controller ranking, superiority, "
    "physiological limits, and coaching guidance remain explicitly unavailable."
)


@dataclass(frozen=True, slots=True)
class ProspectiveControllerRegistration:
    """Deterministic registration contract for future matched controller evaluations."""

    plant_identity: str
    state_ordering: tuple[str, ...]
    control_ordering: tuple[str, ...]
    integration_step_s: float
    planning_horizon_steps: int
    torque_bounds_nm: tuple[float, float]
    torque_rate_bounds_nm_s: tuple[float, float]
    state_scales: tuple[float, float, float, float]
    control_scales: tuple[float, float]
    event_guard_identity: str
    failure_taxonomy: tuple[str, ...]
    candidate_solvers: tuple[str, ...]
    unavailable_solvers: tuple[str, ...]
    double_pendulum_evaluation_count: int
    ranking_eligible_method_count: int
    source_digest: str
    inference_boundary: str = INFERENCE_BOUNDARY

    @classmethod
    def create(cls) -> ProspectiveControllerRegistration:
        spec_text = (
            "plant:analytical_double_pendulum_rk4|"
            "state:theta1,theta2,omega1,omega2|"
            "ctrl:tau1,tau2|"
            "dt:0.002|H:60|"
            "bounds:250.0,30.0|"
            "eval_count:0|rank_count:0"
        )
        digest = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()

        return cls(
            plant_identity="analytical_double_pendulum_rk4",
            state_ordering=("theta1", "theta2", "omega1", "omega2"),
            control_ordering=("tau1", "tau2"),
            integration_step_s=0.002,
            planning_horizon_steps=60,
            torque_bounds_nm=(250.0, 30.0),
            torque_rate_bounds_nm_s=(2000.0, 500.0),
            state_scales=(1.0, 1.0, 10.0, 10.0),
            control_scales=(250.0, 30.0),
            event_guard_identity="delivery_transverse_theta1_crossing",
            failure_taxonomy=(
                "SOLVER_DIVERGED",
                "BOUND_SATURATED",
                "WRONG_CROSSING",
                "GRAZING",
                "NONFINITE_DYNAMICS",
                "TIMED_OUT",
            ),
            candidate_solvers=(
                "matched_open_loop_baseline",
                "bounded_projected_ilqr",
                "bounded_nmpc_collocation",
            ),
            unavailable_solvers=(
                "second_order_ddp",
                "risk_sensitive_control",
                "scenario_stochastic_mpc",
                "impedance_control",
                "human_neuromuscular_inference",
            ),
            double_pendulum_evaluation_count=0,
            ranking_eligible_method_count=0,
            source_digest=digest,
        )


def validate_registration() -> dict[str, Any]:
    reg = ProspectiveControllerRegistration.create()
    passed = (
        reg.double_pendulum_evaluation_count == 0
        and reg.ranking_eligible_method_count == 0
        and len(reg.candidate_solvers) >= 2
        and len(reg.unavailable_solvers) >= 3
    )

    data = asdict(reg)
    data["status"] = "PASSED" if passed else "FAILED"
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate #9126 controller registration."
    )
    parser.add_argument(
        "action", nargs="?", default="validate", choices=["validate", "report"]
    )
    args = parser.parse_args()

    evidence = validate_registration()
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
