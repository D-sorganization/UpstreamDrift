"""Register a matched nonlinear-controller comparison without running it.

This prospective contract freezes comparability, qualification, execution,
and inference rules before a nonlinear optimizer sees the evaluation grid.
It is protocol evidence, not controller-performance evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REPORT_PATH = ARTICLE / "data/nonlinear_controller_comparison_registration.json"
ENVIRONMENT_LOCK_PATH = ROOT / "requirements.lock"
SCHEMA_VERSION = "proximal-distal-nonlinear-controller-registration/v2"

PARENT_PATHS = (
    Path(
        "docs/research/proximal_distal_energy_transfer/data/"
        "trajectory_control_authority.json"
    ),
    Path(
        "docs/research/proximal_distal_energy_transfer/data/"
        "bounded_event_reachability.json"
    ),
    Path(
        "docs/research/proximal_distal_energy_transfer/data/"
        "event_topology_channel_matrix.json"
    ),
)
SOURCE_PATHS = (
    Path("src/shared/python/simulation_backends/model_params.py"),
    Path("src/shared/python/simulation_backends/ode_backend.py"),
    Path(
        "src/engines/pendulum_models/python/double_pendulum_model/physics/"
        "double_pendulum.py"
    ),
    Path("scripts/research/proximal_distal_energy/nonlinear_controller_numerics.py"),
    Path("scripts/research/proximal_distal_energy/nonlinear_controller_kernels.py"),
)

FAILURE_TYPES = [
    "retained_event",
    "event_lost",
    "integration_failure",
    "solver_failure",
    "gate_failure",
]
RANKING_RULE = (
    "suppress all controller rankings when any comparability, adequacy, "
    "replay, convergence, optimality, event, or held-out gate fails"
)


def _matched_comparison_contract(
    evaluation_trials: list[dict[str, object]],
) -> dict[str, object]:
    """Return the shared plant, coordinate, event, and pairing contract."""
    return {
        "model_tier": "analytical_double_pendulum",
        "plant_backend": "ode",
        "state_coordinates": [
            "shoulder_angle_rad",
            "wrist_relative_angle_rad",
            "shoulder_rate_rad_s",
            "wrist_relative_rate_rad_s",
        ],
        "control_coordinates": [
            "shoulder_torque_nm",
            "wrist_torque_nm",
        ],
        "state_scales": [3.14159, 3.14159, 10.0, 10.0],
        "control_lower_nm": [-60.0, -15.0],
        "control_upper_nm": [60.0, 15.0],
        "simulation_horizon_steps": 400,
        "simulation_step_s": 0.001,
        "event_definition": (
            "theta_shoulder + theta_wrist_relative = 0, positive crossing"
        ),
        "event_interpolation": "first transverse linear guard crossing",
        "random_stream_rule": (
            "seed 9126 and identical perturbation/noise stream per paired "
            "controller-trial comparison"
        ),
        "evaluation_trial_ids": [str(trial["trial_id"]) for trial in evaluation_trials],
        "evaluation_trials": evaluation_trials,
        "failure_types": FAILURE_TYPES,
        "paired_estimand_rule": (
            "delivery comparisons use only jointly retained evaluation pairs; "
            "event-loss and typed failures remain separately reported"
        ),
        "ranking_rule": RANKING_RULE,
        "no_retuning_rule": (
            "all solver, horizon, scaling, and objective choices freeze after "
            "the disjoint tuning set and before evaluation execution"
        ),
    }


def _objective_program() -> dict[str, object]:
    """Return the frozen primary and prospective secondary objectives."""
    return {
        "primary": {
            "name": "registered_trajectory_tracking",
            "channels": [
                "integrated_scaled_state_tracking_error",
                "integrated_scaled_control_deviation",
                "terminal_scaled_state_tracking_error",
            ],
            "delivery_speed_is_optimized": False,
            "delivery_speed_is_reported_as_outcome": True,
        },
        "secondary": {
            "status": "prospective_after_primary_qualification",
            "channels": [
                "delivery_speed",
                "event_time",
                "effort_proxy",
                "peak_torque_fraction",
                "event_retention",
            ],
            "dominance_rule": (
                "report nondominated held-out points only after every primary "
                "qualification and comparability gate passes"
            ),
        },
    }


def build_registration(root: Path = ROOT) -> dict[str, object]:
    """Build the deterministic, outcome-blind comparison registration."""
    root = root.resolve()
    parents = [_authority(root, path) for path in PARENT_PATHS]
    sources = [_authority(root, path) for path in SOURCE_PATHS]
    environment = root / ENVIRONMENT_LOCK_PATH.relative_to(ROOT)
    evaluation_trials = _evaluation_trials()
    tuning_trials = _tuning_trials()
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "prospective_controller_comparison_registration",
        "evidence_status": "no_controller_outcome_generated",
        "parent_authorities": parents,
        "plant_and_solver_source_authorities": sources,
        "environment_authority": {
            "path": environment.relative_to(root).as_posix(),
            "sha256": _sha256(environment),
            "supported_python_lanes": ["3.11", "3.12"],
        },
        "controller_families": _controller_families(),
        "controller_family_count": 9,
        "matched_comparison_contract": _matched_comparison_contract(evaluation_trials),
        "tuning_set": {
            "purpose": "solver qualification and hyperparameter freezing only",
            "trial_ids": [str(trial["trial_id"]) for trial in tuning_trials],
            "trials": tuning_trials,
            "evaluation_outcomes_blinded_until_freeze": True,
        },
        "tuning_trial_count": len(tuning_trials),
        "evaluation_trial_count": len(evaluation_trials),
        "objective_program": _objective_program(),
        "solver_qualification_gates": [
            "manufactured discrete nonlinear dynamics fixture passes",
            "central derivatives agree with an independent directional difference",
            "every optimization enforces torque bounds during the solve",
            "accepted iterations monotonically reduce the declared objective",
            "cold-start and warm-start solutions satisfy the sensitivity gate",
            "deterministic replay reproduces controls states status and cost",
            "nonfinite dynamics produce a typed failure without a trajectory",
            "canonical plant transport passes all registered step sizes",
            "held-out evaluation executes without hyperparameter changes",
        ],
        "falsifiers": [
            "A controller receives a different plant, trial, bound, horizon, event detector, or random stream.",
            "Evaluation outcomes influence solver tuning or objective weights.",
            "A failed solve, integration, or lost event receives a fabricated terminal score.",
            "Post-hoc clipping is described as an in-solver constraint.",
            "A ranking is emitted after any required gate fails.",
            "A local model result is promoted to participant control or coaching guidance.",
        ],
        "execution_contract": _execution_contract(),
        "controller_evaluation_count": 0,
        "ranking_eligible_method_count": 0,
        "inference_boundary": (
            "This registration creates no controller-performance result. Later "
            "local model outcomes cannot establish global optimality, participant "
            "control, motor intent, anatomical feasibility, passive biological "
            "torque, injury risk, fatigue response, or coaching guidance."
        ),
    }


def validate_registration(
    report: dict[str, object], root: Path = ROOT
) -> dict[str, int]:
    """Fail closed on identity, split, comparability, or ranking drift."""
    if report != build_registration(root):
        raise ValueError("registration differs from deterministic authority")
    parents = report.get("parent_authorities")
    if not isinstance(parents, list) or len(parents) != len(PARENT_PATHS):
        raise ValueError("all current parent authorities are required")
    families = report.get("controller_families")
    if not isinstance(families, list) or len(families) != 9:
        raise ValueError("exactly nine controller families are required")
    if _exact_int(report, "controller_family_count") != len(families):
        raise ValueError("controller family count drifted")
    contract = report.get("matched_comparison_contract")
    tuning = report.get("tuning_set")
    if not isinstance(contract, dict) or not isinstance(tuning, dict):
        raise ValueError("matched and tuning contracts must be objects")
    evaluation = contract.get("evaluation_trial_ids")
    tuning_ids = tuning.get("trial_ids")
    if (
        not isinstance(evaluation, list)
        or len(evaluation) != 24
        or len(set(evaluation)) != 24
    ):
        raise ValueError("evaluation set must contain 24 unique trials")
    if not isinstance(tuning_ids, list) or set(tuning_ids) & set(evaluation):
        raise ValueError("tuning and evaluation trials must be disjoint")
    if _exact_int(report, "evaluation_trial_count") != len(evaluation):
        raise ValueError("evaluation trial count drifted")
    if _exact_int(report, "tuning_trial_count") != len(tuning_ids):
        raise ValueError("tuning trial count drifted")
    if contract.get("failure_types") != FAILURE_TYPES:
        raise ValueError("typed failure authority drifted")
    if contract.get("ranking_rule") != RANKING_RULE:
        raise ValueError("fail-closed ranking rule drifted")
    execution = report.get("execution_contract")
    if not isinstance(execution, dict) or execution.get("maximum_workers") != 1:
        raise ValueError("single-worker checkpointed execution is required")
    evaluations = _exact_int(report, "controller_evaluation_count")
    eligible = sum(item.get("eligible_for_ranking") is True for item in families)
    if evaluations != 0 or eligible != 0:
        raise ValueError("prospective registration cannot rank controllers")
    return {
        "controller_family_count": len(families),
        "evaluation_trial_count": len(evaluation),
        "ranking_eligible_count": eligible,
    }


def _controller_families() -> list[dict[str, object]]:
    reference_status = "reference_pending_current_parent_matched_execution"
    prospective = "prospective_pending_current_parent_qualification"
    return [
        _family("registered_open_loop", "simple_baseline", reference_status),
        _family(
            "perfect_state_ltv", "finite_horizon_linear_feedback", reference_status
        ),
        _family(
            "delayed_noisy_ltv",
            "finite_horizon_linear_feedback_adverse_observation",
            reference_status,
        ),
        _family("zero_command_killswitch", "negative_control", reference_status),
        _family(
            "bounded_nmpc_collocation",
            "direct_collocation_nonlinear_mpc",
            "not_implemented",
        ),
        _family(
            "first_order_ilqr",
            "iterative_linear_quadratic_regulation",
            prospective,
        ),
        _family(
            "second_order_ddp", "differential_dynamic_programming", "not_implemented"
        ),
        _family(
            "risk_sensitive_control",
            "risk_sensitive_optimal_feedback",
            "not_implemented",
        ),
        _family(
            "scenario_stochastic_mpc",
            "scenario_based_stochastic_mpc",
            "not_implemented",
        ),
    ]


def _family(name: str, class_name: str, status: str) -> dict[str, object]:
    return {
        "name": name,
        "class": class_name,
        "status": status,
        "eligible_for_ranking": False,
    }


def _evaluation_trials() -> list[dict[str, object]]:
    directions = (
        ("angle_common", [1.0, 1.0, 0.0, 0.0]),
        ("angle_relative", [1.0, -1.0, 0.0, 0.0]),
        ("rate_common", [0.0, 0.0, 1.0, 1.0]),
        ("rate_relative", [0.0, 0.0, 1.0, -1.0]),
    )
    trials: list[dict[str, object]] = []
    for radius in (0.025, 0.05, 0.10):
        radius_name = str(radius).replace("0.", "r0p")
        for direction_name, direction in directions:
            for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
                trials.append(
                    {
                        "trial_id": (
                            f"eval_{direction_name}_{sign_name}_{radius_name}"
                        ),
                        "scaled_state_perturbation": [
                            sign * radius * value for value in direction
                        ],
                    }
                )
    return trials


def _tuning_trials() -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    directions = (
        ("diag_all", [1.0, 1.0, 1.0, 1.0]),
        ("diag_cross", [1.0, -1.0, -1.0, 1.0]),
        ("angle_rate_split", [1.0, 1.0, -1.0, -1.0]),
        ("alternating", [1.0, -1.0, 1.0, -1.0]),
    )
    for direction_name, direction in directions:
        for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
            trials.append(
                {
                    "trial_id": f"tune_{direction_name}_{sign_name}_r0p075",
                    "scaled_state_perturbation": [
                        sign * 0.075 * value for value in direction
                    ],
                }
            )
    return trials


def _execution_contract() -> dict[str, object]:
    return {
        "maximum_workers": 1,
        "checkpoint_granularity": "one controller-trial pair",
        "resume_requires_exact_identity_match": True,
        "identity_fields": [
            "source_revision",
            "registration_sha256",
            "parent_authority_sha256s",
            "environment_lock_sha256",
            "solver_name_and_version",
            "objective_digest",
            "evaluation_trial_id",
        ],
        "launch_authority": (
            "separate operator-approved run after protected parent merge verification"
        ),
        "deskcomputer_policy": "registration_and_bounded_serial_tests_only",
        "campaign_policy": "no campaign is launched by this registration command",
    }


def _authority(root: Path, relative_path: Path) -> dict[str, str]:
    path = root / relative_path
    return {
        "path": relative_path.as_posix(),
        "sha256": _sha256(path),
    }


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
        report = build_registration(ROOT)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_registration(report, ROOT), indent=2))


if __name__ == "__main__":
    main()
