"""Preregister a current-main distributed event-attribution smoke study.

The registration is prospective protocol evidence.  It freezes a small,
single-worker matrix against the protected evaluator merged by PR #9306 and
contains no executed outcome or scientific promotion authority.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRATION_PATH = ARTICLE / "data/articulated_distributed_smoke_registration.json"
SCHEMA_VERSION = "proximal-distal-articulated-distributed-smoke-registration/1.0.0"
EVALUATOR_REVISION = "f98bf7b382083322c609bfd7d680e4e82d71aed8"
EVALUATOR_TREE = "e04e7357bdb32a2e966549bd61535c38c3874504"
INPUT_PATH = Path(
    "docs/research/proximal_distal_energy_transfer/data/"
    "subject_scaled_closed_contact.npz"
)
EVALUATOR_PATHS = (
    Path("scripts/research/proximal_distal_energy/articulated_contact_events.py"),
    Path(
        "scripts/research/proximal_distal_energy/"
        "articulated_distributed_event_attribution.py"
    ),
    Path("scripts/research/proximal_distal_energy/articulated_distributed_forward.py"),
    Path("scripts/research/proximal_distal_energy/articulated_distributed_grip.py"),
    Path("scripts/research/proximal_distal_energy/articulated_forward_attribution.py"),
    Path("scripts/research/proximal_distal_energy/articulated_forward_integration.py"),
    Path("scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py"),
)
ENGINES = ("mujoco", "pinocchio")
TIME_STEPS_S = (0.001, 0.0005, 0.00025)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority(root: Path, relative: Path) -> dict[str, object]:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"missing registered authority: {relative.as_posix()}")
    return {
        "path": relative.as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _case_id(engine: str, time_step_s: float) -> str:
    step_microseconds = round(time_step_s * 1_000_000)
    return f"event_probe_{engine}_dt_{step_microseconds:04d}us"


def _case(engine: str, time_step_s: float) -> dict[str, object]:
    case_id = _case_id(engine, time_step_s)
    return {
        "case_id": case_id,
        "engine": engine,
        "time_step_s": time_step_s,
        "source_case_index": 0,
        "source_sample_index": 6,
        "initial_velocity_policy": "all_generalized_velocities_zero",
        "initial_club_displacement_m": 0.001,
        "initial_club_velocity_m_s": -0.8,
        "checkpoint_policy": "atomic_per_case",
        "checkpoint_path": (
            "docs/research/proximal_distal_energy_transfer/data/"
            f"articulated_distributed_smoke_checkpoints/{case_id}.npz"
        ),
    }


def _cases() -> list[dict[str, object]]:
    return [_case(engine, step) for engine in ENGINES for step in TIME_STEPS_S]


def _event_contract() -> dict[str, object]:
    return {
        "supported_event_kinds": ["opening", "reattachment"],
        "state_path_model": "linear_generalized_state_interpolant",
        "active_state_rule": "strictly_positive_signed_gap",
        "simultaneous_event_rule": "one_duplicate_time_boundary_per_event_group",
        "gap_tolerance_m": 1.0e-10,
        "time_tolerance_s": 1.0e-12,
        "maximum_root_iterations": 80,
        "discrete_event_impulse_policy": "exact_zero_for_compliant_transition",
        "discrete_event_work_policy": "exact_zero_for_compliant_transition",
        "prohibited_inferences": [
            "friction_limit_entry_or_exit",
            "static_stick_or_slip_transition",
            "discrete_impact_from_compliant_transition",
        ],
    }


def _acceptance_gates() -> dict[str, object]:
    return {
        "identity": [
            "evaluator_revision_and_tree_match_protected_pr_9306",
            "all_evaluator_source_hashes_match",
            "input_path_hash_and_bytes_match",
            "native_engine_versions_match_the_registered_environment",
        ],
        "case_completeness": {
            "registered_case_count": 6,
            "retain_every_case_and_typed_failure": True,
            "missing_or_duplicate_case_fails": True,
        },
        "trajectory": {
            "all_retained_arrays_finite": True,
            "signed_gap_and_active_state_parity_required": True,
            "required_event_kinds_per_case": ["opening", "reattachment"],
            "unsupported_event_kind_fails": True,
            "maximum_absolute_gap_residual_m": 1.0e-10,
            "maximum_final_bracket_width_s": 1.0e-12,
            "quadrature_may_cross_event_boundary": False,
        },
        "force_and_attribution": {
            "maximum_pointwise_generalized_force_closure_residual": 1.0e-12,
            "maximum_absolute_discrete_event_impulse": 0.0,
            "maximum_absolute_discrete_event_work_j": 0.0,
            "momentum_and_work_closure": "finite_and_retained_not_promoted",
            "signed_shares": "retain_without_clipping",
            "undefined_ratio_rule": "suppress_below_registered_denominator_floor",
        },
        "comparison": {
            "same_trajectory_attribution_only": True,
            "cross_engine_event_timing": "report_without_equivalence_claim",
            "time_step_refinement": "report_without_post_hoc_tolerance_change",
            "counterfactual_eligibility": "requires_separate_registration",
        },
    }


def build_registration(root: Path = ROOT) -> dict[str, object]:
    """Return the deterministic, outcome-blind smoke registration."""
    root = root.resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "prospective_current_main_pipeline_smoke",
        "execution_status": "not_started",
        "evidence_status": "prospective_no_smoke_outcome",
        "registration_status": "frozen_before_first_execution",
        "evaluator_authority": {
            "repository": "D-sorganization/UpstreamDrift",
            "pull_request": 9306,
            "revision": EVALUATOR_REVISION,
            "tree": EVALUATOR_TREE,
            "source_authorities": [_authority(root, path) for path in EVALUATOR_PATHS],
        },
        "input_authority": _authority(root, INPUT_PATH),
        "qualified_environment": {
            "operating_system": "linux_x86_64",
            "supported_python": "3.11.x_or_3.12.x",
            "mujoco_distribution": "mujoco",
            "mujoco_version": "3.8.0",
            "pinocchio_distribution": "pin",
            "pinocchio_version": "3.8.0",
            "maximum_workers": 1,
            "thread_limits": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
        "study_design": {
            "purpose": "runtime_and_pipeline_smoke_only",
            "duration_s": 0.05,
            "engines": list(ENGINES),
            "time_steps_s": list(TIME_STEPS_S),
            "distributed_grip": {
                "station_count_per_hand": 1,
                "station_width_m": 0.0,
                "slack_distance_m": 0.0015,
                "friction_coefficient": 0.0,
            },
            "cases": _cases(),
            "execution_order": "case_list_order",
            "checkpoint_policy": "atomic_per_case",
            "aggregate_policy": "only_after_all_cases_are_retained",
        },
        "event_contract": _event_contract(),
        "acceptance_gates": _acceptance_gates(),
        "killswitches": [
            "source_or_environment_identity_drift",
            "active_state_and_signed_gap_disagreement",
            "active_transition_without_a_bracketed_root",
            "gap_evaluator_shape_or_endpoint_mismatch",
            "unsupported_or_invented_event_kind",
            "missing_duplicate_time_event_alignment",
            "missing_duplicate_or reordered_case",
            "nonfinite_trace_force_or_attribution_value",
            "pointwise_generalized_force_closure_failure",
        ],
        "retained_outcomes": [],
        "promotion_eligible": False,
        "promotion_authority": "none_from_smoke_execution",
        "inference_boundary": (
            "This smoke can qualify current-main execution, event retention, and "
            "same-trajectory numerical bookkeeping for the declared synthetic "
            "probe only. It cannot establish a causal counterfactual, stateful "
            "friction, equipment calibration, anatomical force allocation, "
            "biological passivity, human behavior, injury risk, or coaching guidance."
        ),
    }


def validate_registration(
    registration: Mapping[str, Any], root: Path = ROOT
) -> dict[str, object]:
    """Fail closed on identity, scope, case, or prospective-status drift."""
    expected = build_registration(root)
    if dict(registration) != expected:
        raise ValueError("registration differs from deterministic authority")
    design = expected["study_design"]
    if not isinstance(design, dict):  # pragma: no cover - deterministic guard
        raise ValueError("study_design must be a mapping")
    cases = design["cases"]
    if not isinstance(cases, list):  # pragma: no cover - deterministic guard
        raise ValueError("study cases must be a list")
    return {
        "case_count": len(cases),
        "engine_count": len(ENGINES),
        "time_step_count": len(TIME_STEPS_S),
        "promotion_eligible": False,
    }


def registered_smoke_cases(
    registration: Mapping[str, Any], root: Path = ROOT
) -> tuple[dict[str, Any], ...]:
    """Return validated case records in their frozen execution order."""
    validate_registration(registration, root)
    design = registration["study_design"]
    if not isinstance(design, Mapping):  # pragma: no cover - validated above
        raise ValueError("study_design must be a mapping")
    cases = design["cases"]
    if not isinstance(cases, list):  # pragma: no cover - validated above
        raise ValueError("study cases must be a list")
    return tuple(dict(case) for case in cases)


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate"))
    return parser


def main() -> None:
    """Write or validate the prospective registration."""
    action = _parser().parse_args().action
    if action == "write":
        REGISTRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRATION_PATH.write_text(
            json.dumps(build_registration(), indent=2) + "\n", encoding="utf-8"
        )
        print(REGISTRATION_PATH)
        return
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_registration(registration), indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "EVALUATOR_REVISION",
    "REGISTRATION_PATH",
    "build_registration",
    "registered_smoke_cases",
    "validate_registration",
]
