"""Contracts for subject-scaled bilateral closed-contact inverse kinematics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.spatial_full_body import prescribed_state
from scripts.research.proximal_distal_energy.subject_scaled_closed_contact import (
    ClosedContactConfig,
    engineering_joint_bounds,
    run_closed_contact_feasibility_atlas,
    solve_closed_contact_configuration,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    SyntheticSubjectProfile,
    build_subject_scaled_model,
    default_synthetic_profiles,
)

pytestmark = pytest.mark.scientific
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _representative_model():
    profile = SyntheticSubjectProfile("male-mid", 1.75, 75.0, "M")
    return build_subject_scaled_model(profile)


def test_closed_contact_config_and_joint_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="closure_tolerance_m"):
        ClosedContactConfig(closure_tolerance_m=0.0)
    with pytest.raises(ValueError, match="regularization_weight"):
        ClosedContactConfig(regularization_weight=-1.0)

    model, _ = _representative_model()
    lower, upper = engineering_joint_bounds(model)
    assert lower.shape == upper.shape == (model.nq,)
    assert np.all(lower < upper)
    assert np.all(np.isneginf(lower[model.club_dof_indices]))
    assert np.all(np.isposinf(upper[model.club_dof_indices]))


def test_solver_closes_both_contacts_without_moving_the_club() -> None:
    model, metadata = _representative_model()
    q_reference, _, _ = prescribed_state(model, 0.20)
    solution = solve_closed_contact_configuration(
        model,
        q_reference=q_reference,
        grip_span_m=0.18,
        hand_contact_local_x_m=metadata["hand_contact_local_x_m"],
    )

    assert solution.solver_converged
    assert solution.feasible
    assert solution.contact_closed
    assert solution.joint_limits_satisfied
    assert solution.collision_free
    assert np.max(solution.hand_to_grip_distance_m) <= 5.0e-4
    np.testing.assert_array_equal(
        solution.q[model.club_dof_indices], q_reference[model.club_dof_indices]
    )
    assert solution.minimum_joint_limit_margin_rad > 0.05
    assert solution.minimum_collision_clearance_m >= 0.0
    assert solution.constraint_jacobian_rank == 6


def test_unreachable_grip_span_is_retained_as_adverse_control() -> None:
    model, metadata = _representative_model()
    q_reference, _, _ = prescribed_state(model, 0.20)
    solution = solve_closed_contact_configuration(
        model,
        q_reference=q_reference,
        grip_span_m=2.0,
        hand_contact_local_x_m=metadata["hand_contact_local_x_m"],
    )

    assert not solution.feasible
    assert not solution.contact_closed
    assert np.max(solution.hand_to_grip_distance_m) > 0.05


def test_small_closed_contact_atlas_reports_every_feasibility_gate() -> None:
    profiles = (
        SyntheticSubjectProfile("female-short", 1.55, 24.0 * 1.55**2, "F"),
        SyntheticSubjectProfile("male-tall", 1.95, 24.0 * 1.95**2, "M"),
    )
    # This test only checks array shapes/metadata (below), not solver
    # convergence or feasibility outcomes -- see #9204 (the `scientific`
    # marker tier is not deselected from the default lane, so slow
    # research solves compete with unit tests for a bounded CI budget).
    # The default ClosedContactConfig chases ftol=xtol=gtol=1e-11, a relative
    # tolerance about 7 orders of magnitude tighter than the 5e-4 m
    # closure_tolerance_m gate that actually determines feasibility.
    # Finite-difference verification confirms the analytic jacobian is
    # correct (relative error ~2e-10), and hand-to-grip closure is
    # already within the gate by iteration ~10-20; the extra iterations
    # (up to the default max_nfev=1000, empirically ~200-500 per solve)
    # are scipy's trust-region optimizer chasing near-flat regularization
    # (least-norm tie-breaking among the redundant reduced-tree DOF) long
    # past the point of any engineering-relevant improvement. Across this
    # test's 12 solves that inflated wall time enough to trip CI's
    # per-test watchdog on a contended runner (main's `tests (3.12)`
    # job hung inside this exact call stack). A tightly bounded,
    # loosened-tolerance config keeps this shape/metadata check fast and
    # deterministic without touching the module's published default
    # (which the committed evidence bundle and the single-solve
    # convergence tests below intentionally keep at full precision).
    fast_config = ClosedContactConfig(
        solver_tolerance=1.0e-6, maximum_function_evaluations=60
    )
    record, arrays = run_closed_contact_feasibility_atlas(
        profiles=profiles,
        grip_spans_m=np.array([0.12, 0.24]),
        time_s=np.array([0.00, 0.12, 0.24]),
        config=fast_config,
    )

    assert record["schema_version"] == "subject-scaled-closed-contact/v1"
    assert record["design"]["case_count"] == 4
    assert record["design"]["time_sample_count"] == 3
    assert arrays["feasible"].shape == (4, 3)
    assert arrays["hand_to_grip_distance_m"].shape == (4, 3, 2)
    assert arrays["minimum_joint_limit_margin_rad"].shape == (4, 3)
    assert arrays["minimum_collision_clearance_m"].shape == (4, 3)
    assert arrays["adjacent_configuration_change_rad"].shape == (4, 2)
    assert np.all(np.isfinite(arrays["hand_to_grip_distance_m"]))
    assert np.all(arrays["constraint_jacobian_rank"] <= 6)
    assert record["claim_status"]["human_strategy"] == "untested"
    assert record["limitations"][0].startswith("Joint limits")


def test_committed_closed_contact_evidence_and_figure_are_current() -> None:
    record = json.loads(
        (DATA_DIR / "subject_scaled_closed_contact.json").read_text(encoding="utf-8")
    )
    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as arrays:
        assert arrays["feasible"].shape == (18, 13)
        assert np.all(arrays["feasible"])
        assert np.max(arrays["hand_to_grip_distance_m"]) <= 5.0e-4
        assert np.min(arrays["minimum_joint_limit_margin_rad"]) > 0.0
        assert np.min(arrays["minimum_collision_clearance_m"]) > 0.0
        assert np.unique(arrays["constraint_jacobian_rank"]).tolist() == [6]
        for case_index, profile_index in enumerate(arrays["case_profile_index"]):
            model, _ = build_subject_scaled_model(
                default_synthetic_profiles()[int(profile_index)]
            )
            for time_index, sample_time in enumerate(arrays["time_s"]):
                q_reference, _, _ = prescribed_state(model, float(sample_time))
                np.testing.assert_allclose(
                    arrays["solution_q"][
                        case_index, time_index, model.club_dof_indices
                    ],
                    q_reference[model.club_dof_indices],
                    atol=1.0e-12,
                    rtol=1.0e-12,
                )
    for relative, expected in record["source_sha256"].items():
        assert (
            hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest() == expected
        )
    for suffix in ("pdf", "svg"):
        figure = (
            REPO_ROOT
            / "docs/research/proximal_distal_energy_transfer/figures"
            / f"fig_subject_scaled_closed_contact.{suffix}"
        )
        assert figure.stat().st_size > 5_000
