"""Sampled ideal work and update limits use caller-declared total input."""

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.impact_optimization.total_applied_zoh_audit import (
    AppliedEvidenceKind,
    AppliedZOH,
    ZOHLimits,
    audit_total_applied_zoh,
)

pytestmark = pytest.mark.unit


def evidence(**changes):
    obj = AppliedZOH(
        hold_boundaries_s=np.array([0.0, 1.0, 2.0]),
        channel_names=("one", "two"),
        total_applied_nm=np.array([[2.0, 0.0], [2.0, 0.0]]),
        velocity_start_rad_s=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        velocity_end_rad_s=np.array([[1.0, 0.0], [1.0, 0.0]]),
        evidence_kind=AppliedEvidenceKind.TOTAL_APPLIED,
        provenance="caller recorded applied generalized effort",
        mapping_provenance="direct scalar rotational generalized joints",
        effort_units="N m",
        velocity_units="rad/s",
        hold_semantics="left_zoh",
    )
    return replace(obj, **changes)


def limits(**changes):
    obj = ZOHLimits(
        lower_nm=np.array([-10.0, -10.0]),
        upper_nm=np.array([10.0, 10.0]),
        max_update_rate_nm_s=np.array([10.0, 10.0]),
        max_l2_effort_nm2_s=100.0,
        max_positive_work_j=100.0,
        max_abs_signed_work_j=100.0,
    )
    return replace(obj, **changes)


def test_signed_positive_work_are_separate_and_split_zero_crossings():
    audit = audit_total_applied_zoh(evidence(), limits())
    # First hold: power -2→+2, +0.5 J and -0.5 J. Second: +2 J.
    assert audit.positive_work_j == pytest.approx(2.5)
    assert audit.signed_work_j == pytest.approx(2.0)
    assert audit.l2_effort_nm2_s == pytest.approx(8.0)
    assert audit.status == "pass"
    assert not audit.continuous_certified


def test_per_joint_positive_work_does_not_cancel_against_absorption():
    obj = evidence(
        total_applied_nm=np.array([[2.0, -2.0], [2.0, -2.0]]),
        velocity_start_rad_s=np.ones((2, 2)),
        velocity_end_rad_s=np.ones((2, 2)),
    )
    audit = audit_total_applied_zoh(obj, limits(max_positive_work_j=3.0))
    assert audit.signed_work_j == pytest.approx(0.0)
    assert audit.positive_work_j == pytest.approx(4.0)
    assert audit.status == "violation" and "positive_work" in audit.violations


def test_effort_and_update_limits_use_total_applied_including_feedback():
    # Delta may be zero; the given total already includes baseline + feedback.
    obj = evidence(total_applied_nm=np.array([[0.0, 0.0], [9.0, 0.0]]))
    audit = audit_total_applied_zoh(
        obj,
        limits(
            upper_nm=np.array([8.0, 10.0]), max_update_rate_nm_s=np.array([5.0, 10.0])
        ),
    )
    assert audit.status == "violation"
    assert {"torque", "update_rate"} <= set(audit.violations)
    assert audit.peak_update_rate_nm_s[0] == pytest.approx(9.0)


@pytest.mark.parametrize(
    "kind", [AppliedEvidenceKind.COMMAND, AppliedEvidenceKind.DELTA]
)
def test_command_or_delta_cannot_be_relabeled_as_applied(kind):
    with pytest.raises(ValueError, match="total applied"):
        audit_total_applied_zoh(evidence(evidence_kind=kind), limits())


@pytest.mark.parametrize(
    "changes",
    [
        {"hold_boundaries_s": np.array([0.0, 1.0, 1.0])},
        {"total_applied_nm": np.array([[2.0, np.nan], [2.0, 0.0]])},
        {"velocity_start_rad_s": np.ones((1, 2))},
        {"channel_names": ("one", "one")},
        {"provenance": " "},
        {"mapping_provenance": ""},
        {"velocity_units": "deg/s"},
        {"hold_semantics": "linear"},
    ],
)
def test_incomplete_or_malformed_evidence_is_rejected(changes):
    with pytest.raises(ValueError):
        audit_total_applied_zoh(evidence(**changes), limits())


def test_no_clipping_and_no_continuous_slew_certificate():
    obj = evidence(total_applied_nm=np.array([[20.0, 0.0], [20.0, 0.0]]))
    report = audit_total_applied_zoh(obj, limits())
    assert report.status == "violation" and "torque" in report.violations
    assert obj.total_applied_nm[0, 0] == 20.0
    assert not report.continuous_certified


def test_one_hold_does_not_claim_an_observed_update_rate():
    obj = evidence(
        hold_boundaries_s=np.array([0.0, 1.0]),
        total_applied_nm=np.array([[2.0, 0.0]]),
        velocity_start_rad_s=np.array([[1.0, 0.0]]),
        velocity_end_rad_s=np.array([[1.0, 0.0]]),
    )
    report = audit_total_applied_zoh(obj, limits())
    assert report.status == "indeterminate"
    assert report.peak_update_rate_nm_s is None
    assert report.signed_work_j == pytest.approx(2.0)


def test_nonuniform_hold_durations_integrate_without_smearing_jumps():
    obj = evidence(
        hold_boundaries_s=np.array([0.0, 0.25, 1.0]),
        total_applied_nm=np.array([[4.0, 0.0], [-2.0, 0.0]]),
        velocity_start_rad_s=np.ones((2, 2)),
        velocity_end_rad_s=np.ones((2, 2)),
    )
    report = audit_total_applied_zoh(
        obj, limits(max_update_rate_nm_s=np.array([30.0, 30.0]))
    )
    assert report.signed_work_j == pytest.approx(-0.5)
    assert report.positive_work_j == pytest.approx(1.0)
    assert report.negative_work_j == pytest.approx(-1.5)
    assert report.l2_effort_nm2_s == pytest.approx(7.0)
    assert report.peak_update_rate_nm_s[0] == pytest.approx(24.0)
