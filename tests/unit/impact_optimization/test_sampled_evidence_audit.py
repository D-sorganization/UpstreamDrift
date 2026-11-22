"""Sampled rollout evidence must be complete and explicitly mapped."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.sampled_evidence_audit import (
    JointMappingKind,
    JointSignalBinding,
    RateEvidenceKind,
    SampledMotionEvidence,
    TorqueEvidenceStage,
    audit_sampled_evidence,
)

pytestmark = pytest.mark.unit


def _envelope(*, rom: np.ndarray | None = None) -> PhysiologicalEnvelope:
    return PhysiologicalEnvelope(
        names=("shoulder", "wrist"),
        tau_max_negative_nm=np.array([10.0, 20.0]),
        tau_max_positive_nm=np.array([12.0, 18.0]),
        rom_deg=np.array([[-90.0, 120.0], [-45.0, 60.0]]) if rom is None else rom,
        vel_max_deg_s=np.array([300.0, 500.0]),
        assumed_torque_slots=("wrist",),
        assumed_velocity_slots=(),
    )


def _evidence(**changes) -> SampledMotionEvidence:
    values = {
        "times_s": np.array([0.0, 0.1, 0.2]),
        "expected_times_s": np.array([0.0, 0.1, 0.2]),
        "stage_ids": ("step-0", "step-1", "step-2"),
        "expected_stage_ids": ("step-0", "step-1", "step-2"),
        "coordinate_names": ("shoulder_q", "wrist_q"),
        "coordinate_deg": np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]),
        "rate_names": ("shoulder_rate", "wrist_rate"),
        "rate_deg_s": np.array([[0.0, 0.0], [100.0, -200.0], [200.0, -300.0]]),
        "actuator_names": ("shoulder_motor", "wrist_motor"),
        "total_applied_actuator_torque_nm": np.array(
            [[0.0, 0.0], [8.0, -15.0], [11.0, 17.0]]
        ),
        "expected_actuator_names": ("shoulder_motor", "wrist_motor"),
        "rate_kind": RateEvidenceKind.DIRECT_RECORDED_RATE,
        "rate_provenance": "simulator recorded generalized joint rates",
        "torque_stage": TorqueEvidenceStage.TOTAL_POST_SATURATION,
        "mapping_provenance": "fixture scalar map v1",
        "torque_provenance": "simulator applied-input history",
        "limit_provenance": {
            f"{joint}:{kind}": f"source for {joint} {kind}"
            for joint in ("shoulder", "wrist")
            for kind in ("rom", "velocity", "torque")
        },
        "bindings": (
            JointSignalBinding(
                "shoulder",
                "shoulder_q",
                "shoulder_rate",
                "shoulder_motor",
                JointMappingKind.DIRECT_SCALAR_GENERALIZED_JOINT,
                "fixture channel binding shoulder",  # fmt: skip
            ),
            JointSignalBinding(
                "wrist",
                "wrist_q",
                "wrist_rate",
                "wrist_motor",
                JointMappingKind.DIRECT_SCALAR_GENERALIZED_JOINT,
                "fixture channel binding wrist",  # fmt: skip
            ),
        ),
    }
    values.update(changes)
    return SampledMotionEvidence(**values)


def test_complete_sampled_evidence_passes_and_reports_assumed_limits() -> None:
    report = audit_sampled_evidence(_evidence(), _envelope())
    assert report.status == "pass"
    assert report.sampled_only and not report.continuous_certified
    assert report.assumption_limited
    assert report.assumed_limits == (("wrist", "torque"),)
    assert "Euler" in report.scope_note
    assert "caller-asserted" in report.scope_note
    assert report.rate_provenance == "simulator recorded generalized joint rates"
    assert len(report.limit_provenance) == 6


def test_feedback_only_total_torque_exceedance_is_detected() -> None:
    # A baseline of 11 N m plus a 2 N m feedback contribution exceeds the
    # positive 12 N m envelope; audit the supplied total, never the delta.
    evidence = _evidence(
        total_applied_actuator_torque_nm=np.array(
            [[0.0, 0.0], [8.0, -15.0], [13.0, 17.0]]
        )
    )
    report = audit_sampled_evidence(evidence, _envelope())
    assert report.status == "violation"
    assert ("shoulder", "torque", 2, 1.0) in report.violations


@pytest.mark.parametrize(
    "changes",
    [
        {"coordinate_deg": np.array([[0.0, 0.0], [1.0, np.nan], [3.0, 4.0]])},
        {"rate_deg_s": np.array([[0.0, 0.0], [100.0, -200.0]])},
        {"expected_times_s": np.array([0.0, 0.1, 0.2, 0.3])},
    ],
)
def test_nonfinite_or_incomplete_sampled_evidence_is_indeterminate(changes) -> None:
    report = audit_sampled_evidence(_evidence(**changes), _envelope())
    assert report.status == "indeterminate"
    assert report.reasons


def test_stage_only_rom_and_speed_violations_are_found() -> None:
    evidence = _evidence(
        coordinate_deg=np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 61.0]]),
        rate_deg_s=np.array([[0.0, 0.0], [100.0, -200.0], [301.0, -300.0]]),
    )
    report = audit_sampled_evidence(evidence, _envelope())
    assert report.status == "violation"
    assert ("wrist", "rom_upper", 2, 1.0) in report.violations
    assert ("shoulder", "velocity", 2, 1.0) in report.violations


def test_signed_asymmetric_torque_limits_are_used() -> None:
    evidence = _evidence(
        total_applied_actuator_torque_nm=np.array(
            [[0.0, 0.0], [12.1, -20.1], [12.0, 18.0]]
        )
    )
    report = audit_sampled_evidence(evidence, _envelope())
    assert report.status == "violation"
    assert ("shoulder", "torque", 1, pytest.approx(0.1)) in report.violations
    assert ("wrist", "torque", 1, pytest.approx(0.1)) in report.violations


def test_unknown_mapping_missing_actuator_and_missing_bounds_fail_closed() -> None:
    unknown = _evidence(
        bindings=(
            _evidence().bindings[0],
            JointSignalBinding(
                "wrist",
                "wrist_q",
                "wrist_rate",
                "wrist_motor",
                JointMappingKind.COUPLED,
                "unknown",
            ),
        )
    )
    assert audit_sampled_evidence(unknown, _envelope()).status == "indeterminate"
    partial = _evidence(expected_actuator_names=("shoulder_motor",))
    assert audit_sampled_evidence(partial, _envelope()).status == "indeterminate"
    unbounded = _envelope(rom=np.array([[-np.inf, np.inf], [-45.0, 60.0]]))
    assert audit_sampled_evidence(_evidence(), unbounded).status == "indeterminate"


def test_finite_difference_rate_source_is_not_accepted_as_direct_evidence() -> None:
    evidence = _evidence(rate_kind=RateEvidenceKind.FINITE_DIFFERENCE)
    report = audit_sampled_evidence(evidence, _envelope())
    assert report.status == "indeterminate"
    assert any("directly supplied" in reason for reason in report.reasons)


def test_distinct_rk_stages_may_share_a_timestamp_and_are_checked_individually() -> (
    None
):
    report = audit_sampled_evidence(
        _evidence(
            times_s=np.array([0.0, 0.1, 0.1, 0.2]),
            expected_times_s=np.array([0.0, 0.1, 0.1, 0.2]),
            stage_ids=("k0", "k1-mid", "k2-mid", "k3-end"),
            expected_stage_ids=("k0", "k1-mid", "k2-mid", "k3-end"),
            coordinate_deg=np.array([[0.0, 0.0], [1.0, 2.0], [1.0, 61.0], [3.0, 4.0]]),
            rate_deg_s=np.array(
                [[0.0, 0.0], [100.0, -200.0], [100.0, -200.0], [200.0, -300.0]]
            ),
            total_applied_actuator_torque_nm=np.array(
                [[0.0, 0.0], [8.0, -15.0], [8.0, -15.0], [11.0, 17.0]]
            ),
        ),
        _envelope(),
    )
    assert report.status == "violation"
    assert ("wrist", "rom_upper", 2, 1.0) in report.violations


def test_schedule_ids_are_required_to_describe_the_sampled_evidence() -> None:
    report = audit_sampled_evidence(
        _evidence(expected_stage_ids=("step-0", "step-1")), _envelope()
    )
    assert report.status == "indeterminate"
    assert any("stage IDs" in reason for reason in report.reasons)


def test_malformed_limit_provenance_fails_closed_without_raising() -> None:
    report = audit_sampled_evidence(_evidence(limit_provenance=None), _envelope())
    assert report.status == "indeterminate"
    assert any("provenance" in reason for reason in report.reasons)


def test_malformed_numeric_stage_array_fails_closed_without_raising() -> None:
    report = audit_sampled_evidence(
        _evidence(coordinate_deg="not numeric"), _envelope()
    )
    assert report.status == "indeterminate"
    assert any("coordinate stage records" in reason for reason in report.reasons)


def test_duplicate_envelope_names_fail_closed_instead_of_aliasing_first_joint() -> None:
    envelope = replace(_envelope(), names=("shoulder", "shoulder"))
    report = audit_sampled_evidence(_evidence(), envelope)
    assert report.status == "indeterminate"
    assert any("envelope joint names" in reason for reason in report.reasons)
