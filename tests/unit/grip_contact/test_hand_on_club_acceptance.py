"""Unit tests for numeric hand-on-club acceptance gates (#13998)."""

from __future__ import annotations

import pytest

from src.shared.python.grip_contact.hand_on_club_acceptance import (
    MAX_PALM_TO_GRIP_GAP_MM,
    MAX_TRAIL_HAND_GAP_MM,
    MAX_WRIST_ATTACHMENT_RESIDUAL_MM,
    MIN_LEAD_FINGER_WRAP_DEG,
    REPORT_ENGINES,
    HandOnClubFixture,
    HandWrenchSample,
    RenderContactSheetSpec,
    default_engine_hand_on_club_reports,
    evaluate_engine_hand_on_club,
)

pytestmark = pytest.mark.unit


def test_default_qualified_template_passes_all_acceptance_gates() -> None:
    report = evaluate_engine_hand_on_club(HandOnClubFixture(engine="mujoco"))
    assert report.passed is True and report.status == "pass"
    assert report.failed_gates == ()
    assert (
        report.palm_to_grip_gap_max_mm is not None
        and report.palm_to_grip_gap_max_mm <= MAX_PALM_TO_GRIP_GAP_MM
    )
    assert (
        report.lead_finger_wrap_deg is not None
        and report.lead_finger_wrap_deg >= MIN_LEAD_FINGER_WRAP_DEG
    )
    assert report.lead_v_azimuth_error_deg == pytest.approx(0.0)
    assert report.lead_thumb_azimuth_error_deg == pytest.approx(0.0)
    assert (
        report.trail_hand_gap_max_mm is not None
        and report.trail_hand_gap_max_mm <= MAX_TRAIL_HAND_GAP_MM
    )
    assert (
        report.wrist_attachment_residual_max_mm is not None
        and report.wrist_attachment_residual_max_mm <= MAX_WRIST_ATTACHMENT_RESIDUAL_MM
    )
    assert report.wrench_feasible is True and report.render_evidence_compliant is True


def test_floating_hand_10mm_and_90deg_roll_error_both_fail_report() -> None:
    floating = evaluate_engine_hand_on_club(
        HandOnClubFixture(engine="mujoco", extra_palm_gap_mm=10.0)
    )
    assert floating.passed is False and "palm_to_grip_gap" in floating.failed_gates
    assert (
        floating.palm_to_grip_gap_max_mm is not None
        and floating.palm_to_grip_gap_max_mm >= 10.0
    )

    rolled = evaluate_engine_hand_on_club(
        HandOnClubFixture(engine="mujoco", roll_error_deg=90.0)
    )
    assert rolled.passed is False and "lead_thumb_and_v_azimuth" in rolled.failed_gates
    assert rolled.lead_v_azimuth_error_deg == pytest.approx(90.0)


def test_each_engine_reports_through_same_function_and_unavailable_is_never_zero() -> (
    None
):
    reports = default_engine_hand_on_club_reports()
    assert tuple(reports) == REPORT_ENGINES
    for engine, rep in reports.items():
        assert rep.as_document()["engine"] == engine
        if not rep.available:
            assert rep.status == "unavailable" and rep.passed is False
            assert bool(rep.reason.strip())
            assert all(
                v is None
                for v in (
                    rep.palm_to_grip_gap_max_mm,
                    rep.lead_finger_wrap_deg,
                    rep.lead_v_azimuth_error_deg,
                    rep.lead_thumb_azimuth_error_deg,
                    rep.trail_hand_gap_max_mm,
                    rep.wrist_attachment_residual_max_mm,
                    rep.wrench_feasible,
                    rep.render_evidence_compliant,
                )
            )


def test_wrench_wrist_and_render_contract_gates_detect_violations() -> None:
    w_sample = HandWrenchSample(normal_force_n=120.0, grip_force_limit_n=80.0)
    bad_wrench = evaluate_engine_hand_on_club(
        HandOnClubFixture(
            engine="pinocchio",
            wrench_by_hand={"lead": w_sample, "trail": HandWrenchSample()},
        )
    )
    assert not bad_wrench.passed and "wrench_feasibility" in bad_wrench.failed_gates
    bad_wrist = evaluate_engine_hand_on_club(
        HandOnClubFixture(
            engine="drake", wrist_attachment_residual_mm={"lead": 1.0, "trail": 4.2}
        )
    )
    assert (
        not bad_wrist.passed and "wrist_attachment_residual" in bad_wrist.failed_gates
    )
    bad_render = evaluate_engine_hand_on_club(
        HandOnClubFixture(
            engine="opensim", render_spec=RenderContactSheetSpec(overlays_off=False)
        )
    )
    assert not bad_render.passed and "render_contact_sheet" in bad_render.failed_gates


@pytest.mark.xfail(
    strict=True,
    reason="UD #13998 / #13158: Rajagopal trail-hand gap (14.2 mm at top) > 5.0 mm until #13158 lands.",
)
def test_rajagopal_trail_hand_gap_series_within_five_mm_gate() -> None:
    report = evaluate_engine_hand_on_club(
        HandOnClubFixture(
            engine="opensim", trail_hand_gap_series_mm=(1.2, 14.2, 3.8, 6.4)
        )
    )
    assert report.passed is True
