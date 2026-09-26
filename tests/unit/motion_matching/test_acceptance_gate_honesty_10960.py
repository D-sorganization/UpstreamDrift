"""Unit tests for acceptance.py gate honesty and status semantics (#10960 P1-11 + P2)."""

from __future__ import annotations

import pytest

from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    AcceptanceVerdict,
    GateResult,
    GateStatus,
    Horizon,
    _evaluate_dual_terminal_disclosure,
    _evaluate_normal_contact_force,
    evaluate,
    is_verdict_accepted,
)

pytestmark = [pytest.mark.unit]


def test_contact_audit_with_two_window_fit_notes_fails_force_gate() -> None:
    """A contact-audit receipt with 'notes': 'two_window_fit' and missing max normal force still gets a FAILED max_normal_force_n gate."""
    receipt = {
        "notes": "two_window_fit",
        "contact_audit": {
            "max_penetration_m": 0.002,
        },
    }
    gates = _evaluate_normal_contact_force(
        receipt, AcceptanceGates(), receipt.get("contact_audit")
    )
    assert len(gates) == 1
    assert gates[0].name == "max_normal_force_n"
    assert gates[0].status == GateStatus.FAILED
    assert gates[0].reason == "missing max normal force in contact audit"

    verdict = evaluate(
        {
            "shared_metrics": {
                "whole_marker_rmse_m": 0.020,
                "early_marker_rmse_m": 0.010,
                "terminal_marker_rmse_m": 0.020,
                "club_marker_rmse_m": 0.030,
                "pelvis_yaw_rmse_rad": 0.02,
            },
            "notes": "two_window_fit",
            "contact_audit": {
                "max_penetration_m": 0.002,
            },
        },
        horizon=Horizon.G1,
    )
    force_gates = [g for g in verdict.gates if g.name == "max_normal_force_n"]
    assert len(force_gates) == 1
    assert force_gates[0].status == GateStatus.FAILED
    assert force_gates[0].reason == "missing max normal force in contact audit"


def test_native_fit_lane_gets_not_applicable_force_gate() -> None:
    """A receipt with lane == "native" gets a NOT_APPLICABLE force gate with non-empty reason, and does not by itself make acceptance fail."""
    receipt = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.020,
            "club_marker_rmse_m": 0.030,
            "pelvis_yaw_rmse_rad": 0.02,
        },
        "lane": "native",
        "open_loop_replay": {
            "integrator": "rk45",
            "rtol": 1e-6,
            "drift_m": 0.018,
        },
        "max_penetration_m": 0.002,
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    force_gates = [g for g in verdict.gates if g.name == "max_normal_force_n"]
    assert len(force_gates) == 1
    assert force_gates[0].status == GateStatus.NOT_APPLICABLE
    assert force_gates[0].reason != ""
    assert "native kinematic-fit lane has no contact audit" in force_gates[0].reason
    assert verdict.is_physically_accepted is True


def test_dual_terminal_disclosure_emits_disclosed_status() -> None:
    """Dual-terminal disclosure with all three fields -> DISCLOSED, measured is None."""
    receipt = {
        "require_dual_terminal_metrics": True,
        "terminal_full_marker_rmse_m": 0.030,
        "terminal_body_excluding_head_rmse_m": 0.025,
        "terminal_head_cluster_rmse_m": 0.028,
    }
    gates = _evaluate_dual_terminal_disclosure(receipt)
    assert len(gates) == 1
    assert gates[0].name == "dual_terminal_disclosure"
    assert gates[0].status == GateStatus.DISCLOSED
    assert gates[0].measured is None
    assert gates[0].reason == "terminal full/body/head metrics disclosed"


def test_verdict_with_only_disclosed_or_not_applicable_not_accepted() -> None:
    """A verdict whose only gates are DISCLOSED/NOT_APPLICABLE is NOT accepted (needs >= 1 PASSED)."""
    gates = (
        GateResult(
            name="max_normal_force_n",
            status=GateStatus.NOT_APPLICABLE,
            threshold=2400.0,
            reason="native kinematic-fit lane has no contact audit",
        ),
        GateResult(
            name="dual_terminal_disclosure",
            status=GateStatus.DISCLOSED,
            threshold=1.0,
            measured=None,
            unit="match",
            reason="terminal full/body/head metrics disclosed",
        ),
    )
    assert is_verdict_accepted(gates) is False

    verdict = AcceptanceVerdict(
        horizon=Horizon.G1,
        is_physically_accepted=is_verdict_accepted(gates),
        status="PASSED" if is_verdict_accepted(gates) else "REJECTED",
        gates=gates,
        qualification_note="Physical or kinematic thresholds violated",
    )
    assert verdict.is_physically_accepted is False
    assert verdict.status == "REJECTED"


def test_evaluate_with_only_disclosed_or_not_applicable_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end evaluate() with only DISCLOSED / NOT_APPLICABLE gates returns rejected verdict."""
    monkeypatch.setattr(
        "src.shared.python.motion_matching.acceptance._evaluate_marker_rmse",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "src.shared.python.motion_matching.full_marker_terminal.evaluate_full_marker_terminal_disclosure",
        lambda *args, **kwargs: [],
    )
    receipt = {
        "lane": "native",
        "require_dual_terminal_metrics": True,
        "terminal_full_marker_rmse_m": 0.030,
        "terminal_body_excluding_head_rmse_m": 0.025,
        "terminal_head_cluster_rmse_m": 0.028,
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    assert len(verdict.gates) > 0
    assert all(
        g.status in (GateStatus.DISCLOSED, GateStatus.NOT_APPLICABLE)
        for g in verdict.gates
    )
    assert verdict.is_physically_accepted is False
    assert verdict.status == "REJECTED"


@pytest.mark.parametrize("lane", ["native_crossval", "not_native", "native_fit_v2", ""])
def test_lane_substring_does_not_skip_force_gate(lane: str) -> None:
    """Only the exact native-fit lanes are NOT_APPLICABLE; look-alikes are gated."""
    receipt = {"lane": lane, "contact_audit": {"max_penetration_m": 0.002}}
    gates = _evaluate_normal_contact_force(
        receipt, AcceptanceGates(), receipt["contact_audit"]
    )
    assert [g.status for g in gates] == [GateStatus.FAILED]
