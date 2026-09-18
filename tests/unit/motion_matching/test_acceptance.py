"""Unit tests for physical acceptance contract and evaluation gates (MS-01, #10322)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    AcceptanceVerdict,
    GateStatus,
    Horizon,
    evaluate,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
FB6_PARITY_REPORT = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "evidence"
    / "fb6_parity"
    / "parity_report.json"
)
RUN102_RECEIPT = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_102"
    / "receipt.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fb6_parity_report_is_rejected() -> None:
    """TDD Step 1: FB-6 parity report outcomes must be rejected on physical & kinematic gates."""
    data = _load_json(FB6_PARITY_REPORT)
    assert "outcomes" in data

    for engine, outcome in data["outcomes"].items():
        verdict = evaluate(outcome, horizon=Horizon.G3)
        assert isinstance(verdict, AcceptanceVerdict)
        assert verdict.is_physically_accepted is False, (
            f"Expected {engine} outcome in FB-6 parity report to be rejected, but passed"
        )
        # Check that failing gates include whole, contact_force, penetration
        failing_gate_names = {
            g.name for g in verdict.gates if g.status == GateStatus.FAILED
        }
        assert (
            "whole_marker_rmse_m" in failing_gate_names or "whole" in failing_gate_names
        )
        assert (
            "max_normal_force_n" in failing_gate_names
            or "contact_force" in failing_gate_names
        )
        assert (
            "max_penetration_m" in failing_gate_names
            or "penetration" in failing_gate_names
        )


def test_run102_g1_terminal_fails_only_terminal() -> None:
    """TDD Step 2: Run 102 receipt under Horizon G1 passes all gates except terminal."""
    receipt = _load_json(RUN102_RECEIPT)
    verdict = evaluate(receipt, horizon=Horizon.G1)

    assert isinstance(verdict, AcceptanceVerdict)
    assert verdict.is_physically_accepted is False

    failed = [g for g in verdict.gates if g.status == GateStatus.FAILED]
    assert len(failed) == 1
    assert failed[0].name in ("terminal", "terminal_marker_rmse_m")
    assert (
        failed[0].measured is not None and failed[0].measured > 0.035
    )  # 40.3 mm > 35 mm


def test_missing_field_fails_closed() -> None:
    """TDD Step 3: Missing required gate fields fail closed with explicit reason."""
    receipt = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.030,
            "club_marker_rmse_m": 0.040,
            "pelvis_yaw_rmse_rad": 0.02,
        },
        # Missing dynamics / contact / closure
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    assert verdict.is_physically_accepted is False
    missing_gates = [g for g in verdict.gates if g.status == GateStatus.MISSING]
    assert len(missing_gates) > 0
    assert any("max_normal_force" in g.name for g in missing_gates)


def test_thresholds_are_frozen() -> None:
    """TDD Step 4: AcceptanceGates is a frozen dataclass."""
    gates = AcceptanceGates()
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        gates.g1_whole_rmse_m = 0.05  # type: ignore[misc]


def test_open_loop_replay_gate_passes_when_bounded() -> None:
    """Open-loop replay with declared tolerance and bounded drift passes gate."""
    receipt = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.020,
            "club_marker_rmse_m": 0.030,
            "pelvis_yaw_rmse_rad": 0.02,
        },
        "contact_audit": {
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.004,
        },
        "open_loop_replay": {
            "integrator": "rk45",
            "rtol": 1e-6,
            "drift_m": 0.120,
        },
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    gate_names = {g.name: g for g in verdict.gates}
    assert "open_loop_replay" in gate_names
    assert gate_names["open_loop_replay"].status == GateStatus.PASSED


def test_open_loop_replay_fails_on_excessive_drift() -> None:
    """Open-loop replay fails closed when forward drift exceeds allowable threshold."""
    receipt = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.020,
            "club_marker_rmse_m": 0.030,
            "pelvis_yaw_rmse_rad": 0.02,
        },
        "open_loop_replay": {
            "integrator": "rk45",
            "rtol": 1e-6,
            "drift_m": 0.940,  # 940 mm > 500 mm threshold
        },
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    gate_names = {g.name: g for g in verdict.gates}
    assert "open_loop_replay" in gate_names
    assert gate_names["open_loop_replay"].status == GateStatus.FAILED
    assert (
        "0.940" in gate_names["open_loop_replay"].reason
        or "940" in gate_names["open_loop_replay"].reason
    )


def test_collocation_defect_gate_evaluates_correctly() -> None:
    """Collocation defect gate passes when <= 5mm and fails when exceeding threshold."""
    receipt_pass = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "collocation_defect": {
            "max_defect_m": 0.002,  # 2 mm <= 5 mm
            "mean_defect_m": 0.0005,
        },
    }
    verdict_pass = evaluate(receipt_pass, horizon=Horizon.G1)
    gate_names = {g.name: g for g in verdict_pass.gates}
    assert "collocation_defect" in gate_names
    assert gate_names["collocation_defect"].status == GateStatus.PASSED

    receipt_fail = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "collocation_defect": {
            "max_defect_m": 0.015,  # 15 mm > 5 mm
        },
    }
    verdict_fail = evaluate(receipt_fail, horizon=Horizon.G1)
    gate_names_fail = {g.name: g for g in verdict_fail.gates}
    assert "collocation_defect" in gate_names_fail
    assert gate_names_fail["collocation_defect"].status == GateStatus.FAILED


def test_stabilized_replay_gate_evaluates_correctly() -> None:
    """Stabilized replay gate validates low-gain PD tracking marker RMSE."""
    receipt = {
        "shared_metrics": {"whole_marker_rmse_m": 0.020},
        "stabilized_replay": {
            "tracking_kp": 400.0,
            "tracking_kd": 40.0,
            "whole_marker_rmse_m": 0.025,  # 25 mm <= 40 mm
        },
    }
    verdict = evaluate(receipt, horizon=Horizon.G1)
    gate_names = {g.name: g for g in verdict.gates}
    assert "stabilized_replay" in gate_names
    assert gate_names["stabilized_replay"].status == GateStatus.PASSED


def test_g2_and_g3_fail_closed_if_well_posed_artifacts_missing() -> None:
    """Horizon G2 and G3 require all three well-posed artifacts to pass."""
    receipt = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.020,
            "club_marker_rmse_m": 0.030,
            "pelvis_yaw_rmse_rad": 0.02,
        },
        "contact_audit": {
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.004,
            "max_closure_residual_m": 0.002,
        },
    }
    verdict_g2 = evaluate(receipt, horizon=Horizon.G2)
    assert verdict_g2.is_physically_accepted is False
    missing_names = {g.name for g in verdict_g2.gates if g.status == GateStatus.MISSING}
    assert "open_loop_replay" in missing_names
    assert "collocation_defect" in missing_names
    assert "stabilized_replay" in missing_names
