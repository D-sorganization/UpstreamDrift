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
