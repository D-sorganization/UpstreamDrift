"""Regression tests on canonical driver seeds receipt (#13109).

In #12814 (toe-out reseed), driver IK closure regressed to 7.57 mm (above the
5 mm gate) and impact-phase weight fraction fell to 0.046 BW. #13102 restored
closure to 3.86 mm while keeping trail-foot toe-out near 20 deg.

This suite locks those gates against the committed receipt without re-running
the full MuJoCo pipeline in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    GateResult,
    GateStatus,
    _evaluate_ground_and_closure,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence/ground_support"
RECEIPT_PATH = EVIDENCE / "anthro_driver_seeds" / "receipt.json"

#: Observed at #12814 (commit 72a4f96cdb) on the committed driver seeds receipt.
REGRESSED_CLOSURE_M = 0.007574219117303761
#: Impact-phase minimum weight fraction at #12814; v2 baseline was 0.107 BW.
REGRESSED_IMPACT_WEIGHT_MIN_BW = 0.04634830140447866
V2_IMPACT_WEIGHT_MIN_BW = 0.107


def _receipt() -> dict[str, Any]:
    assert RECEIPT_PATH.is_file(), f"Receipt file missing: {RECEIPT_PATH}"
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_driver_seeds_closure_residual_passes_gate() -> None:
    """Committed driver seeds trajectory IK closure residual must not exceed 5 mm."""
    receipt = _receipt()
    gates = AcceptanceGates()
    max_closure_m = receipt["ik"]["closure_error_max_m"]

    assert max_closure_m <= gates.max_closure_residual_m, (
        f"Driver closure residual {max_closure_m * 1e3:.2f} mm exceeds "
        f"gate threshold {gates.max_closure_residual_m * 1e3:.2f} mm (#13109)"
    )
    address_closure_m = receipt["address"]["calibrated"]["closure_error_m"]
    assert address_closure_m <= gates.max_closure_residual_m, (
        f"Driver address closure {address_closure_m * 1e3:.2f} mm exceeds "
        f"gate threshold {gates.max_closure_residual_m * 1e3:.2f} mm"
    )


def test_driver_seeds_acceptance_block_records_passed_closure() -> None:
    """The acceptance block in the receipt must explicitly record passed closure."""
    receipt = _receipt()
    acceptance_gates = receipt.get("acceptance", {}).get("gates", [])
    closure_gates = [
        g for g in acceptance_gates if g.get("name") == "max_closure_residual_m"
    ]
    assert len(closure_gates) == 1, "Missing max_closure_residual_m gate in receipt"
    closure_gate = closure_gates[0]
    assert closure_gate["status"] == "passed"
    assert closure_gate["measured"] <= closure_gate["threshold"]


def test_driver_seeds_trail_foot_toe_out_within_target_window() -> None:
    """Trail foot toe-out must remain at 20 deg +/- 2 deg under style progression."""
    receipt = _receipt()
    assert receipt.get("foot_progression_mode") == "style"
    foot_progression = receipt.get("address", {}).get("foot_progression", {})
    assert foot_progression.get("enabled") is True
    assert foot_progression.get("source") == "style"

    feet = foot_progression.get("feet", {})
    right_foot = feet.get("right", {})
    assert right_foot.get("role") == "trail"
    assert right_foot.get("target_source") == "style"
    assert right_foot.get("target_deg") == pytest.approx(20.0, abs=0.5)

    model_deg = right_foot.get("model_deg")
    assert model_deg is not None
    assert 18.0 <= model_deg <= 22.0, (
        f"Trail-foot toe-out {model_deg:.2f} deg outside 20 +/- 2 deg range (#13109)"
    )

    left_foot = feet.get("left", {})
    assert left_foot.get("role") == "lead"
    assert left_foot.get("target_source") == "capture"


def test_driver_seeds_pre_finish_weight_fractions() -> None:
    """Swing phases through impact maintain vertical force >= 0.20 BW."""
    receipt = _receipt()
    dynamics = receipt.get("dynamics", {})
    weight_fraction = dynamics.get("weight_fraction", {})
    by_phase = weight_fraction.get("by_phase", {})

    for phase in ("address", "backswing", "downswing", "impact"):
        phase_stats = by_phase.get(phase, {})
        min_bw = phase_stats.get("min")
        assert min_bw is not None, f"Missing min weight fraction for {phase}"
        assert min_bw >= AcceptanceGates().weight_fraction_min, (
            f"Phase {phase} min weight fraction {min_bw:.3f} BW below "
            f"{AcceptanceGates().weight_fraction_min:.2f} BW gate"
        )

    backswing_to_1s = dynamics.get("backswing_to_1s", {})
    assert backswing_to_1s.get("weight_fraction_min", 0.0) >= (
        AcceptanceGates().weight_fraction_min
    )


def test_driver_seeds_impact_weight_fraction_not_worse_than_v2() -> None:
    """Impact-phase support must stay at or above the v2 baseline (0.107 BW)."""
    receipt = _receipt()
    impact_min = receipt["dynamics"]["weight_fraction"]["by_phase"]["impact"]["min"]
    assert impact_min >= V2_IMPACT_WEIGHT_MIN_BW, (
        f"Impact min weight fraction {impact_min:.3f} BW regressed below "
        f"v2 baseline {V2_IMPACT_WEIGHT_MIN_BW:.3f} BW (#13109)"
    )
    assert impact_min > REGRESSED_IMPACT_WEIGHT_MIN_BW


def test_closure_evaluator_rejects_historical_regression() -> None:
    """evaluate() flags the #12814 closure residual (7.57 mm) as FAILED."""
    gates = AcceptanceGates()
    assert gates.max_closure_residual_m < REGRESSED_CLOSURE_M

    mock_receipt = {
        "ik": {"closure_error_max_m": REGRESSED_CLOSURE_M},
    }
    gate_results: list[GateResult] = _evaluate_ground_and_closure(
        mock_receipt,
        gates=gates,
        contact_audit=None,
    )
    closure_results = [g for g in gate_results if g.name == "max_closure_residual_m"]
    assert len(closure_results) == 1
    assert closure_results[0].status == GateStatus.FAILED
    assert "closure residual 7.57 mm > 5.00 mm" in closure_results[0].reason
